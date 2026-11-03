"""Hand Detection and Tracking Module.

Hand2Robot stage 1 (hand detection):
Uses the lightweight MediaPipe HandLandmarker (CPU) to detect human hands,
maintain track identity, and extract precise 2D/3D kinematic landmarks across video frames.

Outputs per frame:
- frame_idx
- timestamp
- detections: list of HandDetection objects with:
    - hand_side: "right" or "left"
    - bbox: [x1, y1, x2, y2]
    - confidence: float
    - track_id: int
    - keypoints_2d: np.ndarray (21, 2)
    - landmarks_3d_norm: np.ndarray (21, 3)
    - landmarks_world: np.ndarray (21, 3) metric, hand-centered (meters)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dfpipeline.retarget import paths

log = logging.getLogger("hand2robot.hand_detection")

# MediaPipe confidence thresholds. Measured on top-down desk videos: 0.5 finds
# the hand in ~22% of frames, 0.1 in ~74%. False detections at this level are
# filtered downstream by SingleHandTracker and the PnP fit check.
MIN_DETECTION_CONFIDENCE = 0.1
MIN_PRESENCE_CONFIDENCE = 0.1
MIN_TRACKING_CONFIDENCE = 0.1


@dataclass
class HandDetection:
    """Single hand detection instance."""

    hand_side: str  # 'right' or 'left'
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    track_id: int = 0
    keypoints_2d: np.ndarray | None = None  # (21, 2) in pixel coordinates
    landmarks_3d_norm: np.ndarray | None = None  # (21, 3) normalized hand coordinates
    landmarks_world: np.ndarray | None = None  # (21, 3) metric, origin at hand center


class HandDetector:
    """MediaPipe HandLandmarker detector (VIDEO mode) with forward + backward passes.

    MediaPipe's palm detector often fails to *acquire* a hand (e.g. back of the
    hand, arm entering from the frame edge), but once acquired its landmark
    tracker follows it well. Running the video a second time in reverse lets the
    tracker follow the hand back into those gaps (measured: 74% -> 98% coverage).
    """

    def __init__(
        self,
        task_path: str | Path | None = None,
    ) -> None:
        self.task_path = Path(task_path) if task_path else paths.hand_landmarker()
        if not self.task_path.exists():
            raise FileNotFoundError(
                f"Missing model weights: {self.task_path}. Run: python scripts/download_weights.py"
            )
        log.info("MediaPipe Hand Landmarker: %s (VIDEO mode, CPU)", self.task_path)

    def _create_landmarker(self):
        """On the GPU where it runs (dfpipeline/gpu.py), else the CPU."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        from dfpipeline import gpu

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.task_path), delegate=gpu.mediapipe_delegate(str(self.task_path)),
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        return vision.HandLandmarker.create_from_options(options)

    def detect_video(
        self,
        video_path: str | Path,
        fps: float,
        progress_callback=None,
        hands: int = 1,
    ) -> list[list[HandDetection]]:
        """Detect hands in every frame. Returns one (possibly empty) list per frame.

        `hands=2` keeps every hand either pass found (see `merge_passes`); with
        one, the reverse pass only fills frames the forward pass found nothing in.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        diag = float(np.hypot(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0))

        # Forward pass. Frames are kept JPEG-encoded for the reverse pass to bound memory.
        landmarker = self._create_landmarker()
        forward: list[list[HandDetection]] = []
        encoded: list[bytes] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if progress_callback and idx % 20 == 0:
                progress_callback(10.0 + 22.0 * idx / max(total, 1),
                                  f"Stage 1/3: Hand detection, forward pass ({idx}/{total})...")
            forward.append(self._detect(landmarker, frame, idx, int(round(idx * 1000 / fps))))
            encoded.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            idx += 1
        cap.release()
        landmarker.close()
        n = len(forward)

        # Reverse pass: fills only the frames the forward pass missed.
        landmarker = self._create_landmarker()
        backward: list[list[HandDetection]] = [[] for _ in range(n)]
        for k, i in enumerate(range(n - 1, -1, -1)):
            if progress_callback and k % 20 == 0:
                progress_callback(32.0 + 18.0 * k / max(n, 1),
                                  f"Stage 1/3: Hand detection, reverse pass ({k}/{n})...")
            frame = cv2.imdecode(np.frombuffer(encoded[i], np.uint8), cv2.IMREAD_COLOR)
            backward[i] = self._detect(landmarker, frame, i, int(round(k * 1000 / fps)))
        landmarker.close()

        if hands == 2:
            merged = merge_passes(forward, backward, diag)
        else:
            merged = [f if f else b for f, b in zip(forward, backward)]
        n_fwd = sum(bool(f) for f in forward)
        n_all = sum(bool(m) for m in merged)
        log.info(
            "Hand detection: forward %d/%d frames, with reverse pass %d/%d (%.1f%%)",
            n_fwd, n, n_all, n, 100.0 * n_all / max(n, 1),
        )
        return merged

    @staticmethod
    def _detect(landmarker, frame: np.ndarray, frame_idx: int, ts_ms: int) -> list[HandDetection]:
        """Run the landmarker on one BGR frame. Returns [] when no hand is found."""
        import mediapipe as mp

        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            result = landmarker.detect_for_video(mp_image, ts_ms)
        except Exception as exc:
            log.warning("Hand detector failed on frame %d: %s", frame_idx, exc)
            return []

        if not result.hand_landmarks or not result.handedness:
            return []

        world_list = result.hand_world_landmarks or [None] * len(result.hand_landmarks)
        detections: list[HandDetection] = []
        for cat, lms, wlms in zip(result.handedness, result.hand_landmarks, world_list):
            kpts_2d = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
            lms_3d = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            lms_world = (
                np.array([[p.x, p.y, p.z] for p in wlms], dtype=np.float32)
                if wlms is not None
                else None
            )

            x_min, y_min = kpts_2d.min(axis=0)
            x_max, y_max = kpts_2d.max(axis=0)
            pad_x = (x_max - x_min) * 0.15
            pad_y = (y_max - y_min) * 0.15
            box = [
                max(0.0, float(x_min - pad_x)),
                max(0.0, float(y_min - pad_y)),
                min(float(w), float(x_max + pad_x)),
                min(float(h), float(y_max + pad_y)),
            ]

            detections.append(
                HandDetection(
                    hand_side=cat[0].category_name.lower(),
                    bbox=box,
                    confidence=float(cat[0].score),
                    track_id=1,
                    keypoints_2d=kpts_2d,
                    landmarks_3d_norm=lms_3d,
                    landmarks_world=lms_world,
                )
            )
        return detections


class SingleHandTracker:
    """Keeps the pipeline locked onto ONE hand across frames.

    MediaPipe returns hands in a nondeterministic order, so always taking
    detections[0] makes the trajectory flip between the left and right hand.
    This tracker picks the detection closest to the previously tracked wrist,
    rejects implausible teleports, and re-acquires after a long loss.
    """

    def __init__(
        self,
        frame_diag_px: float,
        max_jump_frac: float = 0.20,
        reacquire_after: int = 15,
    ) -> None:
        self.max_jump_px = frame_diag_px * max_jump_frac
        self.reacquire_after = reacquire_after
        self.prev_wrist: np.ndarray | None = None
        self.side: str | None = None
        self.side_votes: dict[str, float] = {}
        self.frames_lost = 0

    def select(self, detections: list[HandDetection]) -> HandDetection | None:
        if not detections:
            self.frames_lost += 1
            return None

        lost_too_long = self.prev_wrist is None or self.frames_lost > self.reacquire_after
        if lost_too_long:
            # (Re)acquire: prefer the locked side, then the most confident hand.
            pool = [d for d in detections if d.hand_side == self.side] or detections
            best = max(pool, key=lambda d: d.confidence)
        else:
            def cost(d: HandDetection) -> float:
                dist = float(np.linalg.norm(d.keypoints_2d[0] - self.prev_wrist))
                # Penalize a handedness change: it is usually a different hand.
                return dist + (0.5 * self.max_jump_px if d.hand_side != self.side else 0.0)

            best = min(detections, key=cost)
            # Allow larger motion after a few missed frames.
            allowed = self.max_jump_px * (1 + self.frames_lost)
            if float(np.linalg.norm(best.keypoints_2d[0] - self.prev_wrist)) > allowed:
                self.frames_lost += 1
                return None

        self.prev_wrist = best.keypoints_2d[0].copy()
        self.frames_lost = 0
        self.side_votes[best.hand_side] = self.side_votes.get(best.hand_side, 0.0) + best.confidence
        # Majority-voted handedness removes per-frame left/right label flicker.
        self.side = max(self.side_votes, key=self.side_votes.get)
        best.hand_side = self.side
        return best


def merge_passes(
    forward: list[list[HandDetection]],
    backward: list[list[HandDetection]],
    frame_diag_px: float,
    same_hand_frac: float = 0.08,
) -> list[list[HandDetection]]:
    """Both passes' hands, per frame, for a two-hand clip.

    With one hand the reverse pass only fills frames the forward one missed.
    With two, a frame where the forward pass found one hand and the reverse
    pass found the other would lose the second, so each hand the reverse pass
    adds is kept unless its wrist is on a hand already there (the same hand,
    seen by both). At most two per frame, the most confident first.
    """
    near = frame_diag_px * same_hand_frac
    merged = []
    for f, b in zip(forward, backward):
        frame = list(f)
        for d in b:
            if all(np.linalg.norm(d.keypoints_2d[0] - e.keypoints_2d[0]) > near for e in frame):
                frame.append(d)
        merged.append(sorted(frame, key=lambda d: -d.confidence)[:2])
    return merged


class _Slot:
    """One hand followed through a two-hand clip."""

    def __init__(self) -> None:
        self.prev_wrist: np.ndarray | None = None
        self.frames_lost = 0
        self.votes: dict[str, float] = {}
        self.wrist_x: list[float] = []

    @property
    def label(self) -> str | None:
        return max(self.votes, key=self.votes.get) if self.votes else None

    def take(self, d: HandDetection) -> None:
        self.prev_wrist = d.keypoints_2d[0].copy()
        self.frames_lost = 0
        self.votes[d.hand_side] = self.votes.get(d.hand_side, 0.0) + d.confidence
        self.wrist_x.append(float(d.keypoints_2d[0][0]))


class TwoHandTracker:
    """Both hands of a two-hand clip, each kept on its own track.

    The one-hand tracker follows one hand and would hop between the two
    whenever their wrists come close — a handover, exactly — so it is not
    used here. Two slots instead, and every frame the hands are given to the
    slots the cheapest way: how far each wrist moved, plus a penalty for a
    change of MediaPipe's left/right label (a slot keeps its label even when
    MediaPipe names both hands the same). A slot's hand may move at most a
    fifth of the frame per frame (more after missed frames); a slot that lost
    its hand for longer takes any hand not already on the other slot.

    Which slot is the left arm is decided at the end, by where the hands
    worked: the one further left in the picture. MediaPipe's own label is not
    used for that — it assumes a mirrored selfie image, and on a back camera
    it is the wrong way round.
    """

    def __init__(self, frame_diag_px: float, max_jump_frac: float = 0.20,
                 reacquire_after: int = 15, same_hand_frac: float = 0.08) -> None:
        self.max_jump_px = frame_diag_px * max_jump_frac
        self.same_hand_px = frame_diag_px * same_hand_frac
        self.reacquire_after = reacquire_after
        self.slots = [_Slot(), _Slot()]

    def _free(self, slot: _Slot) -> bool:
        return slot.prev_wrist is None or slot.frames_lost > self.reacquire_after

    def _cost(self, k: int, d: HandDetection) -> float | None:
        """None: slot k cannot take this hand."""
        slot, other = self.slots[k], self.slots[1 - k]
        if self._free(slot):
            # Free to (re)acquire, at a fixed price — but not the hand the other slot follows.
            if not self._free(other) and np.linalg.norm(d.keypoints_2d[0] - other.prev_wrist) <= \
                    self.max_jump_px * (1 + other.frames_lost):
                return None
            return self.max_jump_px
        dist = float(np.linalg.norm(d.keypoints_2d[0] - slot.prev_wrist))
        if dist > self.max_jump_px * (1 + slot.frames_lost):
            return None
        label = slot.label
        return dist + (0.5 * self.max_jump_px if label is not None and d.hand_side != label else 0.0)

    def select(self, detections: list[HandDetection]) -> list[HandDetection | None]:
        """This frame's hand for each slot (None where its hand is not seen)."""
        dets = sorted(detections, key=lambda d: -d.confidence)[:2]
        best: tuple[int, float, list] = (0, 0.0, [None, None])
        options = [None, *range(len(dets))]
        for a in options:
            for b in options:
                if a is not None and a == b:
                    continue
                pick = [a, b]
                costs = [self._cost(k, dets[i]) for k, i in enumerate(pick) if i is not None]
                if any(c is None for c in costs):
                    continue
                if a is not None and b is not None and np.linalg.norm(
                        dets[a].keypoints_2d[0] - dets[b].keypoints_2d[0]) <= self.same_hand_px:
                    continue  # one hand seen twice is not two hands
                score = (len(costs), -sum(costs))
                if score > (best[0], best[1]):
                    best = (score[0], score[1], pick)
        chosen = [dets[i] if i is not None else None for i in best[2]]
        for slot, d in zip(self.slots, chosen):
            if d is None:
                slot.frames_lost += 1
            else:
                slot.take(d)
        return chosen

    def sides(self) -> list[str]:
        """Each slot's arm, "left" or "right", by where its hand worked in the picture."""
        x = [float(np.median(s.wrist_x)) if s.wrist_x else np.inf for s in self.slots]
        return ["left", "right"] if x[0] <= x[1] else ["right", "left"]
