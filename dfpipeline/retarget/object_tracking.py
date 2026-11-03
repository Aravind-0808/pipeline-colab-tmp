"""Manipulated-object tracking (lightweight, CPU).

Finds and follows the object the hand manipulates, alongside the hand track:

1. Detection: MediaPipe Object Detector (EfficientDet-Lite0, COCO, CPU). Class names are
   unreliable from a top-down phone view (a white cup is often "bowl" / "frisbee"), so boxes
   are used class-agnostically; the majority label is only reported.
2. Filtering: boxes that are the hand/arm (overlap the MediaPipe hand box) or huge are dropped.
3. Track: the object is the compact box the hand interacts with; it is followed frame to frame
   by nearest-centre association with a size-consistency check. Short gaps (e.g. occlusion
   by the hand) are held; the grasp logic below handles carrying.
4. Grasp state: an object is "held" when the hand's pinch point is on it and the object moves
   with the hand; otherwise "resting". Pick and place events come from these state changes,
   so several pick-and-place cycles in one video are all found.

Outputs per frame: bbox (px), centre (px), detected flag, state; plus object label, size in
pixels and an estimated mass (a class prior: mass cannot be measured from video).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dfpipeline.retarget import paths

log = logging.getLogger("hand2robot.object_tracking")

# Resolved per call rather than at import: see dfpipeline/retarget/paths.py.
IGNORE_CLASSES = ["person", "dining table", "bed", "couch", "chair", "tv", "laptop"]
MIN_SCORE = 0.15
MAX_AREA_FRAC = 0.12            # bigger boxes are arms / table regions, not a graspable object
MAX_HAND_OVERLAP = 0.6          # fraction of the object box covered by the hand box -> it's the hand
ASSOC_MAX_DIST_FRAC = 0.12      # max centre jump between frames (fraction of the image diagonal)
ASSOC_MAX_SIZE_RATIO = 2.2
HOLD_GAP_S = 1.0                # keep the last box this long when the detector misses
GRASP_PINCH_MARGIN = 0.35       # pinch point within the object box enlarged by this fraction
GRASP_MIN_S = 0.3               # a hold must last at least this long

# Rough mass priors (kg) for the reported label; reported with the object, never gated on.
MASS_PRIOR_KG = {"cup": 0.25, "bowl": 0.3, "frisbee": 0.25, "bottle": 0.4, "wine glass": 0.2,
                 "vase": 0.4, "mouse": 0.1, "cell phone": 0.18, "remote": 0.15, "apple": 0.15,
                 "orange": 0.15, "banana": 0.12, "book": 0.4, "scissors": 0.08, "spoon": 0.04}
DEFAULT_MASS_KG = 0.25


@dataclass
class ObjectTrack:
    label: str
    bbox: np.ndarray                    # (T,4) x1,y1,x2,y2 px (NaN where unknown)
    detected: np.ndarray                # (T,) bool: box from the detector this frame
    held: np.ndarray                    # (T,) bool: grasped by the hand
    size_px: float                      # median box width (px) while resting
    mass_kg_estimate: float
    picks: list[int] = field(default_factory=list)    # frames where a hold starts
    places: list[int] = field(default_factory=list)   # frames where a hold ends

    @property
    def center(self) -> np.ndarray:
        return np.column_stack([(self.bbox[:, 0] + self.bbox[:, 2]) / 2, (self.bbox[:, 1] + self.bbox[:, 3]) / 2])


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_frac(obj, hand):
    """Fraction of the object box covered by the hand box."""
    if hand is None:
        return 0.0
    x1, y1 = max(obj[0], hand[0]), max(obj[1], hand[1])
    x2, y2 = min(obj[2], hand[2]), min(obj[3], hand[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(_area(obj), 1e-6)


class ObjectTracker:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.candidates: list[list[tuple[np.ndarray, str, float]]] = []
        self.model_path = Path(model_path) if model_path else paths.object_detector()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing object detector: {self.model_path}. Run: python scripts/download_weights.py")

    def _detector(self):
        # On the CPU always: MediaPipe's GPU path aborts the process on the
        # class filter below (dfpipeline/gpu.py).
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        opts = vision.ObjectDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE, max_results=8, score_threshold=MIN_SCORE,
            category_denylist=IGNORE_CLASSES,
        )
        return vision.ObjectDetector.create_from_options(opts)

    def track(
        self,
        video_path: str | Path,
        hand_bboxes: dict[int, list[list[float]]],
        pinch_px: dict[int, np.ndarray],
        fps: float,
        progress_callback=None,
    ) -> ObjectTrack | None:
        """hand_bboxes: frame -> hand boxes; pinch_px: frame -> (2,) thumb/index-tip midpoint (px)."""
        import mediapipe as mp

        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        diag = float(np.hypot(w, h))
        det = self._detector()

        # 1-2. Per-frame candidate boxes (class-agnostic, hand and huge boxes removed)
        cands: list[list[tuple[np.ndarray, str, float]]] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if progress_callback and i % 30 == 0:
                progress_callback(50.0 + 4.0 * i / max(n, 1), f"Stage 1/3: Object tracking ({i}/{n})...")
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            hands = hand_bboxes.get(i) or []
            keep = []
            for d in res.detections:
                bb = d.bounding_box
                box = np.array([bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height], float)
                if _area(box) > MAX_AREA_FRAC * w * h:
                    continue
                if any(_overlap_frac(box, np.array(hb)) > MAX_HAND_OVERLAP for hb in hands) and \
                        not self._pinch_on(box, pinch_px.get(i)):
                    continue
                keep.append((box, d.categories[0].category_name, float(d.categories[0].score)))
            cands.append(keep)
            i += 1
        cap.release()
        det.close()
        # Every frame's boxes, kept: the grasp check (grasp.py) looks at all the
        # objects, not only the one tracked here.
        self.candidates = cands
        n = len(cands)
        if n == 0:
            return None

        # 3. Seed: the compact box the hand interacts with most (pinch on/near it), else the most
        #    persistent compact box.
        seed_frame, seed_box = self._seed(cands, pinch_px)
        if seed_box is None:
            log.warning("Object tracking: no graspable object found.")
            return None

        bbox = np.full((n, 4), np.nan)
        detected = np.zeros(n, bool)
        labels: list[str] = []
        hold_frames = int(HOLD_GAP_S * fps)

        def follow(order):
            last, missed = seed_box.copy(), 0
            for k in order:
                best, best_d = None, None
                for box, lab, sc in cands[k]:
                    d = np.linalg.norm(((box[:2] + box[2:]) - (last[:2] + last[2:])) / 2)
                    ratio = max(_area(box), 1) / max(_area(last), 1)
                    if d < ASSOC_MAX_DIST_FRAC * diag * (1 + missed / max(fps, 1)) and 1 / ASSOC_MAX_SIZE_RATIO < ratio < ASSOC_MAX_SIZE_RATIO:
                        if best_d is None or d < best_d:
                            best, best_d = (box, lab), d
                if best is not None:
                    bbox[k], detected[k] = best[0], True
                    labels.append(best[1])
                    last, missed = best[0], 0
                else:
                    missed += 1
                    if missed <= hold_frames:
                        bbox[k] = last

        follow(range(seed_frame, n))
        follow(range(seed_frame - 1, -1, -1))

        # Fill remaining gaps by interpolation (object occluded by the hand while carried)
        valid = ~np.isnan(bbox[:, 0])
        if valid.sum() >= 2:
            idx = np.arange(n)
            for c in range(4):
                bbox[:, c] = np.interp(idx, idx[valid], bbox[valid, c])

        label = max(set(labels), key=labels.count) if labels else "object"
        held = self._grasp_state(bbox, pinch_px, fps)
        picks = [int(k) for k in np.where(np.diff(held.astype(int)) == 1)[0] + 1]
        places = [int(k) for k in np.where(np.diff(held.astype(int)) == -1)[0] + 1]
        if held[0]:
            picks.insert(0, 0)
        if held[-1]:
            places.append(n - 1)
        rest = ~held & detected
        widths = (bbox[:, 2] - bbox[:, 0])[rest] if rest.any() else (bbox[:, 2] - bbox[:, 0])
        track = ObjectTrack(
            label=label, bbox=bbox, detected=detected, held=held,
            size_px=float(np.median(widths)), mass_kg_estimate=MASS_PRIOR_KG.get(label, DEFAULT_MASS_KG),
            picks=picks, places=places,
        )
        log.info(
            "Object tracking: '%s' detected in %d/%d frames, %d pick(s) at %s s, %d place(s) at %s s",
            label, int(detected.sum()), n, len(picks), [round(p / fps, 1) for p in picks],
            len(places), [round(p / fps, 1) for p in places],
        )
        return track

    @staticmethod
    def _pinch_on(box, pinch, margin: float = GRASP_PINCH_MARGIN) -> bool:
        """Is a pinch point on the box. `pinch` may hold several (both hands): any will do."""
        if pinch is None:
            return False
        mw, mh = (box[2] - box[0]) * margin, (box[3] - box[1]) * margin
        return any(
            box[0] - mw <= p[0] <= box[2] + mw and box[1] - mh <= p[1] <= box[3] + mh
            for p in np.atleast_2d(pinch)
        )

    def _seed(self, cands, pinch_px):
        best, best_score = (0, None), 0.0
        for k, frame_c in enumerate(cands):
            for box, lab, sc in frame_c:
                bw, bh = box[2] - box[0], box[3] - box[1]
                compact = 0.5 < bw / max(bh, 1) < 2.0
                if not compact:
                    continue
                s = sc * (3.0 if self._pinch_on(box, pinch_px.get(k)) else 1.0)
                if s > best_score:
                    best, best_score = (k, box.copy()), s
        return best

    def _grasp_state(self, bbox, pinch_px, fps) -> np.ndarray:
        """Held = pinch on the object AND the object moves with the hand (vs. resting still)."""
        n = len(bbox)
        c = np.column_stack([(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2])
        size = np.nanmedian(bbox[:, 2] - bbox[:, 0])
        on = np.array([self._pinch_on(bbox[k], pinch_px.get(k)) for k in range(n)])
        # Object motion over ~0.3 s relative to its size (camera motion is small vs a carried object)
        win = max(1, int(0.3 * fps))
        moved = np.zeros(n, bool)
        for k in range(n):
            a, b = max(0, k - win), min(n - 1, k + win)
            moved[k] = np.linalg.norm(c[b] - c[a]) > 0.35 * size
        held = on & moved
        # Close small holes, then drop holds shorter than GRASP_MIN_S
        gap = int(0.5 * fps)
        k = 0
        while k < n:
            if held[k]:
                j = k
                while j + 1 < n and (held[j + 1] or (on[j + 1] and np.any(held[j + 1: j + 1 + gap]))):
                    j += 1
                held[k:j + 1] = True
                k = j + 1
            else:
                k += 1
        runs, k = [], 0
        while k < n:
            if held[k]:
                j = k
                while j + 1 < n and held[j + 1]:
                    j += 1
                runs.append((k, j))
                k = j + 1
            else:
                k += 1
        out = np.zeros(n, bool)
        for a, b in runs:
            if (b - a + 1) >= GRASP_MIN_S * fps:
                # extend the hold back to when the pinch first closed on the object
                s = a
                while s > 0 and on[s - 1]:
                    s -= 1
                out[s:b + 1] = True
        return out


# Height priors (m): a top-down phone view shows the object's top, not its height.
HEIGHT_PRIOR_M = {"cup": 0.09, "frisbee": 0.09, "bowl": 0.07, "bottle": 0.2, "wine glass": 0.15,
                  "vase": 0.2, "mouse": 0.04, "cell phone": 0.01, "remote": 0.02, "book": 0.03,
                  "apple": 0.07, "orange": 0.07}
DEFAULT_HEIGHT_M = 0.08


def lift_to_world(
    track: ObjectTrack,
    camera_poses: np.ndarray,
    K: np.ndarray,
    table_normal_cam0: np.ndarray | None,
    table_distance_m: float,
    pinch_world: dict[int, np.ndarray],
) -> dict:
    """3D object trajectory in the world (= frame-0 camera) frame.

    Resting: the box centre (top of the object from above) is intersected with the plane at the
    object's top height, then moved down to its base on the table. Diameter = box width x depth
    / fx. Held: the object moves rigidly with the hand's pinch point (offset fixed at the pick).
    """
    n_frames = len(track.bbox)
    n = np.array([0.0, 0.0, 1.0]) if table_normal_cam0 is None else np.asarray(table_normal_cam0, float)
    n = n / np.linalg.norm(n)                       # points from the camera toward the table
    height = HEIGHT_PRIOR_M.get(track.label, DEFAULT_HEIGHT_M)
    K_inv = np.linalg.inv(K)
    fx = float(K[0, 0])

    base = np.full((n_frames, 3), np.nan)
    diam = []
    centers = track.center
    for k in range(n_frames):
        if track.held[k] or np.isnan(centers[k, 0]):
            continue
        T = camera_poses[min(k, len(camera_poses) - 1)]
        R, C = T[:3, :3], T[:3, 3]
        ray = R @ (K_inv @ np.array([centers[k, 0], centers[k, 1], 1.0]))
        denom = float(n @ ray)
        if abs(denom) < 1e-6:
            continue
        s = (table_distance_m - height - float(n @ C)) / denom      # plane of the object's top
        if s <= 0:
            continue
        top = C + s * ray
        base[k] = top + n * height                                  # down to the table
        depth = float((R.T @ (top - C))[2])                         # depth in this camera frame
        if track.detected[k]:
            # Shorter box side: the longer one includes the handle / spout.
            side = min(track.bbox[k, 2] - track.bbox[k, 0], track.bbox[k, 3] - track.bbox[k, 1])
            diam.append(side * depth / fx)

    def pinch_at(k: int, near: np.ndarray | None) -> np.ndarray | None:
        # Several pinch points (both hands): the one holding is the one nearest the object.
        p = pinch_world.get(k)
        if p is None:
            return None
        p = np.atleast_2d(p)
        if len(p) == 1 or near is None or np.isnan(near[0]):
            return p[0]
        return p[int(np.argmin(np.linalg.norm(p - near, axis=1)))]

    # Held: rigid offset from the pinch point, fixed at each pick
    offset = None
    for k in range(n_frames):
        if track.held[k]:
            if offset is None:
                prev = k - 1
                while prev >= 0 and np.isnan(base[prev, 0]):
                    prev -= 1
                p = pinch_at(k, base[prev] if prev >= 0 else None)
                if prev >= 0 and p is not None:
                    offset = base[prev] - p
            p = pinch_at(k, base[k - 1] if k > 0 else None)
            if offset is not None and p is not None:
                base[k] = p + offset
        else:
            offset = None

    valid = ~np.isnan(base[:, 0])
    if valid.sum() >= 2:
        idx = np.arange(n_frames)
        for c in range(3):
            base[:, c] = np.interp(idx, idx[valid], base[valid, c])
    diameter = float(np.clip(np.median(diam), 0.02, 0.15)) if diam else 0.07
    return {
        "label": track.label,
        "diameter_m": round(diameter, 4),
        "height_m_prior": height,
        "mass_kg_estimate": track.mass_kg_estimate,
        "base_world": base,
        "held": track.held.copy(),
        "picks": track.picks,
        "places": track.places,
    }
