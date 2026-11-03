"""What a hand detector gives, and keeping each hand's identity across frames.

The detector is WiLoR, on the GPU (hand_wilor.py); this module holds what it
produces and the trackers that follow one or two hands through a clip.

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

import numpy as np

log = logging.getLogger("hand2robot.hand_detection")


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
    # WiLoR's MANO hand surface, to be drawn (research/visualize.py), not
    # measured with: its 778 vertices in pixels, as WiLoR's own camera
    # projects them, and in metres around the wrist (camera axes).
    mesh_uv: np.ndarray | None = None  # (778, 2) float32
    mesh_xyz: np.ndarray | None = None  # (778, 3) float16


class SingleHandTracker:
    """Keeps the pipeline locked onto ONE hand across frames.

    The detector returns hands in a nondeterministic order, so always taking
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
    change of the detector's left/right label (a slot keeps its label even when
    the detector names both hands the same). A slot's hand may move at most a
    fifth of the frame per frame (more after missed frames); a slot that lost
    its hand for longer takes any hand not already on the other slot.

    Which slot is the left arm is decided at the end, by where the hands
    worked: the one further left in the picture. The detector's own label is not
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
