"""Is each hand holding something — frame by frame — and how wide is it.

The gripper was driven by the gap between thumb tip and index tip: 2 cm
closed, 9 cm open. That is a pinch. A hand wrapped around a cup keeps thumb
and fingers a cup's width apart, so it read as open nearly the whole time it
held one (a two-hand dataset: closed in 5–7% of frames), and the fingertips,
hidden by the cup, made it jump. Finger curl alone is no better: a relaxed
empty hand curls too.

So the question asked is the one the gripper needs: is this hand holding an
object? It is when an object is in its grip — inside the box of its thumb and
fingers — and moves with it. A resting object the hand brushes does not move;
an empty hand has nothing in its grip. Per hand, every object the detector
found counts, not one tracked object, so a pick-and-place of many cups works.

What is held comes back with its width, from its box and the hand's distance
from the camera, so the gripper can open to just more than the object and
close to just less (`opening`), rather than swing between its ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The grip: the box of the thumb and finger points, enlarged by this share.
GRIP_PAD = 0.25
#: An object counts as in the grip when this share of its box is inside it.
IN_GRIP = 0.3
#: ... and is no longer than this many hand lengths (wrist to middle knuckle).
MAX_OBJECT_HANDS = 1.6
#: The hand hides what it holds: contact gaps up to this long are bridged.
GAP_FILL_S = 0.5
#: "Moves with the hand": over this window either side, the object and the
#: hand each travel at least MOVE_FRAC of the object's width, the same way.
MOVE_WINDOW_S = 0.3
MOVE_FRAC = 0.35
#: ... while keeping its place in the hand: its offset from the grip changes
#: by less than this share of its width.
STAY_FRAC = 0.5
#: The hand is still (reaching its target, closing, setting down) below this
#: speed: share of its own size per frame, smoothed over a few frames.
STILL = 0.04
#: Setting something down is slow, not still; moving off after is fast.
SETTLE = 0.08
#: Seen this many of its widths from its place in the hand: left behind.
LEFT_BEHIND = 1.0
#: Carries proved less than this apart are one hold.
MERGE_S = 0.8
#: The fingers close this long before the carry starts.
CLOSE_LEAD_S = 0.3
#: A set-down pause counts for at most this long: a hand that rests after
#: putting something down has let go of it.
MAX_SETTLE_S = 1.5
#: Holds shorter than this are touches, not grasps.
MIN_HOLD_S = 0.3

#: The finger and thumb points that make the grip (MediaPipe hand order).
_GRIP_POINTS = list(range(2, 21))


@dataclass
class Holds:
    """One hand through a clip."""

    held: np.ndarray            # (T,) bool: holding an object
    width_px: np.ndarray        # (T,) the held object's width in pixels, NaN when not held
    contact: np.ndarray         # (T,) bool: an object in the grip, held or not


def _grip_box(kp: np.ndarray) -> np.ndarray:
    pts = kp[_GRIP_POINTS]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = (hi - lo) * GRIP_PAD
    return np.concatenate([lo - pad, hi + pad])


def _inside_share(obj: np.ndarray, box: np.ndarray) -> float:
    x1, y1 = max(obj[0], box[0]), max(obj[1], box[1])
    x2, y2 = min(obj[2], box[2]), min(obj[3], box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1e-6, (obj[2] - obj[0]) * (obj[3] - obj[1]))
    return inter / area


def _speed(hand_c: np.ndarray, keypoints: dict[int, np.ndarray], n: int, fps: float) -> np.ndarray:
    """The grip's speed, per frame, as a share of the hand's size; smoothed.
    Frames without the hand are bridged by interpolation (and never "still")."""
    idx = np.arange(n)
    ok = ~np.isnan(hand_c[:, 0])
    if ok.sum() < 2:
        return np.full(n, np.inf)
    c = np.column_stack([np.interp(idx, idx[ok], hand_c[ok, k]) for k in range(2)])
    size = np.median([np.linalg.norm(p[9] - p[0]) for p in keypoints.values()]) or 1.0
    step = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)] / size
    k = max(1, round(0.2 * fps))
    smooth = np.convolve(step, np.ones(k) / k, mode="same")
    return np.where(ok, smooth, np.inf)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] (inclusive) of each run of True."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def find_holds(
    keypoints: dict[int, np.ndarray],
    candidates: list[list[tuple[np.ndarray, str, float]]],
    n_frames: int,
    fps: float,
) -> Holds:
    """When this hand holds an object.

    keypoints: frame -> (21, 2) pixel points of this hand; candidates: per
    frame, the object boxes the detector found (ObjectTracker.candidates).
    """
    contact = np.zeros(n_frames, bool)
    obj_c = np.full((n_frames, 2), np.nan)       # the object in the grip: centre
    obj_w = np.full(n_frames, np.nan)            # and width (shorter side: a handle widens one)
    hand_c = np.full((n_frames, 2), np.nan)      # the grip's centre
    for i in range(min(n_frames, len(candidates))):
        kp = keypoints.get(i)
        if kp is None:
            continue
        box = _grip_box(kp)
        hand_c[i] = kp[_GRIP_POINTS].mean(axis=0)
        hand_px = float(np.linalg.norm(kp[9] - kp[0]))  # wrist to middle knuckle
        best, share = None, IN_GRIP
        for obj, _label, _score in candidates[i]:
            # Bigger than the hand is not in it: the tray it puts things in, the table.
            if max(obj[2] - obj[0], obj[3] - obj[1]) > MAX_OBJECT_HANDS * hand_px:
                continue
            s = _inside_share(obj, box)
            if s >= share:
                best, share = obj, s
        if best is not None:
            contact[i] = True
            obj_c[i] = (best[:2] + best[2:]) / 2
            obj_w[i] = min(best[2] - best[0], best[3] - best[1])

    # Seen by the detector, as opposed to bridged below: only what was seen
    # can prove a carry — a bridged object moves with the hand by construction.
    seen = contact.copy()

    # The hand hides what it holds: bridge short gaps while the hand is there,
    # carrying the object's position along with the hand.
    gap = max(1, round(GAP_FILL_S * fps))
    for a, b in _runs(~contact):
        if a == 0 or b == n_frames - 1 or b - a + 1 > gap:
            continue
        if np.isnan(hand_c[a - 1, 0]) or np.isnan(hand_c[b + 1, 0]):
            continue
        offset = obj_c[a - 1] - hand_c[a - 1]
        for i in range(a, b + 1):
            if not np.isnan(hand_c[i, 0]):
                contact[i] = True
                obj_c[i] = hand_c[i] + offset
                obj_w[i] = obj_w[a - 1]

    # Moving together: between two sightings of the object 0.3–0.9 s apart,
    # in one run of contact, object and hand both travel, and the object keeps
    # its place in the hand. A hand sweeping over resting objects, or reaching
    # for one the other hand carries, moves past them.
    win = max(1, round(MOVE_WINDOW_S * fps))
    moving = np.zeros(n_frames, bool)
    sightings = np.where(seen)[0]
    for k, a in enumerate(sightings):
        for b in sightings[k + 1:]:
            if b - a < win:
                continue
            if b - a > 3 * win or not contact[a:b + 1].all():
                break
            do, dh = obj_c[b] - obj_c[a], hand_c[b] - hand_c[a]
            if np.isnan(dh).any():
                break
            need = MOVE_FRAC * obj_w[a]
            drift = np.linalg.norm((obj_c[b] - hand_c[b]) - (obj_c[a] - hand_c[a]))
            if np.linalg.norm(do) > need and np.linalg.norm(dh) > need and drift < STAY_FRAC * obj_w[a]:
                moving[a:b + 1] = True
            break

    # The object is seen moving with the hand only now and then — the hand
    # covers it, the detector misses it against the tray. Where it is, that is
    # proof of a carry; the hand's own motion gives the rest. A grab is a pause
    # (reach, close the fingers, lift) and so is a set-down (lower, open, leave):
    # a hold runs from the start of the carry, less the time to close, through
    # the set-down pause, to the moment the hand moves off.
    speed = _speed(hand_c, keypoints, n_frames, fps)
    still = speed < STILL
    held = np.zeros(n_frames, bool)
    merge = round(MERGE_S * fps)
    lead, settle = round(CLOSE_LEAD_S * fps), round(MAX_SETTLE_S * fps)
    proof = np.where(moving)[0]
    for group in np.split(proof, np.where(np.diff(proof) > merge)[0] + 1) if len(proof) else []:
        a, b = int(group[0]), int(group[-1])
        # Not before the object was seen in this hand's grip, in one unbroken
        # run: reaching for a cup the other hand carries is not holding it, and
        # a chain of bridged gaps can reach back to some other object.
        first = a
        gap_left = gap
        while first > 0 and gap_left > 0:
            first -= 1
            gap_left = gap if seen[first] else gap_left - 1
        while not seen[first] and first < a:
            first += 1
        while a > first and not still[a - 1] and not np.isnan(hand_c[a - 1, 0]):
            a -= 1                                   # back to where the carry began
        a = max(first, a - lead)
        while b + 1 < n_frames and not still[b + 1] and not np.isnan(hand_c[b + 1, 0]):
            b += 1                                   # on to where the carry ends
        end = b
        # The object's place in the hand when last carried: once it is seen
        # left behind, the hand has let go, however slowly it moves off.
        anchor = next((obj_c[f] - hand_c[f] for f in range(b, max(-1, b - 2 * win), -1) if seen[f]), None)
        while end + 1 < n_frames and speed[end + 1] < SETTLE and end - b < settle:
            f = end + 1
            if anchor is not None and seen[f] and \
                    np.linalg.norm(obj_c[f] - hand_c[f] - anchor) > LEFT_BEHIND * obj_w[f]:
                break                                # the object stayed; the hand is leaving it
            end = f                                  # through the set-down, until the hand moves off
        held[a:end + 1] = True
    for a, b in _runs(held):
        if (b - a + 1) < MIN_HOLD_S * fps:
            held[a:b + 1] = False

    width = np.where(held, obj_w, np.nan)
    return Holds(held=held, width_px=width, contact=contact)


def object_width_m(holds: Holds, depth_m: dict[int, float], fx: float) -> float | None:
    """The held object's width in metres: its box over the camera's focal length,
    at the hand's distance. The median over every held frame; None if never held."""
    vals = [holds.width_px[i] * depth_m[i] / fx for i in np.where(holds.held)[0]
            if i in depth_m and not np.isnan(holds.width_px[i])]
    if not vals:
        return None
    return float(np.clip(np.median(vals), 0.02, 0.15))


# ---- The gripper, from the hand and its holds ----------------------------------------

#: The thumb-to-index gap is read against the clip's own range, these
#: percentiles of it: one wide-open or fully-shut frame of tracking noise must
#: not set the scale.
APERTURE_PERCENTILES = (5, 95)
#: A hand whose gap barely changes over the clip (under this) never opened or
#: closed: it is read against a fixed range instead of its own.
MIN_APERTURE_RANGE_M = 0.015
FIXED_APERTURE_M = (0.02, 0.09)
#: The file a hand's holds are kept in, beside its trajectory.
GRIP_FILE = "grip.json"


def write_grip(folder, holds: Holds, width_m: float | None) -> None:
    """A hand's holds, for the robot stage: [first, last] frames of each, and the object's width."""
    import json
    from pathlib import Path

    runs = [[int(a), int(b)] for a, b in _runs(holds.held)]
    (Path(folder) / GRIP_FILE).write_text(json.dumps({"holds": runs, "object_width_m": width_m}))


def read_grip(folder) -> dict | None:
    import json
    from pathlib import Path

    path = Path(folder) / GRIP_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def opening(frame_idx: np.ndarray, aperture_m: np.ndarray, grip: dict | None) -> np.ndarray:
    """The gripper's opening (0 closed .. 1 fully open) at these video frames,
    as the hand opens and closes.

    The thumb-to-index gap, from the hand's widest in this clip (fully open)
    to its tightest (closed): a hand wrapped round a cup is at its tightest,
    so the gripper is closed on it, as a person driving the arm squeezes it
    shut and the jaws stop at the object. While the hand holds something it
    is closed outright — fingers behind a cup are where tracking is worst.

    It was the object's width over the jaw's span: a cup held at 63%, which
    is a closed hand to a person and an open jaw to the robot and its viewer.
    """
    gap = np.asarray(aperture_m, float)
    lo, hi = np.percentile(gap, APERTURE_PERCENTILES) if len(gap) else FIXED_APERTURE_M
    if hi - lo < MIN_APERTURE_RANGE_M:
        lo, hi = FIXED_APERTURE_M
    out = np.clip((gap - lo) / (hi - lo), 0.0, 1.0)
    for a, b in (grip or {}).get("holds") or []:
        out[(frame_idx >= a) & (frame_idx <= b)] = 0.0
    return out


#: In a handover the taking hand closes this long before the giving hand lets go.
HANDOVER_S = 0.5


def resolve_handovers(holds: dict[str, Holds], fps: float) -> None:
    """Two hands: a hold that begins while the other hand holds — and ends
    while this one still does — is a handover. The taking hand reaches for a
    cup that is moving in the other hand's grip, which looks like holding it;
    it holds it only from just before the other lets go."""
    sides = list(holds)
    if len(sides) != 2:
        return
    lead = round(HANDOVER_S * fps)
    for taker, giver in ((sides[0], sides[1]), (sides[1], sides[0])):
        t, g = holds[taker].held, holds[giver].held
        for a, b in _runs(t):
            if not g[a]:
                continue
            g_start = a
            while g_start > 0 and g[g_start - 1]:
                g_start -= 1
            g_end = a
            while g_end + 1 < len(g) and g[g_end + 1]:
                g_end += 1
            if g_start < a and g_end < b:            # the other held it first, and lets go first
                t[a:max(a, g_end - lead)] = False
