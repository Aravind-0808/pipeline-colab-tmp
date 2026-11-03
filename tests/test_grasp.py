"""The gripper follows whether the hand holds something (dfpipeline/retarget/grasp.py)."""

import numpy as np

from dfpipeline.retarget import grasp

FPS = 30.0
N = 150


def hand_at(x: float, y: float, size: float = 100.0) -> np.ndarray:
    """21 points of a hand whose grip is centred on (x, y), `size` wrist to middle knuckle."""
    rng = np.random.default_rng(1)
    kp = np.tile([x, y], (21, 1)).astype(float) + rng.normal(0, size * 0.25, (21, 2))
    kp[0] = [x, y + size * 0.6]          # wrist
    kp[9] = [x, y + size * 0.6 - size]   # middle knuckle: `size` from the wrist
    return kp


def cup(x: float, y: float, w: float = 60.0):
    return (np.array([x - w / 2, y - w / 2, x + w / 2, y + w / 2]), "cup", 0.9)


def path(t: np.ndarray, start: float, stop: float, t0: int, t1: int) -> np.ndarray:
    """Still at `start` until t0, moving to `stop` by t1, still after."""
    s = np.clip((t - t0) / max(1, t1 - t0), 0, 1)
    return start + (stop - start) * s


def test_a_hand_carrying_a_cup_holds_it():
    t = np.arange(N)
    hx = np.where(t < 40, path(t, 100, 300, 0, 40), path(t, 300, 600, 50, 100))   # reach, pause, carry
    hx = np.where(t > 110, path(t, 600, 800, 110, 140), hx)                      # set down, then leave
    kps = {i: hand_at(hx[i], 200) for i in range(N)}
    cx = np.where(t < 50, 300, np.where(t <= 105, path(t, 300, 600, 50, 100), 600))
    cands = [[cup(cx[i], 200)] for i in range(N)]
    h = grasp.find_holds(kps, cands, N, FPS)
    held = np.where(h.held)[0]
    assert len(held) > 0
    assert 35 <= held[0] <= 55, "grabbed around the pause before the carry"
    assert 95 <= held[-1] <= 125, "let go around the set-down"
    assert not h.held[:30].any() and not h.held[130:].any()


def test_a_hand_passing_over_resting_cups_holds_nothing():
    t = np.arange(N)
    hx = path(t, 100, 800, 0, N)
    kps = {i: hand_at(hx[i], 200) for i in range(N)}
    cands = [[cup(300, 200), cup(500, 200), cup(700, 200)] for _ in range(N)]
    assert not grasp.find_holds(kps, cands, N, FPS).held.any()


def test_a_cup_hidden_by_the_hand_is_still_held():
    t = np.arange(N)
    hx = path(t, 300, 700, 30, 120)
    kps = {i: hand_at(hx[i], 200) for i in range(N)}
    # The detector sees the cup only now and then while it is carried.
    cands = [[cup(hx[i], 200)] if (i < 30 or i % 10 == 0) else [] for i in range(N)]
    h = grasp.find_holds(kps, cands, N, FPS)
    assert h.held[40:110].mean() > 0.9


def test_the_tray_is_not_held():
    t = np.arange(N)
    hx = path(t, 300, 700, 30, 120)
    kps = {i: hand_at(hx[i], 200) for i in range(N)}
    tray = [(np.array([hx[i] - 150, 50, hx[i] + 150, 350]), "tray", 0.9) for i in range(N)]
    assert not grasp.find_holds(kps, [[b] for b in tray], N, FPS).held.any()


def test_the_gripper_opens_and_closes_as_the_hand_does():
    """The thumb-to-index gap, from the hand's widest in the clip (open) to its
    tightest (closed): a hand round a cup is at its tightest, so the gripper is
    shut on it — not 63% open, as it was at the cup's width."""
    frames = np.arange(40)
    gap = np.r_[np.full(10, 0.10), np.linspace(0.10, 0.06, 10), np.full(10, 0.06), np.full(10, 0.10)]
    out = grasp.opening(frames, gap, None)
    assert np.allclose(out[:10], 1.0) and np.allclose(out[20:30], 0.0) and np.allclose(out[30:], 1.0)
    assert np.all(np.diff(out[10:20]) <= 0), "it closes as the hand closes"


def test_the_gripper_is_shut_while_the_hand_holds():
    """Fingers behind a cup are where tracking is worst: a hold is shut outright."""
    frames = np.arange(30)
    gap = np.full(30, 0.09)
    gap[::2] = 0.03                                   # tracking noise
    out = grasp.opening(frames, gap, {"holds": [[10, 20]], "object_width_m": 0.075})
    assert np.all(out[10:21] == 0.0)
    assert out[:10].max() == 1.0 and out[21:].max() == 1.0


def test_a_hand_that_never_opens_or_closes_is_read_against_a_fixed_range():
    out = grasp.opening(np.arange(5), np.full(5, 0.09), None)
    assert np.allclose(out, 1.0), "a steady wide hand is open, not 'its own tightest'"


def test_the_object_width_comes_from_its_box_and_the_hand_distance():
    h = grasp.Holds(held=np.array([True, True, False]), width_px=np.array([100.0, 110.0, np.nan]),
                    contact=np.array([True, True, False]))
    w = grasp.object_width_m(h, {0: 0.5, 1: 0.5}, fx=700.0)
    assert abs(w - 105 * 0.5 / 700) < 1e-6


def test_in_a_handover_the_taker_holds_from_just_before_the_giver_lets_go():
    left = np.zeros(N, bool)
    right = np.zeros(N, bool)
    left[20:80] = True          # the left hand carries the cup to the middle
    right[40:120] = True        # the right hand looks like it holds it from its reach
    holds = {s: grasp.Holds(held=h, width_px=np.full(N, 60.0), contact=h.copy()) for s, h in
             (("left", left), ("right", right))}
    grasp.resolve_handovers(holds, FPS)
    r = np.where(holds["right"].held)[0]
    assert r[0] == 79 - round(grasp.HANDOVER_S * FPS)   # just before the left lets go (its last frame: 79)
    assert r[-1] == 119 and holds["left"].held[20:80].all()   # the giver is untouched
