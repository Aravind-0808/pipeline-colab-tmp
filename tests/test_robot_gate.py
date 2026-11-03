"""The robot gate judges how well the arm follows the hand, not how long it takes."""

from dfpipeline.retarget.service import gate


def _metrics(slowdown, error, reachable=0.98, peak=0.9):
    return {"reachable_fraction": reachable, "slowdown": slowdown,
            "peak_limit_ratio": peak, "position_error_median_m": error}


def test_a_clip_the_arm_follows_closely_passes_though_it_is_slower():
    # 77e7146c: refused at 3.1x under the old 3x limit, followed to 6 mm.
    assert gate(_metrics(3.116, 0.006, peak=0.811)) is None


def test_an_arm_that_cannot_follow_the_hand_is_refused_for_that():
    # The clip whose hand was tracked for 0.4 s: 11x, and 16 cm off the path.
    reason = gate(_metrics(10.95, 0.1608, reachable=1.0, peak=1.0))
    assert reason and "16.1 cm" in reason


def test_a_slowdown_far_past_the_net_is_still_refused():
    reason = gate(_metrics(6.0, 0.01))
    assert reason and "6.0x" in reason


def test_the_clips_that_built_today_still_pass():
    assert gate(_metrics(2.368, 0.0118, reachable=0.94, peak=0.981)) is None
    assert gate(_metrics(2.756, 0.0062, peak=0.975)) is None
