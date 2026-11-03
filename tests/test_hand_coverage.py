"""Hand coverage, measured while the hand was in view — not over the whole clip."""

from dfpipeline.retarget.service import hand_coverage, hand_gate


def test_seconds_before_the_reach_and_after_it_do_not_count():
    # 300 frames; the hand comes in at 60 and leaves after 239, and was
    # reconstructed in 170 of those 180 frames.
    meta = {"total_frames": 300, "reconstructed_frames": 170,
            "hand_first_frame": 60, "hand_last_frame": 239}
    assert hand_coverage(meta) == round(170 / 180, 4)
    assert hand_gate({"hand_coverage": hand_coverage(meta)}) is None


def test_a_hand_lost_while_it_works_still_fails():
    meta = {"total_frames": 300, "reconstructed_frames": 90,
            "hand_first_frame": 60, "hand_last_frame": 239}
    assert hand_coverage(meta) == 0.5
    assert "50%" in hand_gate({"hand_coverage": 0.5})


def test_an_older_bundle_without_the_span_is_measured_over_the_clip():
    meta = {"total_frames": 300, "reconstructed_frames": 150}
    assert hand_coverage(meta) == 0.5


def test_no_reconstruction_is_no_coverage():
    assert hand_coverage({"total_frames": 300}) is None


def test_a_hand_followed_for_a_fraction_of_the_clip_is_refused():
    """A 10.9 s pick and place whose hand was followed for 14 frames came out
    as a 0.4 s episode. The arm then needed 11x the time, which was the error
    anyone saw — about the robot, when the fault was the recording."""
    from dfpipeline.retarget.service import hand_span

    meta = {"total_frames": 327, "reconstructed_frames": 14, "hand_first_frame": 39, "hand_last_frame": 52}
    seconds, share = hand_span(meta, 30.0)
    assert (seconds, share) == (0.47, round(14 / 327, 4))
    reason = hand_gate({"hand_coverage": 1.0, "hand_seconds": seconds, "hand_share": share})
    assert reason and "0.5s" in reason and "4%" in reason


def test_a_hand_followed_through_the_task_passes():
    from dfpipeline.retarget.service import hand_span

    # The good test clip: 407 frames, hand from 28 to 359.
    meta = {"total_frames": 407, "reconstructed_frames": 332, "hand_first_frame": 28, "hand_last_frame": 359}
    seconds, share = hand_span(meta, 29.26)
    assert seconds > 11 and share > 0.8
    assert hand_gate({"hand_coverage": 1.0, "hand_seconds": seconds, "hand_share": share}) is None


def test_an_older_bundle_without_the_span_is_not_refused_for_it():
    from dfpipeline.retarget.service import hand_span

    assert hand_span({"total_frames": 300, "reconstructed_frames": 150}, 30.0) == (None, None)
    assert hand_gate({"hand_coverage": 0.9}) is None
