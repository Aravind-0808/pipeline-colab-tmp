"""The AI check, on the pipeline: two questions, every frame, or no pass."""

import pytest

from dfpipeline.config import settings
from dfpipeline.validation import schemas, vlm


# ------------------------------------------------------ the two questions


class _Model:
    configured = True

    def __init__(self, reply: dict):
        self.reply = reply

    def chat_json(self, messages, max_tokens=400, temperature=0.0):
        return self.reply


def _ask(reply: dict):
    from PIL import Image

    return vlm.verdict_for_sheet(_Model(reply), Image.new("RGB", (8, 8)), None, "v1")


GOOD = {"task_performed": True, "object_visible": True, "view_obstructed": False,
        "hands_visible": True, "hand_in_frame": True, "hand_clear": True}
ALL_FRAMES = [{"t": t, "hand_fully_visible": True} for t in (0.0, 0.6, 1.2, 1.8, 2.4)]


def test_both_questions_must_pass():
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": ALL_FRAMES,
                    "subtasks": [{"name": "grasp", "start": 0, "end": 2}]})
    assert verdict.verdict == "pass"
    assert [s.name for s in verdict.subtasks] == ["grasp"]


def test_one_frame_without_the_whole_hand_rejects_the_clip():
    """The model's summary said pass; one frame says otherwise, and the frame wins."""
    frames = [*ALL_FRAMES[:3], {"t": 1.8, "hand_fully_visible": False}, ALL_FRAMES[4]]
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"
    (finding,) = [f for f in verdict.findings if f.code == "hand_not_visible"]
    assert "1.8s" in finding.detail
    assert schemas.factors([f.as_dict() for f in verdict.findings], "reject")["tracking"] == "fail"


def test_a_pass_without_per_frame_answers_is_not_a_pass():
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD})
    assert verdict.verdict == "flag"
    assert verdict.subtasks == []


def test_a_pass_it_is_not_sure_of_is_not_a_pass():
    verdict = _ask({"verdict": "pass", "confidence": 0.8, "checks": GOOD, "frames": ALL_FRAMES})
    assert verdict.verdict == "flag"
    assert "80%" in verdict.reason


@pytest.mark.parametrize(
    "check, code, factor",
    [
        ("task_performed", "task_not_performed", "task"),
        ("hands_visible", "hand_not_visible", "tracking"),
        ("hand_in_frame", "hand_out_of_frame", "tracking"),
        ("hand_clear", "hand_unclear", "tracking"),
    ],
)
def test_a_failed_requirement_rejects_even_a_pass(check, code, factor):
    """The model said pass, but answered a requirement no: the clip is rejected,
    and the reason says which of the two questions it failed."""
    verdict = _ask({"verdict": "pass", "confidence": 0.9, "checks": {**GOOD, check: False},
                    "frames": ALL_FRAMES})
    assert verdict.verdict == "reject"
    assert code in [f.code for f in verdict.findings]
    assert verdict.subtasks == []
    factors = schemas.factors([f.as_dict() for f in verdict.findings], "reject")
    assert factors[factor] == "fail"
    other = "tracking" if factor == "task" else "task"
    assert factors[other] == "pass"


def test_a_model_that_cannot_be_reached_says_so():
    from dfpipeline.llm import LlmUnavailable

    class Down:
        configured = True

        def chat_json(self, *a, **k):
            raise LlmUnavailable("gateway down")

    from PIL import Image

    verdict = vlm.safe_verdict_for_sheet(Down(), Image.new("RGB", (8, 8)), None, "v1")
    assert verdict.verdict == "unknown"
    assert [f.code for f in verdict.findings] == [schemas.MODEL_UNREACHABLE]




def test_a_real_clip_on_this_disk_is_read_and_measured(tmp_path, monkeypatch):
    """No stand-ins: ffmpeg reads a real four-second clip from this disk — as
    the check reads a clip in local storage — and the free rules measure it."""
    import subprocess

    import imageio_ffmpeg

    from dfpipeline.validation.judge import judge_clips

    monkeypatch.setattr(settings, "llm_api_base", None)
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=30:duration=4", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    out = judge_clips({
        "clips": [{"video_id": "real", "path": "clip.mp4", "size_bytes": clip.stat().st_size,
                   "kind": "local", "local": str(clip)}],
        "task": {"title": "pick", "description": None, "task_flow": None},
        "thresholds": {"enabled": True, "min_width": 240},
    })
    verdict = out["verdicts"][0]
    assert verdict["facts"] is not None, "the clip should have been read"
    assert verdict["facts"]["decodable"] is True
    assert 3.5 < verdict["facts"]["duration"] < 4.5
    assert verdict["facts"]["width"] == 320
    # Readable, moving, long enough: no rule refuses it, and with no model
    # configured it is left undecided — not rejected.
    assert verdict["verdict"] == "unknown" and verdict["findings"] == []


def test_a_clip_that_cannot_be_opened_is_undecided_not_rejected(tmp_path, monkeypatch):
    """A missing file (a path read from the wrong folder, storage that did not
    answer) says nothing about the recording: it is retried and left
    undecided — never rejected as undecodable, which would delete it."""
    from dfpipeline.validation import judge
    from dfpipeline.validation.judge import judge_clips

    monkeypatch.setattr(settings, "llm_api_base", None)
    monkeypatch.setattr(judge, "INSPECT_BACKOFF_SEC", 0.0)
    out = judge_clips({
        "clips": [{"video_id": "gone", "path": "x.mp4", "size_bytes": 7_000_000,
                   "kind": "local", "local": str(tmp_path / "storage" / "x.mp4")}],
        "task": {"title": "pick", "description": None, "task_flow": None},
        "thresholds": {"enabled": True, "min_width": 240},
    })
    verdict = out["verdicts"][0]
    assert verdict["verdict"] == "unknown"
    assert not any(f["code"] == "undecodable" for f in verdict["findings"] or [])


def test_ffmpeg_that_cannot_open_a_link_is_not_a_corrupt_clip():
    """The same for a link: an HTTP error or a refused connection is not the clip's fault."""
    from dfpipeline import frames

    with pytest.raises(frames.SourceUnavailable):
        frames.probe("http://127.0.0.1:9/clip.mp4")


# ------------------------------------------------------ when the hand may be absent


def _frames(*states):
    return [{"t": float(i), "hand": s} for i, s in enumerate(states)]


def test_no_hand_before_it_reaches_in_or_after_it_leaves_is_fine():
    """An arm reaching for a cup starts outside the picture. Rejecting the clip
    for the frame before it arrived threw away good demonstrations."""
    frames = _frames("absent", "full", "full", "full", "absent")
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "pass"


def test_a_hand_that_vanishes_in_the_middle_rejects_the_clip():
    frames = _frames("full", "full", "absent", "full", "full")
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"
    (finding,) = [f for f in verdict.findings if f.code == "hand_not_visible"]
    assert "2s" in finding.detail


def test_a_cut_off_hand_mid_task_rejects_the_clip():
    frames = _frames("absent", "full", "partial", "full", "absent")
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"


def test_a_hand_partly_in_view_as_it_comes_in_and_goes_out_is_fine():
    """dbb2c12a: "fully visible and sharp throughout", refused for its 1s and
    8s frames — the hand coming in from the edge and leaving."""
    frames = _frames("absent", "partial", "full", "full", "full", "full", "full", "full", "partial", "absent")
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "pass"


def test_a_hand_partly_in_view_for_long_at_the_edge_still_fails():
    # Partial for 4 s before it is ever whole: not a reach-in, a hand at the edge.
    frames = _frames("partial", "partial", "partial", "partial", "full", "full", "absent")
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"
    (finding,) = [f for f in verdict.findings if f.code == "hand_not_visible"]
    assert "0s" in finding.detail and "1s" in finding.detail


def test_a_hand_never_whole_rejects_the_clip():
    frames = _frames("absent", "partial", "partial", "absent")
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"


def test_a_clip_with_no_hand_at_all_is_rejected():
    frames = _frames("absent", "absent", "absent")
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"


def test_an_answer_that_is_neither_full_nor_absent_counts_against_the_clip():
    frames = [{"t": 0.0, "hand": "full"}, {"t": 1.0, "hand": "mostly"}, {"t": 2.0, "hand": "full"}]
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"


# ------------------------------------------------------ by video, or by the grid


class _Recorder:
    """A model that answers the video and the grid differently, and remembers which it saw."""

    configured = True

    def __init__(self, video_reply=None, video_error=None, grid_reply=None):
        self.video_reply, self.video_error, self.grid_reply = video_reply, video_error, grid_reply
        self.seen: list[str] = []

    def chat_json(self, messages, max_tokens=400, temperature=0.0):
        kinds = [part["type"] for part in messages[-1]["content"]]
        if "video_url" in kinds:
            self.seen.append("video")
            if self.video_error:
                raise self.video_error
            return self.video_reply
        self.seen.append("grid")
        return self.grid_reply


PASS = {"verdict": "pass", "confidence": 0.95, "checks": GOOD,
        "frames": _frames("absent", "full", "full", "absent")}
REJECT = {"verdict": "reject", "confidence": 0.95, "checks": {**GOOD, "task_performed": False},
          "frames": _frames("full", "full")}


def _grid():
    from PIL import Image

    return Image.new("RGB", (8, 8))


def test_the_video_is_watched_when_there_is_one():
    model = _Recorder(video_reply=PASS, grid_reply=REJECT)
    verdict = vlm.safe_verdict(model, None, "v1", _grid(), b"avi", 4.0)
    assert model.seen == ["video"]
    assert verdict.verdict == "pass"


def test_a_model_that_will_not_take_the_video_is_asked_with_the_grid():
    """The model server cannot read every video. That is not the clip's fault,
    and must not hold it up: the grid answers instead."""
    from dfpipeline.llm import LlmUnavailable

    model = _Recorder(
        video_error=LlmUnavailable('Error code: 400 - {"message":"Could not open video stream"}'),
        grid_reply=PASS,
    )
    verdict = vlm.safe_verdict(model, None, "v1", _grid(), b"avi", 4.0)
    assert model.seen == ["video", "grid"]
    assert verdict.verdict == "pass"


def test_a_model_that_is_down_is_waited_for_not_asked_twice():
    from dfpipeline.llm import LlmUnavailable

    model = _Recorder(video_error=LlmUnavailable("Connection error: timed out"), grid_reply=PASS)
    verdict = vlm.safe_verdict(model, None, "v1", _grid(), b"avi", 4.0)
    assert model.seen == ["video"]
    assert verdict.verdict == "unknown"
    assert [f.code for f in verdict.findings] == [schemas.MODEL_UNREACHABLE]


def test_without_a_video_the_grid_is_used():
    model = _Recorder(video_reply=REJECT, grid_reply=PASS)
    verdict = vlm.safe_verdict(model, None, "v1", _grid(), None, 4.0)
    assert model.seen == ["grid"]
    assert verdict.verdict == "pass"


def test_the_video_prompt_asks_for_every_second():
    prompt = vlm.TEMPLATE_VIDEO.format(title="t", details="d", duration="10.8", last=11)
    assert "up to 11" in prompt and "one per second" in prompt
    assert "grid" not in prompt.lower()


def test_a_clip_is_turned_into_a_small_mjpeg_video(tmp_path):
    """The model server reads MJPEG, not the H.264 phones record."""
    import subprocess

    import imageio_ffmpeg

    from dfpipeline.video import clip_for_model

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=960x720:rate=30:duration=3", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    data = clip_for_model(str(clip), width=448, fps=4, max_frames=120, duration=3.0)
    assert data[:4] == b"RIFF" and data[8:12] == b"AVI ", "an AVI"
    assert b"MJPG" in data[:4096], "with MJPEG frames"
    assert len(data) < clip.stat().st_size * 4


def test_the_arm_reaching_the_edge_is_not_a_cut_off_hand():
    frames = [{"t": float(i), "edge": "arm only", "hand": "full"} for i in range(4)]
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "pass"


def test_fingers_off_the_edge_fail_even_if_called_full():
    frames = [{"t": 0.0, "edge": "arm only", "hand": "full"},
              {"t": 1.0, "edge": "fingers or palm", "hand": "full"},
              {"t": 2.0, "edge": "arm only", "hand": "full"}]
    verdict = _ask({"verdict": "pass", "confidence": 0.95, "checks": GOOD, "frames": frames})
    assert verdict.verdict == "reject"
