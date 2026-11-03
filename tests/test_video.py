"""Frames fitted to a dataset's size keep their shape."""

import subprocess

import numpy as np

from dfpipeline.video import _normalize, ffmpeg_exe


def test_a_wide_clip_is_fitted_into_a_43_dataset_without_being_squashed(tmp_path):
    """A 16:9 take merged onto a 4:3 version was stretched to it. A square in
    the picture must stay square, with black bars where the shapes differ."""
    import cv2

    src = tmp_path / "wide.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "color=white:s=1280x720:d=1",
         "-vf", "drawbox=x=600:y=320:w=80:h=80:color=red:t=fill", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True,
    )
    out = _normalize(src, 30, (540, 720), tmp_path)
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (540, 720)

    ys, xs = np.where((frame[:, :, 2] > 150) & (frame[:, :, 1] < 100))
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert abs(width - height) <= 2, "the square is still a square"
    assert frame[:30].mean() < 20, "padded black above the picture"
