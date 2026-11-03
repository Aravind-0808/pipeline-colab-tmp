"""Fetch the two MediaPipe models the retargeting pipeline needs.

The SO-101 URDF is vendored in `models/so101`
— it is small, Apache-2.0, and a build should not depend on a third-party
URL being up. These two are 14 MB of Google-hosted weights, so they are fetched
instead of committed.

Run at image build time, never on the first job: a fresh container would
otherwise pay for the download inside an export, and fail it if the box is
offline.

    python scripts/download_retarget_weights.py [--dir models]
"""

import argparse
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
log = logging.getLogger("retarget.weights")

MODELS = {
    "efficientdet_lite0.tflite": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
    ),
}


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        log.info("already here: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Downloaded beside the target and moved into place, so an interrupted
    # download cannot leave a truncated file that looks present to paths.py.
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", dest.name)
    urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(dest)
    log.info("%s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="models", help="where the models live")
    args = parser.parse_args()
    for name, url in MODELS.items():
        fetch(url, Path(args.dir) / name)
