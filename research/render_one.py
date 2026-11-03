"""The videos of one clip's run, in a process of its own (research only).

    python research/render_one.py CLIP RUN_DIR OUT_DIR SO101_DIR|-

On Colab the notebook ran the renderer in its own process for an hour and
wrote not one frame: a hang there costs the whole notebook. Here it costs
this process, which the notebook stops after a time limit. The videos are
written to OUT_DIR (local disk: Drive is slow for a file written in small
pieces) and the notebook copies them to Drive when they are done.

`-` for SO101_DIR draws the stick view instead of MuJoCo's.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import visualize


def main() -> int:
    clip, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    so101 = None if sys.argv[4] == "-" else Path(sys.argv[4])
    started = time.time()
    made = visualize.render(clip, run, f"WiLoR + depth  {clip.name}", so101_dir=so101, out_dir=out)
    print({k: v.name for k, v in made.items()}, f"{time.time() - started:.0f}s", flush=True)
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
