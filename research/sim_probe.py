"""Whether MuJoCo renders here, and how fast: ten frames of the SO-101.

    python research/sim_probe.py SO101_DIR

Prints milliseconds a frame. Run by the notebook with a time limit before
any video: where MuJoCo's EGL hangs or crawls (it wrote nothing in an hour on
one Colab runtime), the videos use the stick view instead.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import visualize


def main() -> int:
    sim = visualize._Sim(Path(sys.argv[1]))
    sim.frame({"joints_rad": [0.0] * 6})
    started = time.perf_counter()
    for i in range(10):
        sim.frame({"joints_rad": [0.1 * i] * 6, "target_pos": [0.2, 0.0, 0.1]})
    print(f"{(time.perf_counter() - started) * 100:.1f}", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
