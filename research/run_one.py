"""One clip through one config, in a process of its own (research only).

The notebook runs every (clip, config) through this, each in a fresh process:
MediaPipe, the depth model and WiLoR in one long-lived notebook process ran a
Colab T4 out of memory and the kernel died mid-run. A process per run starts
clean, loads only its own models, and gives everything back when it ends; a
failure is one row, not the notebook.

    python research/run_one.py CLIP CONFIG HANDS OUT_ROOT [TRUTH_CM]

Writes OUT_ROOT/CONFIG/<clip stem>/row.json (and the run's hand/ and robot/).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

CONFIGS = {
    "A_today": {"retarget_hand_model": "mediapipe", "retarget_depth_model": "off"},
    "B_depth": {"retarget_hand_model": "mediapipe", "retarget_depth_model": "moge-2-vitb"},
    "C_wilor_depth": {"retarget_hand_model": "wilor", "retarget_depth_model": "moge-2-vitb"},
}


def main() -> int:
    clip, name, hands, out_root = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    truth = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
    logging.basicConfig(level=logging.WARNING)

    from dfpipeline.config import settings

    for k, v in CONFIGS[name].items():
        setattr(settings, k, v)
    from dfpipeline.retarget import service as rs
    from dfpipeline.retarget.pipeline import run_hand_stage, run_robot_stage

    out = out_root / name / clip.stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    row = {"clip": clip.name, "config": name}
    try:
        cam_file = Path(str(clip) + ".camera.json")
        camera = json.loads(cam_file.read_text()) if cam_file.exists() else None
        t = time.time()
        code = run_hand_stage(clip, out / "hand", hands=hands, camera=camera)
        row.update(hand_ok=code == 0, hand_s=round(time.time() - t, 1))
        if code == 0:
            bundle = json.loads((out / "hand" / "hand_stage.json").read_text())
            perception = bundle.get("perception") or {}
            row.update(rs.read_hand_metrics(out / "hand"))
            row.update({f"p_{k}": v for k, v in perception.items() if k != "notes"})
            row["notes"] = "; ".join(perception.get("notes") or [])
            row["table_distance_m"] = bundle.get("table_distance_m")
            obj = bundle.get("object")
            row["object_width_cm"] = round(100 * obj["diameter_m"], 1) if obj else None
            row["truth_width_cm"] = truth
            t = time.time()
            rcode = run_robot_stage(out / "hand", out / "robot")
            row.update(robot_ok=rcode == 0, robot_s=round(time.time() - t, 1))
            if rcode == 0:
                m = rs.read_metrics(out / "robot")
                row.update({k: m.get(k) for k in ("reachable_fraction", "position_error_median_m", "slowdown")})
    except Exception:  # noqa: BLE001 - reported in the row
        row["error"] = traceback.format_exc()[-800:]
    (out / "row.json").write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps({k: row.get(k) for k in ("clip", "config", "hand_s", "p_lens_source", "error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
