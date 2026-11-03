"""Human hand video -> SO-101 joint trajectory.

Vendored from Hand2Robot (`hand2robot-1.0`), a completed proof of concept:
MediaPipe hand landmarks -> metric 3D hand -> camera motion from the table
plane -> world fusion -> differential IK onto the SO-101 URDF -> motor-limit
retiming. CPU only; no torch, no GPU, no model weights running here.

Changed on the way in, and nothing else:
  - `src.*` imports became `dfpipeline.retarget.*`
  - the relative model paths became `paths.py`, which also makes the LeRobot
    calibration per-robot rather than a module constant
  - the CLI, the FastAPI review UI, the overlay video and the MuJoCo replay
    are gone: a dataset is the IK trajectory, nothing drawn or simulated

`service.py` is ours: it is what the `retarget.clip` job calls, and it holds
the gate that decides whether a trajectory is fit to train on.
"""
