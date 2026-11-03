# Hand2Robot 2.0: a measured hand, not a guessed one

Branch: `pipeline-optimization` (pipeline, Web, Mobile).
**Status: stage 1 ("the house") is built and tested on the CPU. It is switched off by default
and waits for phases 1–2 on Colab.** R&D only, not commercial: non-commercial research models
are allowed (§9).

## 1. The problem

From one phone video we turn a human hand into robot joint angles. The robot ends up in the
wrong place mostly because of **depth**: how far the hand is from the camera and the table.
Until now every distance was guessed:

| Before | How | Weak point |
|---|---|---|
| Lens | `fx = 0.95 × max(w, h)` for every phone | the hand's depth is `f × size / pixels`, so a focal length 20% off puts the hand 20% off |
| Hand | MediaPipe, 21 points | about 48 mm joint error, against about 15 mm for WiLoR in one study (WristPP) |
| Hand's size | an average adult hand | a big hand reads as near, a small one as far |
| Table | the hand's depth (75th percentile), or **0.5 m** | a guess |
| Robot | our URDF IK and retiming | good: kept |

## 2. Decisions taken

1. **Only the phone video.** Operators do nothing new: no board, no ruler, no scan, no second phone.
2. **Keep the robot half:** IK, gripper, retiming, the dataset build and the app flow.
3. **Measure instead of guess:**
   - the phone's own lens data, saved silently by the app;
   - a GPU depth model on a few frames;
   - optionally a better GPU hand model.
4. **Develop and measure on Colab; deploy on the DGX Spark once.** A smoke test runs there first.
5. **Off by default.** With no setting changed, the pipeline is exactly what it was. Each model is switched on by a setting.
6. **Mass is out of scope.** A camera cannot see weight.

## 3. What is built (stage 1)

```
phone video + camera facts (the app, silently)            ── Mobile CameraFacts.kt
  │
  ├─ 1. Lens        from the phone's Camera2 data; else the depth
  │                 model's estimate; else the old guess          lens.py          [CPU]
  ├─ 2. Up          the phone's gravity sensor, in picture axes   lens.py          [CPU]
  ├─ 3. Hand        WiLoR (the only detector, after the A/B/C test) hand_wilor.py    [GPU]
  ├─ 4. 3D hand     PnP with the real lens (unchanged code)       hand_reconstruction.py
  ├─ 5. Depth       MoGe-2 on ~12 frames: table plane (RANSAC,
  │                 facing up) and the hand's real scale          scene.py         [GPU]
  ├─ 6. Scale       one factor per clip: the hand slides along
  │                 its lines of sight, stays on its pixels       pipeline.py      [CPU]
  ├─ 7. World       camera motion with the real lens and the
  │                 measured table distance and normal            camera_reconstruction.py
  └─ 8. Robot       unchanged                                     retarget_*.py
```

### 3.1 Lens (`lens.py`)
- **What the phone sends** (`camera` on upload):
  - Camera2's description of the lens: focal length in mm, sensor size in mm, pixel array, active array;
  - the factory calibration and distortion, where the phone has them;
  - the physical lens behind a logical camera;
  - sampled capture results: focal length, focus, crop region, zoom ratio, active physical lens, stabilisation;
  - the gravity sensor;
  - the display rotation.
- **The arithmetic is here, not in the app,** so a mistake is fixed with a pipeline deploy, not an app release.
- **The sources, in order:**
  1. the factory calibration, where it is non-zero;
  2. focal length ÷ sensor width × pixel array, then the crop to 16:9 and the scale to the video size;
  3. the depth model's estimate;
  4. the old guess.

  The bundle records which one was used (`lens_source`). A field of view outside 25–130° is rejected as bad data.
- **Up:** the mean gravity reading over the first second, turned from device axes into display axes (Android's remap table for the rotation) and then into camera axes. Only for the back camera.

### 3.2 Depth (`scene.py`)
- **The model:** MoGe-2 (Microsoft, MIT licence, plain PyTorch, metric). It is given the lens's field of view when known. Sizes are `moge-2-vits` (35M), `-vitb` (104M) and `-vitl` (331M); the default to try is `vitb`.
- **It runs on `RETARGET_DEPTH_FRAMES` (12) frames per clip,** always including frame 0 (the world frame), never every frame.
- **The table:** RANSAC over the depth points below the top third of frame 0, with the hands masked out. With "up" known, only planes within 35° of up count. Its normal and distance replace the guessed ones.
- **The hand's scale:** for each sampled frame, the median over the joints of (depth at the joint's pixel + 8 mm skin-to-joint) ÷ (the reconstructed depth). The clip's scale is the median over the frames.
  - **Not applied** when it is outside 0.6–1.6, when the frames disagree by more than 25%, or when the hand was in fewer than 2 sampled frames. The bundle says why.

### 3.3 Hand (`hand_wilor.py`): optional, experimental
WiLoR through WiLoR-mini. It is a drop-in for MediaPipe's detector: its metric 3D joints take the place of MediaPipe's world landmarks, and the existing PnP fit places them with the real lens. Its install is the riskiest part (see §8), so it is tried on Colab before anything else.

### 3.4 How each clip's numbers were made
`hand_stage.json` → `perception`, which also reaches the app in `metrics.perception`:
- `hand_model`, `lens_source`, `hfov_deg`, `stabilized`, `up_known`;
- `depth_model`, `table_source`, `table_inliers`, `table_up_angle_deg`;
- `hand_scale`, `hand_scale_measured`, `hand_scale_spread`;
- `depth_seconds`, `notes`.

**Nothing is gated on these yet.** First we see their real distribution on our clips (phase 2), then we set bars.

## 4. How it fits the system

- **Contract:** `track.hand` gains an optional `camera` field, and `metrics` an optional `perception`. **No protocol change:** an older pipeline ignores the field, and an older app sends none. Both `CONTRACT.md` copies are updated.
- **Queue:** the GPU work stays in the existing `track` queue, so there is no new queue.
  - With a model on, `track` runs **one clip at a time**: each worker process holds its own copy of the models.
  - It gets its own memory bar (`PERCEPTION_WORKER_MEMORY_MB`), so it is not restarted and reloaded after every clip.
- **GPU share:** `gpu.torch_device()` caps PyTorch at `GPU_MEMORY_FRACTION` (0.25) of the GPU's memory, because the AI model comes first. Models are loaded once per process, in fp16, only in pipeline hours.
- **Image:** `PERCEPTION` and `TORCH_INDEX` build arguments (docker-compose passes them from the env file). Packages are pinned to commits (`scripts/install_perception.sh`) and weights are baked in at build time (`scripts/download_perception_weights.py`). No job ever downloads a model.
- **Cost per clip:**
  - **Downloads:** none new.
  - **Disk:** about 1 KB more per bundle.
  - **Database:** the camera facts are a few tens of KB of JSON per clip (capped at 256 KB; anything bigger is dropped, never the clip).
  - **GPU:** about 12 depth frames per clip; measured in phase 1.
- **Version:** `RETARGET_PIPELINE_VERSION` stays `hand2robot-1.0` for now. Changing it makes the app re-track every clip it holds, so it is a deliberate deploy decision, taken after phase 2.

## 5. Changes per repository (branch `pipeline-optimization`)

| Repo | Change |
|---|---|
| **pipeline** | new `retarget/lens.py`, `retarget/scene.py`, `retarget/hand_wilor.py`; `pipeline.py`, `camera_reconstruction.py`, `service.py` and `compute.py` wired; `gpu.py` (`torch_device`); `scheduler.py` (one clip at a time, memory bar); `config.py` settings; Dockerfile and compose build arguments; `scripts/install_perception.sh`, `download_perception_weights.py`, `gpu_smoke.py`; `research/hand2robot2_colab.ipynb`; `CONTRACT.md`; tests (`test_lens.py`, `test_scene.py`, `test_perception_wiring.py`, `test_compute.py`) |
| **Web** | `videos.camera_info` (migration `b7e3f1a9c2d4`); the upload takes `camera` (`session/camera_info.py`: dropped, never refused, when bad or over 256 KB); `track.hand` sends it; `PIPELINE_CONTRACT.md`; tests |
| **Mobile** | `CameraFacts.kt` (Camera2 description, capture-result samples, gravity; side file beside the video); attached to the preview at bind, begun and ended with each take; sent as `camera` on upload; deleted with the video. **Not built here**: build and test on a phone |

## 6. Phases, each with a go/no-go

| # | What | Where | Go when |
|---|---|---|---|
| 0 | Stage 1 code, tests, notebook | local | ✅ done: pipeline 235 tests, Web 500 tests, a real clip through the new hand stage on the CPU |
| 1 | `gpu_smoke.py` + the notebook on 10–15 clips: do MoGe-2 and WiLoR install, run, and how fast | Colab | both run; GPU seconds per clip known |
| 2 | **A (today) vs B (depth) vs C (WiLoR + depth)**, using the notebook's `summary.csv` | Colab + robot | B or C clearly better: object width nearer the ruler, arm reaches more of the path, SO-101 replays pass more often |
| 3 | Build the app, record the test clips with it, check the camera facts arrive (`videos.camera_info`) | phone | lens source `phone` on most clips |
| 4 | The DGX: build with `PERCEPTION=depth`, `TORCH_INDEX=…/cu130`, run `gpu_smoke.py`, then turn `RETARGET_DEPTH_MODEL` on | DGX | smoke test passes, within the GPU share |
| 5 | Later (polish): quality bars from phase 2's numbers; the thumb–index gripper (Phantom/HumanEgo); show `perception` in Video Review; object 3D + tracking (SAM 3D + FoundationPose) | Colab first | phase 2 says where the error still is |

## 7. Test data (only for testing, never for operators)

- **10–15 clips**, 2–3 tasks, recorded as usual (with the new app, so they carry camera facts).
- **Truth, by the team only:** the object's width, measured with a ruler (`truth.csv`: `clip,object_width_cm`). The pipeline's object width is computed from its distances, so it is a direct check of the depth.
- **Robot replay:** for each clip, play A's and B/C's trajectory on the SO-101 and note pass or fail.
- **Where:** a `hand2robot2` folder in Google Drive (the notebook says what goes in it).

## 8. DGX Spark: one careful deploy

**The risk:** the Spark is ARM (aarch64) with a new GPU type (sm_121, CUDA 13). Colab is x86, so a pass on Colab does not prove a pass on the Spark.
- **MoGe-2:** plain PyTorch, no compiled code. Installed `--no-deps` at its 2.0.0 commit, because its own list pulls gradio and the desktop opencv.
- **WiLoR-mini:** pins `torch<=2.5`, which has no build for this GPU, so it is installed `--no-deps`. It needs `chumpy`, which is old. If it does not import on Python 3.12, WiLoR waits and MediaPipe stays: the depth model alone carries most of the gain.
- **PyTorch:** `TORCH_INDEX` set to the CUDA 13 wheels. Whether they include sm_121 is unverified; the smoke test says.
- **`scripts/gpu_smoke.py`:** run once on the DGX inside the built image. It changes nothing, loads each model, runs a made-up frame, and prints the device, the seconds per frame and the peak GPU memory.

## 9. Licences (R&D only)

- **MoGe-2:** MIT.
- **WiLoR:** CC BY-NC-ND 4.0. MANO is non-commercial. The detector is Ultralytics YOLO (AGPL, unverified).
- **Fine for R&D. Datasets made with WiLoR are not for commercial use** until it is replaced.

## 10. Risks

| Risk | Answer |
|---|---|
| A model does not run on the Spark | everything is off by default; the smoke test runs before turning it on; without it the pipeline is unchanged |
| GPU memory clashes with the AI model | one clip at a time, a 25% memory cap, pipeline hours only, measured by the smoke test |
| Depth wrong on a scene (glass, shiny, plain white table) | the scale is applied only within 0.6–1.6 and when frames agree; the table only with ≥30% inliers facing up; otherwise the old estimate, and the bundle says so |
| Phone sends no or wrong camera data | the depth model's estimate, else the guess; impossible fields of view rejected; never a refused clip |
| Gravity axes wrong on some phone | gravity is used only to choose among table planes (35° tolerance) and to report an angle; check `table_up_angle_deg` in phase 3 |
| New is not better | phase 2 is the go/no-go: nothing is switched on unless the numbers say so |

## Sources

- Phantom (CoRL 2025), https://arxiv.org/abs/2503.00779
- Hand2Robot (2026), https://arxiv.org/html/2610.15769
- HumanEgo, https://arxiv.org/html/2605.24934 · R+X, https://arxiv.org/pdf/2407.12957
- MoGe, https://github.com/microsoft/MoGe (MoGe-2 at commit b942f00, 2.0.0)
- WiLoR-mini, https://github.com/warmshao/WiLoR-mini · WiLoR, https://github.com/rolpotamias/WiLoR
- WristPP (MediaPipe vs WiLoR), https://arxiv.org/pdf/2603.00606
- Android Camera2: focus-distance calibration, `LENS_INTRINSIC_CALIBRATION` (only with DEPTH_OUTPUT), `SENSOR_INFO_TIMESTAMP_SOURCE`
- DGX Spark: https://simonwillison.net/2025/Oct/14/nvidia-dgx-spark/
