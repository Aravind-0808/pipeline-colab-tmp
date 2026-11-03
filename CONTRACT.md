# The contract between the app and the pipeline

Two projects, two repositories, no shared code: the **app** (the app server —
[RedMindRD/data-collection-platform](https://github.com/RedMindRD/data-collection-platform))
and the **pipeline** (the pipeline server, the DGX — this repository,
[RedMindRD/dataforge-pipeline](https://github.com/RedMindRD/dataforge-pipeline)).
They meet in **Redis** (work and results) and **MinIO** (files), both on the app
server, and this is everything they agree on.

The app takes uploads, keeps the database and decides. The pipeline does the
heavy work — the AI check, hand tracking, robot conversion, building and
publishing datasets — **only during its hours**.

Each repository keeps a copy of this file (this is the pipeline's; the app's
is its `docs/PIPELINE_CONTRACT.md`) and pins these values in a contract test
(`tests/test_contract.py` here, `backend/tests/test_contract.py` there), so a
change on one side that the other did not make fails a test instead of a job.
**Change it in both repositories, together.**

**Protocol version: 4.** The app stamps it on every message; the pipeline
refuses any other with a clear error. Change anything below in a way an old
peer would misread, and bump it on both sides. Deploy the pipeline first.

## Transport

- Broker: **the app server's Redis**, `redis://:<password>@<the app server's private address>:6379/0`.
- Celery, JSON serializer, `task_acks_late`, `task_reject_on_worker_lost`,
  `worker_prefetch_multiplier=1`, `task_ignore_result`, visibility timeout 9 h.

| Queue | Consumed by | Carries |
|---|---|---|
| `check` | pipeline | `pipeline.compute` for `validate.judge` |
| `track` | pipeline | `pipeline.compute` for `track.hand`, `retarget.solve` |
| `build` | pipeline | `pipeline.compute` for `export.build`, `dataset.publish`, `dataset.convert` |
| `db` | app (`celery-db`) | `pipeline.finish`, `pipeline.export_progress` |

## Hours

The pipeline works only inside **`PIPELINE_HOURS`** (default `22:00-09:00`,
zone `PIPELINE_TZ`, default `Asia/Kolkata`) — the same values on both sides.

- **Pipeline:** one worker per queue (`check`, `track`, `build`), each limited
  (`CHECK_CONCURRENCY` 2, `TRACK_CONCURRENCY` 2, `BUILD_CONCURRENCY` 1), started
  when the hours begin and stopped when they end: running work gets
  `STOP_GRACE_SEC`, the rest is not acknowledged and is handed out again the
  next night.
- **App:** outside the hours it does not send — the job waits in its jobs table
  and is sent when they begin. Its Monitor shows the pipeline asleep, not down.

## Messages

**1. A job's heavy half** — a Celery chain the app builds:

```
pipeline.compute(name, spec, protocol=4)     queue: per name, above   → pipeline
  → pipeline.finish(result, job_id, attempt, payload, offload)   queue: db → app
```

The pipeline defines only `pipeline.compute`. Celery passes its return value
to `pipeline.finish` by name; the app writes it. `pipeline.compute` returns:

```json
{"ok": true,  "value": { ...the compute's result... }}
{"ok": false, "error": "Type: message"}
```

**2. Progress** — `pipeline.export_progress(export_id, fraction, step)` on
queue `db`, at most every 5 s per step, for builds and publishing. Never retried.

## Clips

A `<clip>` — where to read a clip, built by the app. The pipeline reads it from
MinIO with **its own access key**; no credential travels in a message.

```json
{"video_id": str, "path": str, "size_bytes": int|null, "kind": "s3", "bucket": str}
{"video_id": str, "path": str, "size_bytes": int|null, "kind": "local", "local": str}   development only
```

## The computes

### `validate.judge` — the AI check (queue `check`)

```json
spec   {"clips": [<clip>...],
        "task": {"title": str, "description": str|null, "task_flow": any|null,
                 "hands": 1|2} | null,
        "thresholds": {<Thresholds fields>}}
value  {"verdicts": [<verdict>...]}            one per clip, same order
```

`<verdict>`:

```json
{"video_id": str, "verdict": "pass"|"flag"|"reject"|"unknown",
 "confidence": float, "source": "rules"|"vlm"|"none",
 "findings": [{"code": str, "detail": str}],
 "subtasks": [{"name": str, "start": float, "end": float}],
 "facts": {...} | null}
```

A pass means **both** questions held in **every** frame — the task, and hand
tracking — at confidence ≥ 0.9. Finding codes the app reads: task
(`task_not_performed`, `object_not_visible`, `view_obstructed`), tracking
(`hand_not_visible`, `hand_out_of_frame`, `hand_unclear`), and
`model_unreachable` (the app waits and asks again; it never settles for it).

### `track.hand` — hand tracking, the second gate (queue `track`)

```json
spec   {"video_id": str, "clip": <clip>, "bundle_path": str, "hands": 1|2,
        "camera": object|null}                  what the phone said of its camera, or null
value  {"metrics": {..., "perception": {...}?, "hands": {"left": {...}, "right": {...}}?},
        "bundle_path": str, "reason": str|null} reason: why the gate refused, or null
     | {"failed": str}                          the clip could not be read
```

A `reason` or `failed` rejects the clip: the app deletes its file and track.

**`camera`** is what the Android app recorded of its camera with the clip
(Camera2 and the gravity sensor), stored by the app as the phone sent it and
read only by the pipeline (`lens.py`): `schema` (1), `lens_facing`,
`display_rotation` (0-3, `Surface.ROTATION_*`), `start_ns`; `logical` and
`physical` (id → camera) each with `focal_lengths_mm`, `sensor_size_mm`,
`pixel_array`, `active_array`, `intrinsics` (or null), `distortion`,
`focus_calibration`; `frames` (sampled capture results: `t_ns`, `focal_mm`,
`focus_diopters`, `crop`, `zoom_ratio`, `physical_id`, `stabilization`) and
`gravity` (`t_ns`, `x`, `y`, `z` in the device's axes). Every field is
optional; null, or anything the pipeline cannot read, means a guessed lens, as
before. No protocol change: an older pipeline ignores it.

**`metrics.perception`** says how the clip's distances were made: `hand_model`
(`wilor`), `lens_source` (`phone-calibration` | `phone` | `depth-model`),
`hfov_deg`, `up_known`, `depth_model` (the model), and with a depth model the table's and the hand's measurement
(`table_source`, `hand_scale`, `hand_scale_spread`, ...). Absent from tracks
made before it.

**`hands`** is the task's: 2 when it is done with both hands. Each hand is then
tracked on its own and must clear the hand bars on its own; `metrics` keeps the
weaker hand's numbers at the top and each hand's under `hands`. The left hand is
the one that worked on the left of the picture.

### `retarget.solve` — a clip's hand onto the robot's arm (queue `track`)

```json
spec   {"bundle_path": str, "calibration": object|null, "store_path": str,
        "robot": {"package_path": str, "urdf_file": str, "profile": {...}, "name": str}|null}
value  {"metrics": {...}, "trajectory_path": str, "reason": str|null}
     | {"bundle_pending": true}                 bundle not in storage yet — app retries later
     | {"failed": str}
```

The spec does not change for two hands: the bundle says so, and the
trajectory is then for **two SO-101 arms** on one clock — `"arms": ["left",
"right"]`, and each sample's `action` named and scaled as a two-arm SO-101
dataset records it (`left_arm_shoulder_pan.pos` … `right_arm_gripper.pos`,
joints -100 .. 100 of their calibrated range, gripper 0 .. 100). Both arms use
the robot's calibration.

With `robot` (an imported robot: its package and profile, as `dataset.convert`
describes one), the arm is that robot's, solved from its URDF
(`retarget_urdf.py`): the trajectory is `robot_trajectory.json`, the same
shape, with `action` under the robot's own LeRobot names (`<name>.pos`, or
`<side_prefix><name>.pos` for two hands, onto the two sides of a two-arm URDF
or the same arm twice), joints in its `units` from its URDF, the gripper in
its dataset values. Without `robot`, the SO-101 as above.

### `export.build` — a LeRobot dataset, into storage (queue `build`)

```json
spec   {"payload": <the export job's payload: episodes (each with its <clip>,
                    video_id, trajectory_path), store_prefix "datasets/<dataset>/<export>/",
                    repo_id, version, fps, robot_*, arms (1|2), …,
                    "base": null                           first version
                          | {"store_prefix": str}          previous version, still in storage
                          | {"repo_id": str, "revision": "v<version>"}   previous version, on the Hub>}
value  {"episodes": int, "video_ids": [str], "total_frames": int, "lerobot_version": str,
        "store_prefix": str, "files": int, "repo_id": str, "root": str}
     | {"pending": int}                         trajectories not in storage yet
```

`arms` is 2 for a dataset of a two-hand task: twelve state/action columns,
both arms, and robot type `bi_so101_follower`. A dataset never mixes the two.
For an imported robot the payload also carries `names` (its state/action
columns, those its trajectories' `action` holds) and `robot_type` (`bi_…` for
two arms); without them, the SO-101's.

Nothing goes to Hugging Face here. The episodes are the clips new in this
version; with a `base`, they are merged onto the previous version (made at its
frame size), so the stored dataset is the whole of it. `episodes` counts every
episode in it, `video_ids` only the clips built this time — the app then
deletes those clips' raw files, tracks and trajectories (their rows stay).

### `dataset.publish` — a built dataset, to Hugging Face (queue `build`)

```json
spec   {"export_id": str, "store_prefix": str, "repo_id": str, "private": bool,
        "version": str}
value  {"repo_id": str, "hub_url": str, "commit_sha": str|null, "tag": "v<version>"|null}
```

The only thing that ever goes to Hugging Face, and only when an admin asks.
One repo per dataset: each version replaces its files and is tagged
`v<version>`, so every version stays downloadable. Once published, the app
deletes the version from storage.

### `dataset.convert` — a finished dataset, to another robot (queue `build`)

```json
spec   {"export_id": str,
        "source": {"repo_id": str, "revision": "v<version>"|null},
        "target_repo_id": str, "private": bool, "tag": "v0.1.0",
        "source_units": "degrees"|"m100"|null,
        "source_robot": {"package_path": str|null, "urdf_file": str|null,
                         "profile": {...}|null, "name": str},
        "target_robot": {same}}
value  {"repo_id": str, "hub_url": str, "tag": str, "commit_sha": str|null,
        "robot_type": str, "names": [str], "total_episodes": int, "total_frames": int,
        "episodes": [{"episode": int, "rows": int, "position_error_median_mm": float,
                      "position_error_max_mm": float, "clamped_pct": float,
                      "over_speed_pct": float}]}
```

Hub to Hub, for an imported robot (`package_path` + `profile`, read from its
URDF) or the bundled SO-101 (no package). The new repo gets the source's
`videos/` by a server-side copy at the source revision — never downloaded —
and only `data/` and `meta/` are fetched, converted row by row (FK on the
source URDF, moved from the source arm's working point to the target's and
scaled by their spans, IK on the target URDF; open is open on any gripper;
rows and timestamps kept) and uploaded with a card.
`source_units` null reads two-arm (`left_arm_*`) datasets as −100..100 and
one-arm as degrees, as the SO-101 export writes them. Both robots are read
from their URDFs alone — no calibration file: degrees are URDF angles, and
−100..100 spans each joint's URDF limits, 0 halfway.

A `profile` may also carry, all optional (absent: the URDF's names,
`{side}_arm_`, a gripper written 0 closed .. 100 open, the URDF's limits):
`names` (LeRobot's names of the arm joints, then the gripper), `side_prefix`
(how a two-arm dataset prefixes each side, with `{side}`), `gripper.lerobot`
(`{"open", "closed"}`: what a dataset writes for each), `limits`
(`[[lower, upper]]` radians per arm joint: the real arm's, when tighter),
`velocity_limits` (rad/s per arm joint, `null` for the URDF's: a URDF's speed
is often a placeholder), `pair_spacing_m` (a pair of this one-arm robot
stands side by side, facing forward, this far apart base to base; 0.4 when
unset — two hands are placed in one frame across both arms, so a handover
meets), `work_point` ([x, y, z] in the base frame) and `right_arm` (a two-arm URDF's
right arm: its own `ee_link`, `arm_joints`, `gripper`, `home_pose`, `limits`,
`velocity_limits`;
the rest is the left's). A two-arm dataset writes each side with its own arm
under `<side_prefix><name>.pos`, and its `robot_type` gains `bi_`.

## Storage — MinIO on the app server

One bucket (`S3_BUCKET`, default `dataforge`), capped at **20 GB** (its quota).
The app has the root key; the pipeline has its own (read/write, no admin),
reaching MinIO on the app server's private address, port 9000.

| What | Path | Written by |
|---|---|---|
| clip | `<project>/<task>/<video_id>.<ext>` | app |
| hand bundle | `tracks/<pipeline version>/<video_id>.zip` | pipeline |
| robot package (URDF + meshes, imported) | `robots/packages/<sha256>.zip` | app |
| trajectory | `retargets/<pipeline version>/<robot_id>/<video_id>.json.gz` | pipeline |
| built dataset | `datasets/<dataset_id>/<export_id>/...` | pipeline |

The app builds these paths and sends them in the spec; `RETARGET_PIPELINE_VERSION`
must be the same on both servers. A rejected clip's file and track are deleted
by the app; its database row stays. So are a clip's file, track and trajectory
once a dataset version holds it, and a dataset version once it is published.

## Shared Redis keys

| Key | Type | What |
|---|---|---|
| `limits:s:llm:gate` | zset | model calls in flight (`LLM_MAX_CONCURRENCY`) — written by the pipeline, read by the app's Monitor |
| `llm:breaker` | hash | `failures`, `paused_until` — the model's circuit breaker; `failed_at` + `error` / `ok_at` — the last real call's outcome, which the Monitor shows (it never asks the model on a timer) |
| `pipeline:hours` | string (JSON) | `hours` ("22:00-09:00", "" = always), `tz` — set on the app's Settings page, written by the app, read by the pipeline every tick; `PIPELINE_HOURS`/`PIPELINE_TZ` are the fallback when absent |
| `pipeline:heartbeat:<NODE_NAME>` | string (JSON), 90 s TTL | written by the pipeline every 30 s, asleep or not: `node`, `at` (unix s), `asleep`, `hours`, `tz`, `opens_in_sec`, `protocol`, `gpu` — read by the app's Monitor |
