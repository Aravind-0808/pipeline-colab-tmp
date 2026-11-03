"""The pipeline's settings, from the environment (and `.env`).

Only what the pipeline reads. The names are the app's own, so one vocabulary
covers both servers: HF_TOKEN, REDIS_URL and the rest mean the same thing in
either project's `.env`.

There is no DATABASE_URL here. The pipeline has no database, by design.
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# `.env.development` unless told otherwise; in production the container gets
# its settings as environment variables (docker-compose reads
# `.env.production`).
ENV_FILE = os.getenv("ENV_FILE", ".env.development")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"

    # --- The app server's Redis: the work arrives here and the results leave here.
    redis_url: str = ""
    # The name this machine shows on the app server's Monitor page.
    node_name: str = ""

    # --- When the pipeline works. The DGX is the pipeline's only at night:
    # its workers run inside this window and are stopped outside it. Work that
    # arrives in the day waits in the queue; work still running at the end of
    # the window is handed back and carried on the next night. "HH:MM-HH:MM",
    # crossing midnight is fine; empty means always.
    pipeline_hours: str = "22:00-09:00"
    pipeline_tz: str = "Asia/Kolkata"
    # At the end of the window, running work has this long to finish before it
    # is stopped and handed back.
    stop_grace_sec: int = 600

    # --- The model, behind one OpenAI-compatible gateway (LiteLLM), for the AI
    # check. Its calls are held to LLM_MAX_CONCURRENCY through Redis.
    llm_api_base: str | None = None
    llm_api_key: str = "EMPTY"
    llm_model: str | None = None
    llm_timeout_sec: float = 90.0
    llm_max_concurrency: int = 2
    # Clips at or below this size are fetched once and seeked locally; larger
    # ones are range-seeked in place so we never pull a whole big file.
    frame_download_max_bytes: int = 16 * 1024 * 1024

    # --- The AI check watches the clip as a video. The model server cannot
    # read phone video (H.264), so each clip is turned into a small MJPEG copy
    # here, sent, and thrown away. If the model will not take the video, the
    # check falls back to the grid of frames. CHECK_WITH_VIDEO=false: grid only.
    check_with_video: bool = True
    check_video_width: int = 448
    check_video_fps: float = 4.0
    check_video_max_frames: int = 120   # long clips get a lower frame rate

    # --- How much at once — never everything at the same time. One worker per
    # kind of work, each with its own number of tasks at once.
    check_concurrency: int = 2   # AI check: frames and one model call per clip (the model has 2 slots)
    track_concurrency: int = 2   # MediaPipe: about a core each, and memory
    build_concurrency: int = 1   # dataset builds and publishing: one at a time

    # --- A fixed share of the machine, never all of it. The DGX also serves
    # models (vLLM, Ollama) and runs training; a pipeline that took every core
    # and all the memory took them down with it. Docker applies the first two
    # (docker-compose.yaml reads the same keys, with the same defaults).
    pipeline_mem_limit: str = "16g"  # the container's memory; past it only the container restarts
    pipeline_cpus: float = 8.0       # the container's cores, in all
    cpu_threads: int = 2           # per process: numpy/BLAS, OpenMP, OpenCV
    worker_max_tasks: int = 50     # a worker process is replaced after this many tasks
    worker_max_memory_mb: int = 3000  # ... or once it holds this much (leaks never pile up)

    # --- Storage: the app server's MinIO, over the private network, with the
    # pipeline's own access key. Clips are read from it; tracks and datasets
    # are written to it. Empty S3_ENDPOINT: a folder (STORAGE_ROOT) —
    # development and tests.
    s3_endpoint: str = ""          # e.g. http://100.64.0.1:19000 (Tailscale; the app server's MINIO_PORT)
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "dataforge"
    storage_root: str = "./storage"

    # --- Hugging Face: only to publish a finished dataset, when asked.
    hf_token: str | None = None

    # --- Dataset builds (dfpipeline/lerobot_builder.py)
    # How many episodes go up in one commit to the Hub: what a failure costs,
    # against re-hashing the dataset built so far on every commit.
    export_batch_size: int = 10
    export_data_dir: str = "./_datasets"
    # Keep the built dataset on disk after it is safely on the Hub.
    export_keep_local: bool = False
    # Consecutive episode failures before an export stops trying.
    export_give_up_after: int = 5
    # How long to wait before uploading when clips were failing.
    export_recover_pause_sec: float = 5.0
    # An episode shorter than this (frames) is left out: one second at 30 fps.
    # A one-frame episode cannot be trained on and breaks LeRobot's statistics.
    export_min_episode_frames: int = 30

    # NVENC video encoding (dfpipeline/gpu.py): "off" (the default) never
    # tries; "auto" uses it where it works; "on" tries and warns when it falls
    # back. Hand tracking is on the GPU whatever this says: it has no CPU path.
    pipeline_gpu: str = "off"
    # The share of the GPU's memory one worker process's PyTorch models may
    # hold (gpu.torch_device). Hand tracking with models runs one clip at a
    # time (scheduler.py), so this is the whole of the pipeline's share.
    gpu_memory_fraction: float = 0.25

    # --- Robot retargeting (dfpipeline/retarget)
    # Where efficientdet_lite0.tflite, so101/, moge/ and wilor/ live.
    retarget_models_dir: str = "./models"
    # Stamped on every result. Must match the app server's.
    retarget_pipeline_version: str = "hand2robot-1.0"

    # How the hand is seen (docs/HAND2ROBOT_2.md): WiLoR's 3D hand
    # (hand_wilor.py) and real distances from a depth model on a few frames
    # (scene.py), both on the GPU. The depth model: "moge-2-vits" |
    # "moge-2-vitb" | "moge-2-vitl".
    retarget_depth_model: str = "moge-2-vitb"
    # Frames of each clip the depth model looks at, and its detail (0-9:
    # MoGe's resolution_level; lower is faster).
    retarget_depth_frames: int = 12
    retarget_depth_resolution: int = 9
    # The memory bar of a track worker process that holds those models
    # (instead of WORKER_MAX_MEMORY_MB): past it the process is replaced.
    perception_worker_memory_mb: int = 8000

    # The gates: whether coordinates and trajectories are fit to train on.
    retarget_min_hand_coverage: float = 0.70
    # How long the hand must be followed, in seconds and as a share of the
    # clip. Under either, the episode is a fragment of the task: a pick and
    # place came out as a 0.4 s episode from a 10.9 s clip.
    retarget_min_hand_seconds: float = 1.5
    retarget_min_hand_share: float = 0.40
    # The object must be seen in this share of frames. It was 50%, when the
    # gripper's timing came from the object; it now follows the hand itself
    # (grasp.opening) and the object only steadies it. White cups on a white
    # cloth were seen in 25-49% of frames — the hand in 100% — and every one
    # of those clips was refused.
    retarget_min_object_frames: float = 0.20
    retarget_min_reachable: float = 0.85
    # How well the arm follows the hand is the measure: the median distance
    # between the gripper and the hand's path. Slowdown is only a loose net —
    # the dataset runs on the robot's own clock, and a 3x limit refused a clip
    # the arm followed to 6 mm (3.1x) while a 16 cm miss read 11x.
    retarget_max_position_error_m: float = 0.03
    retarget_max_slowdown: float = 5.0
    # The SO-101's jaw gap, in metres, with the gripper fully open (gripper.pos
    # 100): the gripper's opening is the object's width over this (grasp.py).
    # Measure your arm's and set it; 0.09 is the hand-to-gripper scale used so far.
    retarget_gripper_max_gap_m: float = 0.09
    retarget_max_peak_limit: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
