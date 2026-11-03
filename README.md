# DataForge pipeline

The dataset work of the DataForge platform, as a project of its own: hand
tracking, arm retargeting and LeRobot dataset builds — run when a dataset is
built, and idle otherwise. It has **no database**, calls **no model** (the
clip check is the app's, done as each clip arrives), and shares **no code** with the app
([RedMindRD/data-collection-platform](https://github.com/RedMindRD/data-collection-platform));
it takes work from the app server's Redis and sends the results back.
[`CONTRACT.md`](CONTRACT.md) is everything the two agree on.

```
 app server                                          pipeline server (this project)
 ──────────                                          ──────────────────────────────
 prepares a job ──── pipeline.compute ── Redis ────▶ celery-compute
                                                       track · export
 writes the result ◀─ pipeline.finish ── Redis ─────  (result travels back)
 serves clips not yet on the Hub ◀───── HTTP ───────  reads clips (or from the Hub)
                                                     committer → Hugging Face
```

## What runs

| Service | What it does |
|---|---|
| `celery-compute` | Celery workers for the queues `track` and `export`. `COMPUTE_CONCURRENCY` tasks at once. |
| `committer` | Sends what this machine produced (hand bundles, trajectories) to the Hub, many files per commit, under the commit lock and hourly budget it shares with the app server. |

No ports are published.

## What it needs to reach

| What | Why |
|---|---|
| the app server's Redis (`6379`, private address) | its work, and where results go |
| the app server's `:8000` (private address) | clips still in the app server's staging folder, through signed links |
| huggingface.co | committed clips; committing its own output |

## Deploy on the pipeline server

```bash
ssh <pipeline server>
git clone https://github.com/RedMindRD/dataforge-pipeline.git /srv/dataforge-pipeline
cd /srv/dataforge-pipeline
cp .env.example .env.production   # REDIS_URL, HF_* — see the comments in it

# check the link to the app server first
docker run --rm redis:7-alpine redis-cli -u "$(grep ^REDIS_URL .env | cut -d= -f2-)" ping   # PONG
curl -fsS http://10.0.0.1:8000/health                                                     # the app server

./deploy.sh                   # builds here, for this machine's architecture, and starts
```

Then the app's admin panel → **Monitor** → Pipeline should list this machine
(`NODE_NAME`, default `pipeline`) with the queues `track, export`.

To update: `git pull` and run `./deploy.sh` again — **before** updating
the app server, when both change (see CONTRACT.md).

## The pipeline server's architecture

The image is built on the pipeline server itself, so it is always for its own
architecture, `x86_64` or `aarch64` (`uname -m` says which). The pipeline
runs on the CPU cores. The Dockerfile's import checks for MediaPipe
and OpenCV fail the build — not the first job — if a library has no
wheel for the machine; watch the first build on an `aarch64` server.

## Sizing

- `COMPUTE_CONCURRENCY` — tasks at once. A track is about a core while it
  runs. Start at 8 and raise it while the Monitor's queue lengths stay above
  zero during an export.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"      # add ,ml for the full stack
.venv/bin/pytest
REDIS_URL=redis://localhost:6379/0 .venv/bin/celery -A dfpipeline.worker worker -Q track,export
```

The tests stand in for the machine-learning calls, so they run without torch,
MediaPipe or lerobot installed.

## Layout

| | |
|---|---|
| `dfpipeline/worker.py` | the Celery app and its one task, `pipeline.compute` |
| `dfpipeline/compute.py` | the three computes, by name |
| `dfpipeline/retarget/` | hand tracking and SO-101 retargeting (vendored from Hand2Robot) |
| `dfpipeline/lerobot_builder.py` | LeRobot dataset builds |
| `dfpipeline/committer.py`, `staging.py`, `storage.py` | staging and batched commits to the Hub |
| `dfpipeline/redis.py` | the one Redis connection, shared with the committer |
| `models/so101/` | the SO-101 URDF (joints only — the IK reads nothing else); the MediaPipe weights are downloaded at build |
