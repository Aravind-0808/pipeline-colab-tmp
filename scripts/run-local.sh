#!/bin/bash
# Run the pipeline on this machine: its Celery worker for track and export —
# hand tracking, arm retargeting and dataset builds — taking work from the
# app's Redis. (Clip checks are the app's; they do not come here.)
#
#   scripts/run-local.sh          the worker (tracking and exports need --ml
#                                 to do anything)
#   scripts/run-local.sh --ml     with the machine-learning stack (torch,
#                                 MediaPipe, lerobot — a large download)
#
# The app is its own project with its own script (Web/scripts/run-local.sh),
# which starts the Redis this connects to; run-local.sh at the project root
# runs both. If Redis is not up yet, this waits for it.
#
# Settings come from .env.development. Dependencies are installed on the
# first run and whenever pyproject.toml changes. Ctrl+C stops it. The worker
# does not reload: restart the script after changing the code.
set -uo pipefail

PIPELINE="$(cd "$(dirname "$0")/.." && pwd)"

EXTRAS=dev
case "${1:-}" in
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --ml) EXTRAS=dev,ml ;;
    "") ;;
    *) echo "unknown option: $1 (try --help)"; exit 64 ;;
esac

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\033[31m!!  %s\033[0m\n' "$*"; exit 1; }

# run NAME COLOUR DIR COMMAND... — in the background, in a process group of
# its own, every line labelled.
PIDS=()
run() {
    local name=$1 colour=$2 dir=$3; shift 3
    local label
    label="$(printf '\033[%sm%-9s\033[0m| ' "$colour" "$name")"
    setsid bash -c 'cd "$1" && shift && exec "$@"' _ "$dir" "$@" \
        > >(sed -u "s/^/$label/") 2>&1 &
    PIDS+=("$!")
}

# ensure_python_env DIR EXTRAS — DIR/.venv, created if missing (or broken by a
# move) and reinstalled when pyproject.toml or the extras change. A stamp in
# the venv remembers what was installed, so an unchanged run installs nothing.
ensure_python_env() {
    local dir=$1 extras=$2
    local venv="$dir/.venv" stamp="$dir/.venv/.deps-stamp" want
    want="$(sha256sum "$dir/pyproject.toml" | cut -c1-16) [$extras]"

    if [ -e "$venv" ] && ! "$venv/bin/python" -c "" 2>/dev/null; then
        warn "the venv in $dir is broken (moved?) — rebuilding it"
        rm -rf "$venv"
    fi
    if [ -x "$venv/bin/python" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$want" ]; then
        return 0
    fi
    if [ ! -x "$venv/bin/python" ]; then
        say "creating the Python environment in $dir/.venv"
        if command -v uv >/dev/null; then
            uv venv -q --python python3 "$venv" || die "could not create $venv"
        else
            python3 -m venv "$venv" \
                || die "could not create $venv — install uv, or python3-venv (sudo apt install python3-venv)"
        fi
    fi
    say "installing Python dependencies for $(basename "$dir") [$extras]"
    if command -v uv >/dev/null; then
        uv pip install -q --python "$venv/bin/python" -e "$dir[$extras]" || die "installing failed"
    else
        "$venv/bin/python" -m pip install -q -e "$dir[$extras]" || die "installing failed"
    fi
    echo "$want" > "$stamp"
}

# Ctrl+C first, which Celery shuts down on cleanly; anything still running ten
# seconds later is killed.
stop_all() {
    local code=$1
    trap - INT TERM EXIT
    [ "${#PIDS[@]}" -gt 0 ] || exit "$code"
    echo
    say "stopping the pipeline"
    local pid alive
    for pid in "${PIDS[@]}"; do kill -INT -- "-$pid" 2>/dev/null; done
    for _ in $(seq 1 20); do
        alive=0
        for pid in "${PIDS[@]}"; do kill -0 -- "-$pid" 2>/dev/null && alive=1; done
        [ "$alive" = 0 ] && break
        sleep 0.5
    done
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null; done
    say "pipeline stopped"
    exit "$code"
}
trap 'stop_all 130' INT
trap 'stop_all 143' TERM
trap 'stop_all $?' EXIT

# --------------------------------------------------------------- checks

[ -f "$PIPELINE/.env.development" ] || die "no .env.development — copy .env.example and fill it in"

# Dependencies: installed when missing, brought up to date when they change.
ensure_python_env "$PIPELINE" "$EXTRAS"

# Where the app's Redis is, from the same settings the worker will read.
read -r HOST PORT < <(cd "$PIPELINE" && .venv/bin/python -c "
from urllib.parse import urlparse
from dfpipeline.config import settings
url = urlparse(settings.redis_url or 'redis://127.0.0.1:6379/0')
print(url.hostname or '127.0.0.1', url.port or 6379)
") || die "could not read REDIS_URL from .env.development"

port_open() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; }

if ! port_open "$HOST" "$PORT"; then
    say "waiting for the app's Redis at $HOST:$PORT (start the app: Web/scripts/run-local.sh)"
    for _ in $(seq 1 120); do
        port_open "$HOST" "$PORT" && break
        sleep 1
    done
    port_open "$HOST" "$PORT" || die "no Redis at $HOST:$PORT after two minutes — is the app running?"
fi

# ---------------------------------------------------------------- start

say "starting the pipeline (Redis at $HOST:$PORT)"
run pipeline 33 "$PIPELINE" .venv/bin/celery -A dfpipeline.worker worker \
    -Q check,track,build -c 2 -n local-pipeline@%h --loglevel=INFO

echo
say "the pipeline is running — Ctrl+C stops it"
echo "    queues       check · track · build (always on here; the DGX keeps PIPELINE_HOURS)"
echo "    on the app   Monitor → Pipeline lists local-pipeline"
echo

wait
