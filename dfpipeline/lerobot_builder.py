import contextlib
import hashlib
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

from dfpipeline import sources
from dfpipeline.config import settings
from dfpipeline import video

log = logging.getLogger("export.build")

# LeRobot SO-101 follower motors 1-6, in the order the retargeter writes them.
# Joints 1-5 are LeRobot degrees (SO101FollowerConfig(use_degrees=True)); the
# gripper is 0-100, 0 closed. Reordering or renaming these breaks every policy
# trained on what we publish.
SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# The names LeRobot's SO-101 follower gives these motors in a recorded
# dataset, and the robot_type it records. The visualizer's 3D replay matches
# both: without them it shows "robot type unknown" and cannot pose the arm.
SO101_FEATURE_NAMES = [f"{j}.pos" for j in SO101_JOINTS]
# Two arms, one per hand, named and scaled as a two-arm SO-101 dataset records
# them (robot_type "bi_so101_follower"; e.g. Qiu-Xinchuan/so101_towel): the
# left arm's six motors then the right's, `left_arm_shoulder_pan.pos` …
# `right_arm_gripper.pos`, joints in -100 .. 100 of their range.
TWO_ARM_SIDES = ("left", "right")
TWO_ARM_FEATURE_NAMES = [f"{side}_arm_{name}" for side in TWO_ARM_SIDES for name in SO101_FEATURE_NAMES]


def feature_names(arms: int) -> list[str]:
    """The state/action names of a dataset for this many arms."""
    return TWO_ARM_FEATURE_NAMES if arms == 2 else SO101_FEATURE_NAMES


def _robot_type(robot_kind: str | None, arms: int = 1) -> str | None:
    """LeRobot's robot_type for one of our robot kinds: "so101" -> "so101_follower",
    and for two of them, one per hand, "bi_so101_follower"."""
    if not robot_kind:
        return None
    kind = robot_kind.lower()
    single = f"{kind}_follower" if kind in ("so100", "so101") else kind
    return f"bi_{single}" if arms == 2 else single


def _info_total_frames(root: Path) -> int | None:
    try:
        return json.loads((root / "meta" / "info.json").read_text()).get("total_frames")
    except Exception:
        return None


def _resolve_repo_id(repo_id: str, token: str | None) -> str:
    if not token:
        return repo_id
    try:
        from huggingface_hub import HfApi

        me = HfApi(token=token).whoami()
        username = me.get("name")
        writable = {username, *[o["name"] for o in me.get("orgs", [])]}
        owner = repo_id.split("/", 1)[0] if "/" in repo_id else None
        if owner in writable:
            return repo_id
        name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
        return f"{username}/{name}"
    except Exception:
        return repo_id


def _dataset_class():
    """LeRobot moved this module, and the old path is not a fallback — it is
    simply gone.

    Up to 0.5 it lived under `lerobot.common.datasets`; from 0.6 it is
    `lerobot.datasets`. We imported the old one, so every export died on its
    first line with ModuleNotFoundError before touching a single clip. Both are
    tried so the builder works either side of that move.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import importlib
        mod = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        return getattr(mod, "LeRobotDataset")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot is not installed in the worker's Python environment. "
            "Please run the worker with the virtualenv Python (backend/.venv) or install with `pip install \".[export]\"`."
        ) from exc


def build_to_store(payload: dict, root: Path, progress=lambda p, s: None) -> dict:
    """Build the LeRobot dataset and keep it in the store (MinIO), at
    payload["store_prefix"]. Nothing goes to Hugging Face here: that is
    `publish`, when an admin asks."""
    try:
        from lerobot import __version__ as lerobot_version
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot is not installed in the worker's Python environment. "
            "Please run the worker with the virtualenv Python (backend/.venv) or install with `pip install \".[export]\"`."
        ) from exc

    LeRobotDataset = _dataset_class()

    from dfpipeline.storage import get_storage

    fps = int(payload["fps"])
    image_key = payload["image_key"]
    repo_id = payload["repo_id"]  # the name it will be published under
    episodes = payload["episodes"]
    robot_kind = payload.get("robot_kind")
    # Arms per episode: 2 for a dataset of two-hand tasks. An older app sends none.
    arms = int(payload.get("arms") or 1)
    # An imported robot's own LeRobot names and robot_type (the app sends them);
    # the SO-101's otherwise.
    names = list(payload.get("names") or feature_names(arms))
    robot_type = payload.get("robot_type") or _robot_type(robot_kind, arms)
    prefix = payload["store_prefix"].rstrip("/") + "/"

    # info.json is created before encoding starts; it alone is not proof that
    # a build is complete. Only reuse a finalized build with matching inputs.
    fingerprint = _build_fingerprint(payload)
    base = payload.get("base")
    resumed = (
        _find_completed_build(root, fingerprint) if payload.get("resume") and not base else None
    )
    if resumed is not None:
        root = resumed
        progress(0.85, "storing")
        files = get_storage().upload_dir(root, prefix)
        return {
            "total_frames": _info_total_frames(root),
            "lerobot_version": lerobot_version,
            "store_prefix": prefix,
            "files": files,
            "root": str(root),
            "repo_id": repo_id,
        }

    root = _clear_or_sidestep(root)

    progress(0.02, "starting")
    batch = max(1, settings.export_batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        # The frame size every other clip is scaled to comes from the first one
        # that can actually be read. Taking it from episode zero regardless
        # meant a single bad clip at the front killed an export whose other
        # forty-nine were fine — and it was the one download outside the loop's
        # tolerance.
        # Adding clips to a dataset: the previous version is fetched first, and
        # the new episodes are made at its frame size so the two can be merged.
        # A version on the Hub is added to with new files only: its index
        # (meta/) is fetched, never its data or videos (_APPEND_MARKER).
        base_root, base_shape, onto = None, None, None
        if base:
            progress(0.03, "fetching the previous version's index")
            base_root, onto = _fetch_base(base, work / "base")
            base_shape = _video_shape(base_root, image_key)

        skipped: list[dict] = []
        first, first_index = None, -1
        for i, ep in enumerate(episodes):
            try:
                candidate = sources.materialize(ep["clip"], work)
                h, w = video.probe_size(candidate)
                if not h or not w:
                    raise RuntimeError("could not read the frame size")
                first, first_index = candidate, i
                break
            except Exception as exc:
                log.warning("skipping episode %d while looking for a readable clip: %s", i + 1, exc)
                skipped.append({"episode": i, "reason": str(exc)[:200]})
        if first is None:
            raise RuntimeError(
                f"No episode could be built: all {len(episodes)} failed. "
                f"First reason: {skipped[0]['reason'] if skipped else 'unknown'}"
            )
        if base_shape:
            h, w = base_shape
        features = {image_key: {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"]}}
        if robot_kind:
            # The columns that make this a dataset a policy can be trained on.
            # Without them it is a video corpus with task strings: ACT and
            # everything like it read `observation.state` and `action`, and
            # there was nothing to read.
            features["observation.state"] = {
                "dtype": "float32", "shape": [len(names)], "names": names,
            }
            features["action"] = {
                "dtype": "float32", "shape": [len(names)], "names": names,
            }

        # Deliberately not lerobot's own batch_encoding_size: with it set,
        # finalize() dies in flush_pending_videos looking up an episode that is
        # not in its metadata yet. Encoding per episode is what the default
        # does, and it is already the behaviour we want — one video at a time.
        # H.264 on the GPU where it works, else AV1 on the CPU; a version merged
        # onto an older one keeps that one's codec (dfpipeline/gpu.py).
        from lerobot.configs.video import RGBEncoderConfig

        from dfpipeline import gpu

        vcodec = gpu.video_codec(_video_codec(base_root, image_key) if base_root else None)
        if onto is not None and (why := _cannot_append(base_root, features, fps,
                                                       robot_type)):
            if not base.get("repo_id") or base.get("store_prefix"):
                raise RuntimeError(
                    f"The previous version cannot take these episodes ({why}), and it is not on "
                    "the Hub yet to be merged in full: publish it first."
                )
            log.warning("adding to %s in full, not as new files: %s", repo_id, why)
            progress(0.04, "fetching the previous version")
            base_root, onto = _fetch_base(base, work / "base-full", whole=True)
        language = False
        if onto is not None:
            # New files beside the old ones: LeRobot's own resume, on the index
            # alone. It starts a new data, video and index file after the last
            # ones and never opens an old one.
            shutil.copytree(base_root, root, dirs_exist_ok=True)
            dataset = LeRobotDataset.resume(
                repo_id, root=str(root), rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
            )
            # Timeline columns are written after the episodes, as on a fresh build.
            language = _strip_language(dataset.meta)
            dataset.clear_episode_buffer()  # it was laid out with them
        else:
            dataset = LeRobotDataset.create(
                repo_id, fps, features=features, root=str(root), use_videos=True,
                robot_type=robot_type,
                rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
            )
        # Episode indexes continue after the previous version's when added to it.
        first_episode = int(dataset.meta.total_episodes) if onto is not None else 0

        total = 0
        kept = 0
        in_a_row = 0
        # Episode indexes are assigned as clips are accepted, and skipped clips
        # do not get one — so this cannot be derived from the input order.
        timelines: dict[int, list[dict]] = {}
        # Per episode, the clip second each row was taken from: a robot
        # episode runs slower than the video, so the check's step times
        # have to be found on this clock rather than multiplied by fps.
        clip_times: dict[int, list[float]] = {}
        # The clips that became episodes: the app releases their raw files.
        built_ids: list[str] = []
        for i, ep in enumerate(episodes):
            if i < first_index:
                continue  # already tried while looking for a readable clip
            # One clip at a time. Every episode used to be downloaded before
            # any work started — fifty clips on disk, and fifty more once each
            # was re-encoded, for a job that only ever looks at one of them.
            src = None
            try:
                src = first if i == first_index else sources.materialize(ep["clip"], work)
                frames = 0
                times: list[float] = []
                if robot_kind:
                    frames = _add_robot_episode(dataset, src, ep, fps, (h, w), work, image_key, times, names, arms)
                else:
                    for frame in video.iter_frames(src, fps, (h, w), work):
                        dataset.add_frame({image_key: frame, "task": ep["task"]})
                        times.append(frames / fps)
                        frames += 1
                if frames < settings.export_min_episode_frames:
                    raise RuntimeError(
                        f"only {frames} frame(s); an episode needs at least "
                        f"{settings.export_min_episode_frames}"
                    )
                dataset.save_episode()
                if ep.get("subtasks"):
                    timelines[first_episode + kept] = ep["subtasks"]
                    clip_times[first_episode + kept] = times
                if ep.get("video_id"):
                    built_ids.append(ep["video_id"])
                total += frames
                kept += 1
                in_a_row = 0
                if kept % batch == 0:
                    progress(
                        0.05 + 0.75 * (i + 1) / len(episodes),
                        f"built {kept} of {len(episodes)}",
                    )
            except Exception as exc:
                # One clip must not cost the other forty-nine. A link that
                # 500s, a file that will not decode — that is a fact about
                # that episode, not about the dataset, and an export that
                # threw it all away meant starting the whole build again.
                log.warning("skipping episode %d of %d: %s", i + 1, len(episodes), exc)
                skipped.append({"episode": i, "reason": str(exc)[:200]})
                dataset.clear_episode_buffer()
                in_a_row += 1
                if in_a_row >= settings.export_give_up_after:
                    # Skipping is for the odd bad clip. When this many fail one
                    # after another the fault is not in the clips, and grinding
                    # through the remaining three hundred to prove it costs
                    # hours of downloading and a disk full of half-built
                    # episodes. One run did exactly that: 52 failures, no
                    # video written, 117MB left behind.
                    raise RuntimeError(
                        f"Gave up after {in_a_row} episodes failed in a row "
                        f"({kept} of {len(episodes)} built). This is not one bad clip. "
                        f"Last reason: {exc}"
                    ) from exc
            finally:
                # Gone as soon as its frames are in, so the temp directory
                # holds one clip rather than the whole dataset.
                # Only a copy fetched for this build: a clip read where it lies
                # (a local file, the development store) is someone's original.
                if src is not None and _fetched_copy(src, work):
                    with contextlib.suppress(OSError):
                        src.unlink()
            progress(0.05 + 0.75 * (i + 1) / len(episodes), f"episode {i + 1}/{len(episodes)}")

        if kept == 0:
            raise RuntimeError(
                f"No episode could be built: all {len(episodes)} failed. "
                f"First reason: {skipped[0]['reason'] if skipped else 'unknown'}"
            )
        if skipped:
            # Pause before the upload when clips were failing. Whatever was
            # refusing them is very likely the thing we are about to ask for a
            # large transfer.
            time.sleep(settings.export_recover_pause_sec)

        progress(0.82, "finalizing")
        dataset.finalize()
        log.info("built %d of %d episodes (%d frames)", kept, len(episodes), total)

        # The sub-task timeline, from what the check already saw. This used to
        # be a second pass over the finished dataset asking the model a question
        # every couple of seconds of video — 1,257 calls where the check itself
        # made 349, none of them bounded by our own limits because it ran as a
        # subprocess talking to the gateway directly.
        annotation = {"annotated": False, "skipped": "no timelines on these clips"}
        if onto is not None:
            # The new files must have the columns the old ones have, no more.
            annotation = _annotate_new_files(root, timelines, clip_times, language, base_root)
        elif timelines:
            progress(0.84, "writing the timeline")
            try:
                written = _write_subtasks(root, timelines, clip_times)
                annotation = {"annotated": True, "episodes": written, "source": "validation"}
                log.info("wrote sub-tasks onto %d of %d episodes", written, kept)
            except Exception as exc:
                # The episodes are already on the Hub; a missing timeline is
                # not worth failing an export that otherwise worked.
                log.warning("could not write the sub-task timeline: %s", exc)
                annotation = {"annotated": False, "skipped": str(exc)[:200]}

        (root / ".export-complete.json").write_text(
            json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
        )

        # The new episodes go after the previous version's, in one dataset.
        stored, episodes_total = root, kept
        if onto is not None:
            meta = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            total = int(meta.get("total_frames") or total)
            episodes_total = int(meta.get("total_episodes") or kept)
            (root / _APPEND_MARKER).write_text(json.dumps(onto), encoding="utf-8")
        elif base_root is not None:
            progress(0.86, "merging with the previous version")
            stored = _merge(base_root, root, repo_id, work / "merged")
            meta = json.loads((stored / "meta" / "info.json").read_text(encoding="utf-8"))
            total = int(meta.get("total_frames") or total)
            episodes_total = int(meta.get("total_episodes") or kept)
            (stored / ".export-complete.json").write_text(
                json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
            )
        if robot_kind:
            _write_card(stored, repo_id, payload, episodes_total, total, fps)

        progress(0.88, "storing")
        files = get_storage().upload_dir(stored, prefix)

    return {
        # Every episode in this version, and the clips that are new in it.
        "episodes": episodes_total,
        "video_ids": built_ids,
        "total_frames": total,
        "lerobot_version": lerobot_version,
        "store_prefix": prefix,
        "files": files,
        "annotation": annotation,
        "repo_id": repo_id,
        # Where it was actually built, which is not always where it was asked
        # to build — a locked leftover sends it next door.
        "root": str(root),
    }


def _fetched_copy(path: Path, work: Path) -> bool:
    """Whether `path` is a copy made for this build — in its work directory, or
    downloaded from MinIO (storage.S3Storage.fetch_local) — and so safe to delete."""
    fetched = Path(tempfile.gettempdir()) / "dataforge-fetch"
    path = Path(path).resolve()
    return any(path.is_relative_to(d.resolve()) for d in (work, fetched))


# A stored version that holds only its new files, and the Hub version they are
# added to ({"repo_id", "revision"}). Its files are published beside that
# version's, never in place of them.
_APPEND_MARKER = ".export-append.json"


def _fetch_base(base: dict, dest: Path, whole: bool = False) -> tuple[Path, dict | None]:
    """The previous version, on this machine, and the Hub version new files are
    added to — None when it has to be merged in full.

    On the Hub: its index (meta/) only, a few MB at any size, unless `whole`.
    In the store (built, not yet pushed): all of it — a version of new files
    only is small, and its files go out again with the next one.
    """
    dest.mkdir(parents=True, exist_ok=True)
    onto = None
    if base.get("store_prefix"):
        from dfpipeline.storage import get_storage

        get_storage().download_prefix(base["store_prefix"], dest)
        marker = dest / _APPEND_MARKER
        if marker.is_file():
            onto = json.loads(marker.read_text(encoding="utf-8"))
            marker.unlink()
    elif base.get("repo_id"):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=base["repo_id"], repo_type="dataset", revision=base.get("revision"),
            local_dir=dest, token=settings.hf_token, allow_patterns=None if whole else ["meta/**"],
        )
        if not whole:
            onto = {"repo_id": base["repo_id"], "revision": base.get("revision")}
    else:
        raise RuntimeError("the previous version has no copy to merge onto")
    if not (dest / "meta" / "info.json").is_file():
        raise RuntimeError("the previous version could not be read (no meta/info.json)")
    return dest, onto


def _cannot_append(base_root: Path, features: dict, fps: int, robot_type: str | None) -> str | None:
    """Why the new episodes cannot go beside the previous version's as they
    are, or None. New files are read with the old ones' index, so everything
    that shapes a file has to be the same."""
    from lerobot.datasets.language import is_language_column

    info = json.loads((base_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if int(info.get("fps") or 0) != fps:
        return f"its fps is {info.get('fps')}, these are {fps}"
    if info.get("robot_type") != robot_type:
        return f"its robot is {info.get('robot_type')}, these are {robot_type}"
    theirs = {k: v for k, v in (info.get("features") or {}).items()
              if k not in _DEFAULT_COLUMNS and not is_language_column(k)}
    if set(theirs) != set(features):
        return f"its columns are {sorted(theirs)}, these are {sorted(features)}"
    for key, ours in features.items():
        for field in ("dtype", "shape", "names"):
            mine, old = ours.get(field), theirs[key].get(field)
            if (mine != old) if field == "dtype" else (list(mine or []) != list(old or [])):
                return f"its {key} {field} is {old}, these are {mine}"
    return None


_DEFAULT_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _strip_language(meta) -> bool:
    """Take the timeline columns off while episodes are added — LeRobot would
    want them on every frame — and say whether there were any. They are put
    back on the new files by `_annotate_new_files`."""
    from lerobot.datasets.language import is_language_column

    language = [k for k in meta.info.features if is_language_column(k)]
    meta.info.features = {k: v for k, v in meta.info.features.items() if k not in language}
    return bool(language)


def _annotate_new_files(root: Path, timelines: dict, clip_times: dict, language: bool,
                        base_root: Path) -> dict:
    """The timeline on the new files, in the old files' columns exactly.

    LeRobot reads every index file as one table, so a new one with a column
    the old ones lack (or without one they have) breaks the whole dataset.
    """
    spans = {
        ep: rows
        for ep, steps in timelines.items()
        if (rows := _step_rows(steps, clip_times.get(ep) or []))
    }
    old_index = {p.relative_to(base_root) for p in (base_root / "meta" / "episodes").rglob("*.parquet")}
    written = 0
    try:
        if language:  # the old files have the columns: the new ones too, empty or not
            _write_language(root, spans)
        if spans:
            written = _write_episode_subtasks(root, spans)
        _match_index_columns(root, old_index)
    except Exception as exc:  # noqa: BLE001 - the episodes are fine without a timeline
        log.warning("could not write the sub-task timeline: %s", exc)
        return {"annotated": False, "skipped": str(exc)[:200]}
    if not spans:
        return {"annotated": False, "skipped": "no timelines on these clips"}
    return {"annotated": True, "episodes": written, "source": "validation"}


def _match_index_columns(root: Path, old_index: set[Path]) -> None:
    """Give the new episode index files the old ones' columns and types."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not old_index:
        return
    schema = pq.read_schema(root / max(old_index))
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        if path.relative_to(root) in old_index:
            continue
        table = pq.read_table(path)
        if table.schema.equals(schema):
            continue
        columns = [
            table.column(f.name).cast(f.type) if f.name in table.column_names
            else pa.nulls(table.num_rows, type=f.type)
            for f in schema
        ]
        pq.write_table(pa.Table.from_arrays(columns, schema=schema), path, compression="snappy")


def _video_codec(root: Path, image_key: str) -> str | None:
    """The codec a built dataset's videos are in, from its info.json."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
        return (info["features"][image_key].get("info") or {}).get("video.codec")
    except (OSError, KeyError, ValueError):
        return None


def _video_shape(root: Path, image_key: str) -> tuple[int, int] | None:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    shape = ((info.get("features") or {}).get(image_key) or {}).get("shape")
    return (int(shape[0]), int(shape[1])) if shape and len(shape) >= 2 else None


def _merge(base_root: Path, new_root: Path, repo_id: str, out: Path) -> Path:
    """The previous version and the new episodes, as one LeRobot dataset."""
    from lerobot.datasets.aggregate import aggregate_datasets

    base_root = _bring_up_to_date(base_root, new_root, repo_id)
    # A version built before the subtask timeline was written as language
    # columns has none; give it empty ones, or LeRobot refuses the merge.
    if _has_language(base_root) != _has_language(new_root):
        _write_language(new_root if _has_language(base_root) else base_root, {})
    aggregate_datasets(
        [f"{repo_id}-base", f"{repo_id}-new"], repo_id,
        roots=[base_root, new_root], aggr_root=out,
    )
    return out


def _bring_up_to_date(base_root: Path, new_root: Path, repo_id: str) -> Path:
    """The previous version, made mergeable with — and as clean as — the new one.

    Versions built before these rules have no robot_type, joint names without
    LeRobot's `.pos`, and may hold episodes too short to train on. LeRobot
    merges only identical robot types and features, and a short episode merged
    in is published again in every later version.
    """
    info_path = base_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    new = json.loads((new_root / "meta" / "info.json").read_text(encoding="utf-8"))

    changed = False
    if info.get("robot_type") != new.get("robot_type"):
        info["robot_type"] = new.get("robot_type")
        changed = True
    for key in ("observation.state", "action"):
        theirs, ours = info["features"].get(key), new["features"].get(key)
        if theirs and ours and theirs.get("names") != ours.get("names") \
                and len(theirs.get("names") or []) == len(ours.get("names") or []):
            theirs["names"] = ours["names"]
            changed = True
    if changed:
        info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")

    short = _short_episodes(base_root)
    if not short:
        return base_root
    log.info("dropping %d episode(s) under %d frames from the previous version: %s",
             len(short), settings.export_min_episode_frames, short)
    return _drop_episodes(base_root, short, repo_id)


def _short_episodes(root: Path) -> list[int]:
    import pandas as pd

    lengths = pd.concat(
        pd.read_parquet(p, columns=["episode_index", "length"])
        for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))
    )
    return sorted(int(e) for e in lengths[lengths["length"] < settings.export_min_episode_frames]["episode_index"])


def _drop_episodes(root: Path, episodes: list[int], repo_id: str) -> Path:
    """`root` without these episodes, as a new dataset beside it.

    LeRobot's own tool, which renumbers what is left and cuts the videos. It
    keeps the language columns but not the older per-episode subtask lists,
    so those are written again from the language columns.
    """
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = root.parent / f"{root.name}-trimmed"
    dataset = LeRobotDataset(f"{repo_id}-base", root=root, video_backend="pyav")
    delete_episodes(dataset, episodes, output_dir=out, repo_id=f"{repo_id}-base")
    if _has_language(out):
        _write_episode_subtasks(out, _spans_from_language(out))
    return out


def _spans_from_language(root: Path) -> dict[int, list[tuple[str, int, int]]]:
    """Each episode's subtask steps as (name, first row, last row), read back
    from its `language_persistent` column: a step runs until the next begins."""
    import pandas as pd

    data = pd.concat(
        pd.read_parquet(p, columns=["episode_index", "timestamp", "language_persistent"])
        for p in sorted((root / "data").rglob("*.parquet"))
    )
    spans: dict[int, list[tuple[str, int, int]]] = {}
    for ep, rows in data.groupby("episode_index", sort=True):
        stamps = [round(float(t), 4) for t in rows["timestamp"]]
        first = rows["language_persistent"].iloc[0]
        steps = list(first) if first is not None else []
        starts = [stamps.index(round(float(s["timestamp"]), 4)) for s in steps
                  if round(float(s["timestamp"]), 4) in stamps]
        names = [str(s["content"]) for s in steps if round(float(s["timestamp"]), 4) in stamps]
        if starts:
            ends = [*(b - 1 for b in starts[1:]), len(stamps) - 1]
            spans[int(ep)] = list(zip(names, starts, ends, strict=True))
    return spans


# LeRobot gripper.pos: 0 is closed. Commanded fully closed while holding, as a
# person teleoperating squeezes it: the object stops the jaws, not the command.
# Open is where teleoperated SO-101 data opens to before a grasp: 36 is the
# 90th percentile of gripper.pos in Qiu-Xinchuan/so101_box (max 71).
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 35.0
_GRASP_WORDS = ("pick", "grasp", "grab", "lift", "take", "hold")
_RELEASE_WORDS = ("place", "put", "set", "release", "drop", "return", "lower")


def _hold_window(steps: list[dict]) -> tuple[float, float] | None:
    """When the hand holds the object, in clip seconds, from the check's steps.

    It closes as the first step that takes the object begins, and opens half
    way through the last step that puts it down — the moment it touches the
    table, not the end of the clip.

    The retargeter closes the gripper on the thumb-index distance, which is a
    pinch. A hand around a cup keeps thumb and index a cup apart, so the grasp
    went unseen: the gripper was closed in 6% of a pick-and-place. The check
    named the steps it watched — "pick up the cup" ... "place the cup back" —
    so the hold runs from the first step that takes the object to the end of
    the last that puts it down. None when the steps do not say.
    """
    def says(step: dict, words) -> bool:
        name = str(step.get("name") or "").lower()
        return any(w in name for w in words)

    ordered = sorted(steps, key=lambda s: float(s.get("start") or 0.0))
    grasp = next((s for s in ordered if says(s, _GRASP_WORDS)), None)
    if grasp is None:
        return None
    start = float(grasp["start"])
    after = [s for s in ordered if float(s["start"]) >= start]
    release = [s for s in after if says(s, _RELEASE_WORDS)]
    if release:
        last = release[-1]
        end = (float(last["start"]) + float(last["end"])) / 2
    else:
        end = float([s for s in after if says(s, _GRASP_WORDS)][-1]["end"])
    return (start, end) if end > start else None


def _add_robot_episode(dataset, src, ep: dict, fps: int, size, work: Path, image_key: str,
                       times: list[float], names: list[str] = SO101_FEATURE_NAMES, arms: int = 1) -> int:
    """One LeRobot row per robot sample, with the video frame that sample came from.

    The two clocks do not agree. The retargeter's samples are on the robot's
    own motor-limited clock — about 1.6 to 2 times slower than the human, so
    there are more of them than there are video frames — and each one records
    the source frame it was solved from.

    So the row count follows the robot, and a video frame is repeated for as
    many samples as point at it. The episode then lasts as long as the arm
    would really take, and every row is a consistent (image, state, action)
    triple at a fixed rate, which is what a policy needs. Because the samples
    are ordered by source frame, the video is still walked once, streaming, as
    it is for a video-only export.

    The home phases are left out. They are the arm travelling to and from its
    calibration pose with no video behind them, and a policy trained on them
    learns to move while nothing in the image changes.
    """
    import numpy as np

    from dfpipeline.retarget.service import load

    trajectory = load(ep["trajectory_path"])
    two_arms = arms == 2
    if bool(trajectory.get("arms")) != two_arms:
        # A one-arm trajectory in a two-arm dataset, or the other way round:
        # the task was changed between hands after its clips were solved.
        raise RuntimeError(
            f"its trajectory is for {'two arms' if trajectory.get('arms') else 'one arm'}, "
            f"this dataset is {'two-arm' if two_arms else 'one-arm'}"
        )
    records = [
        r for r in (trajectory.get("trajectory") or [])
        if r.get("phase") == "action" and r.get("action")
    ]
    if not records:
        raise RuntimeError("its trajectory has no action frames")
    records.sort(key=lambda r: (int(r.get("frame") or 0), float(r.get("timestamp") or 0.0)))

    def vector(record: dict, key: str) -> "np.ndarray":
        action = record["action"]
        return np.asarray([float(action[n]) for n in names], dtype=np.float32)

    # The steps' hold window is one gripper's; with two, each follows its own
    # hand. And a gripper that already follows the hand and its holds
    # (grasp.py) is better than the steps' guess: only an older trajectory's,
    # read off a pinch alone ("fingertips"), is replaced by it.
    from_fingertips = (trajectory.get("gripper") or {}).get("source", "fingertips") == "fingertips"
    hold = _hold_window(ep.get("subtasks") or []) if from_fingertips and not two_arms else None

    # Walk the video once, holding each decoded frame for as long as the robot
    # is still on it.
    written = 0
    cursor = 0
    previous_action = None
    for index, frame in enumerate(video.iter_frames(src, fps, size, work)):
        while cursor < len(records) and int(records[cursor].get("frame") or 0) == index:
            record = records[cursor]
            action = vector(record, "action")
            if hold:
                # The steps say when it is held, so the fingers do not: a hand
                # reaching with its fingers together read as a closed gripper,
                # and the arm came at the cup already shut.
                held = hold[0] <= index / fps <= hold[1]
                action[-1] = GRIPPER_CLOSED if held else GRIPPER_OPEN
            # No robot ran this, so the state is the arm where the last
            # command put it: the IK trajectory, assuming perfect tracking.
            state = previous_action if previous_action is not None else action
            previous_action = action
            dataset.add_frame(
                {
                    image_key: frame,
                    "observation.state": state,
                    "action": action,
                    "task": ep["task"],
                }
            )
            times.append(index / fps)
            written += 1
            cursor += 1
        if cursor >= len(records):
            break

    if written == 0:
        raise RuntimeError(
            "no robot sample lined up with a video frame — the trajectory and the clip disagree"
        )
    return written


def _write_card(root: Path, repo_id: str, payload: dict, episodes: int, frames: int, fps: int) -> None:
    """Say what this dataset is, and what is wrong with it.

    Two things have to be on the card or whoever trains on this is misled.
    The state column is not a measurement — there was no robot in the loop,
    only the IK trajectory. And the geometry
    is uncalibrated: the focal length is approximated from the frame size and
    the hand is MediaPipe's average, so absolute distances carry a few
    centimetres of error.

    Best effort. The episodes are built by the time this runs, and a missing
    README is not worth failing an export over.
    """
    calibrated = sum(1 for ep in payload["episodes"] if ep.get("calibrated"))
    if int(payload.get("arms") or 1) == 2:
        joint_order = (
            f"Two arms, one per hand of the person: 12 values, the left arm's six "
            f"({', '.join(SO101_JOINTS)}) then the right arm's, named "
            f"`left_arm_shoulder_pan.pos` … `right_arm_gripper.pos` as a two-arm SO-101 "
            f"(`bi_so101_follower`) records them. Joints 1-5 are -100 to 100 of each "
            f"joint's calibrated range (`use_degrees=False`), the gripper 0-100. The left "
            f"arm is the hand that worked on the left of the picture. Both arms share one clock."
        )
    else:
        joint_order = f"Joint order: {', '.join(SO101_JOINTS)} — SO-101 follower motors 1 to 6."
    # An imported robot (its own URDF, names and robot_type): say so, not SO-101.
    names = payload.get("names")
    robot = payload.get("robot_name") or str(payload.get("robot_kind") or "so101").upper()
    tag = payload.get("robot_type") or "so101"
    state = "Joint positions, LeRobot degrees for joints 1-5 and 0-100 for the gripper"
    units = ("The arm units are LeRobot degrees, as `SO101FollowerConfig(use_degrees=True)`\n"
             "reports them; the gripper is 0-100 of its own travel, 0 closed.")
    made = "the SO-101 by", "the SO-101 URDF", "the servos' speed\nand acceleration limits"
    if names:
        arms = int(payload.get("arms") or 1)
        state = "Joint positions in the robot's dataset units (its URDF's angles), the gripper in its own values"
        joint_order = (
            f"{len(names)} values, `{names[0]}` … `{names[-1]}`, robot type `{tag}`"
            + (": the left arm is the hand that worked on the left of the picture, "
               "and both arms share one clock." if arms == 2 else ".")
        )
        units = ("Each robot is read from its own URDF: no calibration file. The gripper runs\n"
                 "between the robot's own dataset values for open and closed.")
        made = f"the {robot} by", "its URDF", "its joints' speed\nlimits"
    card = f"""---
tags:
- lerobot
- dataforge
- robotics
- {tag}
task_categories:
- robotics
---

# {repo_id}

A [LeRobot](https://github.com/huggingface/lerobot) dataset for the
**{robot}**, built by DataForge from human
demonstration video. {episodes} episodes, {frames} frames at {fps} fps.

## Columns

| Column | What it is |
|---|---|
| `{payload.get('image_key', 'observation.images.top')}` | The operator's video, re-encoded |
| `observation.state` | {state} |
| `action` | The commanded joint positions, same units |

{joint_order}
{units}

## How it was made

Nobody teleoperated a robot to record this. Each episode is a phone video of a
person doing the task with their own hand, retargeted onto {made[0]}
`{payload.get('pipeline_version', 'hand2robot')}`: a 3D hand reconstruction, then
differential IK onto {made[1]}, then retiming inside {made[2]}.

Episodes are on the **robot's** clock, which runs about 1.6 to 2 times slower
than the human's, so video frames repeat where the arm moves slowly. The
travel to and from the calibration pose is not included.

Only clips that passed both an automatic content check and a trajectory
quality gate are here. The gate rejects, among other things, clips where the
manipulated object was tracked in too few frames, where too much of the hand
path lies outside the arm's reach.

## What to know before you train on it

- **`observation.state` is not measured.** It is the previous commanded
  position: the IK trajectory, assuming the servos track it perfectly. No
  physical robot executed these trajectories.
- **The gripper closes when the object is held**, timed from the steps the
  automatic check named ("pick up ...", "place ..."). A hand around an object
  keeps thumb and index apart, so it is not read off the fingers. Episodes
  whose steps do not say follow the thumb-index distance.
- **The geometry is uncalibrated.** The camera focal length is approximated
  from the frame size and the hand scale is a population average, so expect a
  few centimetres of absolute error.
- **{calibrated} of {episodes} episodes** were solved against a real
  `lerobot-calibrate` file. Episodes without one assume symmetric joint
  ranges: fine to train on, wrong to replay on a particular arm.
- **Open loop.** The object is where the video said it was. A real arm needs
  the same layout relative to its base, or its own perception.
"""
    try:
        (root / "README.md").write_text(card, encoding="utf-8")
    except OSError as exc:
        log.warning("could not write the dataset card: %s", exc)


def _clear_or_sidestep(root: Path) -> Path:
    """Reserve a private workspace; LeRobot must create its own absent root.

    Never delete a shared repository directory: another worker may still be
    writing it, and Windows may hold its parquet files open after a failure.
    mkdtemp reserves the parent atomically even for simultaneous exports.
    """
    root.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f"{root.name}-build-", dir=root.parent))
    return workspace / "dataset"


def _build_fingerprint(payload: dict) -> str:
    inputs = {
        # robot_kind and pipeline_version belong here: without them a resume
        # would hand back a video-only build for a robot export, or a build
        # made by an older retargeter, and call it complete.
        key: payload.get(key)
        for key in ("fps", "image_key", "robot_id", "robot_kind", "pipeline_version")
    }
    # Signed media URLs change between requests, but the clip path is stable.
    inputs["episodes"] = [
        {"video": ep["clip"]["path"],
         "task": ep["task"],
         "trajectory": ep.get("trajectory_path")}
        for ep in payload["episodes"]
    ]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _find_completed_build(root: Path, fingerprint: str) -> Path | None:
    candidates = [root, *root.parent.glob(f"{root.name}-build-*/dataset")]
    completed = []
    for candidate in candidates:
        marker = candidate / ".export-complete.json"
        try:
            if (json.loads(marker.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
                    and (candidate / "meta" / "info.json").is_file()):
                completed.append((marker.stat().st_mtime_ns, candidate))
        except (OSError, ValueError, AttributeError):
            continue
    return max(completed, key=lambda entry: entry[0])[1] if completed else None


def _step_rows(steps: list[dict], clip_times: list[float]) -> list[tuple[str, int, int]]:
    """Each step as (name, first row, last row) of the episode.

    A step's times are seconds of the clip; a row's clip second is in
    `clip_times`. A step starts on the first row taken at or after its start
    and ends on the last row taken before its end. A step no row falls in is
    dropped, and so is one that starts on the same row as the one before it —
    LeRobot keeps one subtask per moment.
    """
    import bisect

    out: list[tuple[str, int, int]] = []
    for step in sorted(steps, key=lambda s: float(s["start"])):
        first = bisect.bisect_left(clip_times, float(step["start"]))
        last = bisect.bisect_left(clip_times, float(step["end"])) - 1
        if first >= len(clip_times) or last < first:
            continue
        if out and out[-1][1] == first:
            out.pop()
        out.append((str(step["name"]), first, last))
    return out


def _write_subtasks(root: Path, timelines: dict[int, list[dict]],
                    clip_times: dict[int, list[float]]) -> int:
    """Write the sub-task timeline onto the episodes, from what the check saw.

    This is what lerobot-annotate produced by walking the finished dataset and
    asking the model a question every couple of seconds of video — 1,257 calls
    on a 42-minute dataset, against 349 for the check itself. The check already
    looks at a timestamped grid of every clip and can name the steps in the
    same answer, so the timeline arrives with the verdict and this only has to
    write it down.

    Two places, because LeRobot reads two:

    - `language_persistent` on every frame, `style: "subtask"` rows — the
      LeRobot 0.6 format, the one its annotation viewer and its chat-template
      processors read;
    - the older `subtask_*` lists on each episode's metadata row.

    Best effort. A dataset without a timeline is still a dataset.
    """
    if not timelines:
        return 0

    spans = {
        ep: rows
        for ep, steps in timelines.items()
        if (rows := _step_rows(steps, clip_times.get(ep) or []))
    }
    if not spans:
        return 0

    _write_language(root, spans)
    return _write_episode_subtasks(root, spans)


def _subtask_row(name: str, timestamp: float) -> dict:
    return {"role": "assistant", "content": name, "style": "subtask",
            "timestamp": float(timestamp), "camera": None, "tool_calls": None}


def _write_language(root: Path, spans: dict[int, list[tuple[str, int, int]]]) -> None:
    """The `language_persistent` and `language_events` columns, laid out as
    lerobot-annotate lays them out: every frame of an episode carries the whole
    persistent list, each row stamped with the frame time it begins at.

    The column types are fixed rather than inferred, so a file whose episodes
    have no steps has the same schema as one whose episodes do — versions are
    merged, and LeRobot merges only identical features.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.io_utils import (
        load_info,
        write_info,
        write_table_one_row_group_per_episode,
    )
    from lerobot.datasets.language import (
        LANGUAGE_COLUMNS,
        LANGUAGE_EVENTS,
        LANGUAGE_PERSISTENT,
        language_feature_info,
    )

    persistent_type = pa.array([[_subtask_row("", 0.0)]]).type
    events_type = pa.array([[]]).type

    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        episodes = [int(e) for e in table.column("episode_index").to_pylist()]
        stamps = table.column("timestamp").to_pylist()

        # Each episode's frame times, in row order, to stamp its steps with.
        frame_times: dict[int, list[float]] = {}
        for ep, ts in zip(episodes, stamps, strict=True):
            frame_times.setdefault(ep, []).append(float(ts))
        persistent_by_ep = {
            ep: [_subtask_row(name, times[first]) for name, first, _ in spans.get(ep, []) if first < len(times)]
            for ep, times in frame_times.items()
        }

        keep = [n for n in table.column_names if n not in LANGUAGE_COLUMNS]
        table = pa.Table.from_arrays(
            [table.column(n) for n in keep] + [
                pa.array([persistent_by_ep[ep] for ep in episodes], type=persistent_type),
                pa.array([[] for _ in episodes], type=events_type),
            ],
            names=[*keep, LANGUAGE_PERSISTENT, LANGUAGE_EVENTS],
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        write_table_one_row_group_per_episode(table, tmp)
        tmp.replace(path)

    info = load_info(root)
    info.features = {**info.features, **language_feature_info()}
    write_info(info, root)


def _has_language(root: Path) -> bool:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return "language_persistent" in (info.get("features") or {})


# The five list columns LeRobot keeps per episode, and the legacy copies it
# writes alongside them. Taken from lerobot's own writer so the file we produce
# is the file its tools expect.
_SUBTASK_COLUMNS = (
    "subtask_names",
    "subtask_start_times",
    "subtask_end_times",
    "subtask_start_frames",
    "subtask_end_frames",
)


def _write_episode_subtasks(root: Path, spans: dict[int, list[tuple[str, int, int]]]) -> int:
    """The older per-episode `subtask_*` lists, on the dataset's own clock."""
    import pandas as pd

    fps = float(json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"])
    written = 0
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if "episode_index" not in frame.columns:
            continue

        # Both spellings: lerobot writes the prefixed columns and a legacy copy
        # without the prefix, and its readers look for one or the other.
        columns = [f"sparse_{c}" for c in _SUBTASK_COLUMNS] + list(_SUBTASK_COLUMNS)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None

        touched = False
        for row in frame.index:
            steps = spans.get(int(frame.at[row, "episode_index"]))
            if not steps:
                continue
            values = [
                [name for name, _, _ in steps],
                [first / fps for _, first, _ in steps],
                [(last + 1) / fps for _, _, last in steps],
                [first for _, first, _ in steps],
                [last + 1 for _, _, last in steps],
            ]
            for column, value in zip(columns, values * 2, strict=True):
                frame.at[row, column] = value
            touched = True
            written += 1

        if touched:
            frame.to_parquet(path, engine="pyarrow", compression="snappy")

    return written


def _existing_tag(api, repo_id: str, tag: str) -> str | None:
    """The commit a tag points at, or None if the repo or the tag is not there."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return None
    return next((t.target_commit for t in refs.tags if t.name == tag), None)


def _require_complete(root: Path, store_prefix: str) -> None:
    """Refuse to publish a copy that is not a whole dataset.

    The upload replaces the repo's data, meta and videos with what is here, so
    an empty or partial download would delete the published dataset.
    """
    missing = [
        what for what, ok in (
            ("meta/info.json", (root / "meta" / "info.json").is_file()),
            ("data files", any((root / "data").rglob("*.parquet"))),
            ("episode metadata", any((root / "meta" / "episodes").rglob("*.parquet"))),
        ) if not ok
    ]
    if missing:
        raise RuntimeError(
            f"The stored copy at {store_prefix} is incomplete (no {', '.join(missing)}); "
            "refusing to publish it over the repo."
        )


def _ensure_private(api, repo_id: str) -> None:
    """A private dataset is private before anything is uploaded to it.

    Creating a repo does not make an existing one private, and a repo asked
    for as private once came back public (huggingface_hub 1.33). So it is
    checked, made private if it is not, and nothing is uploaded unless it is.
    """
    if api.repo_info(repo_id, repo_type="dataset").private:
        return
    log.warning("%s is public; making it private before uploading", repo_id)
    api.update_repo_settings(repo_id, private=True, repo_type="dataset")
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise RuntimeError(f"{repo_id} could not be made private, so nothing was uploaded to it")


def publish(store_prefix: str, repo_id: str, private: bool, progress=lambda p, s: None,
            version: str | None = None) -> dict:
    """Send a stored dataset version to its Hugging Face repo — on an admin's
    request, and the only thing that ever goes to Hugging Face."""
    from huggingface_hub import HfApi

    from dfpipeline.storage import get_storage

    token = settings.hf_token
    if not token:
        raise RuntimeError("HF_TOKEN is required to publish")
    repo_id = _resolve_repo_id(repo_id, token)
    api = HfApi(token=token)
    tag = f"v{version.lstrip('v')}" if version else None

    # Pushed already: its tag is there. A second push of the same version — a
    # double click, a task redelivered after a restart — runs after the first
    # has deleted the stored copy, and uploading what is left of it would
    # replace the repo's files with nothing. It did, once.
    if tag:
        pushed = _existing_tag(api, repo_id, tag)
        if pushed is not None:
            log.warning("%s %s is already on the Hub; not pushing it again", repo_id, tag)
            progress(1.0, "already published")
            return {"repo_id": repo_id, "hub_url": f"https://huggingface.co/datasets/{repo_id}",
                    "commit_sha": pushed, "tag": tag}

    with tempfile.TemporaryDirectory(prefix="publish-") as tmp:
        progress(0.1, "fetching")
        get_storage().download_prefix(store_prefix, Path(tmp))
        _require_complete(Path(tmp), store_prefix)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        if private:
            _ensure_private(api, repo_id)
        progress(0.4, "publishing")
        message = (f"Version {version}" if version else
                   f"Publish {store_prefix.rstrip('/').rsplit('/', 1)[-1]}")
        marker = Path(tmp) / _APPEND_MARKER
        if marker.is_file():
            # New files beside the version they add to: nothing is deleted, and
            # only what is new (and the index) goes up.
            onto = json.loads(marker.read_text(encoding="utf-8"))
            if _existing_tag(api, repo_id, onto.get("revision") or "") is None:
                raise RuntimeError(
                    f"{repo_id} has no {onto.get('revision')} on the Hub, and this version only "
                    "adds to it: publish that one first."
                )
            info = api.upload_folder(
                repo_id=repo_id, repo_type="dataset", folder_path=tmp, commit_message=message,
                ignore_patterns=[".export-*"],
            )
        else:
            # A whole dataset replaces the repo's files: a layout that changed
            # between versions must not leave stale ones.
            info = api.upload_folder(
                repo_id=repo_id, repo_type="dataset", folder_path=tmp, commit_message=message,
                delete_patterns=["data/**", "videos/**", "meta/**"],
            )
        if tag:
            # Every version stays downloadable at its tag after the next is pushed.
            api.create_tag(repo_id=repo_id, tag=tag, repo_type="dataset",
                           revision=getattr(info, "oid", None), exist_ok=True)
    progress(1.0, "published")
    return {
        "repo_id": repo_id,
        "hub_url": f"https://huggingface.co/datasets/{repo_id}",
        "commit_sha": getattr(info, "oid", None),
        "tag": tag,
    }
