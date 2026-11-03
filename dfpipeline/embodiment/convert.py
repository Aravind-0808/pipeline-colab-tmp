"""A finished dataset, converted to another robot — on Hugging Face only.

The source is a LeRobot dataset on the Hub at a version tag (e.g. the SO-101
version of a task). The result is a new repo for another robot:

  1. the new repo is created, and the videos are copied into it server-side
     (`HfApi.copy_files`) — not one video byte reaches this machine. Our
     videos show the human hand, so they are right for any robot as they are;
  2. only `data/` and `meta/` are downloaded — the joints, a few KB an episode;
  3. every row is converted: the source arm's joints -> its end-effector pose
     (FK on its URDF) -> scaled into the target's reach -> the target's joints
     (IK on its URDF). The gripper's 0–100 % open carries over. Rows and
     timestamps are kept exactly, so the copied videos stay in sync;
  4. `info.json` (robot type, joint names and count) and the statistics are
     rewritten, and the new `data/` + `meta/` + card are uploaded and tagged.

Each episode is checked — how far the target's end-effector lands from where
it should, how many rows were clamped into its reach, how many ask more speed
than its joints have — and the numbers are kept in the card and the result.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dfpipeline.embodiment.robot import (
    SO101_PROFILE,
    W_FOLLOW_ROLL,
    W_FOLLOW_TILT,
    W_REST,
    Robot,
    load,
    reach_approach,
)

log = logging.getLogger("embodiment.convert")

SIDES = ("left", "right")
#: The gripper point of an arm on a table never goes lower than this above it.
MIN_EE_HEIGHT_M = 0.01
#: How another arm's IK follows a pose: as the clip retargeter follows a hand.
NATURAL_IK = {"soft": True, "w_home": W_REST, "w_axis": W_FOLLOW_TILT, "w_jaw": W_FOLLOW_ROLL}


# ----------------------------------------------------------------- encodings


@dataclass
class Encoding:
    """How one robot's arm joints are written in a dataset — from its URDF
    alone, never a calibration file: degrees are the URDF's angles, and
    -100..100 spans each joint's URDF limits, 0 halfway."""

    units: str                 # "degrees" or "m100" (LeRobot -100..100)
    mid_deg: np.ndarray        # m100: the angle 0 stands for (the middle of the limits)
    half_deg: np.ndarray       # m100: the half range 100 stands for

    def to_rad(self, v: np.ndarray) -> np.ndarray:
        deg = self.mid_deg + v / 100.0 * self.half_deg if self.units == "m100" else v
        return np.radians(deg)

    def from_rad(self, q: np.ndarray) -> np.ndarray:
        deg = np.degrees(q)
        if self.units == "m100":
            return np.clip((deg - self.mid_deg) / self.half_deg * 100.0, -100.0, 100.0)
        return deg


def encoding(robot: Robot, units: str) -> Encoding:
    """A robot's encoding, from its URDF's joint limits."""
    lower, upper = np.degrees(robot.lower), np.degrees(robot.upper)
    half = (upper - lower) / 2
    half[half == 0] = 180.0
    return Encoding(units, (upper + lower) / 2, half)


# ------------------------------------------------------------------- columns


def _layout(names: list[str], robot: Robot) -> dict[str, dict]:
    """Where each arm's joints and gripper sit in a row, by the robot's own
    LeRobot names: one arm (`x.pos`) or two (`left_arm_x.pos` …, or the
    robot's own side prefix, `left_x.pos` for an OpenArm). Keyed by side:
    "" for one arm, else "left" and "right"."""
    two = any(n.startswith(robot.prefix("left")) for n in names)
    out = {}
    for side in (SIDES if two else ("",)):
        arm_robot = robot.side(side)
        p = arm_robot.prefix(side) if side else ""
        wanted = [f"{p}{n}.pos" for n in arm_robot.names]
        arm_names = wanted[: arm_robot.dof]
        missing = [n for n in arm_names if n not in names]
        if missing:
            raise ValueError(f"The dataset's columns are not {robot.name}'s: expected {', '.join(arm_names)}")
        grip = wanted[-1] if arm_robot.gripper else None
        out[side] = {"arm": [names.index(n) for n in arm_names],
                     "gripper": names.index(grip) if grip in names else None}
    return out


def _names(robot: Robot, sides: list[str]) -> list[str]:
    names = []
    for side in sides:
        arm_robot = robot.side(side)
        p = arm_robot.prefix(side) if side else ""
        names += [f"{p}{n}.pos" for n in arm_robot.names]
    return names


# --------------------------------------------------------------- the convert


@dataclass
class Pair:
    source: Robot
    target: Robot
    src_enc: Encoding
    dst_enc: Encoding

    def arms(self, side: str) -> tuple[Robot, Robot, Encoding, Encoding]:
        """The source and target arm on that side, with their encodings (the
        right arm of a two-arm URDF has its own limits)."""
        src, dst = self.source.side(side), self.target.side(side)
        src_enc = self.src_enc if src is self.source else encoding(src, self.src_enc.units)
        dst_enc = self.dst_enc if dst is self.target else encoding(dst, self.dst_enc.units)
        return src, dst, src_enc, dst_enc

    @property
    def scale(self) -> float:
        return self.target.span / self.source.span


class EpisodeConverter:
    """One episode's rows, in order: each solved from the row before."""

    def __init__(self, pair: Pair, layout: dict[str, dict], sides: list[str], centres: dict | None = None):
        self.pair, self.layout, self.sides = pair, layout, sides
        self.arms = {side: pair.arms(side) for side in sides}
        # Each side's (source centre, target centre): where the recording's
        # hand mostly is, and where that lands on the target.
        self.centres = centres or {side: (a[0].work_point, a[1].work_point) for side, a in self.arms.items()}
        self.same = {side: _same_arm(a[0], a[1]) for side, a in self.arms.items()}
        self.seed: dict[str, np.ndarray | None] = dict.fromkeys(sides)
        self.cache: dict[bytes, np.ndarray] = {}
        self.errors: list[float] = []
        self.clamped = 0
        self.rows = 0

    def row(self, values: np.ndarray) -> np.ndarray:
        key = np.round(values, 4).tobytes()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        out = []
        for side in self.sides:
            cols = self.layout[side]
            src, dst, src_enc, dst_enc = self.arms[side]
            q_src = src_enc.to_rad(values[cols["arm"]].astype(float))
            pose = src.fk(q_src)
            # Where the source hand is from the middle of the recording,
            # scaled to the target's size, from the target's working point.
            src_centre, dst_centre = self.centres[side]
            target = dst_centre + (pose[:3, 3] - src_centre) * (dst.span / src.span)
            if dst.on_table and not self.same[side]:
                # Never into the table. When both stand on it, height is from
                # it, scaled: from the middle of the recording the table lay
                # below, and a bigger arm went through it (YAM: 42% of rows,
                # down to 24 cm under).
                z = pose[2, 3] * (dst.span / src.span) if src.on_table else target[2]
                target[2] = max(z, MIN_EE_HEIGHT_M)
            out_of_reach = float(np.linalg.norm(target - dst.anchor))
            if out_of_reach > dst.span:
                target = dst.anchor + (target - dst.anchor) * (dst.span / out_of_reach)
                self.clamped += 1
            approach, jaw = src.approach(pose), src.jaw(pose)
            if not self.same[side]:
                # The source gripper's tilt, from this arm's own shoulder.
                approach = reach_approach(dst.anchor, target, approach[2])
            seed = self.seed[side]
            # Another arm follows the pose as a hand's path is followed: off
            # its stops, its spare joints settled, position before the wrist's
            # angle. Without it a 7-joint arm parked joints on their stops in
            # 27-86% of rows, and a 5-joint one bent to copy a wrist it lacks.
            # The same arm keeps its own poses exactly.
            ik = {} if self.same[side] else NATURAL_IK
            q = dst.ik(target, approach, seed, jaw, **ik) if seed is not None else dst.first_ik(target, approach, jaw, **ik)
            self.seed[side] = q
            self.errors.append(float(np.linalg.norm(dst.fk(q)[:3, 3] - target)))
            out.extend(dst_enc.from_rad(q))
            if dst.gripper:
                # Open is open on any robot: through 0 (closed) .. 100 (open).
                pct = src.gripper_pct(float(values[cols["gripper"]])) if cols["gripper"] is not None else 100.0
                out.append(dst.gripper_value(pct))
        self.rows += 1
        result = np.asarray(out, dtype=np.float32)
        self.cache[key] = result
        return result


def _same_arm(a: Robot, b: Robot) -> bool:
    """Two descriptions of one arm: the same joints, placed the same way."""
    return a.arm == b.arm and len(a.path) == len(b.path) and all(
        np.allclose(x.origin, y.origin) for x, y in zip(a.path, b.path, strict=True))


def _centres(files: list[Path], layout: dict, sides: list[str], pair: Pair, every: int = 10) -> dict:
    """Each side's (source centre, target centre). The source centre is
    where the recording's hand mostly is — its median, from FK over the
    action rows; the target's is its arm's working point, so a recording
    made high in one arm's reach lands where the other works. Converting an
    arm to itself keeps every point where it was."""
    import pyarrow.parquet as pq

    points: dict[str, list[np.ndarray]] = {side: [] for side in sides}
    arms = {side: pair.arms(side) for side in sides}
    for path in files:
        rows = np.array(pq.read_table(path, columns=["action"]).column("action").to_pylist(), dtype=np.float64)
        for values in rows[::every]:
            for side in sides:
                src, _, src_enc, _ = arms[side]
                points[side].append(src.fk(src_enc.to_rad(values[layout[side]["arm"]]))[:3, 3])
    centres = {}
    for side in sides:
        src, dst = arms[side][0], arms[side][1]
        mid = np.median(points[side], axis=0) if points[side] else src.work_point
        centres[side] = (mid, mid.copy() if _same_arm(src, dst) else dst.work_point)
    return centres


def convert_folder(src: Path, dst: Path, pair: Pair, progress=lambda f, s: None) -> dict:
    """Convert a LeRobot dataset folder (data/ + meta/ only) into `dst` for the
    pair's target robot. Returns the per-episode report."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    columns = [k for k in ("action", "observation.state") if k in info["features"]]
    if "action" not in columns:
        raise ValueError("The dataset has no action column to convert")
    names = info["features"]["action"]["names"]
    layout = _layout(names, pair.source)
    sides = list(layout)
    new_names = _names(pair.target, sides)
    files = sorted((src / "data").rglob("*.parquet"))
    centres = _centres(files, layout, sides, pair)

    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)
    totals: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    per_episode: dict[int, dict[str, list[np.ndarray]]] = {}
    report: dict[int, dict] = {}
    converters: dict[int, EpisodeConverter] = {}
    speeds: dict[int, list[float]] = {}

    for n, path in enumerate(files):
        progress(0.2 + 0.6 * n / max(1, len(files)), f"converting {path.name}")
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_numpy()
        stamps = table.column("timestamp").to_numpy() if "timestamp" in table.column_names else None
        new_cols = {}
        work = len(episodes) * len(columns)
        for c, col in enumerate(columns):
            raw = np.array(table.column(col).to_pylist(), dtype=np.float64)
            out = np.zeros((len(raw), len(new_names)), dtype=np.float32)
            for i, (ep, values) in enumerate(zip(episodes, raw, strict=True)):
                if i and i % 1000 == 0:
                    # Along the file too: a file can hold most of a dataset.
                    done = (c * len(episodes) + i) / work
                    progress(0.2 + 0.6 * (n + done) / max(1, len(files)),
                             f"converting {path.name} · {c * len(episodes) + i}/{work} rows")
                ep = int(ep)
                conv = converters.setdefault(ep, EpisodeConverter(pair, layout, sides, centres)) if col == "action" \
                    else converters.setdefault(-ep - 1, EpisodeConverter(pair, layout, sides, centres))
                out[i] = conv.row(values)
            new_cols[col] = out
            totals[col].append(out)
            for ep in np.unique(episodes):
                per_episode.setdefault(int(ep), {}).setdefault(col, []).append(out[episodes == ep])

        # How fast the target's joints would have to move, row to row.
        if stamps is not None:
            act = new_cols["action"]
            arm_cols, speed = [], []
            at = 0
            for side in sides:
                arm = pair.target.side(side)
                arm_cols += list(range(at, at + arm.dof))
                speed += list(arm.velocity)
                at += len(arm.names)
            limit = np.degrees(np.array(speed)) if pair.dst_enc.units == "degrees" else None
            for ep in np.unique(episodes):
                rows = episodes == ep
                if limit is None or rows.sum() < 2:
                    continue
                dt = np.diff(stamps[rows])
                dq = np.abs(np.diff(act[rows][:, arm_cols], axis=0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    over = (dq / np.maximum(dt[:, None], 1e-6)) > limit[: len(arm_cols)]
                speeds.setdefault(int(ep), []).append(float(over.any(axis=1).mean()))

        width = len(new_names)
        arrays, fields = [], []
        for name in table.column_names:
            if name in new_cols:
                typ = pa.list_(pa.float32(), width)
                arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(new_cols[name].reshape(-1)), width))
                fields.append(pa.field(name, typ))
            else:
                arrays.append(table.column(name))
                fields.append(table.schema.field(name))
        dest = dst / path.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_table_one_row_group_per_episode(pa.Table.from_arrays(arrays, schema=pa.schema(fields)), dest)

    for ep, conv in converters.items():
        if ep < 0:
            continue
        errs = np.array(conv.errors or [0.0])
        report[ep] = {
            "episode": ep,
            "rows": conv.rows,
            "position_error_median_mm": round(float(np.median(errs)) * 1000, 2),
            "position_error_max_mm": round(float(errs.max()) * 1000, 2),
            "clamped_pct": round(100.0 * conv.clamped / max(1, conv.rows * len(sides)), 1),
            "over_speed_pct": round(100.0 * float(np.mean(speeds.get(ep, [0.0]))), 1),
        }

    # info.json: the target robot and its columns.
    progress(0.82, "writing the metadata")
    for col in columns:
        info["features"][col]["names"] = new_names
        info["features"][col]["shape"] = [len(new_names)]
    robot_type = pair.target.profile.get("robot_type") or pair.target.name
    # Two arms of the same robot: LeRobot's bimanual name (bi_so101_follower …).
    if len(sides) == 2 and not robot_type.startswith("bi_"):
        robot_type = f"bi_{robot_type}"
    info["robot_type"] = robot_type
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    # The statistics of the new columns, overall and per episode.
    def plain(stats: dict) -> dict:
        return {k: np.asarray(v).tolist() for k, v in stats.items()}

    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for col in columns:
        stats[col] = plain(get_feature_stats(np.concatenate(totals[col]), axis=0, keepdims=False))
    stats_path.write_text(json.dumps(stats, indent=4), encoding="utf-8")
    _rewrite_episode_stats(dst, per_episode, columns, get_feature_stats)

    return {"names": new_names, "robot_type": info["robot_type"], "episodes": [report[k] for k in sorted(report)],
            "total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames")}


def _rewrite_episode_stats(dst: Path, per_episode: dict, columns: list[str], get_feature_stats) -> None:
    """The `stats/<column>/<stat>` columns of meta/episodes, for the new values."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        eps = table.column("episode_index").to_pylist()
        for col in columns:
            per = {ep: get_feature_stats(np.concatenate(per_episode[ep][col]), axis=0, keepdims=False)
                   for ep in eps if ep in per_episode}
            for stat in next(iter(per.values()), {}):
                name = f"stats/{col}/{stat}"
                values = [np.asarray(per[ep][stat]).astype(float).reshape(-1).tolist() if ep in per else None
                          for ep in eps]
                arr = pa.array(values, type=pa.list_(pa.float64()))
                if name in table.column_names:
                    table = table.set_column(table.column_names.index(name), name, arr)
                else:
                    table = table.append_column(name, arr)
        pq.write_table(table, path, compression="snappy")


# ------------------------------------------------------------------ the robots


def robot_from_spec(spec: dict, work: Path, name: str) -> Robot:
    """A robot as the app describes it: its package in the store and its
    profile, or the SO-101 the pipeline ships with."""
    from dfpipeline.retarget import paths
    from dfpipeline.storage import get_storage

    if spec.get("package_path"):
        local = get_storage().fetch_local(spec["package_path"])
        folder = work / name
        with zipfile.ZipFile(local) as z:
            z.extractall(folder)
        return load(folder / spec["urdf_file"], spec["profile"], spec.get("name") or name)
    return load(paths.urdf(), spec.get("profile") or SO101_PROFILE, spec.get("name") or "SO-101")


def _card(spec: dict, result: dict, source_name: str, target_name: str) -> str:
    rows = "\n".join(
        f"| {e['episode']} | {e['rows']} | {e['position_error_median_mm']} | {e['position_error_max_mm']} | "
        f"{e['clamped_pct']} | {e['over_speed_pct']} |" for e in result["episodes"]
    )
    src = f"{spec['source']['repo_id']}@{spec['source'].get('revision') or 'main'}"
    return f"""---
tags:
- LeRobot
- robotics
- cross-embodiment
---

# {spec['target_repo_id'].split('/')[-1]}

**{target_name}** version of [{src}](https://huggingface.co/datasets/{spec['source']['repo_id']}), converted
from **{source_name}** by forward kinematics on the source robot's URDF and inverse kinematics on this
robot's URDF, row by row. The videos are the source dataset's, unchanged; rows and timestamps are kept,
so they stay in sync. The gripper's open and closed carry over, in each robot's own values.

- robot_type: `{result['robot_type']}`
- joints: {', '.join(f'`{n}`' for n in result['names'])}

| episode | rows | position error median (mm) | max (mm) | clamped into reach (%) | over joint speed (%) |
|---|---|---|---|---|---|
{rows}
"""


def convert_on_hub(spec: dict, progress=lambda f, s: None) -> dict:
    """The whole conversion, Hub to Hub. See the module docstring."""
    from huggingface_hub import HfApi, snapshot_download

    from dfpipeline.config import settings
    from dfpipeline.lerobot_builder import _ensure_private, _existing_tag

    api = HfApi(token=settings.hf_token)
    src = spec["source"]["repo_id"]
    rev = spec["source"].get("revision")
    dst = spec["target_repo_id"]
    tag = spec.get("tag") or "v0.1.0"
    private = spec.get("private", True)

    if _existing_tag(api, dst, tag) is not None:
        return {"repo_id": dst, "hub_url": f"https://huggingface.co/datasets/{dst}", "tag": tag, "already": True}

    with tempfile.TemporaryDirectory(prefix="convert-") as tmp:
        work = Path(tmp)
        progress(0.02, "reading the robots")
        source = robot_from_spec(spec["source_robot"], work, "source")
        target = robot_from_spec(spec["target_robot"], work, "target")

        progress(0.05, "fetching the joints (no videos)")
        snapshot_download(repo_id=src, repo_type="dataset", revision=rev, local_dir=work / "src",
                          allow_patterns=["data/**", "meta/**"], token=settings.hf_token)
        info = json.loads((work / "src" / "meta" / "info.json").read_text(encoding="utf-8"))
        names = info["features"]["action"]["names"]
        src_units = spec.get("source_units") or ("m100" if any(n.startswith("left_arm_") for n in names) else "degrees")
        pair = Pair(
            source, target,
            encoding(source, src_units),
            encoding(target, target.profile.get("units") or "degrees"),
        )
        result = convert_folder(work / "src", work / "dst", pair, progress)
        (work / "dst" / "README.md").write_text(_card(spec, result, source.name, target.name), encoding="utf-8")

        progress(0.88, "creating the repo")
        api.create_repo(repo_id=dst, repo_type="dataset", private=private, exist_ok=True)
        if private:
            _ensure_private(api, dst)
        progress(0.9, "copying the videos on the Hub")
        api.copy_files(f"hf://datasets/{src}@{rev}/videos/" if rev else f"hf://datasets/{src}/videos/",
                       f"hf://datasets/{dst}/videos/")
        progress(0.95, "uploading the converted joints")
        commit = api.upload_folder(
            repo_id=dst, repo_type="dataset", folder_path=work / "dst",
            commit_message=f"{target.name} version of {src}@{rev or 'main'}",
        )
        api.create_tag(repo_id=dst, tag=tag, repo_type="dataset", revision=getattr(commit, "oid", None), exist_ok=True)

    progress(1.0, "converted")
    return {
        "repo_id": dst,
        "hub_url": f"https://huggingface.co/datasets/{dst}",
        "tag": tag,
        "commit_sha": getattr(commit, "oid", None),
        "robot_type": result["robot_type"],
        "names": result["names"],
        "total_episodes": result["total_episodes"],
        "total_frames": result["total_frames"],
        "episodes": result["episodes"],
    }
