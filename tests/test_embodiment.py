"""A finished dataset converted to another robot (dfpipeline/embodiment)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="needs the 'ml' extra")

from dfpipeline.embodiment import convert as C
from dfpipeline.embodiment.robot import SO101_PROFILE, Robot, load

SO101_URDF = Path(__file__).parents[1] / "models" / "so101" / "so101_new_calib.urdf"

ARM6 = """<?xml version="1.0"?>
<robot name="arm6">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>
  <link name="l5"/><link name="l6"/><link name="tool"/><link name="finger"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.12"/>
    <axis xyz="0 0 1"/><limit lower="-3.0" upper="3.0" velocity="3" effort="1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.05"/>
    <axis xyz="0 1 0"/><limit lower="-2.0" upper="2.0" velocity="3" effort="1"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.30"/>
    <axis xyz="0 1 0"/><limit lower="-2.5" upper="2.5" velocity="3" effort="1"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0.25 0 0"/>
    <axis xyz="1 0 0"/><limit lower="-3.0" upper="3.0" velocity="4" effort="1"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0.05 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-2.0" upper="2.0" velocity="4" effort="1"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0.04 0 0"/>
    <axis xyz="1 0 0"/><limit lower="-3.0" upper="3.0" velocity="4" effort="1"/></joint>
  <joint name="tcp" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0.08 0 0" rpy="0 1.5708 0"/></joint>
  <joint name="slide" type="prismatic"><parent link="l6"/><child link="finger"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.04" velocity="0.1" effort="1"/></joint>
</robot>
"""
ARM6_PROFILE = {
    "base_link": "base", "ee_link": "tool", "arm_joints": ["j1", "j2", "j3", "j4", "j5", "j6"],
    "gripper": {"joint": "slide", "open": 0.04, "closed": 0.0}, "home_pose": [0, 0.3, 0.6, 0, 0.4, 0],
    "approach_axis": [0, 0, 1], "units": "degrees", "robot_type": "arm6_follower",
}


@pytest.fixture
def so101() -> Robot:
    return load(SO101_URDF, SO101_PROFILE, "SO-101")


@pytest.fixture
def arm6() -> Robot:
    return Robot(ARM6, ARM6_PROFILE, "Arm6")


@pytest.fixture
def source_dataset(tmp_path, monkeypatch) -> Path:
    """A one-arm SO-101 dataset the builder made: smooth joint motion, in LeRobot degrees."""
    from test_lerobot_builder import _payload, _robot_episode, _world

    B = _world(monkeypatch, tmp_path, frames=40)
    from dfpipeline.retarget import service as rs

    def load_traj(path):
        n = 60
        t = np.linspace(0, 1, n)
        rows = []
        for i in range(n):
            deg = [30 * np.sin(2 * t[i]), -40 + 30 * t[i], 50 - 20 * t[i], 20 * np.sin(3 * t[i]), 10 * t[i]]
            action = {f"{j}.pos": float(v) for j, v in zip(B.SO101_JOINTS[:5], deg, strict=True)}
            action["gripper.pos"] = float(100 * t[i])
            rows.append({"phase": "action", "frame": i, "timestamp": i / 30, "action": action})
        return {"trajectory": rows}

    monkeypatch.setattr(rs, "load", load_traj)
    result = B.build_to_store(
        _payload("src", [_robot_episode("a"), _robot_episode("b")], robot_id="r1", robot_kind="so101"),
        tmp_path / "built",
    )
    return Path(result["root"])


def _pair(source, target, units="degrees"):
    return C.Pair(source, target, C.encoding(source, units), C.encoding(target, "degrees"))


def _values(root: Path, col: str = "action") -> np.ndarray:
    import pyarrow.parquet as pq

    return np.concatenate([np.array(pq.read_table(p).column(col).to_pylist())
                           for p in sorted((root / "data").rglob("*.parquet"))])


def test_converting_to_the_same_robot_gives_the_joints_back(source_dataset, so101, tmp_path):
    """The round trip: SO-101 -> its pose -> SO-101 lands on the same joints."""
    out = tmp_path / "same"
    report = C.convert_folder(source_dataset, out, _pair(so101, load(SO101_URDF, SO101_PROFILE, "SO-101")))
    before, after = _values(source_dataset), _values(out)
    assert before.shape == after.shape
    assert max(e["position_error_max_mm"] for e in report["episodes"]) < 1.0
    # Same pose, so the same joints (a 5-joint arm has one solution near the seed).
    assert np.abs(before[:, :5] - after[:, :5]).max() < 1.0, "degrees"
    assert np.allclose(before[:, 5], after[:, 5]), "the gripper carries over"


def test_a_dataset_becomes_another_robots_and_loads_with_its_videos(source_dataset, so101, arm6, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = tmp_path / "arm6"
    report = C.convert_folder(source_dataset, out, _pair(so101, arm6))
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["robot_type"] == "arm6_follower"
    assert info["features"]["action"]["names"] == [f"j{i}.pos" for i in range(1, 7)] + ["slide.pos"]
    assert info["features"]["action"]["shape"] == [7]
    values = _values(out)
    assert values.shape[1] == 7
    # Where the SO-101's gripper was, scaled into the bigger arm's reach, Arm6's gripper is.
    assert np.median([e["position_error_median_mm"] for e in report["episodes"]]) < 2.0
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 7

    # The videos are the source's, untouched (on the Hub: copied server-side).
    shutil.copytree(source_dataset / "videos", out / "videos")
    ds = LeRobotDataset("me/arm6", root=out)
    assert ds.meta.total_episodes == 2 and ds.meta.robot_type == "arm6_follower"
    row = ds[len(ds) - 1]
    assert tuple(row["action"].shape) == (7,) and "observation.images.top" in row


def test_a_dataset_of_another_robot_is_refused_by_its_columns(source_dataset, arm6, so101, tmp_path):
    with pytest.raises(ValueError, match="not Arm6's"):
        C.convert_folder(source_dataset, tmp_path / "x", _pair(arm6, so101))


def test_minus_100_to_100_is_the_urdf_limits(so101):
    """From the URDF alone: -100 is each joint's lower limit, 100 its upper."""
    enc = C.encoding(so101, "m100")
    q = np.array([0.3, -0.5, 0.8, -0.2, 0.1])
    assert np.allclose(enc.to_rad(enc.from_rad(q)), q, atol=1e-6)
    assert np.allclose(enc.from_rad(so101.lower), -100) and np.allclose(enc.from_rad(so101.upper), 100)


def test_on_the_hub_the_videos_are_copied_and_never_downloaded(source_dataset, tmp_path, monkeypatch):
    """Only data/ and meta/ come down; the videos go repo to repo on the Hub."""
    from dfpipeline.config import settings
    from dfpipeline.retarget import paths

    hub = tmp_path / "hub-src"
    shutil.copytree(source_dataset, hub, ignore=shutil.ignore_patterns(".export-*"))
    calls = {"downloaded": [], "copied": None, "uploaded": None, "tag": None}

    def snapshot_download(repo_id, repo_type, revision, local_dir, allow_patterns=None, token=None):
        import fnmatch

        for path in hub.rglob("*"):
            rel = path.relative_to(hub).as_posix()
            if path.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, Path(local_dir) / rel)
                calls["downloaded"].append(rel)

    class Hub:
        def __init__(self, token=None):
            pass

        def list_repo_refs(self, repo_id, repo_type):
            return type("Refs", (), {"tags": []})()

        def create_repo(self, **k):
            calls["repo"] = k

        def repo_info(self, repo_id, repo_type):
            return type("Info", (), {"private": True})()

        def copy_files(self, source, destination, token=None):
            calls["copied"] = (source, destination)

        def upload_folder(self, repo_id, repo_type, folder_path, commit_message):
            calls["uploaded"] = sorted(p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*") if p.is_file())
            return type("Commit", (), {"oid": "abc"})()

        def create_tag(self, repo_id, tag, repo_type, revision, exist_ok):
            calls["tag"] = tag

    monkeypatch.setattr("huggingface_hub.HfApi", Hub)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr(settings, "hf_token", "hf_test")
    monkeypatch.setattr(paths, "urdf", lambda: SO101_URDF)
    package = tmp_path / "store" / "robots" / "packages" / "arm6.zip"
    package.parent.mkdir(parents=True, exist_ok=True)
    import zipfile

    with zipfile.ZipFile(package, "w") as z:
        z.writestr("arm6/arm6.urdf", ARM6)

    out = C.convert_on_hub({
        "export_id": "e1",
        "source": {"repo_id": "me/cups", "revision": "v0.4.0"},
        "target_repo_id": "me/cups-arm6",
        "source_robot": {"profile": None},
        "target_robot": {"package_path": "robots/packages/arm6.zip", "urdf_file": "arm6/arm6.urdf",
                         "profile": ARM6_PROFILE, "name": "Arm6"},
    })

    assert calls["downloaded"] and not any(f.startswith("videos/") for f in calls["downloaded"])
    assert calls["copied"] == ("hf://datasets/me/cups@v0.4.0/videos/", "hf://datasets/me/cups-arm6/videos/")
    assert not any(f.startswith("videos/") for f in calls["uploaded"]), "no video goes up either"
    assert "README.md" in calls["uploaded"] and "meta/info.json" in calls["uploaded"]
    assert calls["tag"] == "v0.1.0" and calls["repo"]["private"] is True
    assert out["robot_type"] == "arm6_follower" and len(out["episodes"]) == 2


def _bi_arm(side: str) -> str:
    """One Arm6 on a torso, at +y (left) or -y (right). The right one is a
    mirror: its shoulder turns the other way, so its limits are mirrored too."""
    y, lift = (0.2, "0 1 0") if side == "left" else (-0.2, "0 -1 0")
    lo, hi = (-0.3, 2.0) if side == "left" else (-2.0, 0.3)
    p = f"{side}_"
    return f"""
  <link name="{p}l1"/><link name="{p}l2"/><link name="{p}l3"/><link name="{p}l4"/><link name="{p}l5"/>
  <link name="{p}l6"/><link name="{p}tool"/><link name="{p}finger"/>
  <joint name="{p}j1" type="revolute"><parent link="torso"/><child link="{p}l1"/><origin xyz="0 {y} 0.12"/>
    <axis xyz="0 0 1"/><limit lower="-3.0" upper="3.0" velocity="3" effort="1"/></joint>
  <joint name="{p}j2" type="revolute"><parent link="{p}l1"/><child link="{p}l2"/><origin xyz="0 0 0.05"/>
    <axis xyz="{lift}"/><limit lower="{lo}" upper="{hi}" velocity="3" effort="1"/></joint>
  <joint name="{p}j3" type="revolute"><parent link="{p}l2"/><child link="{p}l3"/><origin xyz="0 0 0.30"/>
    <axis xyz="0 1 0"/><limit lower="-2.5" upper="2.5" velocity="3" effort="1"/></joint>
  <joint name="{p}j4" type="revolute"><parent link="{p}l3"/><child link="{p}l4"/><origin xyz="0.25 0 0"/>
    <axis xyz="1 0 0"/><limit lower="-3.0" upper="3.0" velocity="4" effort="1"/></joint>
  <joint name="{p}j5" type="revolute"><parent link="{p}l4"/><child link="{p}l5"/><origin xyz="0.05 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-2.0" upper="2.0" velocity="4" effort="1"/></joint>
  <joint name="{p}j6" type="revolute"><parent link="{p}l5"/><child link="{p}l6"/><origin xyz="0.04 0 0"/>
    <axis xyz="1 0 0"/><limit lower="-3.0" upper="3.0" velocity="4" effort="1"/></joint>
  <joint name="{p}tcp" type="fixed"><parent link="{p}l6"/><child link="{p}tool"/><origin xyz="0.08 0 0" rpy="0 1.5708 0"/></joint>
  <joint name="{p}slide" type="prismatic"><parent link="{p}l6"/><child link="{p}finger"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.04" velocity="0.1" effort="1"/></joint>"""


BI_ARM6 = f"""<?xml version="1.0"?>
<robot name="bi_arm6"><link name="torso"/>{_bi_arm("left")}{_bi_arm("right")}
</robot>
"""


def _bi_arm_profile(side: str) -> dict:
    # The real arm's limits, tighter than the URDF's on the shoulder, mirrored on the right.
    deg = [(-150, 150), (-9, 100) if side == "left" else (-100, 9), (-140, 140), (-170, 170), (-110, 110), (-170, 170)]
    return {
        "limits": np.radians(deg).tolist(),
        "ee_link": f"{side}_tool",
        "arm_joints": [f"{side}_j{i}" for i in range(1, 7)],
        # Written as an OpenArm-style gripper: 0 closed, -65 fully open.
        "gripper": {"joint": f"{side}_slide", "open": 0.04, "closed": 0.0, "lerobot": {"open": -65.0, "closed": 0.0}},
    }


# LeRobot-style names and side prefix (left_joint_1.pos … left_gripper.pos, right_…), as OpenArm's.
BI_ARM6_PROFILE = {
    "base_link": "torso", **_bi_arm_profile("left"), "right_arm": _bi_arm_profile("right"),
    "names": [f"joint_{i}" for i in range(1, 7)] + ["gripper"], "side_prefix": "{side}_",
    "units": "degrees", "robot_type": "biarm6_follower",
}


def _two_arms(src: Path, dst: Path) -> Path:
    """The one-arm source as a two-arm (bi_so101_follower) dataset: the same
    motion on both sides, the right arm's pan mirrored."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shutil.copytree(src, dst)
    info = json.loads((dst / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    for path in (dst / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        for col in ("action", "observation.state"):
            v = np.array(table.column(col).to_pylist(), dtype=np.float32)
            right = v.copy()
            right[:, 0] *= -1
            both = np.concatenate([v, right], axis=1)
            arr = pa.FixedSizeListArray.from_arrays(pa.array(both.reshape(-1)), both.shape[1])
            table = table.set_column(table.column_names.index(col), col, arr)
        pq.write_table(table, path)
    for col in ("action", "observation.state"):
        info["features"][col]["names"] = [f"{s}_arm_{n}" for s in ("left", "right") for n in names]
        info["features"][col]["shape"] = [2 * len(names)]
    info["robot_type"] = "bi_so101_follower"
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return dst


def test_two_so101_arms_become_a_two_arm_robot_each_side_its_own_arm(source_dataset, so101, tmp_path):
    """A two-arm URDF: each side is its own arm (the right one a mirror of the
    left), written under the robot's LeRobot names, side prefix and gripper
    values — what LeRobot, and its dataset viewer, match on."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = _two_arms(source_dataset, tmp_path / "bi")
    target = Robot(BI_ARM6, BI_ARM6_PROFILE, "BiArm6")
    out = tmp_path / "biarm6"
    report = C.convert_folder(src, out, _pair(so101, target))

    info = json.loads((out / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    assert info["robot_type"] == "bi_biarm6_follower"
    assert names == [f"{s}_{n}.pos" for s in ("left", "right") for n in BI_ARM6_PROFILE["names"]]

    assert np.median([e["position_error_median_mm"] for e in report["episodes"]]) < 2.0
    values = _values(out)
    assert values.shape[1] == 14
    left, right = values[:, :7], values[:, 7:]
    # Each side within its own arm's limits: the shoulder is -9..100° on the left, -100..9° on the right.
    for arm, side in ((left, "left"), (right, "right")):
        lo, hi = np.degrees(np.array(_bi_arm_profile(side)["limits"])).T
        assert (arm[:, :6] >= lo - 0.01).all() and (arm[:, :6] <= hi + 0.01).all()
    # The mirrored source moves the right shoulder the other way round.
    assert np.sign(np.median(left[:, 1])) == -np.sign(np.median(right[:, 1]))
    # The source gripper ran 0 (closed) → 100 (open); this one runs 0 → -65.
    before = _values(src)
    assert np.allclose(left[:, 6], -0.65 * before[:, 5], atol=1e-3)
    assert left[:, 6].min() >= -65.0 and left[:, 6].max() <= 0.0

    shutil.copytree(source_dataset / "videos", out / "videos")
    ds = LeRobotDataset("me/biarm6", root=out)
    assert ds.meta.robot_type == "bi_biarm6_follower" and tuple(ds[0]["action"].shape) == (14,)


def test_a_bigger_arm_on_a_table_never_goes_through_it(so101, arm6):
    """Both stand on the table: height is from the table, scaled, not from
    the middle of the recording — a bigger arm (YAM) went 24 cm under it."""
    pair = _pair(so101, arm6)
    layout = C._layout([f"{n}.pos" for n in so101.names], so101)
    enc = pair.src_enc
    # The SO-101's gripper from table height up to 25 cm and back down.
    rows, q = [], None
    for z in np.r_[np.linspace(0.01, 0.25, 15), np.linspace(0.25, 0.01, 15)]:
        target = np.array([0.25, 0.0, z])
        q = so101.ik(target, None, q) if q is not None else so101.first_ik(target, None)
        rows.append(np.r_[enc.from_rad(q), 50.0])
    rows = np.array(rows)
    src_z = np.array([so101.fk(enc.to_rad(r[:5]))[2, 3] for r in rows])
    centre = np.array([so101.fk(enc.to_rad(r[:5]))[:3, 3] for r in rows]).mean(axis=0)
    conv = C.EpisodeConverter(pair, layout, [""], {"": (centre, arm6.work_point)})
    dst_z = np.array([arm6.fk(pair.dst_enc.to_rad(conv.row(r)[:6].astype(float)))[2, 3] for r in rows])

    assert dst_z.min() >= 0.008, "never into the table"
    ratio = arm6.span / so101.span
    assert np.allclose(dst_z[src_z > 0.05], src_z[src_z > 0.05] * ratio, atol=0.01), "height from the table, scaled"


def test_the_profiles_joint_speeds_override_the_urdfs():
    """A URDF's speed is often a placeholder (YAM: 1 rad/s, datasets ~5x
    slower than the hand): the profile's, where set, is the arm's."""
    from dfpipeline.embodiment.robot import SO101_PROFILE, load

    urdf = load(SO101_URDF, SO101_PROFILE, "SO-101").velocity
    faster = load(SO101_URDF, {**SO101_PROFILE, "velocity_limits": [6.0, None, 6.0, None, 6.0]}, "SO-101")
    assert list(faster.velocity) == [6.0, urdf[1], 6.0, urdf[3], 6.0]
