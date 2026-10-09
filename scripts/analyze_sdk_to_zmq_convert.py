#!/usr/bin/env python3
"""校验 SDK→ZMQ 转换结果是否正确（相对源几何 + 相对 KIAN 参考分布）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_sdk_replay_to_zmq import (  # noqa: E402
    DEFAULT_EEF_OFFSET,
    DEFAULT_GRIPPER_MAX,
    DEFAULT_GRIPPER_MIN,
    invert_gripper,
    shift_tcp_along_z,
)


def _sample(path: Path, n: int = 20000):
    root = zarr.open(str(path), mode="r")
    n_total = int(root["data/action"].shape[0])
    idx = np.linspace(0, n_total - 1, min(n, n_total), dtype=int)
    d = root["data"]
    return {
        "n_total": n_total,
        "n_ep": int(root["meta/episode_ends"].shape[0]),
        "pos": np.asarray(d["robot0_eef_pos"].get_orthogonal_selection((idx, slice(None)))),
        "rot": np.asarray(
            d["robot0_eef_rot_axis_angle"].get_orthogonal_selection((idx, slice(None)))
        ),
        "grip": np.asarray(d["robot0_gripper_width"].get_orthogonal_selection((idx,))),
        "action": np.asarray(d["action"].get_orthogonal_selection((idx, slice(None)))),
        "joint": np.asarray(
            d["robot0_joint_pos"].get_orthogonal_selection((idx, slice(None)))
        ),
        "idx": idx,
    }


def _no_symlinks(path: Path) -> tuple[bool, int]:
    n = 0
    for p in path.rglob("*"):
        if p.is_symlink():
            return False, n
        if p.is_file():
            n += 1
    return True, n


def _inode_overlap(src: Path, dst: Path, sample: int = 64) -> int:
    overlap = 0
    files = [p for p in dst.rglob("*") if p.is_file() and not p.is_symlink()]
    checked = 0
    for p in files:
        if checked >= sample:
            break
        rel = p.relative_to(dst)
        sp = src / rel
        if not sp.is_file():
            continue
        checked += 1
        if p.stat().st_ino == sp.stat().st_ino and p.stat().st_dev == sp.stat().st_dev:
            overlap += 1
    return overlap


@click.command()
@click.option("--src", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dst", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--ref",
    default=Path("/data1/projects/umi-arx-kian/data/collect_demo_zmq/replay_buffer.zarr"),
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--eef_offset", default=DEFAULT_EEF_OFFSET, type=float)
@click.option("--gripper_min", default=DEFAULT_GRIPPER_MIN, type=float)
@click.option("--gripper_max", default=DEFAULT_GRIPPER_MAX, type=float)
@click.option("--name", default="", help="报告标题")
def main(
    src: Path,
    dst: Path,
    ref: Path,
    eef_offset: float,
    gripper_min: float,
    gripper_max: float,
    name: str,
):
    title = name or dst.parent.name
    print("=" * 64)
    print(f"ANALYZE: {title}")
    print(f"  src: {src}")
    print(f"  dst: {dst}")
    print(f"  ref: {ref}")
    print("=" * 64)

    checks = {}

    ok_link, n_files = _no_symlinks(dst)
    checks["no_symlinks"] = ok_link
    print(f"[files] no_symlinks={ok_link}  n_files≈{n_files}")

    overlap = _inode_overlap(src, dst)
    checks["no_hardlink_to_src"] = overlap == 0
    print(f"[files] hardlink overlap with src (sample): {overlap}  PASS={overlap==0}")

    s = _sample(src)
    d = _sample(dst)
    r = _sample(ref)
    print(f"[size] src frames={s['n_total']} ep={s['n_ep']} | dst frames={d['n_total']} ep={d['n_ep']}")
    checks["frame_count_match"] = s["n_total"] == d["n_total"] and s["n_ep"] == d["n_ep"]

    # expected conversion from src sample
    exp_pos = shift_tcp_along_z(s["pos"], s["rot"], eef_offset)
    exp_grip = invert_gripper(
        s["grip"], gripper_min=gripper_min, gripper_max=gripper_max
    )
    pos_err = np.abs(d["pos"] - exp_pos).max()
    grip_err = np.abs(d["grip"] - exp_grip).max()
    # rot unchanged
    rot_err = np.abs(d["rot"] - s["rot"]).max()
    joint_err = np.abs(d["joint"] - s["joint"]).max()
    act_pos_err = np.abs(d["action"][:, :3] - d["pos"]).max()
    act_grip_err = np.abs(d["action"][:, 6] - d["grip"]).max()
    act_rot_err = np.abs(d["action"][:, 3:6] - d["rot"]).max()

    print(f"[geom] max|dst_pos - expected| = {pos_err:.3e} m")
    print(f"[geom] max|dst_grip - expected| = {grip_err:.3e} m")
    print(f"[geom] max|dst_rot - src_rot| = {rot_err:.3e}  (expect 0)")
    print(f"[geom] max|dst_joint - src_joint| = {joint_err:.3e}  (expect 0)")
    print(f"[sync] max|action[:3]-pos| = {act_pos_err:.3e}")
    print(f"[sync] max|action[3:6]-rot| = {act_rot_err:.3e}")
    print(f"[sync] max|action[6]-grip| = {act_grip_err:.3e}")

    checks["pos_matches_formula"] = pos_err < 1e-9
    checks["grip_matches_formula"] = grip_err < 1e-9
    checks["rot_unchanged"] = rot_err < 1e-15
    checks["joint_unchanged"] = joint_err < 1e-15
    checks["action_pos_sync"] = act_pos_err < 1e-12
    checks["action_grip_sync"] = act_grip_err < 1e-12

    delta = d["pos"] - s["pos"]
    along = np.einsum("ij,ij->i", delta, R.from_rotvec(s["rot"]).as_matrix()[:, :, 2])
    print(
        f"[geom] Δpos·TCP_Z mean/std = {along.mean():.6f}/{along.std():.6f} "
        f"(expect {eef_offset})"
    )
    checks["offset_along_tcp_z"] = abs(float(along.mean()) - eef_offset) < 1e-5

    # vs KIAN ref distribution
    mean_raw = np.linalg.norm(s["pos"].mean(0) - r["pos"].mean(0))
    mean_cvt = np.linalg.norm(d["pos"].mean(0) - r["pos"].mean(0))
    g_raw = abs(float(s["grip"].mean()) - float(r["grip"].mean()))
    g_cvt = abs(float(d["grip"].mean()) - float(r["grip"].mean()))
    print(f"[vs KIAN] pos mean L2: src={mean_raw:.4f}  dst={mean_cvt:.4f}  Δ={mean_raw-mean_cvt:+.4f}")
    print(f"[vs KIAN] |grip mean|: src={g_raw:.4f}  dst={g_cvt:.4f}  Δ={g_raw-g_cvt:+.4f}")
    print(f"[vs KIAN] grip dst [{d['grip'].min():.4f},{d['grip'].max():.4f}] mean={d['grip'].mean():.4f}")
    print(f"[vs KIAN] grip ref [{r['grip'].min():.4f},{r['grip'].max():.4f}] mean={r['grip'].mean():.4f}")
    checks["pos_closer_to_kian"] = mean_cvt <= mean_raw + 1e-9
    checks["grip_closer_to_kian"] = g_cvt <= g_raw + 1e-9

    print("-" * 64)
    all_ok = True
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
        all_ok = all_ok and v
    print(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")

    report = {
        "name": title,
        "src": str(src),
        "dst": str(dst),
        "ref": str(ref),
        "checks": {k: bool(v) for k, v in checks.items()},
        "metrics": {
            "pos_err": float(pos_err),
            "grip_err": float(grip_err),
            "along_tcp_z_mean": float(along.mean()),
            "pos_mean_l2_src_to_ref": float(mean_raw),
            "pos_mean_l2_dst_to_ref": float(mean_cvt),
            "grip_mean_abs_src_to_ref": float(g_raw),
            "grip_mean_abs_dst_to_ref": float(g_cvt),
        },
        "overall": bool(all_ok),
    }
    out_json = dst.parent / "sdk_to_zmq_analyze.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
