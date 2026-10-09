#!/usr/bin/env python3
"""SDK→ZMQ 坐标转换冒烟测试。

对 HZL(SDK/link6) replay_buffer 做 link6→eef_link + 夹爪极性翻转，
再与 KIAN(ZMQ/eef_link) 参考集对比分布，并可选写出少量 episode 验写盘。

用法:
  conda run -n umi-arx-kian python scripts/smoke_test_sdk_to_zmq_convert.py

  python scripts/smoke_test_sdk_to_zmq_convert.py \\
      --sdk /data1/projects/umi-arx-hzl/data/collect_demo/replay_buffer.zarr \\
      --zmq /data1/projects/umi-arx-kian/data/collect_demo_zmq/replay_buffer.zarr \\
      --write_episodes 2
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.replay_buffer import ReplayBuffer  # noqa: E402
from scripts.convert_sdk_replay_to_zmq import (  # noqa: E402
    DEFAULT_EEF_OFFSET,
    DEFAULT_GRIPPER_MAX,
    DEFAULT_GRIPPER_MIN,
    convert_episode,
    invert_gripper,
    shift_tcp_along_z,
)


def _sample_arrays(zarr_path: Path, n_sample: int = 20000):
    root = zarr.open(str(zarr_path), mode="r")
    n = int(root["data/action"].shape[0])
    idx = np.linspace(0, n - 1, min(n_sample, n), dtype=int)
    data = root["data"]
    return {
        "n_total": n,
        "n_ep": int(root["meta/episode_ends"].shape[0]),
        "idx": idx,
        "pos": np.asarray(data["robot0_eef_pos"].get_orthogonal_selection((idx, slice(None)))),
        "rot": np.asarray(
            data["robot0_eef_rot_axis_angle"].get_orthogonal_selection((idx, slice(None)))
        ),
        "grip": np.asarray(data["robot0_gripper_width"].get_orthogonal_selection((idx,))),
        "action": np.asarray(data["action"].get_orthogonal_selection((idx, slice(None)))),
        "joint": np.asarray(
            data["robot0_joint_pos"].get_orthogonal_selection((idx, slice(None)))
        ),
    }


def _range_str(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float64)
    return f"[{x.min():.4f}, {x.max():.4f}] mean={x.mean():.4f}"


def _overlap_1d(a: np.ndarray, b: np.ndarray) -> float:
    lo = max(float(a.min()), float(b.min()))
    hi = min(float(a.max()), float(b.max()))
    union = max(float(a.max()), float(b.max())) - min(float(a.min()), float(b.min()))
    if union <= 0:
        return 0.0
    return max(0.0, hi - lo) / union


def _mean_rot_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    ra = R.from_rotvec(rot_a.mean(axis=0))
    rb = R.from_rotvec(rot_b.mean(axis=0))
    rel = ra.inv() * rb
    return float(np.degrees(rel.magnitude()))


def convert_sample(
    pos: np.ndarray,
    rot: np.ndarray,
    grip: np.ndarray,
    action: np.ndarray,
    *,
    eef_offset: float,
    gripper_min: float,
    gripper_max: float,
    do_invert: bool,
):
    pos2 = shift_tcp_along_z(pos, rot, eef_offset)
    grip2 = (
        invert_gripper(grip, gripper_min=gripper_min, gripper_max=gripper_max)
        if do_invert
        else grip.copy()
    )
    act2 = action.copy()
    act2[:, :3] = pos2
    act2[:, 6] = grip2.reshape(-1)
    return pos2, grip2, act2


def print_section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


@click.command()
@click.option(
    "--sdk",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/data1/projects/umi-arx-hzl/data/collect_demo/replay_buffer.zarr"),
    show_default=True,
)
@click.option(
    "--zmq",
    "zmq_ref",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(
        "/data1/projects/umi-arx-kian/data/collect_demo_zmq/replay_buffer.zarr"
    ),
    show_default=True,
)
@click.option("--eef_offset", default=DEFAULT_EEF_OFFSET, show_default=True, type=float)
@click.option("--gripper_min", default=DEFAULT_GRIPPER_MIN, show_default=True, type=float)
@click.option("--gripper_max", default=DEFAULT_GRIPPER_MAX, show_default=True, type=float)
@click.option("--no_invert_gripper", is_flag=True, default=False)
@click.option("--n_sample", default=20000, show_default=True, type=int)
@click.option(
    "--write_episodes",
    default=2,
    show_default=True,
    type=int,
    help="写出前 N 个 episode 到临时目录（0=不写盘）",
)
@click.option(
    "--out_dir",
    type=click.Path(path_type=Path),
    default=Path("data/collect_demo_hzl_as_zmq_smoke"),
    show_default=True,
)
def main(
    sdk: Path,
    zmq_ref: Path,
    eef_offset: float,
    gripper_min: float,
    gripper_max: float,
    no_invert_gripper: bool,
    n_sample: int,
    write_episodes: int,
    out_dir: Path,
):
    do_invert = not no_invert_gripper
    print_section("SDK → ZMQ convert smoke test")
    print(f"  sdk (HZL)   : {sdk}")
    print(f"  zmq (KIAN)  : {zmq_ref}")
    print(f"  eef_offset  : {eef_offset} m")
    print(f"  invert_grip : {do_invert}  [{gripper_min}, {gripper_max}]")
    print(f"  n_sample    : {n_sample}")

    sdk_d = _sample_arrays(sdk, n_sample)
    zmq_d = _sample_arrays(zmq_ref, n_sample)
    print(f"  sdk frames  : {sdk_d['n_total']}  episodes={sdk_d['n_ep']}")
    print(f"  zmq frames  : {zmq_d['n_total']}  episodes={zmq_d['n_ep']}")

    pos_c, grip_c, act_c = convert_sample(
        sdk_d["pos"],
        sdk_d["rot"],
        sdk_d["grip"],
        sdk_d["action"],
        eef_offset=eef_offset,
        gripper_min=gripper_min,
        gripper_max=gripper_max,
        do_invert=do_invert,
    )

    # --- geometry sanity ---
    print_section("1) 几何自检（单帧偏移应 ≈ eef_offset）")
    delta = pos_c - sdk_d["pos"]
    delta_norm = np.linalg.norm(delta, axis=1)
    tcp_z = R.from_rotvec(sdk_d["rot"]).as_matrix()[:, :, 2]
    along_z = np.einsum("ij,ij->i", delta, tcp_z)
    print(f"  |Δpos| mean/std/max = {delta_norm.mean():.6f} / {delta_norm.std():.6f} / {delta_norm.max():.6f} m")
    print(f"  Δpos·TCP_Z mean/std = {along_z.mean():.6f} / {along_z.std():.6f} m  (期望 {eef_offset})")
    ok_offset = abs(float(along_z.mean()) - eef_offset) < 1e-6 and float(delta_norm.std()) < 1e-9
    # std can be tiny numerical noise; use looser bound
    ok_offset = abs(float(along_z.mean()) - eef_offset) < 1e-5 and float(delta_norm.std()) < 1e-6
    print(f"  PASS offset geometry: {ok_offset}")

    # action synced with state
    pos_act_err = np.abs(act_c[:, :3] - pos_c).max()
    grip_act_err = np.abs(act_c[:, 6] - grip_c.reshape(-1)).max()
    print(f"  max |action[:3]-pos| after = {pos_act_err:.3e}")
    print(f"  max |action[6]-grip| after = {grip_act_err:.3e}")
    ok_sync = pos_act_err < 1e-12 and grip_act_err < 1e-12
    print(f"  PASS action↔state sync: {ok_sync}")

    # --- vs KIAN ---
    print_section("2) 与 KIAN(ZMQ) 参考集分布对比")
    mean_raw = np.linalg.norm(sdk_d["pos"].mean(0) - zmq_d["pos"].mean(0))
    mean_cvt = np.linalg.norm(pos_c.mean(0) - zmq_d["pos"].mean(0))
    print(f"  pos mean L2 → KIAN:  raw={mean_raw:.4f} m   converted={mean_cvt:.4f} m")
    print(f"  improvement: {(mean_raw - mean_cvt)*1000:.1f} mm")

    print("  axis overlap (1D IoU of ranges):")
    for i, name in enumerate("XYZ"):
        o_raw = _overlap_1d(sdk_d["pos"][:, i], zmq_d["pos"][:, i])
        o_cvt = _overlap_1d(pos_c[:, i], zmq_d["pos"][:, i])
        print(f"    {name}: raw={o_raw:.3f}  converted={o_cvt:.3f}  Δ={o_cvt-o_raw:+.3f}")

    print("  position ranges:")
    for i, name in enumerate("XYZ"):
        print(f"    {name} SDK raw : {_range_str(sdk_d['pos'][:, i])}")
        print(f"    {name} SDK→ZMQ: {_range_str(pos_c[:, i])}")
        print(f"    {name} KIAN   : {_range_str(zmq_d['pos'][:, i])}")

    ang = _mean_rot_angle_deg(sdk_d["rot"], zmq_d["rot"])
    print(f"  mean-orientation angle SDK vs KIAN: {ang:.2f} deg  (期望小，姿态本已对齐)")
    print(f"  rotvec mean SDK : {sdk_d['rot'].mean(0)}")
    print(f"  rotvec mean KIAN: {zmq_d['rot'].mean(0)}")

    print("  gripper:")
    print(f"    SDK raw : {_range_str(sdk_d['grip'])}")
    print(f"    SDK→ZMQ: {_range_str(grip_c)}")
    print(f"    KIAN   : {_range_str(zmq_d['grip'])}")
    grip_mean_err_raw = abs(float(sdk_d["grip"].mean()) - float(zmq_d["grip"].mean()))
    grip_mean_err_cvt = abs(float(grip_c.mean()) - float(zmq_d["grip"].mean()))
    print(
        f"  |grip mean − KIAN|: raw={grip_mean_err_raw:.4f}  converted={grip_mean_err_cvt:.4f}"
    )

    # joints should be comparable without conversion (same robot)
    joint_mean_l2 = np.linalg.norm(sdk_d["joint"].mean(0) - zmq_d["joint"].mean(0))
    print(f"  joint_pos mean L2 SDK vs KIAN (no convert): {joint_mean_l2:.4f} rad")

    print_section("3) 可逆性（convert 再 -offset / 再 invert 应回到原值）")
    pos_back = shift_tcp_along_z(pos_c, sdk_d["rot"], -eef_offset)
    grip_back = (
        invert_gripper(grip_c, gripper_min=gripper_min, gripper_max=gripper_max)
        if do_invert
        else grip_c
    )
    pos_err = np.abs(pos_back - sdk_d["pos"]).max()
    grip_err = np.abs(grip_back - sdk_d["grip"]).max()
    print(f"  roundtrip max |pos| err  = {pos_err:.3e} m")
    print(f"  roundtrip max |grip| err = {grip_err:.3e} m")
    ok_rt = pos_err < 1e-9 and grip_err < 1e-12
    print(f"  PASS roundtrip: {ok_rt}")

    # --- optional write ---
    write_ok = True
    written = 0
    if write_episodes > 0:
        print_section(f"4) 写盘冒烟（前 {write_episodes} episode）")
        out_dir = out_dir.resolve()
        out_zarr = out_dir / "replay_buffer.zarr"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        src = ReplayBuffer.create_from_path(str(sdk), mode="r")
        n_write = min(int(write_episodes), int(src.n_episodes))
        dst = ReplayBuffer.create_from_path(str(out_zarr), mode="a")
        for i in range(n_write):
            ep = src.get_episode(i, copy=True)
            # drop camera to keep smoke light; pose fields still full-res
            ep_light = {
                k: v
                for k, v in ep.items()
                if not (k.startswith("camera") and k.endswith("_rgb"))
            }
            converted = convert_episode(
                ep_light,
                eef_offset=eef_offset,
                invert_grip=do_invert,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )
            dst.add_episode(converted, compressors="disk")
            written += 1
            print(f"  wrote episode {i}  steps={converted['action'].shape[0]}")

        # verify first frame on disk
        chk = ReplayBuffer.create_from_path(str(out_zarr), mode="r")
        ep0_src = src.get_episode(0, copy=True)
        ep0_dst = chk.get_episode(0, copy=True)
        p0 = np.asarray(ep0_src["robot0_eef_pos"][0])
        r0 = np.asarray(ep0_src["robot0_eef_rot_axis_angle"][0])
        expect = shift_tcp_along_z(p0, r0, eef_offset)
        got = np.asarray(ep0_dst["robot0_eef_pos"][0])
        disk_err = float(np.linalg.norm(got - expect))
        print(f"  disk ep0 pos vs expected: err={disk_err:.3e} m")
        write_ok = disk_err < 1e-9 and chk.n_episodes == n_write
        print(f"  PASS write+reload: {write_ok}")

        meta = {
            "source": str(sdk),
            "reference_zmq": str(zmq_ref),
            "eef_offset_tcp_z": eef_offset,
            "invert_gripper": do_invert,
            "gripper_min": gripper_min,
            "gripper_max": gripper_max,
            "n_episodes_written": written,
            "note": "smoke: camera*_rgb dropped to keep output small",
        }
        meta_path = out_dir / "sdk_to_zmq_smoke_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  output: {out_zarr}")
        print(f"  meta  : {meta_path}")

    print_section("SUMMARY")
    checks = {
        "offset_geometry": ok_offset,
        "action_state_sync": ok_sync,
        "roundtrip": ok_rt,
        "mean_pos_closer_to_kian": mean_cvt <= mean_raw + 1e-9,
        "grip_mean_closer_to_kian": grip_mean_err_cvt <= grip_mean_err_raw + 1e-9,
    }
    if write_episodes > 0:
        checks["write_reload"] = write_ok
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")

    all_ok = all(checks.values())
    print(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")
    print(
        "\n提示: 全量转换请用:\n"
        "  python scripts/convert_sdk_replay_to_zmq.py \\\n"
        f"    -i {sdk} \\\n"
        "    -o data/collect_demo_hzl_as_zmq/replay_buffer.zarr"
    )
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
