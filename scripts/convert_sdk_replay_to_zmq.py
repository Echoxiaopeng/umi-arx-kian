#!/usr/bin/env python3
"""
把 SDK / collect_drag_umi.py 采集的 replay_buffer 转成 ZMQ 语义，
便于与 collect_drag_umi_zmq.py 新采的数据混训、并用 ZMQ 评测（通常可关
eef_offset / invert_gripper）。

转换内容（与 eval_arx5_fast_wam 对齐）:
  1. 末端位置 link6 → eef_link：沿 TCP Z 前移 --eef_offset（默认 0.145m）
     姿态 rotvec 不变
  2. 夹爪极性：SDK「越大越夹紧」→ ZMQ 物理开口「越大越张开」
     g' = gripper_min + gripper_max - g

会改写的字段（每臂）:
  - robot{i}_eef_pos
  - robot{i}_gripper_width
  - action 中对应的 xyz / gripper 维
不改: eef_rot、joint_*、timestamp、camera*_rgb（若有）

用法:
  # 写出新 zarr（推荐）
  python scripts/convert_sdk_replay_to_zmq.py \\
      -i data/sdk_demo/replay_buffer.zarr \\
      -o data/sdk_demo_as_zmq/replay_buffer.zarr

  # 同时复制 videos/（编号不变）
  python scripts/convert_sdk_replay_to_zmq.py \\
      -i data/sdk_demo/replay_buffer.zarr \\
      -o data/sdk_demo_as_zmq/replay_buffer.zarr \\
      --copy_videos

  # 只做位置偏移 / 只做夹爪（调试）
  python scripts/convert_sdk_replay_to_zmq.py -i ... -o ... --no_invert_gripper
  python scripts/convert_sdk_replay_to_zmq.py -i ... -o ... --eef_offset 0
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click
import numpy as np
import scipy.spatial.transform as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.replay_buffer import ReplayBuffer  # noqa: E402

# 与 eval_arx5_fast_wam 一致
DEFAULT_EEF_OFFSET = 0.145
DEFAULT_GRIPPER_MIN = 0.011
DEFAULT_GRIPPER_MAX = 0.085


def shift_tcp_along_z(
    pos: np.ndarray,
    rotvec: np.ndarray,
    dz: float,
) -> np.ndarray:
    """p' = p + R_tcp @ [0,0,dz]，支持 (3,) 或 (T,3)。"""
    pos = np.asarray(pos, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if abs(float(dz)) < 1e-12:
        return pos.copy()

    single = pos.ndim == 1
    if single:
        pos = pos.reshape(1, 3)
        rotvec = rotvec.reshape(1, 3)

    delta = np.array([0.0, 0.0, float(dz)], dtype=np.float64)
    # 批量旋转，避免 Python for-loop（30万帧也能秒级完成）
    out = pos + st.Rotation.from_rotvec(rotvec).apply(delta)
    return out[0] if single else out


def _assert_no_links(path: Path, *, label: str) -> None:
    """拒绝输出里出现软链接；并抽查不应与源共享 inode（硬链接）。"""
    symlinks = [p for p in path.rglob("*") if p.is_symlink()]
    if symlinks:
        raise RuntimeError(
            f"{label} 含软链接（禁止）: 例 {symlinks[0]}"
        )


def copy_tree_no_links(src: Path, dst: Path) -> None:
    """物理拷贝目录树：跟随源侧软链接内容，但不创建软/硬链接。"""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _copy_file(src_f: str, dst_f: str) -> str:
        # copy2：新 inode，不保留硬链接关系
        return shutil.copy2(src_f, dst_f, follow_symlinks=True)

    shutil.copytree(src, dst, symlinks=False, copy_function=_copy_file)
    _assert_no_links(dst, label=str(dst))


def patch_zarr_sdk_to_zmq(
    zarr_path: Path,
    *,
    eef_offset: float,
    invert_grip: bool,
    gripper_min: float,
    gripper_max: float,
) -> dict:
    """在已拷贝的 zarr 上原地改写 eef_pos / gripper / action（不碰 RGB）。"""
    import zarr

    root = zarr.open(str(zarr_path), mode="r+")
    data = root["data"]
    n_robots = 0
    while f"robot{n_robots}_eef_pos" in data:
        n_robots += 1
    if n_robots == 0:
        raise KeyError("zarr 中没有 robot*_eef_pos")

    action = np.asarray(data["action"][:], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 7 * n_robots:
        raise ValueError(f"action shape 异常: {action.shape}")

    stats = {"n_robots": n_robots, "n_steps": int(action.shape[0])}
    for ridx in range(n_robots):
        pos_key = f"robot{ridx}_eef_pos"
        rot_key = f"robot{ridx}_eef_rot_axis_angle"
        grip_key = f"robot{ridx}_gripper_width"
        pos = np.asarray(data[pos_key][:], dtype=np.float64)
        rot = np.asarray(data[rot_key][:], dtype=np.float64)
        grip = np.asarray(data[grip_key][:], dtype=np.float64)

        pos_before_mean = pos.mean(axis=0).tolist()
        grip_before_mean = float(grip.mean())

        if eef_offset != 0.0:
            pos = shift_tcp_along_z(pos, rot, eef_offset)
            data[pos_key][:] = pos
            action[:, 7 * ridx : 7 * ridx + 3] = pos

        if invert_grip:
            grip = invert_gripper(
                grip, gripper_min=gripper_min, gripper_max=gripper_max
            )
            data[grip_key][:] = grip
            action[:, 7 * ridx + 6] = grip.reshape(-1)

        stats[f"robot{ridx}"] = {
            "pos_mean_before": pos_before_mean,
            "pos_mean_after": pos.mean(axis=0).tolist(),
            "grip_mean_before": grip_before_mean,
            "grip_mean_after": float(grip.mean()),
        }

    data["action"][:] = action
    return stats


def assert_independent_from_source(src: Path, dst: Path, sample: int = 32) -> None:
    """抽查 dst 文件与 src 无相同 inode（排除硬链接共用）。"""
    src = src.resolve()
    dst = dst.resolve()
    files = [p for p in dst.rglob("*") if p.is_file() and not p.is_symlink()]
    if not files:
        raise RuntimeError(f"输出为空: {dst}")
    # 优先抽查会被改写的数组 + 若干 RGB chunk
    prefer = []
    for sub in (
        "data/robot0_eef_pos",
        "data/robot0_gripper_width",
        "data/action",
        "data/camera0_rgb",
    ):
        prefer.extend(sorted((dst / sub).glob("*"))[:8])
    candidates = [p for p in prefer if p.is_file()] + files
    checked = 0
    for p in candidates:
        if checked >= sample:
            break
        rel = p.relative_to(dst)
        src_p = src / rel
        if not src_p.is_file():
            continue
        if p.stat().st_ino == src_p.stat().st_ino and p.stat().st_dev == src_p.stat().st_dev:
            raise RuntimeError(
                f"输出与源共享 inode（硬链接）: {rel}  ino={p.stat().st_ino}"
            )
        checked += 1
    print(f"[check] independent inodes ok (sampled {checked} files)")


def invert_gripper(
    grip: np.ndarray,
    *,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray:
    return gripper_min + gripper_max - np.asarray(grip, dtype=np.float64)


def discover_n_robots(episode: dict) -> int:
    n = 0
    while f"robot{n}_eef_pos" in episode:
        n += 1
    if n == 0:
        raise KeyError("episode 中没有 robot*_eef_pos")
    return n


def convert_episode(
    episode: dict,
    *,
    eef_offset: float,
    invert_grip: bool,
    gripper_min: float,
    gripper_max: float,
) -> dict:
    out = {k: np.asarray(v).copy() for k, v in episode.items()}
    n_robots = discover_n_robots(out)
    action = out.get("action")
    if action is None:
        raise KeyError("episode 缺少 action")
    action = np.asarray(action, dtype=np.float64).copy()
    if action.ndim != 2 or action.shape[1] < 7 * n_robots:
        raise ValueError(
            f"action shape 异常: {action.shape}，期望 (*, {7 * n_robots}+)"
        )

    for ridx in range(n_robots):
        pos_key = f"robot{ridx}_eef_pos"
        rot_key = f"robot{ridx}_eef_rot_axis_angle"
        grip_key = f"robot{ridx}_gripper_width"
        if rot_key not in out:
            raise KeyError(f"缺少 {rot_key}")

        pos = np.asarray(out[pos_key], dtype=np.float64)
        rot = np.asarray(out[rot_key], dtype=np.float64)
        if eef_offset != 0.0:
            pos = shift_tcp_along_z(pos, rot, eef_offset)
            out[pos_key] = pos
            action[:, 7 * ridx : 7 * ridx + 3] = pos

        if invert_grip and grip_key in out:
            grip = invert_gripper(
                out[grip_key],
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )
            out[grip_key] = grip.reshape(out[grip_key].shape)
            action[:, 7 * ridx + 6] = np.asarray(grip, dtype=np.float64).reshape(-1)

    out["action"] = action
    return out


def copy_video_dirs(src_zarr: Path, dst_zarr: Path) -> int:
    """复制与 episode 编号对应的 videos/ 目录。"""
    src_videos = src_zarr.parent / "videos"
    dst_videos = dst_zarr.parent / "videos"
    if not src_videos.is_dir():
        print(f"[warn] 无视频目录可复制: {src_videos}")
        return 0
    dst_videos.mkdir(parents=True, exist_ok=True)
    n = 0
    for child in sorted(src_videos.iterdir()):
        if not child.is_dir() or not child.name.isdigit():
            continue
        target = dst_videos / child.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(child, target)
        n += 1
    return n


def summarize_gripper(buf: ReplayBuffer, tag: str) -> None:
    if "robot0_gripper_width" not in buf.keys():
        return
    g = np.asarray(buf["robot0_gripper_width"])
    a = np.asarray(buf["action"])
    print(
        f"[{tag}] robot0_gripper_width "
        f"min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}"
    )
    print(
        f"[{tag}] action[...,6] "
        f"min={a[:, 6].min():.4f} max={a[:, 6].max():.4f} mean={a[:, 6].mean():.4f}"
    )


@click.command()
@click.option(
    "--input",
    "-i",
    "input_zarr",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="SDK 采集的 replay_buffer.zarr 路径",
)
@click.option(
    "--output",
    "-o",
    "output_zarr",
    required=True,
    type=click.Path(path_type=Path),
    help="输出 replay_buffer.zarr（勿与 input 相同）",
)
@click.option(
    "--eef_offset",
    default=DEFAULT_EEF_OFFSET,
    show_default=True,
    type=float,
    help="link6→eef_link 沿 TCP Z 的位移（米）；0 表示不做位置转换",
)
@click.option(
    "--invert_gripper/--no_invert_gripper",
    default=True,
    help="是否做夹爪极性对翻（默认开）",
)
@click.option(
    "--gripper_min",
    default=DEFAULT_GRIPPER_MIN,
    show_default=True,
    type=float,
    help="对翻用的开口下界（米）",
)
@click.option(
    "--gripper_max",
    default=DEFAULT_GRIPPER_MAX,
    show_default=True,
    type=float,
    help="对翻用的开口上界（米）",
)
@click.option(
    "--copy_videos/--no_copy_videos",
    "do_copy_videos",
    default=False,
    help="是否把 input 旁 videos/ 复制到 output 旁",
)
@click.option(
    "--dry_run",
    is_flag=True,
    default=False,
    help="只打印将转换的 episode 数与 gripper 统计，不写盘",
)
@click.option(
    "--mode",
    type=click.Choice(["copy_patch", "episode"]),
    default="copy_patch",
    show_default=True,
    help="copy_patch=物理拷贝后只改位姿/夹爪（推荐，无软硬链接）；"
    "episode=逐集重写全部字段（慢，含 RGB 重压缩）",
)
def main(
    input_zarr: Path,
    output_zarr: Path,
    eef_offset: float,
    invert_gripper: bool,
    gripper_min: float,
    gripper_max: float,
    do_copy_videos: bool,
    dry_run: bool,
    mode: str,
):
    input_zarr = input_zarr.resolve()
    output_zarr = output_zarr.resolve()
    if input_zarr == output_zarr:
        raise click.ClickException("output 不能与 input 相同，请写到新路径")

    src = ReplayBuffer.create_from_path(str(input_zarr), mode="r")
    n_ep = src.n_episodes
    if n_ep <= 0:
        raise click.ClickException(f"输入没有 episode: {input_zarr}")

    print("=" * 60)
    print("SDK → ZMQ replay_buffer convert")
    print(f"  input        : {input_zarr}")
    print(f"  output       : {output_zarr}")
    print(f"  n_episodes   : {n_ep}")
    print(f"  n_steps      : {src.n_steps}")
    print(f"  mode         : {mode}")
    print(f"  eef_offset   : {eef_offset} m (link6 → eef_link along TCP Z)")
    print(f"  invert_grip  : {invert_gripper}  (range [{gripper_min}, {gripper_max}])")
    print(f"  copy_videos  : {do_copy_videos}")
    print("=" * 60)
    summarize_gripper(src, "before")

    if dry_run:
        ep0 = src.get_episode(0, copy=True)
        # dry_run 不读相机，避免大数组
        ep0.pop("camera0_rgb", None)
        for k in list(ep0.keys()):
            if k.startswith("camera") and k.endswith("_rgb"):
                ep0.pop(k, None)
        demo = convert_episode(
            ep0,
            eef_offset=eef_offset,
            invert_grip=invert_gripper,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        )
        g0 = float(np.asarray(ep0["robot0_gripper_width"]).reshape(-1)[0])
        g1 = float(np.asarray(demo["robot0_gripper_width"]).reshape(-1)[0])
        p0 = np.asarray(ep0["robot0_eef_pos"])[0]
        p1 = np.asarray(demo["robot0_eef_pos"])[0]
        print(f"[dry_run] ep0 grip {g0:.4f} → {g1:.4f}")
        print(
            f"[dry_run] ep0 pos  {p0} → {p1}  "
            f"(Δ={np.linalg.norm(p1 - p0)*1000:.2f} mm)"
        )
        print("dry_run 完成，未写盘。")
        return

    meta = {
        "source": str(input_zarr),
        "convert": "sdk_to_zmq",
        "mode": mode,
        "eef_offset_tcp_z": float(eef_offset),
        "invert_gripper": bool(invert_gripper),
        "gripper_min": float(gripper_min),
        "gripper_max": float(gripper_max),
        "no_symlinks": True,
        "no_hardlinks_to_source": True,
    }

    if mode == "copy_patch":
        print("[1/3] physically copying zarr (no symlinks / no hardlinks)...")
        copy_tree_no_links(input_zarr, output_zarr)
        print("[2/3] patching eef_pos / gripper / action...")
        patch_stats = patch_zarr_sdk_to_zmq(
            output_zarr,
            eef_offset=eef_offset,
            invert_grip=invert_gripper,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        )
        meta["patch_stats"] = patch_stats
        print("[3/3] verifying independence from source...")
        assert_independent_from_source(input_zarr, output_zarr)
        dst = ReplayBuffer.create_from_path(str(output_zarr), mode="r")
    else:
        if output_zarr.exists():
            shutil.rmtree(output_zarr)
        output_zarr.parent.mkdir(parents=True, exist_ok=True)
        dst = ReplayBuffer.create_from_path(str(output_zarr), mode="a")
        for i in range(n_ep):
            ep = src.get_episode(i, copy=True)
            converted = convert_episode(
                ep,
                eef_offset=eef_offset,
                invert_grip=invert_gripper,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )
            dst.add_episode(converted, compressors="disk")
            if (i + 1) % 10 == 0 or i + 1 == n_ep:
                print(f"  converted {i + 1}/{n_ep}")
        assert_independent_from_source(input_zarr, output_zarr)

    summarize_gripper(dst, "after")

    meta_path = output_zarr.parent / "sdk_to_zmq_convert_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote meta: {meta_path}")

    if do_copy_videos:
        src_videos = input_zarr.parent / "videos"
        dst_videos = output_zarr.parent / "videos"
        if src_videos.is_dir():
            print(f"Copying videos: {src_videos} -> {dst_videos}")
            copy_tree_no_links(src_videos, dst_videos)
            print(f"Copied videos dir ({sum(1 for _ in dst_videos.iterdir() if _.is_dir())} episodes).")
        else:
            print(f"[warn] 无视频目录可复制: {src_videos}")

    print(f"Done. output episodes={dst.n_episodes} steps={dst.n_steps}")


if __name__ == "__main__":
    main()
