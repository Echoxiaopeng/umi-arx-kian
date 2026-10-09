#!/usr/bin/env python3
"""把多套已对齐的 replay_buffer.zarr 直接合并为一份 LeRobot v2.1。

默认语义（对齐 VLA 训练）:
  - Action 10D: T_cur^{-1} @ T_next（pos3 + rot6d）+ 绝对 gripper
  - State 20D: hist/curr 均相对 episode 第一帧（pos3+rot6d）+ 绝对 gripper
  - rot6d: 旋转矩阵前两行 flatten
  - FPS: 60 → 20

默认输入顺序:
  1. data/collect_demo_zmq
  2. data/collect_demo_as_zmq
  3. data/collect_demo_pro_as_zmq

用法:
  # 冒烟：每源只转 episode 0
  python scripts/merge_replay_buffers_to_lerobot_rel.py --smoke

  # 全量
  python scripts/merge_replay_buffers_to_lerobot_rel.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_replay_buffer_to_lerobot_rel import (  # noqa: E402
    DEFAULT_TASK,
    build_info_json,
    convert_episode_arrays,
    encode_video,
    episode_slices,
    write_episodes_stats,
)

DEFAULT_SOURCES = [
    {
        "name": "collect_demo_zmq",
        "zarr": ROOT / "data" / "collect_demo_zmq" / "replay_buffer.zarr",
    },
    {
        "name": "collect_demo_as_zmq",
        "zarr": ROOT / "data" / "collect_demo_as_zmq" / "replay_buffer.zarr",
    },
    {
        "name": "collect_demo_pro_as_zmq",
        "zarr": ROOT / "data" / "collect_demo_pro_as_zmq" / "replay_buffer.zarr",
    },
]

DEFAULT_DST = ROOT / "data" / "collect_demo_merged_lerobot_rel"


def merge_sources_to_lerobot(
    sources: list[dict],
    dst_dir: Path,
    *,
    fps_src: int = 60,
    fps_dst: int = 20,
    task: str = DEFAULT_TASK,
    state_ref: str = "episode_first",
    smoke: bool = False,
    skip_videos: bool = False,
    skip_video_stats: bool = False,
) -> dict:
    if state_ref != "episode_first":
        raise ValueError(
            "merge script requires state_ref='episode_first' for VLA training"
        )
    if fps_src % fps_dst != 0:
        raise ValueError(f"fps_src ({fps_src}) must be divisible by fps_dst ({fps_dst})")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    dst_meta = dst_dir / "meta"
    dst_data = dst_dir / "data" / "chunk-000"
    dst_video_dir = dst_dir / "videos" / "chunk-000" / "observation.images.wrist_image"
    dst_meta.mkdir(parents=True, exist_ok=True)
    dst_data.mkdir(parents=True, exist_ok=True)
    if not skip_videos:
        dst_video_dir.mkdir(parents=True, exist_ok=True)

    stride = fps_src // fps_dst
    global_ep = 0
    global_index = 0
    total_frames = 0
    episodes_meta: list[dict] = []
    manifest_rows: list[dict] = []

    print("=" * 64)
    print("Merge replay_buffer.zarr → LeRobot v2.1")
    print(f"  dst       : {dst_dir}")
    print(f"  fps       : {fps_src} → {fps_dst} (stride={stride})")
    print("  action    : relative to current EE + abs gripper")
    print("  state     : hist/curr relative to episode first frame + abs gripper")
    print(f"  smoke     : {smoke}")
    print("=" * 64)

    for src_info in sources:
        src_name = src_info["name"]
        src_zarr = Path(src_info["zarr"])
        if not src_zarr.exists():
            raise FileNotFoundError(f"missing source zarr: {src_zarr}")

        root = zarr.open(str(src_zarr), mode="r")
        ends = np.asarray(root["meta/episode_ends"][:])
        slices = episode_slices(ends)
        local_ids = [0] if smoke else list(range(len(slices)))

        print(f"\n[{src_name}] episodes={len(local_ids)}/{len(slices)}  zarr={src_zarr}")

        pos_arr = root["data"]["robot0_eef_pos"]
        rot_arr = root["data"]["robot0_eef_rot_axis_angle"]
        grip_arr = root["data"]["robot0_gripper_width"]
        rgb_arr = root["data"]["camera0_rgb"]

        for local_ep in local_ids:
            start, end = slices[local_ep]
            table, n_dst = convert_episode_arrays(
                eef_pos=np.asarray(pos_arr[start:end]),
                eef_rot_aa=np.asarray(rot_arr[start:end]),
                gripper=np.asarray(grip_arr[start:end]),
                episode_index=global_ep,
                fps_src=fps_src,
                fps_dst=fps_dst,
                index_start=global_index,
                state_ref=state_ref,
            )
            pq.write_table(table, str(dst_data / f"episode_{global_ep:06d}.parquet"))

            if not skip_videos:
                src_indices = np.arange(0, end - start, stride)
                encode_video(
                    np.asarray(rgb_arr[start + src_indices]),
                    dst_video_dir / f"episode_{global_ep:06d}.mp4",
                    fps=fps_dst,
                )

            episodes_meta.append(
                {"episode_index": global_ep, "tasks": [task], "length": n_dst}
            )
            manifest_rows.append(
                {
                    "episode_index": global_ep,
                    "source_name": src_name,
                    "source_zarr": str(src_zarr),
                    "source_episode_index": local_ep,
                    "source_start": int(start),
                    "source_end": int(end),
                    "n_frames_dst": n_dst,
                    "index_start": global_index,
                }
            )

            if (global_ep + 1) % 20 == 0 or local_ep == local_ids[-1]:
                print(
                    f"  wrote global_ep={global_ep:03d} "
                    f"(src_ep={local_ep}) frames={n_dst}"
                )

            global_ep += 1
            global_index += n_dst
            total_frames += n_dst

    # meta files
    with open(dst_meta / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with open(dst_meta / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

    info = build_info_json(
        total_episodes=len(episodes_meta),
        total_frames=total_frames,
        total_tasks=1,
        fps_dst=fps_dst,
    )
    info["state_ref"] = state_ref
    info["action_ref"] = "current_ee"
    info["gripper"] = "absolute"
    info["rot6d"] = "first_two_rows"
    info["sources"] = [s["name"] for s in sources]
    with open(dst_meta / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    manifest_path = dst_meta / "source_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_episodes_stats(
        dst_dir,
        compute_video_stats=not skip_video_stats,
        episode_filter=None,
    )

    summary = {
        "dst_dir": str(dst_dir),
        "n_episodes": len(episodes_meta),
        "n_frames": total_frames,
        "fps_src": fps_src,
        "fps_dst": fps_dst,
        "state_ref": state_ref,
        "smoke": smoke,
        "sources": [s["name"] for s in sources],
    }
    with open(dst_meta / "merge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n合并完成!")
    print(f"  episodes = {summary['n_episodes']}")
    print(f"  frames   = {summary['n_frames']} @ {fps_dst}fps")
    print(f"  output   = {dst_dir}")
    print(f"  manifest = {manifest_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并多套对齐 zarr 为 LeRobot v2.1（episode-first state + current-EE action）"
    )
    parser.add_argument("--dst-dir", type=Path, default=DEFAULT_DST)
    parser.add_argument("--fps-src", type=int, default=60)
    parser.add_argument("--fps-dst", type=int, default=20)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="每源只转换 episode 0，用于冒烟测试",
    )
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-video-stats", action="store_true")
    args = parser.parse_args()

    merge_sources_to_lerobot(
        DEFAULT_SOURCES,
        args.dst_dir,
        fps_src=args.fps_src,
        fps_dst=args.fps_dst,
        task=args.task,
        state_ref="episode_first",
        smoke=args.smoke,
        skip_videos=args.skip_videos,
        skip_video_stats=args.skip_video_stats,
    )


if __name__ == "__main__":
    main()
