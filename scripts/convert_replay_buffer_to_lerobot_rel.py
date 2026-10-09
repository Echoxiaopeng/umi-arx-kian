#!/usr/bin/env python3
"""
将真机采集（collect_drag_umi.py）的 replay_buffer.zarr 转为 Light-WAM 用的 LeRobot v2.1 格式。

源数据特点：
- camera0_rgb 是 float32 [0, 1]，不是 uint8
- robot0_gripper_width 是 (N,)，不是 (N, 1)
- 有 data/action（指令 TCP），但含单帧毛刺，本脚本不用它
- 用 robot0_eef_pos / rot_axis_angle / gripper 作为实际轨迹

转换语义：
这里的rot6d,是按照旋转矩阵的前两行flatten得到的

1. State (20维) = flatten(eef_pos[2,3], eef_rot[2,6], gripper[2,1])
   默认 state_ref=current（相对当前 EE）:
   - hist_pos  = R_t^T (p_{t-s} - p_t)
   - curr_pos  = 0
   - hist_rot6 = rot6d(R_t^T R_{t-s})
   - curr_rot6 = identity
   - grip      = [g_{t-s}, g_t]（绝对值）
   state_ref=episode_first（相对 episode 第一帧）:
   - hist_pos/rot6 = T_0^{-1} @ T_{t-s}
   - curr_pos/rot6 = T_0^{-1} @ T_t
   - grip      = [g_{t-s}, g_t]（绝对值）
2. Action (10维) = 相对当前 EE 的下一步实际 TCP（两种 state 模式共用）
   - pos  = R_t^T (p_{t+s} - p_t)
   - rot6 = rot6d(R_t^T R_{t+s})
   - grip = g_{t+s}（绝对值）
3. FPS: 默认 60→20（stride=3）
4. 视频: camera0_rgb → uint8 → observation.images.wrist_image

用法:
    python scripts/convert_replay_buffer_to_lerobot_rel.py
    python scripts/convert_replay_buffer_to_lerobot_rel.py --episode 0
    python scripts/convert_replay_buffer_to_lerobot_rel.py --state-ref episode_first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

ROOT = Path("/data1/projects/umi-arx-hzl")
DEFAULT_SRC = ROOT /"data"/"collect_demo"/"replay_buffer.zarr"
DEFAULT_DST = ROOT / "data" / "collect_demo"/"replay_buffer_lerobot_rel"
DEFAULT_TASK = "pick up the electronic component and place it into the designated box"

IDENTITY_ROT6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

STATE_NAMES = [
    "pos_hist_x",
    "pos_hist_y",
    "pos_hist_z",
    "pos_curr_x",
    "pos_curr_y",
    "pos_curr_z",
    "rot_hist_0",
    "rot_hist_1",
    "rot_hist_2",
    "rot_hist_3",
    "rot_hist_4",
    "rot_hist_5",
    "rot_curr_0",
    "rot_curr_1",
    "rot_curr_2",
    "rot_curr_3",
    "rot_curr_4",
    "rot_curr_5",
    "gripper_hist",
    "gripper_curr",
]

ACTION_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "rot6_0",
    "rot6_1",
    "rot6_2",
    "rot6_3",
    "rot6_4",
    "rot6_5",
    "gripper",
]


def episode_slices(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def gripper_as_1d(gripper: np.ndarray) -> np.ndarray:
    g = np.asarray(gripper)
    if g.ndim == 2 and g.shape[-1] == 1:
        g = g[:, 0]
    if g.ndim != 1:
        raise ValueError(f"Unexpected gripper shape {g.shape}, expected (N,) or (N, 1)")
    return g


def rgb_to_uint8(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB [N,H,W,3], got {arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr_f = arr.astype(np.float32, copy=False)
    vmax = float(np.nanmax(arr_f)) if arr_f.size else 1.0
    if vmax <= 1.0 + 1e-3:
        arr_f = arr_f * 255.0
    return np.ascontiguousarray(np.clip(np.rint(arr_f), 0, 255).astype(np.uint8))


def matrix_to_rot6d(R_mat: np.ndarray) -> np.ndarray:
    """UMI / Light-WAM rot6d: flatten the first two rows of R."""
    return np.concatenate([R_mat[0, :], R_mat[1, :]]).astype(np.float32)


def pose_rel_to_current(
    pos_ref: np.ndarray,
    rot_aa_ref: np.ndarray,
    pos_cur: np.ndarray,
    rot_aa_cur: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """p' = R_c^T (p_r - p_c), R' = R_c^T R_r."""
    R_cur = R.from_rotvec(rot_aa_cur.astype(np.float64)).as_matrix()
    R_ref = R.from_rotvec(rot_aa_ref.astype(np.float64)).as_matrix()
    pos_rel = (R_cur.T @ (pos_ref.astype(np.float64) - pos_cur.astype(np.float64))).astype(
        np.float32
    )
    return pos_rel, matrix_to_rot6d(R_cur.T @ R_ref)


def convert_episode_arrays(
    eef_pos: np.ndarray,
    eef_rot_aa: np.ndarray,
    gripper: np.ndarray,
    episode_index: int,
    fps_src: int = 60,
    fps_dst: int = 20,
    index_start: int = 0,
    state_ref: str = "current",
) -> tuple[pa.Table, int]:
    """Convert one episode to LeRobot parquet columns.

    state_ref:
      - current: hist/curr relative to current EE (legacy Light-WAM)
      - episode_first: hist/curr relative to episode first frame
    Action is always relative to current EE; gripper is absolute.
    """
    if state_ref not in ("current", "episode_first"):
        raise ValueError(f"state_ref must be 'current' or 'episode_first', got {state_ref}")
    if fps_src % fps_dst != 0:
        raise ValueError(f"fps_src ({fps_src}) must be divisible by fps_dst ({fps_dst})")
    stride = fps_src // fps_dst

    gripper = gripper_as_1d(gripper)
    n_frames = len(eef_pos)
    if len(eef_rot_aa) != n_frames or len(gripper) != n_frames:
        raise ValueError(
            f"Length mismatch: pos={n_frames}, rot={len(eef_rot_aa)}, grip={len(gripper)}"
        )

    pos_first = eef_pos[0]
    rot_first = eef_rot_aa[0]

    dst_indices = list(range(0, n_frames, stride))
    n_dst = len(dst_indices)
    new_actions = []
    new_states = []
    new_timestamps = []
    new_frame_indices = []
    new_episode_indices = []
    new_task_indices = []

    for dst_idx, src_idx in enumerate(dst_indices):
        hist_idx = max(0, src_idx - stride)
        next_idx = src_idx + stride
        pos_cur = eef_pos[src_idx]
        rot_cur = eef_rot_aa[src_idx]
        grip_cur = float(gripper[src_idx])
        grip_hist = float(gripper[hist_idx])

        if state_ref == "current":
            hist_pos, hist_rot6 = pose_rel_to_current(
                eef_pos[hist_idx], eef_rot_aa[hist_idx], pos_cur, rot_cur
            )
            curr_pos = np.zeros(3, dtype=np.float32)
            curr_rot6 = IDENTITY_ROT6.copy()
        else:
            # Both hist and curr expressed in episode-first frame.
            hist_pos, hist_rot6 = pose_rel_to_current(
                eef_pos[hist_idx], eef_rot_aa[hist_idx], pos_first, rot_first
            )
            curr_pos, curr_rot6 = pose_rel_to_current(
                pos_cur, rot_cur, pos_first, rot_first
            )

        state = np.concatenate(
            [
                hist_pos,
                curr_pos,
                hist_rot6,
                curr_rot6,
                np.array([grip_hist, grip_cur], dtype=np.float32),
            ]
        ).astype(np.float32)
        if state.shape[0] != 20:
            raise RuntimeError(f"state dim {state.shape[0]} != 20")

        if next_idx < n_frames:
            act_pos, act_rot6 = pose_rel_to_current(
                eef_pos[next_idx], eef_rot_aa[next_idx], pos_cur, rot_cur
            )
            action = np.concatenate(
                [act_pos, act_rot6, np.array([float(gripper[next_idx])], dtype=np.float32)]
            ).astype(np.float32)
        else:
            action = np.concatenate(
                [
                    np.zeros(3, dtype=np.float32),
                    IDENTITY_ROT6.copy(),
                    np.array([grip_cur], dtype=np.float32),
                ]
            )

        new_actions.append(action)
        new_states.append(state)
        new_timestamps.append(float(dst_idx / fps_dst))
        new_frame_indices.append(dst_idx)
        new_episode_indices.append(episode_index)
        new_task_indices.append(0)

    columns = {
        "action": pa.array(
            np.asarray(new_actions, dtype=np.float32).tolist(),
            type=pa.list_(pa.float32()),
        ),
        "observation.state": pa.array(
            np.asarray(new_states, dtype=np.float32).tolist(),
            type=pa.list_(pa.float32()),
        ),
        "timestamp": pa.array(new_timestamps, type=pa.float32()),
        "frame_index": pa.array(new_frame_indices, type=pa.int64()),
        "episode_index": pa.array(new_episode_indices, type=pa.int64()),
        "index": pa.array(list(range(index_start, index_start + n_dst)), type=pa.int64()),
        "task_index": pa.array(new_task_indices, type=pa.int64()),
    }
    return pa.table(columns), n_dst


def encode_video(rgb_frames: np.ndarray, output_path: Path, fps: int) -> None:
    rgb_frames = rgb_to_uint8(rgb_frames)
    if len(rgb_frames) == 0:
        raise ValueError(f"No frames to encode for {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, height, width, _ = rgb_frames.shape
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    for frame_arr in rgb_frames:
        frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def get_feature_stats(array: np.ndarray, axis: int | tuple, keepdims: bool) -> dict:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def _to_serializable(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def sample_video_frames_chw(
    video_path: Path, num_frames: int, n_total_hint: int | None = None
) -> np.ndarray:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    n_total = n_total_hint or stream.frames or 0
    if n_total <= 0:
        frames = []
        for frame in container.decode(stream):
            frames.append(np.transpose(frame.to_ndarray(format="rgb24"), (2, 0, 1)))
        container.close()
        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")
        n = min(num_frames, len(frames))
        indices = np.round(np.linspace(0, len(frames) - 1, n)).astype(int)
        return np.stack([frames[i] for i in indices], axis=0)

    n = min(num_frames, n_total)
    target = set(np.round(np.linspace(0, n_total - 1, n)).astype(int).tolist())
    sampled = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in target:
            sampled[i] = np.transpose(frame.to_ndarray(format="rgb24"), (2, 0, 1))
            if len(sampled) == len(target):
                break
    container.close()
    if not sampled:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack([sampled[i] for i in sorted(sampled)], axis=0)


def compute_episode_stats_from_parquet(
    parquet_path: Path,
    video_path: Path | None,
    compute_video_stats: bool = True,
) -> dict:
    table = pq.read_table(str(parquet_path))
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.float64)
    episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.float64)
    index = np.asarray(table["index"].to_pylist(), dtype=np.float64)
    task_index = np.asarray(table["task_index"].to_pylist(), dtype=np.float64)

    ep_stats = {
        "action": _to_serializable(get_feature_stats(action, axis=0, keepdims=action.ndim == 1)),
        "observation.state": _to_serializable(
            get_feature_stats(state, axis=0, keepdims=state.ndim == 1)
        ),
        "timestamp": _to_serializable(get_feature_stats(timestamp, axis=0, keepdims=True)),
        "frame_index": _to_serializable(get_feature_stats(frame_index, axis=0, keepdims=True)),
        "episode_index": _to_serializable(
            get_feature_stats(episode_index, axis=0, keepdims=True)
        ),
        "index": _to_serializable(get_feature_stats(index, axis=0, keepdims=True)),
        "task_index": _to_serializable(get_feature_stats(task_index, axis=0, keepdims=True)),
    }

    if compute_video_stats and video_path is not None and video_path.exists():
        n_sample = estimate_num_samples(len(action))
        images = sample_video_frames_chw(video_path, n_sample, n_total_hint=None)
        _, _, h, w = images.shape
        if max(h, w) >= 300:
            factor = int(w / 150) if w > h else int(h / 150)
            factor = max(factor, 1)
            images = images[:, :, ::factor, ::factor]
        img_stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
        ep_stats["observation.images.wrist_image"] = _to_serializable(
            {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                for k, v in img_stats.items()
            }
        )
    return ep_stats


def write_episodes_stats(
    dst_dir: Path,
    compute_video_stats: bool = True,
    episode_filter: set[int] | None = None,
) -> int:
    dst_data = dst_dir / "data" / "chunk-000"
    dst_meta = dst_dir / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    video_dir = dst_dir / "videos" / "chunk-000" / "observation.images.wrist_image"
    pq_files = sorted(dst_data.glob("episode_*.parquet"))
    if episode_filter is not None:
        pq_files = [p for p in pq_files if int(p.stem.split("_")[1]) in episode_filter]

    out_path = dst_meta / "episodes_stats.jsonl"
    existing: dict[int, dict] = {}
    if episode_filter is not None and out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing[int(row["episode_index"])] = row

    with open(out_path, "w") as f:
        written = 0
        for pq_file in tqdm(pq_files, desc="写 episodes_stats"):
            ep_idx = int(pq_file.stem.split("_")[1])
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
            stats = compute_episode_stats_from_parquet(
                pq_file,
                video_path=video_path,
                compute_video_stats=compute_video_stats,
            )
            existing[ep_idx] = {"episode_index": ep_idx, "stats": stats}
            written += 1
        for ep_idx in sorted(existing):
            f.write(json.dumps(existing[ep_idx], ensure_ascii=False) + "\n")
    return written if episode_filter is not None else len(existing)


def build_info_json(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps_dst: int,
) -> dict:
    video_feature = {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps_dst,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    features = {
        "observation.images.wrist_image": dict(video_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": [20],
            "names": {"motors": STATE_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": [10],
            "names": {"motors": ACTION_NAMES},
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "umi",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps_dst,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def convert_replay_buffer(
    src_zarr: Path,
    dst_dir: Path,
    *,
    fps_src: int = 60,
    fps_dst: int = 20,
    task: str = DEFAULT_TASK,
    episode: int | None = None,
    skip_videos: bool = False,
    stats_only: bool = False,
    skip_video_stats: bool = False,
    state_ref: str = "current",
    episode_index_offset: int = 0,
    index_start_offset: int = 0,
) -> dict:
    """Convert one zarr replay buffer to LeRobot v2.1.

    Returns summary dict with n_episodes / n_frames / global index offset used.
    """
    dst_meta = dst_dir / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    episode_filter = (
        {episode + episode_index_offset} if episode is not None else None
    )

    if stats_only:
        n = write_episodes_stats(
            dst_dir,
            compute_video_stats=not skip_video_stats,
            episode_filter=episode_filter,
        )
        print(f"\n已写入 episodes_stats.jsonl ({n} episodes) -> {dst_meta / 'episodes_stats.jsonl'}")
        return {"n_episodes": n, "n_frames": 0}

    src = zarr.open(str(src_zarr), mode="r")
    episode_ends = np.asarray(src["meta"]["episode_ends"])
    slices = episode_slices(episode_ends)

    if episode is not None:
        if episode < 0 or episode >= len(slices):
            raise ValueError(f"episode {episode} out of range [0, {len(slices)})")
        local_episode_indices = [episode]
    else:
        local_episode_indices = list(range(len(slices)))

    stride = fps_src // fps_dst
    dst_data = dst_dir / "data" / "chunk-000"
    dst_video_dir = dst_dir / "videos" / "chunk-000" / "observation.images.wrist_image"
    dst_data.mkdir(parents=True, exist_ok=True)

    # Precompute per-local-episode frame counts for global index offsets.
    local_n_dst: dict[int, int] = {}
    for i, (s, e) in enumerate(slices):
        n_src = e - s
        local_n_dst[i] = (n_src + stride - 1) // stride

    print(f"源 zarr: {src_zarr}")
    print(f"Episodes: {len(local_episode_indices)}")
    print(f"FPS: {fps_src} → {fps_dst} (stride={stride})")
    print("Action: 10D = next EE pose relative to current EE (from actual TCP, not data/action)")
    if state_ref == "episode_first":
        print("State:  20D = hist/curr relative to episode first frame + abs gripper")
    else:
        print("State:  20D = hist/curr relative to current EE + abs gripper")
    print(f"Output: {dst_dir}")
    if "action" in src["data"]:
        print("注意: 忽略 data/action（指令流），只用 robot0_eef_* + gripper")

    pos_arr = src["data"]["robot0_eef_pos"]
    rot_arr = src["data"]["robot0_eef_rot_axis_angle"]
    grip_arr = src["data"]["robot0_gripper_width"]
    rgb_arr = src["data"]["camera0_rgb"]

    total_frames = 0
    new_episodes = []
    next_index = index_start_offset
    # Build index_start for each local episode in conversion order.
    running = index_start_offset
    index_starts: dict[int, int] = {}
    for local_ep in local_episode_indices:
        index_starts[local_ep] = running
        running += local_n_dst[local_ep]

    for local_ep in tqdm(local_episode_indices, desc="转换 episodes"):
        start, end = slices[local_ep]
        global_ep = local_ep + episode_index_offset
        table, n_dst = convert_episode_arrays(
            eef_pos=np.asarray(pos_arr[start:end]),
            eef_rot_aa=np.asarray(rot_arr[start:end]),
            gripper=np.asarray(grip_arr[start:end]),
            episode_index=global_ep,
            fps_src=fps_src,
            fps_dst=fps_dst,
            index_start=index_starts[local_ep],
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

        total_frames += n_dst
        next_index = index_starts[local_ep] + n_dst
        new_episodes.append(
            {"episode_index": global_ep, "tasks": [task], "length": n_dst}
        )

    episodes_path = dst_meta / "episodes.jsonl"
    if episode_filter is not None and episodes_path.exists():
        by_idx = {}
        for line in episodes_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                by_idx[int(row["episode_index"])] = row
        for ep in new_episodes:
            by_idx[int(ep["episode_index"])] = ep
        new_episodes = [by_idx[i] for i in sorted(by_idx)]

    with open(episodes_path, "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with open(dst_meta / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

    info_n_eps = len(new_episodes)
    info_n_frames = total_frames
    if episode_filter is not None:
        info_n_eps = len(list(dst_data.glob("episode_*.parquet")))
        info_n_frames = 0
        for p in dst_data.glob("episode_*.parquet"):
            info_n_frames += pq.read_metadata(str(p)).num_rows

    info = build_info_json(
        total_episodes=info_n_eps,
        total_frames=info_n_frames,
        total_tasks=1,
        fps_dst=fps_dst,
    )
    info["state_ref"] = state_ref
    assert info["features"]["observation.state"]["names"]["motors"] == STATE_NAMES
    assert info["features"]["action"]["names"]["motors"] == ACTION_NAMES
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    write_episodes_stats(
        dst_dir,
        compute_video_stats=not skip_video_stats,
        episode_filter=episode_filter,
    )

    print("\n转换完成!")
    print(f"  输出目录: {dst_dir}")
    print(f"  Episodes (本次): {len(local_episode_indices)}")
    print(f"  本次帧数: {total_frames} ({total_frames / fps_dst:.1f}s @ {fps_dst}fps)")
    print("  Action 维度: 10 (下一步相对当前 EE)")
    if state_ref == "episode_first":
        print("  State 维度:  20 (hist/curr 相对 episode 第一帧 + abs gripper)")
    else:
        print("  State 维度:  20 (hist/curr 相对当前 EE + abs gripper)")
    print("  视频: observation.images.wrist_image")
    print("  Meta: info.json / episodes.jsonl / tasks.jsonl / episodes_stats.jsonl")
    print(f"  FPS: {fps_dst}")
    return {
        "n_episodes": len(local_episode_indices),
        "n_frames": total_frames,
        "next_episode_index": episode_index_offset + (
            (local_episode_indices[-1] + 1) if local_episode_indices else 0
        ),
        "next_index": next_index,
        "state_ref": state_ref,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="转换真机 replay_buffer.zarr 为 LeRobot v2.1（相对 proprio 20D + 相对 action 10D）"
    )
    parser.add_argument("--src-zarr", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst-dir", type=Path, default=DEFAULT_DST)
    parser.add_argument("--fps-src", type=int, default=60)
    parser.add_argument(
        "--fps-dst",
        type=int,
        default=20,
        help="目标帧率。默认 20（stride=3，对齐 UMI / Light-WAM 管线）",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--episode", type=int, default=None, help="只转换指定 episode")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--skip-video-stats", action="store_true")
    parser.add_argument(
        "--state-ref",
        type=str,
        choices=["current", "episode_first"],
        default="current",
        help="state 参考系：current=相对当前 EE；episode_first=相对 episode 第一帧",
    )
    args = parser.parse_args()

    if not args.stats_only and not args.src_zarr.exists():
        raise FileNotFoundError(f"源 zarr 不存在: {args.src_zarr}")

    convert_replay_buffer(
        args.src_zarr,
        args.dst_dir,
        fps_src=args.fps_src,
        fps_dst=args.fps_dst,
        task=args.task,
        episode=args.episode,
        skip_videos=args.skip_videos,
        stats_only=args.stats_only,
        skip_video_stats=args.skip_video_stats,
        state_ref=args.state_ref,
    )


if __name__ == "__main__":
    main()