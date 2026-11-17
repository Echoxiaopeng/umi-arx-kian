"""Convert bimanual UMI-style replay_buffer.zarr to Kai0 Agilex LeRobot v2.1.

Source fields (per frame):
  robot{0,1}_joint_pos [6], robot{0,1}_gripper_width, camera{0,1,2}_rgb [H,W,3]
  action in zarr is EE pose — NOT used for joint training.

Target:
  observation.state [14] = [j0(6), g0, j1(6), g1]
  action [14]           = next-frame absolute joint state (last frame repeats)
  images: top_head <- camera1, hand_left <- camera0, hand_right <- camera2
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import zarr

try:
    import av
except ImportError:  # pragma: no cover
    av = None  # type: ignore

ProgressCb = Optional[Callable[[dict], None]]


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "转换为 LeRobot 需要 pyarrow。请在当前环境安装: pip install pyarrow"
        ) from exc
    return pa, pq

DEFAULT_PROMPT = "Flatten and fold the cloth."
DEFAULT_FPS = 20
IMAGE_HW = (224, 224)

CAM_MAP = (
    ("camera1_rgb", "observation.images.top_head"),
    ("camera0_rgb", "observation.images.hand_left"),
    ("camera2_rgb", "observation.images.hand_right"),
)

FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(DEFAULT_FPS),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(DEFAULT_FPS),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(DEFAULT_FPS),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {"dtype": "float32", "shape": [14]},
    "action": {"dtype": "float32", "shape": [14]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
}


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    if float(np.nanmax(x)) <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def encode_mp4(images: np.ndarray, path: Path, fps: int = 30) -> None:
    if av is None:
        raise ImportError(
            "PyAV (av) is required to encode LeRobot videos. "
            "Install with: pip install av"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = images[0].shape
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = int(w)
        stream.height = int(h)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "28", "preset": "ultrafast"}
        for img in images:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def load_cam_uint8(
    data,
    key: str,
    start: int,
    end: int,
    batch: int = 256,
    out_hw: tuple[int, int] = IMAGE_HW,
) -> np.ndarray:
    """Load RGB frames in batches; resize to out_hw if needed."""
    T = int(end) - int(start)
    oh, ow = int(out_hw[0]), int(out_hw[1])
    out = np.empty((T, oh, ow, 3), dtype=np.uint8)
    for i in range(0, T, batch):
        sl = slice(start + i, min(end, start + i + batch))
        chunk = to_uint8_rgb(np.asarray(data[key][sl]))
        for j in range(chunk.shape[0]):
            frame = chunk[j]
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"{key} frame shape unexpected: {frame.shape}")
            if frame.shape[0] != oh or frame.shape[1] != ow:
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            out[i + j] = frame
    return out


def build_joint_state(data, start: int, end: int) -> np.ndarray:
    j0 = np.asarray(data["robot0_joint_pos"][start:end], dtype=np.float32)
    j1 = np.asarray(data["robot1_joint_pos"][start:end], dtype=np.float32)
    g0 = np.asarray(data["robot0_gripper_width"][start:end], dtype=np.float32).reshape(
        -1, 1
    )
    g1 = np.asarray(data["robot1_gripper_width"][start:end], dtype=np.float32).reshape(
        -1, 1
    )
    return np.concatenate([j0, g0, j1, g1], axis=-1)


def find_zarr(extract_dir: Path) -> Path:
    extract_dir = Path(extract_dir)
    if extract_dir.name == "replay_buffer.zarr" and extract_dir.exists():
        return extract_dir
    direct = extract_dir / "replay_buffer.zarr"
    if direct.exists():
        return direct
    cands = list(extract_dir.rglob("replay_buffer.zarr"))
    if not cands:
        raise FileNotFoundError(f"No replay_buffer.zarr under {extract_dir}")
    return cands[0]


def default_lerobot_out_dir(zarr_dataset_dir: Path) -> Path:
    """Sibling `{name}_lerobot` next to the zarr dataset directory."""
    p = Path(zarr_dataset_dir)
    if p.name == "replay_buffer.zarr":
        p = p.parent
    return p.parent / f"{p.name}_lerobot"


def _emit(progress_cb: ProgressCb, **payload) -> None:
    if progress_cb is not None:
        progress_cb(dict(payload))


def convert_one_zarr(
    zarr_path: Path,
    out_dir: Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    fps: int = DEFAULT_FPS,
    episode_offset: int = 0,
    global_index_offset: int = 0,
    overwrite: bool = False,
    progress_cb: ProgressCb = None,
) -> tuple[int, int]:
    """Convert one replay_buffer.zarr into a LeRobot dataset at ``out_dir``.

    Returns (n_episodes, n_frames) written on disk.
    """
    pa, pq = _require_pyarrow()
    zarr_path = Path(zarr_path)
    out_dir = Path(out_dir)
    if zarr_path.name != "replay_buffer.zarr":
        zarr_path = find_zarr(zarr_path)

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    n_src = int(len(ends))

    data_chunk = out_dir / "data" / "chunk-000"
    vid_chunk = out_dir / "videos" / "chunk-000"
    meta_dir = out_dir / "meta"
    for _zkey, vkey in CAM_MAP:
        (vid_chunk / vkey).mkdir(parents=True, exist_ok=True)
    data_chunk.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    feats = json.loads(json.dumps(FEATURES))
    for vkey in (
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ):
        feats[vkey]["video_info"]["video.fps"] = float(fps)

    total_frames_written = 0
    for p in sorted(data_chunk.glob("episode_*.parquet")):
        total_frames_written += len(pq.read_table(p, columns=["index"]))

    if global_index_offset == 0 and episode_offset > 0:
        global_index = total_frames_written
    else:
        global_index = int(global_index_offset)

    # work units: each episode * (3 cams + 1 parquet write)
    n_units = max(n_src * 4, 1)
    done_units = 0

    for local_i in range(n_src):
        s, e = int(starts[local_i]), int(ends[local_i])
        T = e - s
        ep_idx = int(episode_offset) + local_i
        ep_name = f"episode_{ep_idx:06d}"
        pq_path = data_chunk / f"{ep_name}.parquet"

        if T <= 1:
            done_units += 4
            _emit(
                progress_cb,
                phase="convert",
                key=ep_name,
                key_i=local_i + 1,
                n_keys=n_src,
                frac=float(done_units) / float(n_units),
                message=f"跳过过短 ep{ep_idx} (T={T})",
            )
            continue

        if pq_path.exists() and not overwrite:
            # resume: still count as done
            done_units += 4
            _emit(
                progress_cb,
                phase="convert",
                key=ep_name,
                key_i=local_i + 1,
                n_keys=n_src,
                frac=float(done_units) / float(n_units),
                message=f"已存在，跳过 {ep_name}",
            )
            continue

        _emit(
            progress_cb,
            phase="convert",
            key=ep_name,
            key_i=local_i + 1,
            n_keys=n_src,
            frac=float(done_units) / float(n_units),
            message=f"转换 {ep_name} ({local_i + 1}/{n_src}) · T={T}",
        )

        state = build_joint_state(data, s, e)
        action = np.concatenate([state[1:], state[-1:]], axis=0)

        for zkey, vkey in CAM_MAP:
            cams = load_cam_uint8(data, zkey, s, e)
            encode_mp4(cams, vid_chunk / vkey / f"{ep_name}.mp4", fps)
            del cams
            done_units += 1
            _emit(
                progress_cb,
                phase="convert",
                key=f"{ep_name}/{vkey}",
                key_i=local_i + 1,
                n_keys=n_src,
                frac=float(done_units) / float(n_units),
                message=f"转换 {ep_name} · 视频 {vkey.split('.')[-1]}",
            )

        timestamps = np.arange(T, dtype=np.float32) / float(fps)
        frame_index = np.arange(T, dtype=np.int64)
        episode_index = np.full(T, ep_idx, dtype=np.int64)
        index = np.arange(global_index, global_index + T, dtype=np.int64)
        task_index = np.zeros(T, dtype=np.int64)
        global_index += T
        total_frames_written += T

        state_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(state.reshape(-1), type=pa.float32()), 14
        )
        action_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(action.reshape(-1), type=pa.float32()), 14
        )
        table = pa.table(
            {
                "observation.state": state_arr,
                "action": action_arr,
                "timestamp": timestamps,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": index,
                "task_index": task_index,
            }
        )
        pq.write_table(table, pq_path)
        done_units += 1
        _emit(
            progress_cb,
            phase="convert",
            key=ep_name,
            key_i=local_i + 1,
            n_keys=n_src,
            frac=float(done_units) / float(n_units),
            message=f"已写 {ep_name} T={T}",
        )
        print(f"  wrote {ep_name} T={T}", flush=True)

    parquets = sorted(data_chunk.glob("episode_*.parquet"))
    lengths: list[int] = []
    total = 0
    episodes_meta: list[dict] = []
    for p in parquets:
        n = len(pq.read_table(p, columns=["episode_index"]))
        ep_id = int(p.stem.split("_")[-1])
        episodes_meta.append(
            {"episode_index": ep_id, "tasks": [prompt], "length": n}
        )
        lengths.append(n)
        total += n

    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "codebase_version": "v2.1",
                "robot_type": "agilex",
                "total_episodes": len(parquets),
                "total_frames": total,
                "total_tasks": 1,
                "total_videos": len(parquets) * 3,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": int(fps),
                "splits": {"train": f"0:{len(parquets)}"},
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                ),
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4"
                ),
                "features": feats,
            },
            f,
            indent=2,
        )

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": prompt}) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for row, n in zip(episodes_meta, lengths):
            f.write(
                json.dumps(
                    {
                        "episode_index": row["episode_index"],
                        "stats": {
                            "observation.state": {
                                "min": [],
                                "max": [],
                                "mean": [],
                                "std": [],
                                "count": [n],
                            },
                            "action": {
                                "min": [],
                                "max": [],
                                "mean": [],
                                "std": [],
                                "count": [n],
                            },
                        },
                    }
                )
                + "\n"
            )

    _emit(
        progress_cb,
        phase="convert",
        key="(done)",
        key_i=n_src,
        n_keys=n_src,
        frac=1.0,
        message=f"转换完成 · {len(parquets)} episodes / {total} frames → {out_dir}",
    )
    return len(parquets), total
