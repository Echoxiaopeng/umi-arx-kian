"""LeRobot v2.1 dataset helpers for offline viewer: detect / list / series / frame / trim / delete."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

ProgressCb = Optional[Callable[[dict], None]]


def _require_pyarrow():
    """Lazy import so zarr-only manage / browse works without pyarrow installed."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "读取/裁剪 LeRobot 需要 pyarrow。请在当前环境安装: pip install pyarrow"
        ) from exc
    return pa, pq

LEROBOT_CAM_KEYS = (
    "observation.images.hand_left",
    "observation.images.top_head",
    "observation.images.hand_right",
)


def is_zarr_dataset(path: str | Path) -> bool:
    p = Path(path)
    if p.name == "replay_buffer.zarr" and p.exists():
        return True
    return (p / "replay_buffer.zarr").exists()


def is_lerobot_dataset(path: str | Path) -> bool:
    p = Path(path)
    info = p / "meta" / "info.json"
    if not info.is_file():
        return False
    try:
        meta = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(meta.get("codebase_version") or meta.get("data_path"))


def detect_dataset_format(path: str | Path) -> Optional[str]:
    p = Path(path)
    if is_zarr_dataset(p):
        return "zarr"
    if is_lerobot_dataset(p):
        return "lerobot"
    return None


def load_info(root: Path) -> dict:
    path = Path(root) / "meta" / "info.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _chunk_id(episode_index: int, chunks_size: int = 1000) -> int:
    return int(episode_index) // max(int(chunks_size), 1)


def episode_parquet_path(root: Path, episode_index: int, info: Optional[dict] = None) -> Path:
    root = Path(root)
    info = info or load_info(root)
    chunks_size = int(info.get("chunks_size") or 1000)
    chunk = _chunk_id(episode_index, chunks_size)
    tmpl = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel = tmpl.format(episode_chunk=chunk, episode_index=int(episode_index))
    return root / rel


def episode_video_path(
    root: Path,
    episode_index: int,
    video_key: str,
    info: Optional[dict] = None,
) -> Path:
    root = Path(root)
    info = info or load_info(root)
    chunks_size = int(info.get("chunks_size") or 1000)
    chunk = _chunk_id(episode_index, chunks_size)
    tmpl = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    rel = tmpl.format(
        episode_chunk=chunk,
        episode_index=int(episode_index),
        video_key=video_key,
    )
    return root / rel


def list_episode_ids(root: Path, info: Optional[dict] = None) -> list[int]:
    root = Path(root)
    info = info or load_info(root)
    eps_path = root / "meta" / "episodes.jsonl"
    ids: list[int] = []
    if eps_path.is_file():
        for line in eps_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ids.append(int(row["episode_index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        if ids:
            return sorted(ids)
    # fallback: scan parquet
    for p in sorted((root / "data").rglob("episode_*.parquet")):
        try:
            ids.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(ids)


def episode_length(root: Path, episode_index: int, info: Optional[dict] = None) -> int:
    root = Path(root)
    info = info or load_info(root)
    eps_path = root / "meta" / "episodes.jsonl"
    if eps_path.is_file():
        for line in eps_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if int(row["episode_index"]) == int(episode_index):
                    return int(row.get("length") or 0)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    pq_path = episode_parquet_path(root, episode_index, info)
    if not pq_path.is_file():
        return 0
    _pa, pq = _require_pyarrow()
    return len(pq.read_table(pq_path, columns=["frame_index"]))


def list_lerobot_episodes(root: Path) -> list[dict]:
    root = Path(root)
    info = load_info(root)
    fps = float(info.get("fps") or 20.0)
    out: list[dict] = []
    for eid in list_episode_ids(root, info):
        n_steps = episode_length(root, eid, info)
        has_video = any(
            episode_video_path(root, eid, k, info).is_file() for k in LEROBOT_CAM_KEYS
        )
        duration = (n_steps - 1) / fps if n_steps >= 2 and fps > 1e-6 else None
        out.append(
            {
                "id": int(eid),
                "n_steps": int(n_steps),
                "duration": duration,
                "has_video": bool(has_video),
            }
        )
    return out


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    n = int(n)
    max_points = max(int(max_points), 1)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_points).astype(np.int64)


def _fixed_list_to_np(col) -> np.ndarray:
    """Convert pyarrow FixedSizeList / list column to (N, D) float64."""
    try:
        return np.asarray(col.to_pylist(), dtype=np.float64)
    except Exception:
        arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
        return np.asarray(arr.to_pylist(), dtype=np.float64)


def episode_series_lerobot(
    root: Path, episode_id: int, *, max_points: int = 2000
) -> dict:
    root = Path(root)
    info = load_info(root)
    pq_path = episode_parquet_path(root, episode_id, info)
    if not pq_path.is_file():
        raise IndexError(f"episode {episode_id} parquet missing: {pq_path}")
    _pa, pq = _require_pyarrow()
    table = pq.read_table(pq_path)
    n = table.num_rows
    if n <= 0:
        return {
            "episode_id": int(episode_id),
            "n_steps": 0,
            "n_robots": 2,
            "steps": [],
            "t": [],
            "xyz": [],
            "rot": [],
            "grip": [],
            "robots": [],
            "action": [],
        }

    state = _fixed_list_to_np(table["observation.state"])
    action = _fixed_list_to_np(table["action"])
    if state.ndim == 1:
        state = state.reshape(n, -1)
    if action.ndim == 1:
        action = action.reshape(n, -1)
    if state.shape[-1] < 14:
        pad = np.zeros((n, 14), dtype=np.float64)
        pad[:, : state.shape[-1]] = state
        state = pad
    if action.shape[-1] < 14:
        pad = np.zeros((n, 14), dtype=np.float64)
        pad[:, : action.shape[-1]] = action
        action = pad

    if "timestamp" in table.column_names:
        t = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
        if t.size == n:
            t = t - t[0]
        else:
            t = np.arange(n, dtype=np.float64)
    else:
        fps = float(info.get("fps") or 20.0)
        t = np.arange(n, dtype=np.float64) / max(fps, 1e-6)

    idx = _downsample_indices(n, max_points)
    # Plot joint pos as "xyz" (first 3 joints) so existing chart still useful;
    # rot = joints 3:6; grip = joint gripper channel.
    robots_out = []
    for ridx, label in ((0, "L"), (1, "R")):
        base = 7 * ridx
        joints = state[:, base : base + 6]
        grip = state[:, base + 6]
        robots_out.append(
            {
                "label": label,
                "xyz": joints[idx, :3].tolist(),
                "rot": joints[idx, 3:6].tolist(),
                "grip": grip[idx].tolist(),
            }
        )

    r0 = robots_out[0]
    return {
        "episode_id": int(episode_id),
        "n_steps": int(n),
        "n_robots": 2,
        "steps": idx.tolist(),
        "t": t[idx].tolist(),
        "xyz": r0["xyz"],
        "rot": r0["rot"],
        "grip": r0["grip"],
        "robots": robots_out,
        "action": action[idx].tolist(),
    }


def _read_video_frame_bgr(path: Path, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        vidx = int(np.clip(index, 0, max(n_frames - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, vidx)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            return bgr
        return None
    finally:
        cap.release()


def episode_frame_jpeg_lerobot(
    root: Path, episode_id: int, index: int, quality: int = 80
) -> tuple[bytes, dict]:
    root = Path(root)
    info = load_info(root)
    n_steps = episode_length(root, episode_id, info)
    if n_steps <= 0:
        raise FileNotFoundError(f"episode {episode_id} has 0 steps")
    idx = int(np.clip(index, 0, max(n_steps - 1, 0)))

    tiles_bgr: list[np.ndarray] = []
    keys_used: list[str] = []
    for vkey in LEROBOT_CAM_KEYS:
        path = episode_video_path(root, episode_id, vkey, info)
        if not path.is_file():
            continue
        bgr = _read_video_frame_bgr(path, idx)
        if bgr is None:
            continue
        tiles_bgr.append(bgr)
        keys_used.append(vkey.split(".")[-1])

    if not tiles_bgr:
        raise FileNotFoundError(f"no videos for episode {episode_id}")

    h0 = min(t.shape[0] for t in tiles_bgr)
    resized = []
    for t in tiles_bgr:
        if t.shape[0] != h0:
            scale = h0 / float(t.shape[0])
            w2 = max(1, int(round(t.shape[1] * scale)))
            t = cv2.resize(t, (w2, h0), interpolation=cv2.INTER_AREA)
        resized.append(t)
    strip = np.concatenate(resized, axis=1)
    ok, buf = cv2.imencode(
        ".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("jpeg encode failed")
    h, w = int(strip.shape[0]), int(strip.shape[1])
    return bytes(buf), {
        "source": "lerobot_mp4",
        "key": "+".join(keys_used),
        "width": w,
        "height": h,
        "warning": "",
    }


def _trim_one_mp4_ffmpeg(mp4: Path, v0: int, v1: int, fps: float) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    tmp = mp4.with_suffix(".trim_tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    vf = f"trim=start_frame={v0}:end_frame={v1},setpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[lerobot-trim] ffmpeg failed for {mp4.name}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(mp4)
        return True
    if tmp.exists():
        tmp.unlink()
    return False


def _trim_one_mp4_cv2(mp4: Path, v0: int, v1: int, fps: float) -> bool:
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return False
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return False
        use_fps = float(fps) if fps > 1e-3 else float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        tmp = mp4.with_suffix(".trim_tmp.mp4")
        if tmp.exists():
            tmp.unlink()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp), fourcc, use_fps, (width, height))
        if not writer.isOpened():
            return False
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, v0)
            for _ in range(v0, v1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
        finally:
            writer.release()
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(mp4)
            return True
        if tmp.exists():
            tmp.unlink()
        return False
    finally:
        cap.release()


def _map_steps_to_video_frames(
    start_step: int, end_step: int, n_steps: int, n_frames: int
) -> tuple[int, int]:
    n_steps = max(int(n_steps), 1)
    n_frames = max(int(n_frames), 0)
    if n_frames <= 0:
        return 0, 0
    if n_frames != n_steps and n_steps > 1 and n_frames > 1:
        v0 = int(round(int(start_step) * (n_frames - 1) / (n_steps - 1)))
        v1 = int(round((int(end_step) - 1) * (n_frames - 1) / (n_steps - 1))) + 1
    else:
        v0 = int(start_step)
        v1 = int(end_step)
    v0 = int(np.clip(v0, 0, n_frames))
    v1 = int(np.clip(v1, v0 + 1 if v0 < n_frames else v0, n_frames))
    return v0, v1


def _rewrite_episodes_meta(
    root: Path,
    *,
    lengths_by_id: dict[int, int],
    prompt: Optional[str] = None,
) -> None:
    meta_dir = Path(root) / "meta"
    eps_path = meta_dir / "episodes.jsonl"
    rows: list[dict] = []
    if eps_path.is_file():
        for line in eps_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                eid = int(row["episode_index"])
                if eid in lengths_by_id:
                    row["length"] = int(lengths_by_id[eid])
                rows.append(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    else:
        for eid, n in sorted(lengths_by_id.items()):
            rows.append(
                {
                    "episode_index": eid,
                    "tasks": [prompt or DEFAULT_PROMPT_FALLBACK],
                    "length": int(n),
                }
            )

    with open(eps_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    stats_path = meta_dir / "episodes_stats.jsonl"
    stats_rows: list[dict] = []
    if stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                eid = int(row["episode_index"])
                if eid in lengths_by_id:
                    n = int(lengths_by_id[eid])
                    stats = row.setdefault("stats", {})
                    for key in ("observation.state", "action"):
                        if key not in stats:
                            stats[key] = {
                                "min": [],
                                "max": [],
                                "mean": [],
                                "std": [],
                                "count": [n],
                            }
                        else:
                            stats[key]["count"] = [n]
                stats_rows.append(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    with open(stats_path, "w", encoding="utf-8") as f:
        for row in stats_rows:
            f.write(json.dumps(row) + "\n")

    info = load_info(root)
    info["total_episodes"] = len(rows)
    info["total_frames"] = int(sum(int(r.get("length") or 0) for r in rows))
    info["total_videos"] = int(info["total_episodes"]) * 3
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


DEFAULT_PROMPT_FALLBACK = "Flatten and fold the cloth."


def _table_slice_rewrite(
    table,
    start: int,
    end: int,
    *,
    fps: float,
    index_start: int,
):
    pa, _pq = _require_pyarrow()
    n_old = table.num_rows
    start = int(start)
    end = int(end)
    if start < 0 or end > n_old or start >= end:
        raise ValueError(f"invalid trim [{start}, {end}) for T={n_old}")
    sliced = table.slice(start, end - start)
    T = sliced.num_rows
    frame_index = np.arange(T, dtype=np.int64)
    timestamps = (frame_index.astype(np.float32)) / float(max(fps, 1e-6))
    index = np.arange(int(index_start), int(index_start) + T, dtype=np.int64)
    # keep episode_index / task_index / state from sliced rows
    cols = {name: sliced[name] for name in sliced.column_names}
    cols["frame_index"] = pa.array(frame_index, type=pa.int64())
    cols["timestamp"] = pa.array(timestamps, type=pa.float32())
    cols["index"] = pa.array(index, type=pa.int64())
    # 裁切后必须重算 action=next-state（末帧重复），否则去尾后最后一帧 action 仍指向被删掉的下一步
    if "observation.state" in sliced.column_names and "action" in sliced.column_names:
        state = np.asarray(sliced["observation.state"].to_pylist(), dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(T, -1)
        action = np.concatenate([state[1:], state[-1:]], axis=0).astype(np.float32)
        dim = int(action.shape[-1])
        cols["action"] = pa.FixedSizeListArray.from_arrays(
            pa.array(action.reshape(-1), type=pa.float32()), dim
        )
    return pa.table(cols)


def trim_lerobot_episode(
    root: Path,
    episode_id: int,
    start: int,
    end: int,
    progress_cb: ProgressCb = None,
) -> bool:
    """Keep half-open [start, end) for one episode; rewrite parquet + videos + meta."""
    root = Path(root)
    info = load_info(root)
    fps = float(info.get("fps") or 20.0)
    ids = list_episode_ids(root, info)
    if int(episode_id) not in ids:
        raise IndexError(f"episode {episode_id} not in dataset")

    pa, pq = _require_pyarrow()
    pq_path = episode_parquet_path(root, episode_id, info)
    if not pq_path.is_file():
        raise FileNotFoundError(f"missing parquet {pq_path}")

    table = pq.read_table(pq_path)
    n_steps = table.num_rows
    start = int(start)
    end = int(end)
    if start < 0 or end > n_steps or start >= end:
        raise ValueError(f"invalid trim [{start}, {end}) for T={n_steps}")
    if start == 0 and end == n_steps:
        if progress_cb is not None:
            progress_cb(
                {
                    "phase": "lerobot",
                    "key": "(noop)",
                    "frac": 1.0,
                    "message": f"ep{episode_id} 无需裁剪",
                }
            )
        return True

    keep = end - start
    if progress_cb is not None:
        progress_cb(
            {
                "phase": "lerobot",
                "key": "parquet",
                "frac": 0.05,
                "message": f"裁剪 ep{episode_id} parquet → [{start},{end})",
            }
        )

    # index_start = sum of lengths of earlier episodes
    index_start = 0
    lengths_by_id = {eid: episode_length(root, eid, info) for eid in ids}
    for eid in ids:
        if eid == int(episode_id):
            break
        index_start += int(lengths_by_id.get(eid, 0))

    new_table = _table_slice_rewrite(
        table, start, end, fps=fps, index_start=index_start
    )
    tmp_pq = pq_path.with_suffix(".trim_tmp.parquet")
    pq.write_table(new_table, tmp_pq)
    tmp_pq.replace(pq_path)
    lengths_by_id[int(episode_id)] = keep

    # trim videos
    cam_list = list(LEROBOT_CAM_KEYS)
    for ci, vkey in enumerate(cam_list):
        mp4 = episode_video_path(root, episode_id, vkey, info)
        if progress_cb is not None:
            progress_cb(
                {
                    "phase": "video",
                    "key": vkey,
                    "key_i": ci + 1,
                    "n_keys": len(cam_list),
                    "frac": 0.2 + 0.5 * (ci / max(len(cam_list), 1)),
                    "message": f"裁剪视频 {vkey.split('.')[-1]} ({ci + 1}/{len(cam_list)})…",
                }
            )
        if not mp4.is_file():
            continue
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            continue
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            vfps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
        finally:
            cap.release()
        v0, v1 = _map_steps_to_video_frames(start, end, n_steps, n_frames)
        if v1 <= v0:
            continue
        if not _trim_one_mp4_ffmpeg(mp4, v0, v1, vfps):
            _trim_one_mp4_cv2(mp4, v0, v1, vfps)

    # reindex subsequent episodes' global `index`
    if progress_cb is not None:
        progress_cb(
            {
                "phase": "lerobot",
                "key": "reindex",
                "frac": 0.85,
                "message": f"重排后续 episode index…",
            }
        )
    running = index_start + keep
    for eid in ids:
        if eid <= int(episode_id):
            continue
        path = episode_parquet_path(root, eid, info)
        if not path.is_file():
            continue
        tab = pq.read_table(path)
        T = tab.num_rows
        new_index = np.arange(running, running + T, dtype=np.int64)
        cols = {name: tab[name] for name in tab.column_names}
        cols["index"] = pa.array(new_index, type=pa.int64())
        tmp = path.with_suffix(".reindex_tmp.parquet")
        pq.write_table(pa.table(cols), tmp)
        tmp.replace(path)
        running += T

    _rewrite_episodes_meta(root, lengths_by_id=lengths_by_id)

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "lerobot",
                "key": "(done)",
                "frac": 1.0,
                "message": f"已裁剪 ep{episode_id} → [{start},{end}) · {keep} 步",
            }
        )
    return True


def _unlink_episode_assets(root: Path, episode_id: int, info: dict) -> None:
    pq_path = episode_parquet_path(root, episode_id, info)
    if pq_path.is_file():
        pq_path.unlink()
    for vkey in LEROBOT_CAM_KEYS:
        mp4 = episode_video_path(root, episode_id, vkey, info)
        if mp4.is_file():
            mp4.unlink()


def _relocate_episode_assets(
    root: Path,
    src_eid: int,
    dst_eid: int,
    info: dict,
) -> None:
    """Move parquet + videos src→dst and set parquet ``episode_index`` to dst."""
    pa, pq = _require_pyarrow()
    src_pq = episode_parquet_path(root, src_eid, info)
    dst_pq = episode_parquet_path(root, dst_eid, info)
    if not src_pq.is_file():
        raise FileNotFoundError(f"missing parquet for ep{src_eid}: {src_pq}")
    if dst_pq.exists():
        raise FileExistsError(f"destination parquet already exists: {dst_pq}")
    dst_pq.parent.mkdir(parents=True, exist_ok=True)

    tab = pq.read_table(src_pq)
    T = int(tab.num_rows)
    cols = {name: tab[name] for name in tab.column_names}
    if "episode_index" in cols:
        cols["episode_index"] = pa.array(
            np.full(T, int(dst_eid), dtype=np.int64), type=pa.int64()
        )
    tmp_pq = dst_pq.with_suffix(".move_tmp.parquet")
    pq.write_table(pa.table(cols), tmp_pq)
    tmp_pq.replace(dst_pq)
    src_pq.unlink()

    for vkey in LEROBOT_CAM_KEYS:
        src_v = episode_video_path(root, src_eid, vkey, info)
        dst_v = episode_video_path(root, dst_eid, vkey, info)
        if not src_v.is_file():
            continue
        dst_v.parent.mkdir(parents=True, exist_ok=True)
        if dst_v.exists():
            dst_v.unlink()
        shutil.move(str(src_v), str(dst_v))


def _rewrite_meta_after_delete(
    root: Path,
    *,
    deleted_eid: int,
    id_map: dict[int, int],
    lengths_by_new_id: dict[int, int],
) -> None:
    """Rewrite episodes.jsonl / episodes_stats.jsonl / info.json after a delete."""
    meta_dir = Path(root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    eps_path = meta_dir / "episodes.jsonl"
    rows_out: list[dict] = []
    if eps_path.is_file():
        for line in eps_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                old = int(row["episode_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if old == int(deleted_eid) or old not in id_map:
                continue
            new = int(id_map[old])
            row["episode_index"] = new
            row["length"] = int(lengths_by_new_id.get(new, row.get("length") or 0))
            rows_out.append(row)
    if not rows_out:
        for new in sorted(lengths_by_new_id):
            rows_out.append(
                {
                    "episode_index": int(new),
                    "tasks": [DEFAULT_PROMPT_FALLBACK],
                    "length": int(lengths_by_new_id[new]),
                }
            )
    rows_out.sort(key=lambda r: int(r["episode_index"]))
    with open(eps_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    stats_path = meta_dir / "episodes_stats.jsonl"
    stats_out: list[dict] = []
    if stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                old = int(row["episode_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if old == int(deleted_eid) or old not in id_map:
                continue
            new = int(id_map[old])
            row["episode_index"] = new
            n = int(lengths_by_new_id.get(new, 0))
            stats = row.setdefault("stats", {})
            for key in ("observation.state", "action"):
                if key not in stats:
                    stats[key] = {
                        "min": [],
                        "max": [],
                        "mean": [],
                        "std": [],
                        "count": [n],
                    }
                else:
                    stats[key]["count"] = [n]
            stats_out.append(row)
    stats_out.sort(key=lambda r: int(r["episode_index"]))
    with open(stats_path, "w", encoding="utf-8") as f:
        for row in stats_out:
            f.write(json.dumps(row) + "\n")

    info = load_info(root)
    info["total_episodes"] = len(rows_out)
    info["total_frames"] = int(sum(int(r.get("length") or 0) for r in rows_out))
    n_cams = len(LEROBOT_CAM_KEYS)
    info["total_videos"] = int(info["total_episodes"]) * n_cams
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def delete_lerobot_episode(
    root: Path,
    episode_id: int,
    progress_cb: ProgressCb = None,
) -> bool:
    """Delete one episode; renumber higher ids down; rewrite parquet/videos/meta."""
    root = Path(root)
    info = load_info(root)
    ids = list_episode_ids(root, info)
    eid = int(episode_id)
    if eid not in ids:
        raise IndexError(f"episode {eid} not in dataset (have {ids[:8]}…)")

    lengths_old = {i: int(episode_length(root, i, info)) for i in ids}
    remaining_old = [i for i in ids if i != eid]
    id_map = {old: new for new, old in enumerate(remaining_old)}

    def _prog(frac: float, message: str, **extra) -> None:
        if progress_cb is None:
            return
        payload = {
            "phase": "lerobot_delete",
            "frac": float(frac),
            "message": message,
            "episode_id": eid,
        }
        payload.update(extra)
        progress_cb(payload)

    _prog(0.05, f"删除 ep{eid} 文件…")
    _unlink_episode_assets(root, eid, info)

    # 从高到低搬迁，避免覆盖尚未搬走的目标
    to_move = [(o, n) for o, n in id_map.items() if o != n]
    to_move.sort(key=lambda x: x[0], reverse=True)
    for mi, (src, dst) in enumerate(to_move):
        _prog(
            0.1 + 0.55 * ((mi + 1) / max(len(to_move), 1)),
            f"重编号 ep{src} → ep{dst} ({mi + 1}/{len(to_move)})…",
            key="relocate",
        )
        _relocate_episode_assets(root, src, dst, info)

    lengths_by_new = {id_map[o]: lengths_old[o] for o in remaining_old}

    _prog(0.75, "重写 meta…")
    _rewrite_meta_after_delete(
        root,
        deleted_eid=eid,
        id_map=id_map,
        lengths_by_new_id=lengths_by_new,
    )

    # 全局 index 连续重排
    pa, pq = _require_pyarrow()
    info = load_info(root)
    new_ids = list(range(len(remaining_old)))
    running = 0
    for ni, new_eid in enumerate(new_ids):
        _prog(
            0.8 + 0.15 * ((ni + 1) / max(len(new_ids), 1)),
            f"重排 index ep{new_eid}…",
            key="reindex",
        )
        path = episode_parquet_path(root, new_eid, info)
        if not path.is_file():
            continue
        tab = pq.read_table(path)
        T = int(tab.num_rows)
        cols = {name: tab[name] for name in tab.column_names}
        cols["index"] = pa.array(
            np.arange(running, running + T, dtype=np.int64), type=pa.int64()
        )
        if "episode_index" in cols:
            cols["episode_index"] = pa.array(
                np.full(T, int(new_eid), dtype=np.int64), type=pa.int64()
            )
        tmp = path.with_suffix(".reindex_tmp.parquet")
        pq.write_table(pa.table(cols), tmp)
        tmp.replace(path)
        running += T

    _prog(
        1.0,
        f"已删除 ep{eid} · 剩余 {len(remaining_old)} 条 · {running} 帧",
        key="(done)",
    )
    return True


class LerobotReplayShim:
    """Minimal replay_buffer-like surface for status / trim pre-checks.

    Episode ids are the real LeRobot ``episode_index`` values (as shown in the UI),
    not necessarily dense 0..n-1.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = load_info(self.root)
        self._ids = list_episode_ids(self.root, self.info)
        self._id_set = set(self._ids)
        self.data = {}  # no zarr keys

    @property
    def n_episodes(self) -> int:
        return len(self._ids)

    def has_episode(self, episode_index: int) -> bool:
        return int(episode_index) in self._id_set

    def get_episode_slice(self, episode_index: int) -> slice:
        eid = int(episode_index)
        if eid not in self._id_set:
            raise IndexError(f"episode {eid} out of range")
        T = episode_length(self.root, eid, self.info)
        return slice(0, T)

    def refresh(self) -> None:
        self.info = load_info(self.root)
        self._ids = list_episode_ids(self.root, self.info)
        self._id_set = set(self._ids)


class OfflineLerobotEnv:
    """Offline LeRobot dataset handle for the web viewer."""

    dataset_format = "lerobot"

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        if not is_lerobot_dataset(self.output_dir):
            raise FileNotFoundError(
                f"不是 LeRobot 数据集（缺少 meta/info.json）: {self.output_dir}"
            )
        self.replay_buffer = LerobotReplayShim(self.output_dir)
        # video_dir unused for lerobot layout; keep attr for getattr compatibility
        self.video_dir = self.output_dir / "videos"
