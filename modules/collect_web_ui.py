"""
ZMQ 拖拽采数网页服务（标准库 HTTP，无 FastAPI 依赖）。

主循环写入 SharedCollectState / 入队命令；浏览器只读状态、发命令，
不直接控制机械臂。
"""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# 仓库根下的静态页
_DEFAULT_STATIC = Path(__file__).resolve().parent.parent / "web" / "collect_zmq"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def list_fs_dir(path: str | Path | None = None) -> dict:
    """
    列目录（仅子文件夹），供网页文件夹浏览器使用。
    返回 path / parent / entries[{name,path,is_dataset,n_children}]。
    """
    if path is None or str(path).strip() == "":
        target = _REPO_ROOT
    else:
        p = Path(str(path).strip()).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        target = p

    if not target.exists():
        raise FileNotFoundError(f"路径不存在: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"不是目录: {target}")

    parent = target.parent if target.parent != target else None
    entries: list[dict] = []
    try:
        children = sorted(target.iterdir(), key=lambda x: x.name.lower())
    except PermissionError as exc:
        raise PermissionError(f"无权限读取: {target}") from exc

    for child in children:
        name = child.name
        if name.startswith("."):
            continue
        if not child.is_dir():
            continue
        # zarr 组目录无需再点进去浏览
        if name.endswith(".zarr"):
            continue
        # zarr (replay_buffer.zarr) 或 LeRobot (meta/info.json)
        from modules.lerobot_dataset_io import detect_dataset_format

        fmt = detect_dataset_format(child)
        is_dataset = fmt is not None
        n_children = None
        try:
            n_children = sum(
                1
                for c in child.iterdir()
                if c.is_dir()
                and not c.name.startswith(".")
                and not c.name.endswith(".zarr")
            )
        except (PermissionError, OSError):
            n_children = None
        entries.append(
            {
                "name": name,
                "path": str(child),
                "is_dataset": bool(is_dataset),
                "dataset_format": fmt,
                "n_children": n_children,
            }
        )

    from modules.lerobot_dataset_io import detect_dataset_format

    fmt_self = detect_dataset_format(target)
    return {
        "path": str(target),
        "parent": str(parent) if parent is not None else None,
        "is_dataset": fmt_self is not None,
        "dataset_format": fmt_self,
        "repo_root": str(_REPO_ROOT),
        "entries": entries,
    }


def _resolve_fs_path(path: str | Path | None = None) -> Path:
    if path is None or str(path).strip() == "":
        return _REPO_ROOT
    p = Path(str(path).strip()).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _dataset_light_summary(path: Path, fmt: str) -> dict:
    """Cheap per-dataset stats for the library table (no full env open)."""
    out: dict[str, Any] = {
        "n_episodes": None,
        "total_frames": None,
        "fps": None,
    }
    try:
        if fmt == "lerobot":
            from modules.lerobot_dataset_io import load_info

            info = load_info(path)
            out["n_episodes"] = int(info.get("total_episodes") or 0) or None
            out["total_frames"] = int(info.get("total_frames") or 0) or None
            if info.get("fps") is not None:
                out["fps"] = float(info["fps"])
            return out
        if fmt == "zarr":
            import zarr

            zpath = path / "replay_buffer.zarr"
            if not zpath.exists() and path.name == "replay_buffer.zarr":
                zpath = path
            root = zarr.open(str(zpath), mode="r")
            ends = None
            try:
                ends = root["meta"]["episode_ends"][:]
            except Exception:  # noqa: BLE001
                ends = None
            if ends is not None and len(ends) > 0:
                out["n_episodes"] = int(len(ends))
                out["total_frames"] = int(ends[-1])
            return out
    except Exception:  # noqa: BLE001
        return out
    return out


def list_library_datasets(path: str | Path | None = None) -> dict:
    """
    库管理：列出父目录下一层子目录中的数据集 + 非数据集文件夹。
    返回 path / parent / datasets[] / other_dirs[] / repo_root。
    """
    target = _resolve_fs_path(path)
    if not target.exists():
        raise FileNotFoundError(f"路径不存在: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"不是目录: {target}")

    from modules.lerobot_dataset_io import detect_dataset_format

    parent = target.parent if target.parent != target else None
    datasets: list[dict] = []
    other_dirs: list[dict] = []
    try:
        children = sorted(target.iterdir(), key=lambda x: x.name.lower())
    except PermissionError as exc:
        raise PermissionError(f"无权限读取: {target}") from exc

    for child in children:
        name = child.name
        if name.startswith(".") or not child.is_dir() or name.endswith(".zarr"):
            continue
        fmt = detect_dataset_format(child)
        try:
            mtime = float(child.stat().st_mtime)
        except OSError:
            mtime = None
        if fmt is None:
            other_dirs.append({"name": name, "path": str(child), "mtime": mtime})
            continue
        summary = _dataset_light_summary(child, fmt)
        datasets.append(
            {
                "name": name,
                "path": str(child),
                "dataset_format": fmt,
                "is_dataset": True,
                "n_episodes": summary.get("n_episodes"),
                "total_frames": summary.get("total_frames"),
                "fps": summary.get("fps"),
                "mtime": mtime,
            }
        )

    return {
        "path": str(target),
        "parent": str(parent) if parent is not None else None,
        "is_dataset": detect_dataset_format(target) is not None,
        "dataset_format": detect_dataset_format(target),
        "repo_root": str(_REPO_ROOT),
        "datasets": datasets,
        "other_dirs": other_dirs,
    }


class CommandQueue:
    """线程安全命令队列：网页 POST → 主循环 drain。"""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def put(self, action: str, **payload: Any) -> None:
        item = {"action": str(action), **payload}
        self._q.put(item)

    def drain(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


class SharedCollectState:
    """主循环写、HTTP 读的共享状态。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.mode: str = "IDLE"
        self.zero_force_active: bool = False
        self.recording: bool = False
        self.no_zero_force: bool = True
        self.n_steps: int = 0
        self.t_elapsed: float = 0.0
        self.n_episodes: int = 0
        self.episode_id: Optional[int] = None
        self.pos: list[float] = [0.0, 0.0, 0.0]
        self.rotvec: list[float] = [0.0, 0.0, 0.0]
        self.grip: float = 0.0
        # Dual-arm: [{pos, rotvec, grip, label}, ...]；单臂时长度 1（与 pos/rotvec/grip 同步）
        self.robots: list[dict] = [
            {
                "pos": [0.0, 0.0, 0.0],
                "rotvec": [0.0, 0.0, 0.0],
                "grip": 0.0,
                "label": "L",
            }
        ]
        self.n_robots: int = 1
        self.camera_latency: list[dict] = []
        self.delete_pending: Optional[int] = None
        self.deleting: bool = False
        # 删除进度：{episode_id, phase, key, key_i, n_keys, frac, message} 或 None
        self.delete_progress: Optional[dict] = None
        # 末尾开合预览：{episode_id, n, cut, drop, start, end} 或 None
        self.end_grip_preview: Optional[dict] = None
        self.message: str = ""
        self.frequency: float = 60.0
        self.viewer_only: bool = False
        # DAgger：当前目录是否含 intervention/ sidecar
        self.has_intervention: bool = False
        self.dagger_stats: Optional[dict] = None
        # zarr | lerobot | ""
        self.dataset_format: str = ""
        self._jpeg: bytes = b""
        self._live_series: list[dict] = []
        self.logs: list[str] = []
        # 由主循环注入，只读访问 replay / videos
        self.env: Any = None
        self.output_dir: Optional[Path] = None

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if k == "jpeg":
                    if isinstance(v, (bytes, bytearray)) and len(v) > 0:
                        self._jpeg = bytes(v)
                elif k == "live_series":
                    self._live_series = list(v)
                elif k == "logs":
                    self.logs = list(v)
                elif hasattr(self, k):
                    setattr(self, k, v)

    def clear_logs(self) -> None:
        with self._lock:
            self.logs = []

    def append_log(self, line: str, *, max_lines: int = 200) -> None:
        """追加一条带时间戳的启动/运行日志（网页 CONNECTING 面板用）。"""
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {line}"
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > max_lines:
                self.logs = self.logs[-max_lines:]
            self.message = str(line)

    def clear_live_series(self) -> None:
        with self._lock:
            self._live_series = []

    def append_live_sample(
        self,
        *,
        step: int,
        t: float,
        xyz: np.ndarray,
        rot: np.ndarray,
        grip: float,
        action: Optional[np.ndarray] = None,
        robots: Optional[list[dict]] = None,
        max_points: int = 8000,
    ) -> None:
        sample = {
            "step": int(step),
            "t": float(t),
            "xyz": [float(x) for x in np.asarray(xyz, dtype=np.float64).reshape(3)],
            "rot": [float(x) for x in np.asarray(rot, dtype=np.float64).reshape(3)],
            "grip": float(grip),
        }
        if robots is not None:
            sample["robots"] = robots
        if action is not None:
            a = np.asarray(action, dtype=np.float64).reshape(-1)
            # Keep full action (7 * n_robots); UI decides how many dims to plot.
            sample["action"] = [float(x) for x in a]
        with self._lock:
            self._live_series.append(sample)
            if len(self._live_series) > max_points:
                self._live_series = self._live_series[-max_points:]

    def snapshot(self) -> dict:
        with self._lock:
            out_dir = getattr(self, "output_dir", None)
            robots = list(getattr(self, "robots", []) or [])
            return {
                "mode": self.mode,
                "zero_force_active": self.zero_force_active,
                "recording": self.recording,
                "no_zero_force": self.no_zero_force,
                "n_steps": self.n_steps,
                "t_elapsed": self.t_elapsed,
                "n_episodes": self.n_episodes,
                "episode_id": self.episode_id,
                "pos": list(self.pos),
                "rotvec": list(self.rotvec),
                "grip": float(self.grip),
                "robots": robots,
                "n_robots": int(getattr(self, "n_robots", len(robots) or 1)),
                "camera_latency": list(getattr(self, "camera_latency", []) or []),
                "delete_pending": self.delete_pending,
                "deleting": bool(getattr(self, "deleting", False)),
                "delete_progress": (
                    dict(self.delete_progress)
                    if isinstance(getattr(self, "delete_progress", None), dict)
                    else None
                ),
                "end_grip_preview": (
                    dict(self.end_grip_preview)
                    if isinstance(getattr(self, "end_grip_preview", None), dict)
                    else None
                ),
                "message": self.message,
                "frequency": self.frequency,
                "live_n": len(self._live_series),
                "viewer_only": bool(getattr(self, "viewer_only", False)),
                "has_intervention": bool(
                    getattr(self, "has_intervention", False)
                ),
                "dagger_stats": (
                    dict(self.dagger_stats)
                    if isinstance(getattr(self, "dagger_stats", None), dict)
                    else None
                ),
                "dataset_format": str(
                    getattr(self, "dataset_format", "")
                    or getattr(getattr(self, "env", None), "dataset_format", "")
                    or ""
                ),
                "output_dir": str(out_dir) if out_dir is not None else "",
                "repo_root": str(_REPO_ROOT),
                "logs": list(self.logs),
            }

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def get_live_series(self) -> list[dict]:
        with self._lock:
            return list(self._live_series)


def _encode_jpeg_bgr(bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b""
    return buf.tobytes()


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        bgr = arr[..., ::-1]
    else:
        bgr = arr
    return _encode_jpeg_bgr(bgr, quality=quality)


def vis_rgb_to_jpeg(vis_rgb: np.ndarray, quality: int = 80) -> bytes:
    """make_vis_image 输出为 RGB。"""
    return rgb_to_jpeg(vis_rgb, quality=quality)


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=max_points, dtype=np.int64)


def _rb_slice_array(replay_buffer, key: str, sl: slice) -> np.ndarray:
    """只切片读一个 key，避免 get_episode 把 camera rgb 整段解压进内存。"""
    if key not in replay_buffer.data:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(replay_buffer.data[key][sl])


def _count_robots_in_buffer(replay_buffer) -> int:
    n = 0
    while f"robot{n}_eef_pos" in replay_buffer.data:
        n += 1
    return max(n, 1)


def _slice_robot_pose_series(replay_buffer, robot_idx: int, sl: slice, n_hint: int = 0):
    pos = _rb_slice_array(replay_buffer, f"robot{robot_idx}_eef_pos", sl)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3) if pos.size else np.zeros((0, 3))
    elif pos.ndim >= 2:
        pos = pos.reshape(-1, pos.shape[-1])[:, :3]
    n = int(pos.shape[0]) if pos.size else int(n_hint)

    rot = _rb_slice_array(replay_buffer, f"robot{robot_idx}_eef_rot_axis_angle", sl)
    if rot.size == 0:
        rot = np.zeros((n, 3), dtype=np.float64)
    else:
        rot = np.asarray(rot, dtype=np.float64).reshape(-1, 3)

    grip = _rb_slice_array(replay_buffer, f"robot{robot_idx}_gripper_width", sl)
    if grip.size == 0:
        grip = np.zeros((n,), dtype=np.float64)
    else:
        grip = np.asarray(grip, dtype=np.float64).reshape(-1)
    return pos, rot, grip, n


_ARM_LABELS = ("L", "R", "A2", "A3")


def _live_series_robots(series: list[dict]) -> list[dict]:
    """
    Rebuild per-arm time series for /api/live/series.

    Prefer sample['robots']; if missing, synthesize from action (7 dims each).
    Length always matches len(series).
    """
    if not series:
        return []

    n_r = 1
    for s in series:
        if isinstance(s.get("robots"), list) and len(s["robots"]) > n_r:
            n_r = len(s["robots"])
        act = s.get("action")
        if act is not None:
            try:
                n_r = max(n_r, int(len(act) // 7))
            except Exception:
                pass

    out: list[dict] = []
    for ridx in range(max(n_r, 1)):
        label = _ARM_LABELS[ridx] if ridx < len(_ARM_LABELS) else f"A{ridx}"
        xyz_list: list = []
        rot_list: list = []
        grip_list: list = []
        for s in series:
            robots = s.get("robots")
            if isinstance(robots, list) and ridx < len(robots):
                arm = robots[ridx]
                xyz_list.append(list(arm.get("xyz") or [0.0, 0.0, 0.0]))
                rot_list.append(list(arm.get("rot") or [0.0, 0.0, 0.0]))
                grip_list.append(float(arm.get("grip") or 0.0))
                continue
            act = s.get("action")
            if act is not None and len(act) >= 7 * (ridx + 1):
                base = 7 * ridx
                xyz_list.append(
                    [float(act[base]), float(act[base + 1]), float(act[base + 2])]
                )
                rot_list.append(
                    [
                        float(act[base + 3]),
                        float(act[base + 4]),
                        float(act[base + 5]),
                    ]
                )
                grip_list.append(float(act[base + 6]))
                continue
            if ridx == 0:
                xyz_list.append(list(s.get("xyz") or [0.0, 0.0, 0.0]))
                rot_list.append(list(s.get("rot") or [0.0, 0.0, 0.0]))
                grip_list.append(float(s.get("grip") or 0.0))
            else:
                xyz_list.append([0.0, 0.0, 0.0])
                rot_list.append([0.0, 0.0, 0.0])
                grip_list.append(0.0)
        out.append(
            {
                "label": label,
                "xyz": xyz_list,
                "rot": rot_list,
                "grip": grip_list,
            }
        )
    return out


def episode_series_from_buffer(
    replay_buffer, episode_id: int, *, max_points: int = 2000
) -> dict:
    n_ep = int(replay_buffer.n_episodes)
    if episode_id < 0 or episode_id >= n_ep:
        raise IndexError(f"episode {episode_id} out of range 0..{n_ep - 1}")
    sl = replay_buffer.get_episode_slice(episode_id)

    n_robots = _count_robots_in_buffer(replay_buffer)

    action = _rb_slice_array(replay_buffer, "action", sl)
    if action.size:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            # Ambiguous; prefer known robot count.
            cols = max(7 * n_robots, 7)
            action = action.reshape(-1, cols) if action.size else np.zeros((0, cols))
        # Infer dual-arm from action width even if robot1_* keys missing.
        if action.ndim >= 2 and action.shape[-1] >= 14:
            n_robots = max(n_robots, int(action.shape[-1] // 7))

    act_dim = 7 * n_robots
    if action.size == 0:
        action = np.zeros((0, act_dim), dtype=np.float64)
    elif action.shape[-1] < act_dim:
        pad = np.zeros((action.shape[0], act_dim), dtype=np.float64)
        pad[:, : action.shape[-1]] = action
        action = pad

    robots_raw = []
    n = 0
    for ridx in range(n_robots):
        pos, rot, grip, n_r = _slice_robot_pose_series(replay_buffer, ridx, sl, n)
        if ridx == 0:
            n = n_r if n_r > 0 else int(action.shape[0])
        # Fallback: synthesize pose from action block when robot{i}_* missing.
        if pos.shape[0] == 0 and action.shape[0] > 0 and action.shape[-1] >= 7 * (ridx + 1):
            block = action[:, 7 * ridx : 7 * (ridx + 1)]
            pos = block[:, :3].copy()
            rot = block[:, 3:6].copy()
            grip = block[:, 6].copy()
            if ridx == 0 and n == 0:
                n = int(pos.shape[0])
        robots_raw.append((pos, rot, grip))

    ts = _rb_slice_array(replay_buffer, "timestamp", sl)
    if ts.size == n and n > 0:
        t = np.asarray(ts, dtype=np.float64).reshape(-1)
        t = t - t[0]
    else:
        t = np.arange(n, dtype=np.float64)

    idx = _downsample_indices(n, max_points)
    robots_out = []
    for ridx, (pos, rot, grip) in enumerate(robots_raw):
        pos_d = pos[idx] if n and pos.shape[0] == n else np.zeros((idx.shape[0], 3))
        rot_d = rot[idx] if n and rot.shape[0] == n else np.zeros((idx.shape[0], 3))
        grip_d = grip[idx] if n and grip.shape[0] == n else np.zeros((idx.shape[0],))
        label = _ARM_LABELS[ridx] if ridx < len(_ARM_LABELS) else f"A{ridx}"
        robots_out.append(
            {
                "label": label,
                "xyz": pos_d.tolist(),
                "rot": rot_d.tolist(),
                "grip": grip_d.tolist(),
            }
        )

    act_d = (
        action[idx]
        if action.shape[0] == n and n
        else np.zeros((idx.shape[0], act_dim), dtype=np.float64)
    )
    t_d = t[idx] if t.shape[0] == n and n else np.arange(idx.shape[0], dtype=np.float64)

    # robot0 aliases for older UI / single-arm
    r0 = robots_out[0] if robots_out else {"xyz": [], "rot": [], "grip": []}
    return {
        "episode_id": int(episode_id),
        "n_steps": n,
        "n_robots": int(n_robots),
        "steps": idx.tolist(),
        "t": t_d.tolist(),
        "xyz": r0["xyz"],
        "rot": r0["rot"],
        "grip": r0["grip"],
        "robots": robots_out,
        "action": act_d.tolist(),
    }


def dataset_summary(env, *, frequency_fallback: float = 20.0) -> dict:
    """Dataset-level Frames / Episodes / FPS for the viewer header."""
    fmt = str(getattr(env, "dataset_format", "zarr") or "zarr")
    fps = float(frequency_fallback) if frequency_fallback else 20.0
    n_episodes = 0
    total_frames = 0

    if fmt == "lerobot":
        from modules.lerobot_dataset_io import (
            episode_length,
            list_episode_ids,
            load_info,
        )

        root = Path(env.output_dir)
        info = load_info(root)
        fps = float(info.get("fps") or fps)
        ids = list_episode_ids(root, info)
        n_episodes = int(info.get("total_episodes") or len(ids))
        total_frames = int(info.get("total_frames") or 0)
        if total_frames <= 0:
            total_frames = int(
                sum(int(episode_length(root, eid, info)) for eid in ids)
            )
        if n_episodes <= 0:
            n_episodes = len(ids)
        return {
            "n_episodes": int(n_episodes),
            "total_frames": int(total_frames),
            "fps": float(fps),
            "dataset_format": "lerobot",
        }

    rb = env.replay_buffer
    n_episodes = int(rb.n_episodes)
    for eid in range(n_episodes):
        sl = rb.get_episode_slice(eid)
        total_frames += int(sl.stop - sl.start)
    return {
        "n_episodes": int(n_episodes),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "dataset_format": "zarr",
    }


def list_episodes(env) -> list[dict]:
    fmt = getattr(env, "dataset_format", "zarr")
    if fmt == "lerobot":
        from modules.lerobot_dataset_io import list_lerobot_episodes

        return list_lerobot_episodes(Path(env.output_dir))

    rb = env.replay_buffer
    n = int(rb.n_episodes)
    video_dir = Path(env.video_dir)
    out: list[dict] = []
    for eid in range(n):
        sl = rb.get_episode_slice(eid)
        n_steps = int(sl.stop - sl.start)
        vdir = video_dir / str(eid)
        has_video = vdir.is_dir() and any(vdir.glob("*.mp4"))
        duration = None
        try:
            if n_steps >= 2 and "timestamp" in rb.data:
                ts = np.asarray(rb.data["timestamp"][sl], dtype=np.float64).reshape(-1)
                if ts.size >= 2:
                    duration = float(ts[-1] - ts[0])
        except Exception:
            pass
        out.append(
            {
                "id": eid,
                "n_steps": n_steps,
                "duration": duration,
                "has_video": bool(has_video),
            }
        )
    return out


def episode_series(env, episode_id: int, *, max_points: int = 2000) -> dict:
    """Episode time-series for the chart (zarr or LeRobot)."""
    fmt = getattr(env, "dataset_format", "zarr")
    if fmt == "lerobot":
        from modules.lerobot_dataset_io import episode_series_lerobot

        return episode_series_lerobot(
            Path(env.output_dir), int(episode_id), max_points=max_points
        )
    return episode_series_from_buffer(
        env.replay_buffer, int(episode_id), max_points=max_points
    )


def _open_episode_video(env, episode_id: int) -> Optional[cv2.VideoCapture]:
    vdir = Path(env.video_dir) / str(episode_id)
    if not vdir.is_dir():
        return None
    mp4s = sorted(vdir.glob("*.mp4"))
    if not mp4s:
        return None
    # Prefer 0.mp4 (first camera)
    preferred = vdir / "0.mp4"
    path = preferred if preferred.exists() else mp4s[0]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    return cap


def episode_frame_jpeg(
    env, episode_id: int, index: int, quality: int = 80
) -> tuple[bytes, dict]:
    """
    取 episode 某一帧 JPEG。

    优先 zarr 相机（与训练数据一致）；失败再回退 mp4。
    LeRobot：从三路 mp4 拼条带。
    返回 (jpeg_bytes, meta)，meta 含 source=zarr|mp4|lerobot_mp4 与可选 warning。
    """
    fmt = getattr(env, "dataset_format", "zarr")
    if fmt == "lerobot":
        from modules.lerobot_dataset_io import episode_frame_jpeg_lerobot

        return episode_frame_jpeg_lerobot(
            Path(env.output_dir), int(episode_id), int(index), quality=quality
        )

    n_ep = int(env.replay_buffer.n_episodes)
    if episode_id < 0 or episode_id >= n_ep:
        raise IndexError(f"episode {episode_id} out of range")

    rb = env.replay_buffer
    sl = rb.get_episode_slice(episode_id)
    n_steps = int(sl.stop - sl.start)
    idx = int(np.clip(index, 0, max(n_steps - 1, 0)))

    zarr_err: Optional[str] = None
    cam_keys = sorted(
        k for k in rb.data.keys() if "camera" in k and "rgb" in k
    )
    if cam_keys and n_steps > 0:
        try:
            tiles = []
            for key in cam_keys:
                frame = np.asarray(rb.data[key][sl.start + idx])
                if frame.dtype != np.uint8:
                    if np.issubdtype(frame.dtype, np.floating) and float(
                        np.nanmax(frame)
                    ) <= 1.5:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                tiles.append(frame)
            if len(tiles) == 1:
                strip = tiles[0]
            else:
                # Match heights for hstack (obs may differ slightly).
                h0 = min(t.shape[0] for t in tiles)
                resized = []
                for t in tiles:
                    if t.shape[0] != h0:
                        scale = h0 / float(t.shape[0])
                        w2 = max(1, int(round(t.shape[1] * scale)))
                        t = cv2.resize(t, (w2, h0), interpolation=cv2.INTER_AREA)
                    resized.append(t)
                strip = np.concatenate(resized, axis=1)
            jpeg = rgb_to_jpeg(strip, quality=quality)
            if jpeg:
                h, w = int(strip.shape[0]), int(strip.shape[1])
                return jpeg, {
                    "source": "zarr",
                    "key": "+".join(cam_keys),
                    "width": w,
                    "height": h,
                    "warning": "",
                }
            zarr_err = "jpeg encode failed"
        except Exception as exc:  # noqa: BLE001
            zarr_err = str(exc)
    elif not cam_keys:
        zarr_err = "zarr 无 camera*_rgb"
    else:
        zarr_err = "episode 步数为 0"

    cap = _open_episode_video(env, episode_id)
    if cap is not None:
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if n_frames > 0 and n_steps > 1 and n_frames != n_steps:
                vidx = int(round(idx * (n_frames - 1) / (n_steps - 1)))
            else:
                vidx = idx
            vidx = int(np.clip(vidx, 0, max(n_frames - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, vidx)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                jpeg = _encode_jpeg_bgr(bgr, quality=quality)
                h, w = int(bgr.shape[0]), int(bgr.shape[1])
                reason = zarr_err or "zarr 不可用"
                return jpeg, {
                    "source": "mp4",
                    "key": "",
                    "width": w,
                    "height": h,
                    "warning": f"zarr 不可用，已回退 mp4（{reason}）",
                }
        finally:
            cap.release()

    raise FileNotFoundError(
        f"no video/images for episode {episode_id}"
        + (f" · zarr: {zarr_err}" if zarr_err else "")
    )


class CollectWebServer:
    def __init__(
        self,
        state: SharedCollectState,
        cmd_queue: CommandQueue,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        static_dir: Optional[Path] = None,
    ):
        self.state = state
        self.cmd_queue = cmd_queue
        self.host = host
        self.port = int(port)
        self.static_dir = Path(static_dir) if static_dir else _DEFAULT_STATIC
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        state = self.state
        cmd_queue = self.cmd_queue
        static_dir = self.static_dir

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:  # quieter
                if "/api/status" in str(args[0] if args else "") or "/api/video" in str(
                    args[0] if args else ""
                ):
                    return
                if "/api/live" in str(args[0] if args else ""):
                    return
                super().log_message(fmt, *args)

            def _send(
                self,
                code: int,
                body: bytes,
                content_type: str,
                extra_headers: Optional[dict] = None,
            ) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header(
                    "Access-Control-Expose-Headers",
                    "X-Frame-Source, X-Frame-Warning, X-Frame-Width, X-Frame-Height",
                )
                if extra_headers:
                    for hk, hv in extra_headers.items():
                        if hv is None or hv == "":
                            continue
                        self.send_header(str(hk), str(hv))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, code: int, obj: Any) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self._send(code, body, "application/json; charset=utf-8")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                try:
                    parsed = urlparse(self.path)
                    path = parsed.path
                    qs = parse_qs(parsed.query)

                    if path in ("/", "/index.html"):
                        index = static_dir / "index.html"
                        if not index.exists():
                            self._send_json(404, {"error": "index.html missing"})
                            return
                        self._send(
                            200,
                            index.read_bytes(),
                            "text/html; charset=utf-8",
                        )
                        return

                    if path.startswith("/static/"):
                        rel = path[len("/static/") :]
                        fpath = (static_dir / rel).resolve()
                        if not str(fpath).startswith(str(static_dir.resolve())):
                            self._send_json(403, {"error": "forbidden"})
                            return
                        if not fpath.is_file():
                            self._send_json(404, {"error": "not found"})
                            return
                        ctype = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
                        self._send(200, fpath.read_bytes(), ctype)
                        return

                    if path == "/api/status":
                        snap = state.snapshot()
                        want_eps = qs.get("episodes", ["0"])
                        include_eps = str(want_eps[0]).lower() in ("1", "true", "yes")
                        if state.env is not None:
                            try:
                                snap["n_episodes"] = int(
                                    state.env.replay_buffer.n_episodes
                                )
                                snap["dataset_format"] = str(
                                    getattr(state.env, "dataset_format", "")
                                    or snap.get("dataset_format")
                                    or ""
                                )
                                try:
                                    summary = dataset_summary(
                                        state.env,
                                        frequency_fallback=float(
                                            getattr(state, "frequency", 20.0)
                                            or 20.0
                                        ),
                                    )
                                    snap["n_episodes"] = int(
                                        summary.get("n_episodes")
                                        or snap["n_episodes"]
                                    )
                                    snap["total_frames"] = int(
                                        summary.get("total_frames") or 0
                                    )
                                    snap["dataset_fps"] = float(
                                        summary.get("fps")
                                        or snap.get("frequency")
                                        or 20.0
                                    )
                                    # keep frequency aligned for scrub time labels
                                    if snap.get("dataset_fps"):
                                        snap["frequency"] = float(snap["dataset_fps"])
                                except Exception as exc:  # noqa: BLE001
                                    snap["dataset_summary_error"] = str(exc)
                                if include_eps:
                                    snap["episodes"] = list_episodes(state.env)
                            except Exception as exc:  # noqa: BLE001
                                snap["episodes_error"] = str(exc)
                        else:
                            snap.setdefault("total_frames", 0)
                            snap.setdefault(
                                "dataset_fps",
                                float(snap.get("frequency") or 20.0),
                            )
                        self._send_json(200, snap)
                        return

                    if path == "/api/episodes":
                        if state.env is None:
                            self._send_json(503, {"error": "env not ready"})
                            return
                        self._send_json(200, {"episodes": list_episodes(state.env)})
                        return

                    if path == "/api/fs/list":
                        raw_path = qs.get("path", [""])[0]
                        try:
                            self._send_json(200, list_fs_dir(raw_path or None))
                        except FileNotFoundError as exc:
                            self._send_json(404, {"error": str(exc)})
                        except NotADirectoryError as exc:
                            self._send_json(400, {"error": str(exc)})
                        except PermissionError as exc:
                            self._send_json(403, {"error": str(exc)})
                        except OSError as exc:
                            self._send_json(500, {"error": str(exc)})
                        return

                    if path == "/api/fs/library":
                        raw_path = qs.get("path", [""])[0]
                        try:
                            self._send_json(
                                200, list_library_datasets(raw_path or None)
                            )
                        except FileNotFoundError as exc:
                            self._send_json(404, {"error": str(exc)})
                        except NotADirectoryError as exc:
                            self._send_json(400, {"error": str(exc)})
                        except PermissionError as exc:
                            self._send_json(403, {"error": str(exc)})
                        except OSError as exc:
                            self._send_json(500, {"error": str(exc)})
                        return

                    if path == "/api/video":
                        self._stream_mjpeg()
                        return

                    if path == "/api/live/series":
                        series = state.get_live_series()
                        robots_series = _live_series_robots(series)
                        self._send_json(
                            200,
                            {
                                "n_steps": len(series),
                                "n_robots": len(robots_series) or 1,
                                "steps": [s["step"] for s in series],
                                "t": [s["t"] for s in series],
                                "xyz": [s["xyz"] for s in series],
                                "rot": [s["rot"] for s in series],
                                "grip": [s["grip"] for s in series],
                                "robots": robots_series,
                                "action": [s.get("action") for s in series],
                            },
                        )
                        return

                    if path == "/api/live/frame":
                        jpeg = state.get_jpeg()
                        if not jpeg:
                            self._send_json(404, {"error": "no frame"})
                            return
                        self._send(200, jpeg, "image/jpeg")
                        return

                    # /api/episodes/{id}/series
                    parts = path.strip("/").split("/")
                    if (
                        len(parts) == 4
                        and parts[0] == "api"
                        and parts[1] == "episodes"
                        and parts[3] == "series"
                    ):
                        if state.env is None:
                            self._send_json(503, {"error": "env not ready"})
                            return
                        eid = int(parts[2])
                        max_points = int(qs.get("max_points", ["2000"])[0])
                        try:
                            data = episode_series(
                                state.env,
                                eid,
                                max_points=max_points,
                            )
                            self._send_json(200, data)
                        except IndexError as exc:
                            self._send_json(404, {"error": str(exc)})
                        except Exception as exc:  # noqa: BLE001
                            self._send_json(500, {"error": str(exc)})
                        return

                    if (
                        len(parts) == 4
                        and parts[0] == "api"
                        and parts[1] == "episodes"
                        and parts[3] == "frame"
                    ):
                        if state.env is None:
                            self._send_json(503, {"error": "env not ready"})
                            return
                        eid = int(parts[2])
                        index = int(qs.get("index", ["0"])[0])
                        try:
                            jpeg, meta = episode_frame_jpeg(state.env, eid, index)
                            self._send(
                                200,
                                jpeg,
                                "image/jpeg",
                                extra_headers={
                                    "X-Frame-Source": meta.get("source", ""),
                                    "X-Frame-Warning": meta.get("warning", ""),
                                    "X-Frame-Width": meta.get("width", ""),
                                    "X-Frame-Height": meta.get("height", ""),
                                },
                            )
                        except (IndexError, FileNotFoundError) as exc:
                            self._send_json(404, {"error": str(exc)})
                        except Exception as exc:  # noqa: BLE001
                            self._send_json(500, {"error": str(exc)})
                        return

                    self._send_json(404, {"error": f"unknown path {path}"})
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    try:
                        self._send_json(500, {"error": str(exc)})
                    except Exception:
                        pass

            def _stream_mjpeg(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while True:
                        jpeg = state.get_jpeg()
                        if jpeg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self) -> None:  # noqa: N802
                try:
                    parsed = urlparse(self.path)
                    if parsed.path != "/api/cmd":
                        self._send_json(404, {"error": "unknown path"})
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    try:
                        payload = json.loads(raw.decode("utf-8") or "{}")
                    except json.JSONDecodeError:
                        self._send_json(400, {"error": "invalid json"})
                        return
                    action = payload.get("action")
                    if not action:
                        self._send_json(400, {"error": "action required"})
                        return
                    extra = {
                        k: v
                        for k, v in payload.items()
                        if k != "action"
                    }
                    cmd_queue.put(str(action), **extra)
                    self._send_json(200, {"ok": True, "action": action})
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    self._send_json(500, {"error": str(exc)})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="collect-web", daemon=True
        )
        self._thread.start()
        print(
            f"[collect_web_ui] http://{self.host}:{self.port}/  "
            f"(static={self.static_dir})"
        )

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread is not None:
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass
            self._thread = None


def start_collect_web_ui(
    state: SharedCollectState,
    cmd_queue: CommandQueue,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> CollectWebServer:
    server = CollectWebServer(state, cmd_queue, host=host, port=port)
    server.start()
    return server


def publish_pose_from_env(state: SharedCollectState, env) -> None:
    try:
        states = env.get_robot_state()
        robots = []
        for ridx, st in enumerate(states):
            pose = np.asarray(st["ActualTCPPose"], dtype=np.float64).reshape(6)
            grip = float(st["gripper_position"])
            label = _ARM_LABELS[ridx] if ridx < len(_ARM_LABELS) else f"A{ridx}"
            robots.append(
                {
                    "pos": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "rotvec": [float(pose[3]), float(pose[4]), float(pose[5])],
                    "grip": grip,
                    "label": label,
                }
            )
        r0 = robots[0]
        state.update(
            pos=list(r0["pos"]),
            rotvec=list(r0["rotvec"]),
            grip=float(r0["grip"]),
            robots=robots,
            n_robots=len(robots),
            n_episodes=int(env.replay_buffer.n_episodes),
        )
    except Exception:
        pass
