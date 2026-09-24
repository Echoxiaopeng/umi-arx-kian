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
        self.delete_pending: Optional[int] = None
        self.message: str = ""
        self.frequency: float = 60.0
        self.viewer_only: bool = False
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
        max_points: int = 8000,
    ) -> None:
        sample = {
            "step": int(step),
            "t": float(t),
            "xyz": [float(x) for x in np.asarray(xyz, dtype=np.float64).reshape(3)],
            "rot": [float(x) for x in np.asarray(rot, dtype=np.float64).reshape(3)],
            "grip": float(grip),
        }
        if action is not None:
            a = np.asarray(action, dtype=np.float64).reshape(-1)
            sample["action"] = [float(x) for x in a[:7]]
        with self._lock:
            self._live_series.append(sample)
            if len(self._live_series) > max_points:
                self._live_series = self._live_series[-max_points:]

    def snapshot(self) -> dict:
        with self._lock:
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
                "delete_pending": self.delete_pending,
                "message": self.message,
                "frequency": self.frequency,
                "live_n": len(self._live_series),
                "viewer_only": bool(getattr(self, "viewer_only", False)),
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


def episode_series_from_buffer(
    replay_buffer, episode_id: int, *, max_points: int = 2000
) -> dict:
    n_ep = int(replay_buffer.n_episodes)
    if episode_id < 0 or episode_id >= n_ep:
        raise IndexError(f"episode {episode_id} out of range 0..{n_ep - 1}")
    sl = replay_buffer.get_episode_slice(episode_id)

    pos = _rb_slice_array(replay_buffer, "robot0_eef_pos", sl)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3) if pos.size else np.zeros((0, 3))
    elif pos.ndim >= 2:
        pos = pos.reshape(-1, pos.shape[-1])[:, :3]
    n = int(pos.shape[0])

    rot = _rb_slice_array(replay_buffer, "robot0_eef_rot_axis_angle", sl)
    if rot.size == 0:
        rot = np.zeros((n, 3), dtype=np.float64)
    else:
        rot = np.asarray(rot, dtype=np.float64).reshape(-1, 3)

    grip = _rb_slice_array(replay_buffer, "robot0_gripper_width", sl)
    if grip.size == 0:
        grip = np.zeros((n,), dtype=np.float64)
    else:
        grip = np.asarray(grip, dtype=np.float64).reshape(-1)

    action = _rb_slice_array(replay_buffer, "action", sl)
    if action.size == 0:
        action = np.zeros((n, 7), dtype=np.float64)
    else:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            action = action.reshape(-1, 7)
        elif action.shape[-1] < 7:
            pad = np.zeros((action.shape[0], 7), dtype=np.float64)
            pad[:, : action.shape[-1]] = action
            action = pad

    ts = _rb_slice_array(replay_buffer, "timestamp", sl)
    if ts.size == n and n > 0:
        t = np.asarray(ts, dtype=np.float64).reshape(-1)
        t = t - t[0]
    else:
        t = np.arange(n, dtype=np.float64)

    idx = _downsample_indices(n, max_points)
    pos_d = pos[idx] if n else pos
    rot_d = rot[idx] if rot.shape[0] == n and n else np.zeros((idx.shape[0], 3))
    grip_d = grip[idx] if grip.shape[0] == n and n else np.zeros((idx.shape[0],))
    act_d = (
        action[idx]
        if action.shape[0] == n and n
        else np.zeros((idx.shape[0], 7), dtype=np.float64)
    )
    t_d = t[idx] if t.shape[0] == n and n else np.arange(idx.shape[0], dtype=np.float64)

    return {
        "episode_id": int(episode_id),
        "n_steps": n,
        "steps": idx.tolist(),
        "t": t_d.tolist(),
        "xyz": pos_d.tolist(),
        "rot": rot_d.tolist(),
        "grip": grip_d.tolist(),
        "action": act_d.tolist(),
    }


def list_episodes(env) -> list[dict]:
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


def episode_frame_jpeg(env, episode_id: int, index: int, quality: int = 80) -> bytes:
    n_ep = int(env.replay_buffer.n_episodes)
    if episode_id < 0 or episode_id >= n_ep:
        raise IndexError(f"episode {episode_id} out of range")

    rb = env.replay_buffer
    sl = rb.get_episode_slice(episode_id)
    n_steps = int(sl.stop - sl.start)
    idx = int(np.clip(index, 0, max(n_steps - 1, 0)))

    # 优先 zarr 单帧（随机访问快）；避免每次打开整段 mp4 seek
    cam_keys = sorted(
        k for k in rb.data.keys() if "camera" in k and "rgb" in k
    )
    if cam_keys and n_steps > 0:
        try:
            frame = np.asarray(rb.data[cam_keys[0]][sl.start + idx])
            return rgb_to_jpeg(frame, quality=quality)
        except Exception:
            pass

    cap = _open_episode_video(env, episode_id)
    if cap is not None:
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            vidx = int(np.clip(index, 0, max(n_frames - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, vidx)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                return _encode_jpeg_bgr(bgr, quality=quality)
        finally:
            cap.release()

    raise FileNotFoundError(f"no video/images for episode {episode_id}")


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

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
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
                                if include_eps:
                                    snap["episodes"] = list_episodes(state.env)
                            except Exception as exc:  # noqa: BLE001
                                snap["episodes_error"] = str(exc)
                        self._send_json(200, snap)
                        return

                    if path == "/api/episodes":
                        if state.env is None:
                            self._send_json(503, {"error": "env not ready"})
                            return
                        self._send_json(200, {"episodes": list_episodes(state.env)})
                        return

                    if path == "/api/video":
                        self._stream_mjpeg()
                        return

                    if path == "/api/live/series":
                        series = state.get_live_series()
                        self._send_json(
                            200,
                            {
                                "n_steps": len(series),
                                "steps": [s["step"] for s in series],
                                "t": [s["t"] for s in series],
                                "xyz": [s["xyz"] for s in series],
                                "rot": [s["rot"] for s in series],
                                "grip": [s["grip"] for s in series],
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
                            data = episode_series_from_buffer(
                                state.env.replay_buffer,
                                eid,
                                max_points=max_points,
                            )
                            self._send_json(200, data)
                        except IndexError as exc:
                            self._send_json(404, {"error": str(exc)})
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
                            jpeg = episode_frame_jpeg(state.env, eid, index)
                            self._send(200, jpeg, "image/jpeg")
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
        st = env.get_robot_state()[0]
        pose = np.asarray(st["ActualTCPPose"], dtype=np.float64).reshape(6)
        grip = float(st["gripper_position"])
        state.update(
            pos=[float(pose[0]), float(pose[1]), float(pose[2])],
            rotvec=[float(pose[3]), float(pose[4]), float(pose[5])],
            grip=grip,
            n_episodes=int(env.replay_buffer.n_episodes),
        )
    except Exception:
        pass
