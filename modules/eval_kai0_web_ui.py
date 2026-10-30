"""Kai0 eval live dashboard (stdlib HTTP, no FastAPI).

Main loop publishes JPEG + status/logs; browser is read-only (optional cmds).
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
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import cv2
import numpy as np

_DEFAULT_STATIC = Path(__file__).resolve().parent.parent / "web" / "kai0_eval"


class CommandQueue:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def put(self, action: str, **payload: Any) -> None:
        self._q.put({"action": str(action), **payload})

    def drain(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


class SharedEvalState:
    """Thread-safe snapshot for the eval dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mode: str = "IDLE"
        self.message: str = ""
        self.progress_label: str = ""
        self.progress_pct: int = -1
        self.progress_detail: str = ""
        self.step: int = 0
        self.infer_count: int = 0
        self.buf_len: int = 0
        self.delay_steps: int = 0
        self.exec_h: int = 0
        self.frequency: float = 20.0
        self.inference_rate: float = 4.0
        self.dispatch: str = "schedule"
        self.rtt_s: float = 0.0
        self.joints14: list[float] = [0.0] * 14
        self.prompt: str = ""
        self.controls: str = "C start · S stop · Enter init · Q home"
        self._jpeg: bytes = b""
        self.logs: list[str] = []

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if key == "jpeg":
                    if isinstance(value, (bytes, bytearray)) and len(value) > 0:
                        self._jpeg = bytes(value)
                elif key == "logs":
                    self.logs = list(value)
                elif key == "joints14":
                    self.joints14 = [float(x) for x in np.asarray(value).reshape(-1)]
                elif hasattr(self, key):
                    setattr(self, key, value)

    def append_log(self, line: str, *, max_lines: int = 300) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {line}"
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > max_lines:
                self.logs = self.logs[-max_lines:]
            self.message = str(line)

    def set_progress(
        self, label: str, pct: int, detail: str = ""
    ) -> None:
        with self._lock:
            self.progress_label = str(label)
            self.progress_pct = int(pct)
            self.progress_detail = str(detail)
            if pct >= 0:
                self.message = f"{label} {pct}%"

    def clear_progress(self) -> None:
        with self._lock:
            self.progress_label = ""
            self.progress_pct = -1
            self.progress_detail = ""

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "message": self.message,
                "progress_label": self.progress_label,
                "progress_pct": self.progress_pct,
                "progress_detail": self.progress_detail,
                "step": self.step,
                "infer_count": self.infer_count,
                "buf_len": self.buf_len,
                "delay_steps": self.delay_steps,
                "exec_h": self.exec_h,
                "frequency": self.frequency,
                "inference_rate": self.inference_rate,
                "dispatch": self.dispatch,
                "rtt_s": self.rtt_s,
                "joints14": list(self.joints14),
                "prompt": self.prompt,
                "controls": self.controls,
                "logs": list(self.logs[-80:]),
                "t": time.time(),
            }


def obs_to_jpeg(obs: dict, *, quality: int = 70, max_width: int = 960) -> bytes:
    """Concatenate camera0/1/2 RGB into one JPEG strip."""
    views = []
    for index in (0, 1, 2):
        key = f"camera{index}_rgb"
        if key not in obs:
            continue
        frame = np.asarray(obs[key][-1], dtype=np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[-1] == 3:
            frame = frame[..., ::-1].copy()  # RGB → BGR
        views.append(frame)
    if not views:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(
            ".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        return bytes(buf) if ok else b""
    canvas = np.concatenate(views, axis=1)
    h, w = canvas.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        canvas = cv2.resize(
            canvas,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    return bytes(buf) if ok else b""


class EvalWebServer:
    def __init__(
        self,
        state: SharedEvalState,
        cmd_queue: CommandQueue | None = None,
        *,
        host: str = "0.0.0.0",
        port: int = 8090,
        static_dir: Path | None = None,
    ) -> None:
        self.state = state
        self.cmd_queue = cmd_queue or CommandQueue()
        self.host = host
        self.port = int(port)
        self.static_dir = Path(static_dir) if static_dir else _DEFAULT_STATIC
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        state = self.state
        cmd_queue = self.cmd_queue
        static_dir = self.static_dir.resolve()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:  # noqa: A003
                return

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
                if extra_headers:
                    for k, v in extra_headers.items():
                        self.send_header(k, str(v))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, code: int, payload: dict) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send(code, raw, "application/json; charset=utf-8")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                try:
                    path = urlparse(self.path).path
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
                        if not str(fpath).startswith(str(static_dir)):
                            self._send_json(403, {"error": "forbidden"})
                            return
                        if not fpath.is_file():
                            self._send_json(404, {"error": "not found"})
                            return
                        ctype = (
                            mimetypes.guess_type(str(fpath))[0]
                            or "application/octet-stream"
                        )
                        self._send(200, fpath.read_bytes(), ctype)
                        return
                    if path == "/api/status":
                        self._send_json(200, state.snapshot())
                        return
                    if path == "/api/video":
                        self._stream_mjpeg()
                        return
                    if path == "/api/live/frame":
                        jpeg = state.get_jpeg()
                        if not jpeg:
                            self._send_json(404, {"error": "no frame"})
                            return
                        self._send(200, jpeg, "image/jpeg")
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
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode(
                                    "ascii"
                                )
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self) -> None:  # noqa: N802
                try:
                    path = urlparse(self.path).path
                    if path != "/api/cmd":
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
                    extra = {k: v for k, v in payload.items() if k != "action"}
                    cmd_queue.put(str(action), **extra)
                    self._send_json(200, {"ok": True, "action": action})
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    self._send_json(500, {"error": str(exc)})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="kai0-eval-web", daemon=True
        )
        self._thread.start()
        print(
            f"[eval_web] http://{self.host}:{self.port}/  "
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


def start_eval_web_ui(
    state: SharedEvalState,
    cmd_queue: CommandQueue | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 8090,
) -> EvalWebServer:
    server = EvalWebServer(state, cmd_queue, host=host, port=port)
    server.start()
    return server


def make_progress_cb(
    state: SharedEvalState | None,
    *,
    also_print: bool = False,
) -> Callable[[str, int, str], None]:
    def _cb(label: str, pct: int, detail: str = "") -> None:
        if also_print:
            print(f"[{label}] {pct:3d}%  {detail}", flush=True)
        if state is not None:
            state.set_progress(label, pct, detail)
            if pct >= 100 or pct % 5 == 0:
                state.append_log(f"{label} {pct}% {detail}".strip())

    return _cb
