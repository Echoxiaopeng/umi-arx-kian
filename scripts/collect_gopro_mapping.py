#!/usr/bin/env python3
"""
采集「周围环境建图」视频（GoPro / Elgato / 任意 V4L 相机）。

适用场景：
  - GoPro 开 Webcam 模式，USB 枚举成 /dev/video*
  - 或 GoPro HDMI 输出接到 Elgato 采集卡（本机常见）

用法:
    # 列出可用相机
    python scripts/collect_gopro_mapping.py --list

    # 自动选相机，预览；按 R 开始/停止录制，Q 退出
    python scripts/collect_gopro_mapping.py -o data/mapping_sessions

    # 指定设备（推荐用 by-id 路径，更稳定）
    python scripts/collect_gopro_mapping.py \
      --device /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB52423M5WU-video-index0 \
      -o data/mapping_sessions

键盘:
  R = 开始 / 停止录制
  S = 停止当前录制（同 R 在录制中）
  Q / Esc = 退出

建图拍摄建议:
  - 慢速绕工作台走一圈，尽量匀速
  - 光照稳定，少运动模糊
  - 覆盖有纹理区域（桌面/箱子/边缘），避免大片白墙
  - 不必在场景里贴标定码（内参标定是另一次单独拍）
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np


def list_v4l_devices() -> list[dict]:
    """Enumerate /dev/video* and /dev/v4l/by-id links."""
    devices: list[dict] = []
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        for link in sorted(by_id.glob("*video-index0")):
            devices.append(
                {
                    "path": str(link),
                    "resolved": str(link.resolve()),
                    "name": link.name,
                }
            )
    # Also include raw /dev/videoN if not already covered
    seen = {d["resolved"] for d in devices}
    for path in sorted(Path("/dev").glob("video*")):
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        # Odd-numbered metadata nodes often fail to open for capture; still list them.
        devices.append({"path": str(path), "resolved": resolved, "name": path.name})
    return devices


def open_capture(device: str, width: int, height: int, fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback without forcing V4L2 backend
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: {device}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    # Prefer MJPG when available (USB bandwidth)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def probe_frame_size(
    cap: cv2.VideoCapture,
) -> tuple[int, int, float, np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("相机已打开，但读不到画面。请检查 GoPro 是否在 Webcam/输出模式。")
    h, w = frame.shape[:2]
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return w, h, fps, frame


def make_writer(
    path: Path, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v is widely available; change to avc1 if your OpenCV has it.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {path}")
    return writer


def overlay_hud(
    frame: np.ndarray,
    *,
    recording: bool,
    elapsed: float,
    n_frames: int,
    out_name: str,
    device: str,
) -> np.ndarray:
    vis = frame.copy()
    status = "REC" if recording else "IDLE"
    color = (0, 0, 255) if recording else (0, 200, 0)
    lines = [
        f"{status}  t={elapsed:6.1f}s  frames={n_frames}",
        f"out: {out_name}",
        f"dev: {Path(device).name}",
        "R: record/stop   Q: quit",
    ]
    y = 28
    for text in lines:
        cv2.putText(
            vis,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if y == 28 else 0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if y == 28 else 0.55,
            color if y == 28 else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28
    if recording:
        cv2.circle(vis, (vis.shape[1] - 40, 40), 14, (0, 0, 255), -1)
    return vis


@click.command()
@click.option(
    "--output",
    "-o",
    default="data/mapping_sessions",
    show_default=True,
    help="输出根目录；每次录制写入 mapping_YYYYmmdd_HHMMSS/",
)
@click.option(
    "--device",
    "-d",
    default=None,
    help="V4L 设备路径。默认自动选 /dev/v4l/by-id/*video-index0 或第一个可打开设备",
)
@click.option("--width", default=1920, type=int, show_default=True, help="请求宽度；0=不设置")
@click.option("--height", default=1080, type=int, show_default=True, help="请求高度；0=不设置")
@click.option("--fps", default=30.0, type=float, show_default=True, help="请求帧率；0=不设置")
@click.option("--list", "list_only", is_flag=True, help="只列出相机后退出")
@click.option(
    "--no-preview",
    is_flag=True,
    help="无界面模式：启动后立刻录制，Ctrl+C 结束（适合 SSH）",
)
@click.option(
    "--duration",
    type=float,
    default=None,
    help="无预览时可指定录制秒数；到时自动停止",
)
def main(
    output: str,
    device: str | None,
    width: int,
    height: int,
    fps: float,
    list_only: bool,
    no_preview: bool,
    duration: float | None,
):
    devices = list_v4l_devices()
    if list_only:
        if not devices:
            click.echo("未发现 /dev/video* 或 /dev/v4l/by-id 设备。")
            return
        click.echo("可用相机:")
        for i, d in enumerate(devices):
            click.echo(f"  [{i}] {d['path']}")
            if d["path"] != d["resolved"]:
                click.echo(f"       -> {d['resolved']}")
        return

    pending_frame: np.ndarray | None = None

    if device is None:
        # Prefer by-id index0 nodes
        preferred = [d["path"] for d in devices if "video-index0" in d["path"]]
        candidates = preferred + [d["path"] for d in devices if d["path"] not in preferred]
        if not candidates:
            raise click.ClickException(
                "没有找到相机设备。请确认 GoPro Webcam / Elgato 已连接，"
                "或用 --list 查看。"
            )
        last_err = None
        cap = None
        chosen = None
        for cand in candidates:
            try:
                cap = open_capture(cand, width, height, fps)
                _, _, _, pending_frame = probe_frame_size(cap)
                chosen = cand
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if cap is not None:
                    cap.release()
                cap = None
                pending_frame = None
        if cap is None or chosen is None:
            raise click.ClickException(f"自动打开相机失败: {last_err}")
        device = chosen
    else:
        cap = open_capture(device, width, height, fps)
        _, _, _, pending_frame = probe_frame_size(cap)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height or 0)
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    if actual_w <= 0 or actual_h <= 0:
        if pending_frame is None:
            actual_w, actual_h, _, pending_frame = probe_frame_size(cap)
        else:
            actual_h, actual_w = pending_frame.shape[:2]
    if actual_fps <= 1.0:
        actual_fps = fps if fps > 1.0 else 30.0

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 60)
    click.echo("GoPro / V4L mapping capture")
    click.echo(f"  device : {device}")
    click.echo(f"  size   : {actual_w}x{actual_h} @ {actual_fps:.1f} fps (requested {width}x{height}@{fps})")
    click.echo(f"  output : {out_root}")
    if no_preview:
        click.echo("  mode   : no-preview (Ctrl+C to stop)")
    else:
        click.echo("  keys   : R=record/stop  Q=quit")
    click.echo("=" * 60)

    writer: cv2.VideoWriter | None = None
    session_dir: Path | None = None
    video_path: Path | None = None
    recording = False
    rec_t0 = 0.0
    n_frames = 0

    def start_recording() -> None:
        nonlocal writer, session_dir, video_path, recording, rec_t0, n_frames
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = out_root / f"mapping_{stamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        video_path = session_dir / "raw_video.mp4"
        writer = make_writer(video_path, actual_w, actual_h, actual_fps)
        meta = {
            "created_at": stamp,
            "device": device,
            "width": actual_w,
            "height": actual_h,
            "fps_requested": fps,
            "fps_reported": actual_fps,
            "purpose": "environment_mapping_for_slam",
            "notes": [
                "Walk slowly around the workspace with steady lighting.",
                "Prefer textured surfaces; avoid motion blur.",
                "Camera intrinsics calibration is a separate step.",
            ],
        }
        (session_dir / "session.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        recording = True
        rec_t0 = time.time()
        n_frames = 0
        click.echo(f"[REC] start -> {video_path}")

    def stop_recording() -> None:
        nonlocal writer, recording
        if writer is not None:
            writer.release()
            writer = None
        if recording and session_dir is not None and video_path is not None:
            elapsed = time.time() - rec_t0
            summary = {
                "frames": n_frames,
                "elapsed_sec": round(elapsed, 3),
                "approx_fps": round(n_frames / elapsed, 3) if elapsed > 1e-3 else 0.0,
                "video": str(video_path.name),
            }
            (session_dir / "capture_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            click.echo(
                f"[REC] stop  frames={n_frames}  t={elapsed:.1f}s  "
                f"saved={video_path}"
            )
        recording = False

    try:
        if no_preview:
            start_recording()
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
                    frame = cv2.resize(frame, (actual_w, actual_h))
                writer.write(frame)
                n_frames += 1
                if duration is not None and (time.time() - rec_t0) >= duration:
                    break
                if n_frames % int(max(actual_fps, 1)) == 0:
                    click.echo(f"  ... {n_frames} frames, {time.time() - rec_t0:.1f}s")
        else:
            window = "gopro_mapping"
            while True:
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                    ok = True
                else:
                    ok, frame = cap.read()
                if not ok or frame is None:
                    key = cv2.waitKey(10) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    continue

                if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
                    frame = cv2.resize(frame, (actual_w, actual_h))

                if recording and writer is not None:
                    writer.write(frame)
                    n_frames += 1

                elapsed = (time.time() - rec_t0) if recording else 0.0
                vis = overlay_hud(
                    frame,
                    recording=recording,
                    elapsed=elapsed,
                    n_frames=n_frames,
                    out_name=video_path.name if video_path else "-",
                    device=device,
                )
                # Fit preview to screen-ish width
                max_w = 1280
                if vis.shape[1] > max_w:
                    scale = max_w / vis.shape[1]
                    vis = cv2.resize(vis, None, fx=scale, fy=scale)
                cv2.imshow(window, vis)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    if recording:
                        stop_recording()
                    else:
                        start_recording()
                if key in (ord("s"), ord("S")) and recording:
                    stop_recording()
                if duration is not None and recording and (time.time() - rec_t0) >= duration:
                    stop_recording()
                    break
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
    finally:
        if recording:
            stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        click.echo("Done.")


if __name__ == "__main__":
    main()
