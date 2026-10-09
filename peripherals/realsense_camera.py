"""
Intel RealSense camera process (pyrealsense2), API-aligned with UvcCamera.

Returns BGR color (+ optional uint16 depth in mm, aligned to color).
RGB video recording reuses VideoRecorder; depth is not written to mp4.
"""

from __future__ import annotations

import copy
import enum
import time
from typing import Callable, Dict, Optional

import cv2
import multiprocessing as mp
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from threadpoolctl import threadpool_limits

from modules.timestamp_accumulator import get_accumulate_timestamp_idxs
from peripherals.video_recorder import VideoRecorder
from shared_memory.shared_memory_queue import Empty, Full, SharedMemoryQueue
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


def _import_rs():
    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError as e:
        raise ImportError(
            "需要安装 pyrealsense2 才能使用 D435。"
            " 例: pip install pyrealsense2"
        ) from e
    return rs


def list_realsense_devices() -> list[dict]:
    """Return [{name, serial}, ...] for devices visible to pyrealsense2."""
    rs = _import_rs()
    ctx = rs.context()
    out = []
    for dev in ctx.query_devices():
        out.append(
            {
                "name": str(dev.get_info(rs.camera_info.name)),
                "serial": str(dev.get_info(rs.camera_info.serial_number)),
            }
        )
    return out


def require_realsense_device(serial_number: Optional[str] = None) -> dict:
    """
    Fail fast with a clear message if no D435 (or requested serial) is present.
    Call from the main process before spawning RealSenseCamera.
    """
    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError(
            "未检测到 Intel RealSense（pyrealsense2 devices=0）。\n"
            "请检查：\n"
            "  1) D435 USB 已插紧（尽量 USB3 口，可用另一根线试）\n"
            "  2) lsusb | grep -i RealSense 能看到设备\n"
            "  3) rs-enumerate-devices 能列出相机\n"
            "  4) 没有其它进程占用相机（realsense-viewer / 上一次采数未退出）\n"
            "腕部 Elgato 已就绪不代表 D435 已连接。"
        )
    if serial_number:
        for d in devices:
            if d["serial"] == str(serial_number):
                return d
        available = ", ".join(f"{d['name']}({d['serial']})" for d in devices)
        raise RuntimeError(
            f"找不到序列号为 {serial_number!r} 的 RealSense。"
            f" 当前可见: {available}"
        )
    return devices[0]


class RealSenseCamera(mp.Process):
    """Single RealSense device as a shared-memory camera process."""

    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number: Optional[str] = None,
        resolution=(640, 480),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        get_max_k=30,
        receive_latency=0.0,
        enable_depth=True,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        resolution = tuple(resolution)
        shape = resolution[::-1]  # H, W
        examples: Dict = {
            "color": np.empty(shape=shape + (3,), dtype=np.uint8),
            "camera_capture_timestamp": 0.0,
            "camera_receive_timestamp": 0.0,
            "timestamp": 0.0,
            "step_idx": 0,
        }
        if enable_depth:
            examples["depth"] = np.empty(shape=shape, dtype=np.uint16)

        if transform is not None:
            ring_examples = transform(dict(examples))
        else:
            ring_examples = examples

        vis_examples = copy.deepcopy(examples)
        if vis_transform is not None:
            vis_examples["color"] = np.empty(shape=(720, 960, 3), dtype=np.uint8)
        else:
            vis_examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=vis_examples,
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=ring_examples,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                "cmd": Command.RESTART_PUT.value,
                "put_start_time": 0.0,
                "video_path": np.array("a" * self.MAX_PATH_LENGTH),
                "recording_start_time": 0.0,
            },
            buffer_size=128,
        )

        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=capture_fps,
                input_pix_fmt="bgr24",
            )
        assert video_recorder.fps == capture_fps

        self.shm_manager = shm_manager
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = int(capture_fps)
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.enable_depth = bool(enable_depth)
        self.transform = transform
        self.vis_transform = vis_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape + (3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=self.shm_manager, data_example=data_example
        )
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
        self.video_recorder.start_wait()

    def end_wait(self):
        self.join()
        self.video_recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def start_recording(self, video_path: str, start_time: float = -1):
        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    def run(self):
        threadpool_limits(1)
        cv2.setNumThreads(1)
        rs = _import_rs()

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(str(self.serial_number))

        w, h = self.resolution
        fps = self.capture_fps
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

        align = rs.align(rs.stream.color) if self.enable_depth else None

        print(
            f"[RealSenseCamera] starting serial={self.serial_number!r} "
            f"{w}x{h}@{fps} depth={self.enable_depth}"
        )
        pipeline_started = False
        try:
            try:
                profile = pipeline.start(config)
            except RuntimeError as e:
                devs = list_realsense_devices()
                detail = (
                    "none"
                    if not devs
                    else ", ".join(f"{d['name']}({d['serial']})" for d in devs)
                )
                print(
                    f"[RealSenseCamera] pipeline.start failed: {e}\n"
                    f"  visible devices: {detail}\n"
                    "  请插上 D435 后重试（lsusb / rs-enumerate-devices）。"
                )
                raise
            pipeline_started = True
            device = profile.get_device()
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"[RealSenseCamera] opened device serial={serial}")

            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    print("[RealSenseCamera] wait_for_frames timeout; retrying...")
                    continue

                if align is not None:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                t_recv = time.time()
                frame = np.asanyarray(color_frame.get_data())
                depth_img = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_img = np.asanyarray(depth_frame.get_data())

                got_recorder_buf = False
                try:
                    rec_buf = self.video_recorder.get_img_buffer()
                    rec_buf[:] = frame
                    frame = rec_buf
                    got_recorder_buf = True
                except Full:
                    got_recorder_buf = False

                t_cal = t_recv - self.receive_latency
                if got_recorder_buf and self.video_recorder.is_ready():
                    self.video_recorder.write_img_buffer(frame, frame_time=t_cal)

                data: Dict = {
                    "camera_receive_timestamp": t_recv,
                    "camera_capture_timestamp": t_recv,
                    "color": frame,
                }
                if self.enable_depth:
                    if depth_img is None:
                        depth_img = np.zeros(frame.shape[:2], dtype=np.uint16)
                    data["depth"] = depth_img

                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[t_cal],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        put_data["timestamp"] = t_cal
                        self.ring_buffer.put(put_data, wait=True)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = t_cal
                    self.ring_buffer.put(put_data, wait=True)

                if iter_idx == 0:
                    print(f"[RealSenseCamera] ready serial={serial}")
                    self.ready_event.set()

                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / max(duration, 1e-6), 1)
                t_start = t_end
                if self.verbose:
                    print(f"[RealSenseCamera] FPS {frequency}")

                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command["cmd"]
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command["put_start_time"]
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command["video_path"])
                        start_time = command["recording_start_time"]
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(
                            video_path, start_time=start_time
                        )
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop_recording()

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            if pipeline_started:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            # Do NOT set ready_event on failure — that would unblock env with a dead cam.
