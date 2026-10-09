"""
Hybrid camera facade: 2x UVC (wrist GoPros) + 1x RealSense (middle D435).

Logical indices match Arx5Env / video paths:
  0 = left wrist (UVC[0])
  1 = D435 (RealSense)
  2 = right wrist (UVC[1])
"""

from __future__ import annotations

import pathlib
import time
from typing import Dict, List, Optional, Union

import numpy as np

from peripherals.multi_uvc_camera import MultiUvcCamera
from peripherals.realsense_camera import RealSenseCamera


class HybridMultiCamera:
    """MultiUvcCamera-compatible wrapper with fixed index mapping 0/1/2."""

    def __init__(
        self,
        uvc_camera: MultiUvcCamera,
        realsense_camera: RealSenseCamera,
        *,
        rs_index: int = 1,
    ):
        if uvc_camera.n_cameras != 2:
            raise ValueError(
                f"HybridMultiCamera expects exactly 2 UVC cameras, got {uvc_camera.n_cameras}"
            )
        if rs_index != 1:
            raise ValueError("Only rs_index=1 (middle) is supported")
        self.uvc_camera = uvc_camera
        self.realsense_camera = realsense_camera
        self.rs_index = int(rs_index)
        # logical index -> ("uvc", uvc_i) or ("rs", None)
        self._sources = {
            0: ("uvc", 0),
            1: ("rs", None),
            2: ("uvc", 1),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self) -> int:
        return 3

    @property
    def is_ready(self) -> bool:
        return self.uvc_camera.is_ready and self.realsense_camera.is_ready

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        self.uvc_camera.start(wait=False, put_start_time=put_start_time)
        self.realsense_camera.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.uvc_camera.stop(wait=False)
        self.realsense_camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.uvc_camera.start_wait()
        self.realsense_camera.start_wait()

    def stop_wait(self):
        self.uvc_camera.stop_wait()
        self.realsense_camera.end_wait()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        if out is None:
            out = {}

        uvc_out: Dict[int, Dict] = {}
        for logical_idx, (kind, uvc_i) in self._sources.items():
            if kind != "uvc":
                continue
            if logical_idx in out:
                uvc_out[uvc_i] = out[logical_idx]

        uvc_data = self.uvc_camera.get(k=k, out=uvc_out if uvc_out else None)

        rs_out = out.get(self.rs_index) if self.rs_index in out else None
        rs_data = self.realsense_camera.get(k=k, out=rs_out)

        result: Dict[int, Dict[str, np.ndarray]] = {}
        result[0] = uvc_data[0]
        result[1] = rs_data
        result[2] = uvc_data[1]
        return result

    def get_vis(self, out=None):
        # Stack color frames in logical order for MultiCameraVisualizer.
        uvc_vis = self.uvc_camera.get_vis()
        rs_vis = self.realsense_camera.get_vis()
        colors = [
            uvc_vis["color"][0],
            rs_vis["color"],
            uvc_vis["color"][1],
        ]
        stacked = np.stack(colors, axis=0)
        if out is not None and "color" in out:
            out["color"][:] = stacked
            return out
        return {"color": stacked}

    def start_recording(
        self, video_path: Union[str, List[str]], start_time: float
    ):
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = [
                str(video_dir.joinpath(f"{i}.mp4").absolute())
                for i in range(self.n_cameras)
            ]
        assert len(video_path) == self.n_cameras

        self.uvc_camera.start_recording(
            video_path=[video_path[0], video_path[2]],
            start_time=start_time,
        )
        self.realsense_camera.start_recording(video_path[1], start_time=start_time)

    def stop_recording(self):
        self.uvc_camera.stop_recording()
        self.realsense_camera.stop_recording()

    def restart_put(self, start_time):
        self.uvc_camera.restart_put(start_time)
        self.realsense_camera.restart_put(start_time)
