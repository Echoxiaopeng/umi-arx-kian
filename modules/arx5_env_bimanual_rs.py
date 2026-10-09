"""
Bimanual Arx5Env (ZMQ) with 2 wrist GoPros (V4L) + middle D435 (pyrealsense2).

Logical cameras:
  camera0 = left wrist GoPro
  camera1 = D435 RGB (+ optional depth -> camera1_depth)
  camera2 = right wrist GoPro
"""

from __future__ import annotations

import math
import pathlib
import time
from typing import List, Optional

import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager

from modules.arx5_controller import Arx5Controller
from modules.arx5_env import Arx5Env
from modules.replay_buffer import ReplayBuffer
from peripherals.hybrid_multi_camera import HybridMultiCamera
from peripherals.multi_camera_visualizer import MultiCameraVisualizer
from peripherals.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from peripherals.realsense_camera import RealSenseCamera
from utils.cv2_util import get_image_transform, optimal_row_cols
from utils.cv_util import draw_predefined_mask
from utils.interpolation_util import PoseInterpolator, get_interp1d
from utils.usb_util import get_sorted_v4l_paths, reset_all_elgato_devices


_RS_V4L_MARKERS = ("realsense", "intel_r", "intel_corp", "depth_camera")


def filter_wrist_v4l_paths(v4l_paths: List[str]) -> List[str]:
    """Drop Intel RealSense nodes so they are not opened twice via V4L."""
    kept = []
    for path in v4l_paths:
        name = path.lower()
        if any(m in name for m in _RS_V4L_MARKERS):
            print(f"[Arx5EnvBimanualRS] skip RealSense V4L device: {path}")
            continue
        kept.append(path)
    return kept


def _geometry_for_transform(input_res, output_res):
    iw, ih = input_res
    ow, oh = output_res
    interp_method = cv2.INTER_AREA
    if (iw / ih) >= (ow / oh):
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    w_slice_start = (rw - ow) // 2
    h_slice_start = (rh - oh) // 2
    return rw, rh, w_slice_start, h_slice_start, ow, oh, interp_method


def make_uvc_transform(
    input_res,
    obs_image_resolution,
    *,
    obs_float32: bool,
    no_mirror: bool,
    mirror_swap: bool,
    fisheye_converter=None,
):
    is_mirror = None
    if mirror_swap:
        mirror_mask = np.ones((224, 224, 3), dtype=np.uint8)
        mirror_mask = draw_predefined_mask(
            mirror_mask,
            color=(0, 0, 0),
            mirror=False,
            gripper=False,
            finger=False,
        )
        is_mirror = mirror_mask[..., 0] == 0

    def tf(data, input_res=input_res):
        img = data["color"]
        if fisheye_converter is None:
            f = get_image_transform(
                input_res=input_res,
                output_res=obs_image_resolution,
                bgr_to_rgb=True,
            )
            img = np.ascontiguousarray(f(img))
            if is_mirror is not None:
                img[is_mirror] = img[:, ::-1, :][is_mirror]
            img = draw_predefined_mask(
                img,
                color=(0, 0, 0),
                mirror=no_mirror,
                gripper=False,
                finger=False,
                use_aa=False,
            )
        else:
            img = fisheye_converter.forward(img)
            img = img[..., ::-1]
        if obs_float32:
            img = img.astype(np.float32) / 255
        data["color"] = img
        return data

    return tf


def make_rs_transform(
    input_res,
    obs_image_resolution,
    *,
    obs_float32: bool,
    enable_depth: bool,
):
    color_tf = get_image_transform(
        input_res=input_res,
        output_res=obs_image_resolution,
        bgr_to_rgb=True,
    )
    rw, rh, w0, h0, ow, oh, _ = _geometry_for_transform(
        input_res, obs_image_resolution
    )
    w_slice = slice(w0, w0 + ow)
    h_slice = slice(h0, h0 + oh)

    def tf(data, input_res=input_res):
        img = data["color"]
        # Allow empty example arrays during ring-buffer construction.
        if img.shape[:2] == (input_res[1], input_res[0]):
            out = color_tf(img)
        else:
            out = img
        if obs_float32 and out.dtype != np.float32:
            out = out.astype(np.float32) / 255
        data["color"] = out
        if enable_depth and "depth" in data:
            depth = data["depth"]
            if depth.ndim == 2 and depth.shape == (input_res[1], input_res[0]):
                depth = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
                depth = depth[h_slice, w_slice]
            data["depth"] = depth
        return data

    return tf


class Arx5EnvBimanualRS(Arx5Env):
    """ZMQ dual-arm env: GoPro×2 (V4L) + D435 (pyrealsense2)."""

    def __init__(
        self,
        output_dir,
        robots_config,
        frequency=20,
        obs_image_resolution=(224, 224),
        max_obs_buffer_size=60,
        obs_float32=False,
        camera_reorder=None,
        no_mirror=False,
        fisheye_converter=None,
        mirror_swap=False,
        camera_obs_latency=0.125,
        camera_down_sample_steps=1,
        robot_down_sample_steps=1,
        camera_obs_horizon=2,
        robot_obs_horizon=2,
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        init_joints=False,
        enable_multi_cam_vis=True,
        multi_cam_vis_resolution=(960, 960),
        shm_manager=None,
        # RealSense
        rs_serial: Optional[str] = None,
        rs_resolution=(640, 480),
        rs_fps: int = 30,
        enable_depth: bool = True,
    ):
        # Intentionally do not call Arx5Env.__init__ (different camera stack).
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Check D435 before USB reset / spawning cameras or robots.
        from peripherals.realsense_camera import require_realsense_device

        chosen_rs = require_realsense_device(rs_serial)
        print(
            f"[Arx5EnvBimanualRS] RealSense OK: {chosen_rs['name']} "
            f"serial={chosen_rs['serial']}"
        )

        reset_all_elgato_devices()
        print("[Arx5EnvBimanualRS] waiting for Elgato after USB reset...")
        time.sleep(2.5)

        all_v4l = get_sorted_v4l_paths()
        wrist_v4l = filter_wrist_v4l_paths(all_v4l)
        print(f"[Arx5EnvBimanualRS] wrist V4L candidates ({len(wrist_v4l)}):")
        for i, p in enumerate(wrist_v4l):
            print(f"  [{i}] {p}")

        if camera_reorder is None:
            camera_reorder = [0, 1]
        if len(camera_reorder) != 2:
            raise ValueError(
                f"camera_reorder 应为两个腕部相机索引（如 0,1），收到 {camera_reorder}"
            )
        try:
            wrist_paths = [wrist_v4l[i] for i in camera_reorder]
        except IndexError as e:
            raise RuntimeError(
                f"腕部相机索引越界: camera_reorder={camera_reorder}, "
                f"可用腕部 V4L 数={len(wrist_v4l)}（已过滤 RealSense）"
            ) from e

        rw, rh, col, row = optimal_row_cols(
            n_cameras=3,
            in_wh_ratio=4 / 3,
            max_resolution=multi_cam_vis_resolution,
        )

        resolution = []
        capture_fps = []
        cap_buffer_size = []
        video_recorder = []
        transform = []
        for path in wrist_paths:
            print("wrist path:", path)
            if "Cam_Link_4K" in path:
                res = (3840, 2160)
                fps = 30
                buf = 3
            else:
                res = (1920, 1080)
                fps = 60
                buf = 1
            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(
                VideoRecorder.create_h264(fps=fps, input_pix_fmt="bgr24")
            )
            transform.append(
                make_uvc_transform(
                    res,
                    obs_image_resolution,
                    obs_float32=obs_float32,
                    no_mirror=no_mirror,
                    mirror_swap=mirror_swap,
                    fisheye_converter=fisheye_converter,
                )
            )

        uvc_camera = MultiUvcCamera(
            dev_video_paths=wrist_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            video_recorder=video_recorder,
            verbose=False,
        )

        rs_res = tuple(rs_resolution)
        rs_transform = make_rs_transform(
            rs_res,
            obs_image_resolution,
            obs_float32=obs_float32,
            enable_depth=enable_depth,
        )
        rs_recorder = VideoRecorder.create_h264(
            fps=int(rs_fps), input_pix_fmt="bgr24"
        )
        realsense_camera = RealSenseCamera(
            shm_manager=shm_manager,
            serial_number=rs_serial,
            resolution=rs_res,
            capture_fps=int(rs_fps),
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            enable_depth=enable_depth,
            transform=rs_transform,
            video_recorder=rs_recorder,
            verbose=False,
        )

        camera = HybridMultiCamera(
            uvc_camera=uvc_camera,
            realsense_camera=realsense_camera,
            rs_index=1,
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera, row=row, col=col, rgb_to_bgr=False
            )

        robots: List[Arx5Controller] = []
        for rc in robots_config:
            robots.append(
                Arx5Controller(
                    shm_manager=shm_manager,
                    robot_ip=rc["robot_ip"],
                    robot_port=rc["robot_port"],
                    frequency=200,
                    receive_latency=float(rc.get("robot_obs_latency", 0.0)),
                    verbose=True,
                )
            )

        self.camera = camera
        self.robots = robots
        self.robots_config = robots_config
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.camera_obs_latency = camera_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        self.last_camera_data = None
        self.obs_accumulator = None
        self.action_accumulator = None
        self.start_time = None
        self.last_time_step = 0
        self.enable_depth = bool(enable_depth)
        self.rs_index = 1
        self.last_align_camera_idx: Optional[int] = None
        self.last_align_timestamp: Optional[float] = None

    def get_obs(self) -> dict:
        """Align over all opened cameras; emit camera1_depth when present."""
        assert self.is_ready
        k = (
            math.ceil(
                self.camera_obs_horizon
                * self.camera_down_sample_steps
                * (60 / self.frequency)
            )
            + 2
        )
        self.last_camera_data = self.camera.get(k=k, out=self.last_camera_data)

        last_robots_data = [robot.get_all_state() for robot in self.robots]

        num_obs_cameras = len(self.last_camera_data)
        align_camera_idx = None
        running_best_error = np.inf

        for camera_idx in range(num_obs_cameras):
            this_error = 0
            this_timestamp = self.last_camera_data[camera_idx]["timestamp"][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestamps = self.last_camera_data[other_camera_idx]["timestamp"]
                other_timestep_idx = -1
                while True:
                    if other_timestep_idx < -len(other_timestamps):
                        this_error += other_timestamps[0] - this_timestamp
                        break
                    if other_timestamps[other_timestep_idx] < this_timestamp:
                        this_error += (
                            this_timestamp - other_timestamps[other_timestep_idx]
                        )
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx

        last_timestamp = self.last_camera_data[align_camera_idx]["timestamp"][-1]
        self.last_align_camera_idx = int(align_camera_idx)
        self.last_align_timestamp = float(last_timestamp)
        dt = 1 / self.frequency
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1]
            * self.camera_down_sample_steps
            * dt
        )

        camera_obs = {}
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = [
                int(np.argmin(np.abs(this_timestamps - t)))
                for t in camera_obs_timestamps
            ]
            camera_obs[f"camera{camera_idx}_rgb"] = value["color"][this_idxs]
            if (
                self.enable_depth
                and camera_idx == self.rs_index
                and "depth" in value
            ):
                camera_obs[f"camera{camera_idx}_depth"] = value["depth"][this_idxs]

        obs_data = dict(camera_obs)
        obs_data["timestamp"] = camera_obs_timestamps

        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt
        )
        for robot_idx, last_robot_data in enumerate(last_robots_data):
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data["robot_timestamp"], x=last_robot_data["ActualTCPPose"]
            )
            gripper_pos_interpolator = get_interp1d(
                t=last_robot_data["robot_timestamp"],
                x=last_robot_data["gripper_position"][..., None],
            )
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            gripper_pos = gripper_pos_interpolator(robot_obs_timestamps)
            obs_data.update(
                {
                    f"robot{robot_idx}_eef_pos": robot_pose[..., :3],
                    f"robot{robot_idx}_eef_rot_axis_angle": robot_pose[..., 3:],
                    f"robot{robot_idx}_gripper_width": gripper_pos,
                }
            )

        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(
                    data={
                        f"robot{robot_idx}_eef_pose": last_robot_data["ActualTCPPose"],
                        f"robot{robot_idx}_joint_pos": last_robot_data["ActualQ"],
                        f"robot{robot_idx}_joint_vel": last_robot_data["ActualQd"],
                        f"robot{robot_idx}_gripper_width": last_robot_data[
                            "gripper_position"
                        ],
                    },
                    timestamps=last_robot_data["robot_timestamp"],
                )

        return obs_data
