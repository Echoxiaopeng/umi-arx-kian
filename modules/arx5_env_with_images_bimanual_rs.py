"""
Bimanual ZMQ Arx5Env that writes RGB (+ D435 depth) into replay_buffer.zarr.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from modules.arx5_env_bimanual_rs import Arx5EnvBimanualRS
from utils.interpolation_util import (
    PoseInterpolator,
    continuize_action_rotvecs,
    continuize_rotvecs,
    get_interp1d,
)


class ImageAccumulator:
    """Accumulator for images / depth with timestamps."""

    def __init__(self):
        self.data: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.timestamps: Dict[str, List[float]] = defaultdict(list)

    def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray):
        for key, value in data.items():
            for i, t in enumerate(timestamps):
                if (
                    key not in self.timestamps
                    or len(self.timestamps[key]) == 0
                    or self.timestamps[key][-1] < t
                ):
                    self.timestamps[key].append(t)
                    self.data[key].append(value[i])


class Arx5EnvWithImages(Arx5EnvBimanualRS):
    """Bimanual+RS env that also writes camera RGB/depth into zarr."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_accumulator: Optional[ImageAccumulator] = None

    def start_episode(self, start_time=None):
        super().start_episode(start_time=start_time)
        self.image_accumulator = ImageAccumulator()

    def get_obs(self) -> dict:
        obs_data = super().get_obs()
        if self.image_accumulator is not None:
            camera_data = {}
            timestamps = obs_data.get("timestamp", None)
            if timestamps is not None:
                for key, value in obs_data.items():
                    if key.startswith("camera") and (
                        key.endswith("_rgb") or key.endswith("_depth")
                    ):
                        camera_data[key] = value
                if camera_data:
                    self.image_accumulator.put(camera_data, timestamps)
        return obs_data

    def end_episode(self):
        assert self.is_ready
        self.camera.stop_recording()

        if self.obs_accumulator is not None:
            assert self.action_accumulator is not None

            end_time = float("inf")
            for _, value in self.obs_accumulator.timestamps.items():
                if len(value) > 0:
                    end_time = min(end_time, value[-1])
            if len(self.action_accumulator.timestamps) == 0:
                self.obs_accumulator = None
                self.action_accumulator = None
                self.image_accumulator = None
                return
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = (
                    np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]
                    + 1
                )

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                n_robots = len(self.robots)
                # 去掉 rotvec 在 ±π 附近的表示翻转（欧氏≈2π、测地≈0）
                episode_actions = continuize_action_rotvecs(
                    actions[:n_steps], n_robots=n_robots
                )
                episode = {
                    "timestamp": timestamps,
                    "action": episode_actions,
                }

                for robot_idx in range(n_robots):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_eef_pose"
                            ]
                        ),
                        x=np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_eef_pose"]
                        ),
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f"robot{robot_idx}_eef_pos"] = robot_pose[:, :3]
                    episode[f"robot{robot_idx}_eef_rot_axis_angle"] = continuize_rotvecs(
                        robot_pose[:, 3:]
                    )

                    joint_pos_interpolator = get_interp1d(
                        np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_joint_pos"
                            ]
                        ),
                        np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_joint_pos"]
                        ),
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_joint_vel"
                            ]
                        ),
                        np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_joint_vel"]
                        ),
                    )
                    episode[f"robot{robot_idx}_joint_pos"] = joint_pos_interpolator(
                        timestamps
                    )
                    episode[f"robot{robot_idx}_joint_vel"] = joint_vel_interpolator(
                        timestamps
                    )

                    gripper_interpolator = get_interp1d(
                        t=np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_gripper_width"
                            ]
                        ),
                        x=np.array(
                            self.obs_accumulator.data[
                                f"robot{robot_idx}_gripper_width"
                            ]
                        ),
                    )
                    episode[f"robot{robot_idx}_gripper_width"] = gripper_interpolator(
                        timestamps
                    )

                if self.image_accumulator is not None:
                    self._add_images_to_episode(episode, timestamps)

                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                print(f"Episode {episode_id} saved!")

            self.obs_accumulator = None
            self.action_accumulator = None
            self.image_accumulator = None

    def _add_images_to_episode(self, episode: dict, timestamps: np.ndarray):
        if self.image_accumulator is None:
            return

        for camera_key in self.image_accumulator.data.keys():
            camera_timestamps = np.array(self.image_accumulator.timestamps[camera_key])
            camera_images = self.image_accumulator.data[camera_key]
            if len(camera_timestamps) == 0:
                continue

            aligned_images = []
            for t in timestamps:
                nn_idx = np.argmin(np.abs(camera_timestamps - t))
                aligned_images.append(camera_images[nn_idx])
            episode[camera_key] = np.stack(aligned_images, axis=0)

        print(
            f"Added {len(self.image_accumulator.data)} camera stream(s) "
            f"(rgb/depth) to episode"
        )
