"""
Arx5Env with image recording support.
This module extends Arx5Env to save images directly into zarr replay buffer.
"""

from typing import Optional, List, Dict
import numpy as np
from collections import defaultdict

from modules.arx5_env_sdk import Arx5Env
from utils.interpolation_util import get_interp1d, PoseInterpolator


class ImageAccumulator:
    """Accumulator for storing images with timestamps."""
    
    def __init__(self):
        self.data: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.timestamps: Dict[str, List[float]] = defaultdict(list)
    
    def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray):
        """
        Store image data with timestamps.
        
        Args:
            data: dict with keys like 'camera0_rgb', values are images (T, H, W, C)
            timestamps: array of timestamps (T,)
        """
        for key, value in data.items():
            for i, t in enumerate(timestamps):
                # Only add if timestamp is newer than the last one
                if (key not in self.timestamps) or (len(self.timestamps[key]) == 0) or (self.timestamps[key][-1] < t):
                    self.timestamps[key].append(t)
                    self.data[key].append(value[i])


class Arx5EnvWithImages(Arx5Env):
    """
    Extended Arx5Env that saves images directly into zarr replay buffer.
    
    Usage:
        Replace Arx5Env with Arx5EnvWithImages in your code.
        Images will be saved as numpy arrays in the zarr file.
    """
    
    def __init__(self, *args, drop_uninitialized_actions: bool = False, **kwargs):
        """
        Args:
            drop_uninitialized_actions: If True, end_episode drops action-grid
                slots that were never written (all-zero). Use for segmented
                recording (e.g. cup collect) where wait gaps leave holes in
                TimestampActionAccumulator.
            *args, **kwargs: Arguments passed to Arx5Env
        """
        super().__init__(*args, **kwargs)
        self.image_accumulator: Optional[ImageAccumulator] = None
        self.drop_uninitialized_actions = bool(drop_uninitialized_actions)
    
    def start_episode(self, start_time=None):
        """Start recording and initialize image accumulator."""
        # Call parent start_episode
        super().start_episode(start_time=start_time)
        
        # Initialize image accumulator
        self.image_accumulator = ImageAccumulator()
    
    def get_obs(self) -> dict:
        """Get observation and accumulate images."""
        obs_data = super().get_obs()
        
        # Accumulate images if recording
        if self.image_accumulator is not None:
            # Extract camera data from obs_data
            camera_data = {}
            timestamps = obs_data.get("timestamp", None)
            
            if timestamps is not None:
                for key, value in obs_data.items():
                    if "camera" in key and "rgb" in key:
                        camera_data[key] = value
                
                if camera_data:
                    self.image_accumulator.put(camera_data, timestamps)
        
        return obs_data
    
    def end_episode(self):
        """Stop recording and save images to zarr."""
        assert self.is_ready
        
        # Stop video recorder (from parent)
        self.camera.stop_recording()
        
        if self.obs_accumulator is not None:
            # Recording
            assert self.action_accumulator is not None
            
            # Find end_time (same logic as parent)
            end_time = float("inf")
            for _, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
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
                episode = {
                    "timestamp": timestamps,
                    "action": actions[:n_steps],
                }
                
                # Process robot data (same as parent)
                for robot_idx in range(len(self.robots)):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(
                            self.obs_accumulator.timestamps[f"robot{robot_idx}_eef_pose"]
                        ),
                        x=np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_eef_pose"]
                        ),
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f"robot{robot_idx}_eef_pos"] = robot_pose[:, :3]
                    episode[f"robot{robot_idx}_eef_rot_axis_angle"] = robot_pose[:, 3:]
                    
                    joint_pos_interpolator = get_interp1d(
                        np.array(
                            self.obs_accumulator.timestamps[f"robot{robot_idx}_joint_pos"]
                        ),
                        np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_joint_pos"]
                        ),
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(
                            self.obs_accumulator.timestamps[f"robot{robot_idx}_joint_vel"]
                        ),
                        np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_joint_vel"]
                        ),
                    )
                    episode[f"robot{robot_idx}_joint_pos"] = joint_pos_interpolator(timestamps)
                    episode[f"robot{robot_idx}_joint_vel"] = joint_vel_interpolator(timestamps)
                    
                    gripper_interpolator = get_interp1d(
                        t=np.array(
                            self.obs_accumulator.timestamps[f"robot{robot_idx}_gripper_width"]
                        ),
                        x=np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_gripper_width"]
                        ),
                    )
                    episode[f"robot{robot_idx}_gripper_width"] = gripper_interpolator(timestamps)
                
                # === Save images to zarr ===
                if self.image_accumulator is not None:
                    self._add_images_to_episode(episode, timestamps)

                if self.drop_uninitialized_actions:
                    episode, n_dropped = self._drop_uninitialized_action_steps(
                        episode
                    )
                    n_kept = int(episode["action"].shape[0])
                    if n_dropped > 0:
                        print(
                            f"Dropped {n_dropped} uninitialized (wait-gap) "
                            f"action steps; kept {n_kept}."
                        )
                    if n_kept == 0:
                        print(
                            "No valid action steps after dropping "
                            "uninitialized slots; episode not saved."
                        )
                        self.obs_accumulator = None
                        self.action_accumulator = None
                        self.image_accumulator = None
                        return

                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                print(f"Episode {episode_id} saved!")
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.image_accumulator = None

    @staticmethod
    def _drop_uninitialized_action_steps(episode: dict):
        """
        TimestampActionAccumulator leaves never-written slots as all-zero.
        Segmented record/wait FSMs only put actions during record_*; wait gaps
        must be removed so zarr indices match recorded frames / labels.
        """
        actions = np.asarray(episode["action"])
        if actions.ndim != 2 or actions.shape[0] == 0:
            return episode, 0
        # Unwritten slots are exactly 0; real TCP poses are not all-zero.
        valid = np.any(actions != 0, axis=1)
        n_dropped = int((~valid).sum())
        if n_dropped == 0:
            return episode, 0
        out = {}
        for key, value in episode.items():
            arr = np.asarray(value)
            if arr.shape[0] == actions.shape[0]:
                out[key] = arr[valid]
            else:
                out[key] = value
        return out, n_dropped
    
    def _add_images_to_episode(self, episode: dict, timestamps: np.ndarray):
        """
        Add interpolated images to episode dict for zarr storage.
        
        Args:
            episode: Episode dict to add images to
            timestamps: Target timestamps for interpolation
        """
        if self.image_accumulator is None:
            return
        
        # Process each camera
        for camera_key in self.image_accumulator.data.keys():
            camera_timestamps = np.array(self.image_accumulator.timestamps[camera_key])
            camera_images = self.image_accumulator.data[camera_key]
            
            if len(camera_timestamps) == 0:
                continue
            
            # Find nearest image for each target timestamp
            aligned_images = []
            for t in timestamps:
                nn_idx = np.argmin(np.abs(camera_timestamps - t))
                aligned_images.append(camera_images[nn_idx])
            
            # Stack into array: (T, H, W, C)
            episode[camera_key] = np.stack(aligned_images, axis=0)
        
        print(f"Added {len(self.image_accumulator.data)} camera(s) images to episode")
