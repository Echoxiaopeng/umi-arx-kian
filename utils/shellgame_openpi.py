"""Data-contract helpers for real-wrist ShellGame deployment."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
from scipy.spatial.transform import Rotation

from utils.pose_util import mat_to_pose
from utils.pose_util import mat_to_pose10d
from utils.pose_util import pose10d_to_mat
from utils.pose_util import pose_to_mat

HISTORY_FRAMES = 241
SWAP_FRAMES = 80
NUM_SWAPS = 3
TOTAL_MODEL_FRAMES = 242
ACTION_HORIZON = 16
ACTION_DIM = 10


def parse_csv_vector(value: str, *, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{name} must contain {size} comma-separated numbers") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly {size} finite numbers, got {value!r}")
    return result


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Return a contiguous 224x224 RGB uint8 frame."""
    frame = np.asarray(image)
    if frame.shape != (224, 224, 3):
        raise ValueError(f"Expected RGB image shape (224, 224, 3), got {frame.shape}")
    if np.issubdtype(frame.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(frame)) <= 1.5 else 1.0
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def relative_pose10(start_pose6: np.ndarray, current_pose6: np.ndarray) -> np.ndarray:
    """Convert a world TCP pose into episode-start-relative xyz+row-rot6d."""
    start = np.asarray(start_pose6, dtype=np.float64).reshape(6)
    current = np.asarray(current_pose6, dtype=np.float64).reshape(6)
    relative = np.linalg.inv(pose_to_mat(start)) @ pose_to_mat(current)
    return np.asarray(mat_to_pose10d(relative), dtype=np.float32)


def relative_chunk_to_world(action_chunk: np.ndarray, anchor_pose6: np.ndarray) -> np.ndarray:
    """Restore current-relative EEF10 targets to world xyz+rotvec+gripper."""
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"Expected action chunk {(ACTION_HORIZON, ACTION_DIM)}, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned NaN or infinite actions")

    anchor = pose_to_mat(np.asarray(anchor_pose6, dtype=np.float64).reshape(6))
    relative_mats = pose10d_to_mat(actions[:, :9])
    world_mats = anchor[None] @ relative_mats
    world_pose6 = np.asarray(mat_to_pose(world_mats), dtype=np.float64)
    return np.concatenate([world_pose6, actions[:, 9:10]], axis=-1)


def validate_world_actions(
    world_actions: np.ndarray,
    *,
    anchor_pose6: np.ndarray,
    dt: float,
    max_pos_speed: float,
    max_rot_speed: float,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> None:
    """Reject unsafe chunks instead of silently changing policy geometry."""
    actions = np.asarray(world_actions, dtype=np.float64)
    if actions.shape != (ACTION_HORIZON, 7) or not np.all(np.isfinite(actions)):
        raise ValueError(f"Invalid world action shape/content: {actions.shape}")

    xyz = actions[:, :3]
    outside = (xyz < workspace_min[None]) | (xyz > workspace_max[None])
    if np.any(outside):
        bad = int(np.flatnonzero(np.any(outside, axis=1))[0])
        raise ValueError(f"Waypoint {bad} is outside workspace: {xyz[bad]}")

    poses = np.concatenate(
        [np.asarray(anchor_pose6, dtype=np.float64).reshape(1, 6), actions[:, :6]],
        axis=0,
    )
    position_steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=-1)
    rotations = Rotation.from_rotvec(poses[:, 3:6])
    rotation_steps = np.asarray(
        [(rotations[i].inv() * rotations[i + 1]).magnitude() for i in range(len(rotations) - 1)]
    )
    max_position_step = max_pos_speed * dt
    max_rotation_step = max_rot_speed * dt
    if np.any(position_steps > max_position_step):
        index = int(np.argmax(position_steps))
        raise ValueError(
            f"Waypoint {index} translation step {position_steps[index]:.4f} m exceeds {max_position_step:.4f} m"
        )
    if np.any(rotation_steps > max_rotation_step):
        index = int(np.argmax(rotation_steps))
        raise ValueError(
            f"Waypoint {index} rotation step {rotation_steps[index]:.4f} rad exceeds {max_rotation_step:.4f} rad"
        )


@dataclass
class ShellGameHistory:
    """Strict F-key capture state: 1 initial frame followed by 3x80 swap frames."""

    initial_frame: np.ndarray | None = None
    episode_start_pose6: np.ndarray | None = None
    swap_segments: list[list[np.ndarray]] = field(default_factory=list)
    active_segment: list[np.ndarray] | None = None

    @property
    def ready(self) -> bool:
        return self.initial_frame is not None and len(self.swap_segments) == NUM_SWAPS

    @property
    def locked(self) -> bool:
        return self.initial_frame is not None

    @property
    def collecting(self) -> bool:
        return self.active_segment is not None

    @property
    def frame_count(self) -> int:
        return (
            (1 if self.initial_frame is not None else 0)
            + sum(len(segment) for segment in self.swap_segments)
            + (len(self.active_segment) if self.active_segment is not None else 0)
        )

    def reset(self) -> None:
        self.initial_frame = None
        self.episode_start_pose6 = None
        self.swap_segments.clear()
        self.active_segment = None

    def press_f(self, frame: np.ndarray, current_pose6: np.ndarray) -> str:
        if self.collecting:
            return "正在采集当前交换, 忽略重复 F"
        if self.initial_frame is None:
            self.initial_frame = image_to_uint8(frame).copy()
            self.episode_start_pose6 = np.asarray(current_pose6, dtype=np.float64).reshape(6).copy()
            return "已保存初始帧; 下一次 F 开始采集第一次交换"
        if self.ready:
            return "241 帧历史已完成; 按 C 开始抓取, 或按 P 开始下一轮"
        self.active_segment = []
        return f"开始采集第 {len(self.swap_segments) + 1} 次交换: 0/{SWAP_FRAMES}"

    def capture_tick(self, frame: np.ndarray) -> str | None:
        if self.active_segment is None:
            return None
        self.active_segment.append(image_to_uint8(frame).copy())
        count = len(self.active_segment)
        if count < SWAP_FRAMES:
            return f"采集第 {len(self.swap_segments) + 1} 次交换: {count}/{SWAP_FRAMES}"
        self.swap_segments.append(self.active_segment)
        self.active_segment = None
        if self.ready:
            return "三次交换采集完成: 241/241, 按 C 开始抓取"
        return f"第 {len(self.swap_segments)} 次交换完成; 按 F 采集下一次交换"

    def frames(self) -> list[np.ndarray]:
        if not self.ready or self.initial_frame is None:
            raise RuntimeError(f"History is incomplete: {self.frame_count}/{HISTORY_FRAMES}")
        frames = [self.initial_frame]
        for segment in self.swap_segments:
            frames.extend(segment)
        if len(frames) != HISTORY_FRAMES:
            raise RuntimeError(f"Expected {HISTORY_FRAMES} history frames, got {len(frames)}")
        return frames

    def status(self) -> str:
        if self.collecting:
            return f"SWAP {len(self.swap_segments) + 1}/{NUM_SWAPS} {len(self.active_segment or [])}/{SWAP_FRAMES}"
        if self.ready:
            return "READY 241/241 - C: policy, P: reset"
        if self.initial_frame is None:
            return "F: capture initial frame"
        return f"WAITING F - next swap {len(self.swap_segments) + 1}/{NUM_SWAPS}"

    def build_policy_observation(
        self,
        *,
        current_frame: np.ndarray,
        current_pose6: np.ndarray,
        gripper_width: float,
        prompt: str,
    ) -> dict[str, np.ndarray | str]:
        if self.episode_start_pose6 is None:
            raise RuntimeError("Episode start pose is missing")
        all_frames = [*self.frames(), image_to_uint8(current_frame)]
        if len(all_frames) != TOTAL_MODEL_FRAMES:
            raise RuntimeError(f"Expected {TOTAL_MODEL_FRAMES} total frames")

        state_pose9 = relative_pose10(self.episode_start_pose6, current_pose6)
        observation: dict[str, np.ndarray | str] = {
            "robot0_eef_pos": state_pose9[:3],
            "robot0_eef_rot_axis_angle": state_pose9[3:9],
            "robot0_gripper_width": np.asarray([gripper_width], dtype=np.float32),
            "prompt": prompt,
        }
        for index, frame in enumerate(all_frames):
            observation[f"left_wrist_0_rgb_{index}"] = frame
        return observation


def run_contract_self_test() -> None:
    history = ShellGameHistory()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    start = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    assert "初始帧" in history.press_f(image, start)
    for _ in range(NUM_SWAPS):
        history.press_f(image, start)
        for _ in range(SWAP_FRAMES):
            history.capture_tick(image)
    assert history.ready
    assert history.frame_count == HISTORY_FRAMES

    current = start.copy()
    current[0] += 0.02
    obs = history.build_policy_observation(
        current_frame=image,
        current_pose6=current,
        gripper_width=0.04,
        prompt="test",
    )
    image_keys = [key for key in obs if key.startswith("left_wrist_0_rgb_")]
    assert len(image_keys) == TOTAL_MODEL_FRAMES
    np.testing.assert_allclose(obs["robot0_eef_pos"], [0.02, 0.0, 0.0], atol=1e-6)

    relative_actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float64)
    relative_actions[:, 3] = 1.0
    relative_actions[:, 7] = 1.0
    relative_actions[:, 0] = 0.01
    world = relative_chunk_to_world(relative_actions, start)
    np.testing.assert_allclose(world[:, 0], start[0] + 0.01, atol=1e-6)
    print("ShellGame OpenPI contract self-test passed")
