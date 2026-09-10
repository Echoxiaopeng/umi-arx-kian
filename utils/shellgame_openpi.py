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
# M6 / Stage2 deployment keys (serve_shellgame_real_m6_cached.py).
M6_VIDEO_FRAME_KEY_PREFIX = "left_wrist_0_rgb_0_"
M6_VIDEO_CURRENT_STEP_KEY = "left_wrist_0_rgb_0"
# Legacy semantic-action cached keys.
LEGACY_VIDEO_FRAME_KEY_PREFIX = "left_wrist_0_rgb_"
LEGACY_VIDEO_CURRENT_STEP_KEY = "left_wrist_0_rgb"


def shift_pose_along_tcp_z(pose6: np.ndarray, distance: float) -> np.ndarray:
    """Shift a world pose along its local TCP Z axis without changing rotation."""
    pose = np.asarray(pose6, dtype=np.float64).reshape(6).copy()
    if abs(float(distance)) > 1e-12:
        pose[:3] += Rotation.from_rotvec(pose[3:6]).apply(
            np.array([0.0, 0.0, float(distance)], dtype=np.float64)
        )
    return pose


def eef_pose_to_link6(pose6: np.ndarray, eef_offset: float) -> np.ndarray:
    """Convert a ZMQ eef control-point pose to the Direct-training link6 pose."""
    return shift_pose_along_tcp_z(pose6, -float(eef_offset))


def link6_pose_to_eef(pose6: np.ndarray, eef_offset: float) -> np.ndarray:
    """Convert a Direct-training link6 pose to the ZMQ eef control-point pose."""
    return shift_pose_along_tcp_z(pose6, float(eef_offset))


def link6_actions_to_eef(actions: np.ndarray, eef_offset: float) -> np.ndarray:
    """Convert absolute link6 waypoints [..., xyz, rotvec, ...] to ZMQ eef."""
    result = np.asarray(actions, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] < 6:
        raise ValueError(f"Expected action array shaped (N, >=6), got {result.shape}")
    for index in range(len(result)):
        result[index, :6] = link6_pose_to_eef(result[index, :6], eef_offset)
    return result


def direct_link6_pose_to_zmq_link6(
    direct_pose6: np.ndarray,
    origin_offset: np.ndarray,
) -> np.ndarray:
    """Map Direct link6 xyz+rotvec into the ZMQ link6 frame via a fixed translation."""
    direct = np.asarray(direct_pose6, dtype=np.float64).reshape(6).copy()
    offset = np.asarray(origin_offset, dtype=np.float64).reshape(3)
    out = direct.copy()
    out[:3] = direct[:3] + offset
    return out


def direct_link6_pose_to_zmq_eef(
    direct_pose6: np.ndarray,
    origin_offset: np.ndarray,
    eef_offset: float,
) -> np.ndarray:
    """Convert a Direct-collected init pose into a ZMQ eef command pose."""
    link6_zmq = direct_link6_pose_to_zmq_link6(direct_pose6, origin_offset)
    return link6_pose_to_eef(link6_zmq, eef_offset)


def zmq_eef_pose_to_direct_link6(
    eef_pose6: np.ndarray,
    origin_offset: np.ndarray,
    eef_offset: float,
) -> np.ndarray:
    """Inverse of direct_link6_pose_to_zmq_eef for calibration helpers."""
    link6_zmq = eef_pose_to_link6(eef_pose6, eef_offset)
    direct = link6_zmq.copy()
    direct[:3] = link6_zmq[:3] - np.asarray(origin_offset, dtype=np.float64).reshape(3)
    return direct


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


def relative_action_to_world(action10: np.ndarray, anchor_pose6: np.ndarray) -> np.ndarray:
    """Decode one current-relative EEF10 action to world xyz+rotvec+gripper."""
    action10 = np.asarray(action10, dtype=np.float64).reshape(10)
    anchor = pose_to_mat(np.asarray(anchor_pose6, dtype=np.float64).reshape(6))
    world_mat = anchor @ pose10d_to_mat(action10[:9])
    world_pose6 = np.asarray(mat_to_pose(world_mat), dtype=np.float64)
    return np.concatenate([world_pose6, [float(action10[9])]], dtype=np.float64)


def world_action_to_relative(world7: np.ndarray, anchor_pose6: np.ndarray) -> np.ndarray:
    """Encode one world xyz+rotvec+gripper waypoint as current-relative EEF10."""
    world7 = np.asarray(world7, dtype=np.float64).reshape(7)
    anchor = pose_to_mat(np.asarray(anchor_pose6, dtype=np.float64).reshape(6))
    world_mat = pose_to_mat(world7[:6])
    relative = np.linalg.inv(anchor) @ world_mat
    pose9 = np.asarray(mat_to_pose10d(relative), dtype=np.float32)
    return np.concatenate([pose9, np.asarray([world7[6]], dtype=np.float32)], axis=-1)


def world_chunk_to_relative(world_actions: np.ndarray, anchor_pose6: np.ndarray) -> np.ndarray:
    """Encode a world xyz+rotvec+gripper chunk as current-relative EEF10."""
    world = np.asarray(world_actions, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 7:
        raise ValueError(f"Expected world actions (N, 7), got {world.shape}")
    out = np.empty((len(world), ACTION_DIM), dtype=np.float32)
    for index in range(len(world)):
        out[index] = world_action_to_relative(world[index], anchor_pose6)
    return out


def relative_actions_sequence_to_world(
    actions: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Decode a sequence of current-relative EEF10 actions using per-step anchors."""
    actions = np.asarray(actions, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions shaped (N, {ACTION_DIM}), got {actions.shape}")
    if anchors.shape != (len(actions), 6):
        raise ValueError(f"Expected anchors shaped ({len(actions)}, 6), got {anchors.shape}")

    world = np.empty((len(actions), 7), dtype=np.float64)
    for index in range(len(actions)):
        world[index] = relative_action_to_world(actions[index], anchors[index])
    return world


def relative_chunk_to_world(action_chunk: np.ndarray, anchor_pose6: np.ndarray) -> np.ndarray:
    """Restore current-relative EEF10 targets to world xyz+rotvec+gripper."""
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action chunk (N, {ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] < 1:
        raise ValueError(f"Expected a non-empty action chunk, got {actions.shape}")
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
    max_waypoints: int = ACTION_HORIZON,
) -> None:
    """Reject unsafe chunks instead of silently changing policy geometry."""
    actions = np.asarray(world_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.all(np.isfinite(actions)):
        raise ValueError(f"Invalid world action shape/content: {actions.shape}")
    if len(actions) < max_waypoints:
        raise ValueError(
            f"Need at least {max_waypoints} waypoints for validation, got {len(actions)}"
        )

    max_waypoints = int(max_waypoints)
    if not 1 <= max_waypoints <= len(actions):
        raise ValueError(f"max_waypoints must be in [1, {len(actions)}], got {max_waypoints}")

    selected = actions[:max_waypoints]
    xyz = selected[:, :3]
    outside = (xyz < workspace_min[None]) | (xyz > workspace_max[None])
    if np.any(outside):
        bad = int(np.flatnonzero(np.any(outside, axis=1))[0])
        raise ValueError(f"Waypoint {bad} is outside workspace: {xyz[bad]}")

    poses = np.concatenate(
        [np.asarray(anchor_pose6, dtype=np.float64).reshape(1, 6), selected[:, :6]],
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
            f"Executed segment {index}->{index + 1} translation step "
            f"{position_steps[index]:.4f} m exceeds {max_position_step:.4f} m"
        )
    if np.any(rotation_steps > max_rotation_step):
        index = int(np.argmax(rotation_steps))
        raise ValueError(
            f"Executed segment {index}->{index + 1} rotation step "
            f"{rotation_steps[index]:.4f} rad exceeds {max_rotation_step:.4f} rad"
        )


def select_safe_waypoint_count(
    world_actions: np.ndarray,
    *,
    anchor_pose6: np.ndarray,
    dt: float,
    max_pos_speed: float,
    max_rot_speed: float,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    max_waypoints: int,
) -> tuple[int, str | None]:
    """Return the longest safe prefix length and the rejection reason if truncated."""
    max_waypoints = min(int(max_waypoints), len(world_actions))
    last_error: str | None = None
    for count in range(max_waypoints, 0, -1):
        try:
            validate_world_actions(
                world_actions,
                anchor_pose6=anchor_pose6,
                dt=dt,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                max_waypoints=count,
            )
            if count < max_waypoints:
                return count, last_error
            return count, None
        except ValueError as exc:
            last_error = str(exc)
    return 0, last_error


@dataclass
class ShellGameHistory:
    """Strict F-key capture state: 1 initial frame followed by 3x80 swap frames."""

    initial_frame: np.ndarray | None = None
    episode_start_pose6: np.ndarray | None = None
    swap_segments: list[list[np.ndarray]] = field(default_factory=list)
    active_segment: list[np.ndarray] | None = None
    # auto: one F saves the initial frame and continuously records all swaps.
    # manual: F for initial, then each F starts the next swap.
    auto_capture: bool = False

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

    def seed_dummy_history(self, frame: np.ndarray, current_pose6: np.ndarray) -> None:
        """Fill the required 241-frame history with duplicates for alignment smoke tests."""
        self.reset()
        dummy = image_to_uint8(frame)
        pose6 = np.asarray(current_pose6, dtype=np.float64).reshape(6).copy()
        self.initial_frame = dummy.copy()
        self.episode_start_pose6 = pose6
        self.swap_segments = [[dummy.copy() for _ in range(SWAP_FRAMES)] for _ in range(NUM_SWAPS)]
        self.active_segment = None
        if not self.ready:
            raise RuntimeError(
                f"Failed to seed dummy history: {self.frame_count}/{HISTORY_FRAMES} frames"
            )

    def press_f(self, frame: np.ndarray, current_pose6: np.ndarray) -> str:
        if self.collecting:
            return "正在采集当前交换, 忽略重复 F"
        if self.initial_frame is None:
            self.initial_frame = image_to_uint8(frame).copy()
            self.episode_start_pose6 = np.asarray(current_pose6, dtype=np.float64).reshape(6).copy()
            if self.auto_capture:
                self.active_segment = []
                return (
                    f"已保存初始帧并开始连续采集 "
                    f"(1+{NUM_SWAPS * SWAP_FRAMES}); 采满后自动开始"
                )
            return "已保存初始帧; 下一次 F 开始采集第一次交换"
        if self.ready:
            if self.auto_capture:
                return "241 帧历史已完成; 即将自动开始, 或按 P 开始下一轮"
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
        if self.ready:
            self.active_segment = None
            if self.auto_capture:
                return "三次交换采集完成: 241/241, 即将自动开始"
            return "三次交换采集完成: 241/241, 按 C 开始抓取"
        if self.auto_capture:
            finished = len(self.swap_segments)
            self.active_segment = []
            return (
                f"第 {finished} 次交换完成; 继续采集第 {finished + 1} 次: "
                f"0/{SWAP_FRAMES}"
            )
        self.active_segment = None
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
            if self.auto_capture:
                return "READY 241/241 - auto-start policy, P: reset"
            return "READY 241/241 - C: policy, P: reset"
        if self.initial_frame is None:
            return "F: auto-capture all swaps" if self.auto_capture else "F: capture initial frame"
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

    def build_history_upload_observation(
        self,
        *,
        prompt: str,
        image_key_style: str = "m6",
    ) -> dict[str, np.ndarray | str]:
        """Build a one-time 241-frame history upload for cached server inference."""
        if not self.ready:
            raise RuntimeError(f"History is incomplete: {self.frame_count}/{HISTORY_FRAMES}")
        if image_key_style == "m6":
            prefix = M6_VIDEO_FRAME_KEY_PREFIX
        elif image_key_style == "legacy":
            prefix = LEGACY_VIDEO_FRAME_KEY_PREFIX
        else:
            raise ValueError(f"Unknown image_key_style={image_key_style!r}")
        observation: dict[str, np.ndarray | str] = {
            "mode": "reset_history",
            "prompt": prompt,
        }
        for index, frame in enumerate(self.frames()):
            observation[f"{prefix}{index}"] = frame
        if len([key for key in observation if key.startswith(prefix)]) != HISTORY_FRAMES:
            raise RuntimeError(f"Expected {HISTORY_FRAMES} history frames for upload")
        return observation

    def build_step_observation(
        self,
        *,
        current_frame: np.ndarray,
        current_pose6: np.ndarray,
        gripper_width: float,
        prompt: str,
        image_key_style: str = "m6",
    ) -> dict[str, np.ndarray | str]:
        """Build a single-frame step observation for cached server inference."""
        if self.episode_start_pose6 is None:
            raise RuntimeError("Episode start pose is missing")
        if image_key_style == "m6":
            current_key = M6_VIDEO_CURRENT_STEP_KEY
        elif image_key_style == "legacy":
            current_key = LEGACY_VIDEO_CURRENT_STEP_KEY
        else:
            raise ValueError(f"Unknown image_key_style={image_key_style!r}")
        state_pose9 = relative_pose10(self.episode_start_pose6, current_pose6)
        return {
            "mode": "infer_step",
            "prompt": prompt,
            current_key: image_to_uint8(current_frame),
            "robot0_eef_pos": state_pose9[:3],
            "robot0_eef_rot_axis_angle": state_pose9[3:9],
            "robot0_gripper_width": np.asarray([gripper_width], dtype=np.float32),
        }


class ShellGamePolicyClient:
    """WebSocket client wrapper for cached ShellGame history upload + step infer."""

    def __init__(
        self,
        client,
        *,
        cached_infer: bool = True,
        image_key_style: str = "m6",
    ) -> None:
        self._client = client
        self._cached_infer = cached_infer
        self._image_key_style = image_key_style
        metadata = client.get_server_metadata()
        self._server_supports_cache = bool(metadata.get("supports_cached_infer", False))
        self._server_supports_rtc = bool(metadata.get("supports_rtc", False))
        self._history_uploaded = False

    @property
    def cached_infer_enabled(self) -> bool:
        return self._cached_infer and self._server_supports_cache

    @property
    def rtc_enabled(self) -> bool:
        return self._server_supports_rtc

    def reset_session(self) -> None:
        try:
            if self.cached_infer_enabled:
                self._client.infer({"mode": "reset"})
        except Exception as exc:
            # RTC stop can race a background recv; keep local cache state consistent.
            print(f"[cache] reset_session failed (ignored): {exc}")
        finally:
            self._history_uploaded = False

    def upload_history(self, history: ShellGameHistory, *, prompt: str) -> dict:
        if not self.cached_infer_enabled:
            raise RuntimeError("Cached infer is disabled or unsupported by the server")
        result = self._client.infer(
            history.build_history_upload_observation(
                prompt=prompt,
                image_key_style=self._image_key_style,
            )
        )
        self._history_uploaded = bool(result.get("cache_ready", False))
        if not self._history_uploaded:
            raise RuntimeError(f"Server failed to cache history: {result!r}")
        return result

    def infer_step(
        self,
        history: ShellGameHistory,
        *,
        current_frame: np.ndarray,
        current_pose6: np.ndarray,
        gripper_width: float,
        prompt: str,
        rtc_prev_chunk: np.ndarray | None = None,
        rtc_inference_delay: int = 0,
        rtc_execution_horizon: int = 0,
        rtc_prefix_attention_schedule: str = "exp",
        rtc_max_guidance_weight: float = 5.0,
    ) -> dict:
        if self.cached_infer_enabled:
            if not self._history_uploaded:
                raise RuntimeError("History cache missing; call upload_history first")
            obs = history.build_step_observation(
                current_frame=current_frame,
                current_pose6=current_pose6,
                gripper_width=gripper_width,
                prompt=prompt,
                image_key_style=self._image_key_style,
            )
            if rtc_prev_chunk is not None:
                obs["rtc_prev_chunk"] = np.asarray(rtc_prev_chunk, dtype=np.float32)
                obs["rtc_inference_delay"] = int(rtc_inference_delay)
                obs["rtc_execution_horizon"] = int(rtc_execution_horizon)
                obs["rtc_prefix_attention_schedule"] = str(rtc_prefix_attention_schedule)
                obs["rtc_max_guidance_weight"] = float(rtc_max_guidance_weight)
            return self._client.infer(obs)
        return self.infer_full(
            history,
            current_frame=current_frame,
            current_pose6=current_pose6,
            gripper_width=gripper_width,
            prompt=prompt,
        )

    def infer_full(
        self,
        history: ShellGameHistory,
        *,
        current_frame: np.ndarray,
        current_pose6: np.ndarray,
        gripper_width: float,
        prompt: str,
    ) -> dict:
        return self._client.infer(
            history.build_policy_observation(
                current_frame=current_frame,
                current_pose6=current_pose6,
                gripper_width=gripper_width,
                prompt=prompt,
            )
        )


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

    auto_history = ShellGameHistory(auto_capture=True)
    msg = auto_history.press_f(image, start)
    assert "连续采集" in msg
    assert auto_history.collecting
    for _ in range(NUM_SWAPS * SWAP_FRAMES):
        auto_history.capture_tick(image)
    assert auto_history.ready
    assert auto_history.frame_count == HISTORY_FRAMES
    assert not auto_history.collecting

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

    history_obs = history.build_history_upload_observation(prompt="test")
    assert history_obs["mode"] == "reset_history"
    assert len([key for key in history_obs if key.startswith("left_wrist_0_rgb_")]) == HISTORY_FRAMES

    step_obs = history.build_step_observation(
        current_frame=image,
        current_pose6=current,
        gripper_width=0.04,
        prompt="test",
    )
    assert step_obs["mode"] == "infer_step"
    assert step_obs["left_wrist_0_rgb"].shape == (224, 224, 3)

    relative_actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float64)
    relative_actions[:, 3] = 1.0
    relative_actions[:, 7] = 1.0
    relative_actions[:, 0] = 0.01
    world = relative_chunk_to_world(relative_actions, start)
    np.testing.assert_allclose(world[:, 0], start[0] + 0.01, atol=1e-6)

    eef_offset = 0.145
    eef_anchor = np.array([0.24, 0.03, 0.15, 0.2, -0.3, 0.4], dtype=np.float64)
    link6_anchor = eef_pose_to_link6(eef_anchor, eef_offset)
    np.testing.assert_allclose(
        link6_pose_to_eef(link6_anchor, eef_offset),
        eef_anchor,
        atol=1e-9,
    )

    identity_actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float64)
    identity_actions[:, 3] = 1.0
    identity_actions[:, 7] = 1.0
    identity_link6 = relative_chunk_to_world(identity_actions, link6_anchor)
    identity_eef = link6_actions_to_eef(identity_link6, eef_offset)
    np.testing.assert_allclose(
        identity_eef[:, :6],
        np.tile(eef_anchor, (ACTION_HORIZON, 1)),
        atol=1e-8,
    )

    rotated_actions = identity_actions.copy()
    rotated_actions[:, 3:9] = np.asarray(
        mat_to_pose10d(
            pose_to_mat(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0]))
        )[3:9]
    )
    rotated_link6 = relative_chunk_to_world(rotated_actions, link6_anchor)
    rotated_eef = link6_actions_to_eef(rotated_link6, eef_offset)
    expected_first_eef = link6_pose_to_eef(rotated_link6[0, :6], eef_offset)
    np.testing.assert_allclose(rotated_eef[0, :6], expected_first_eef, atol=1e-9)

    # ZMQ may expose an equivalent long-angle rotvec. Compare geometry, not
    # raw rotvec components, because the decoded waypoint is canonicalized.
    long_angle_eef = np.array(
        [0.241, 0.0332, 0.1468, 2.6911, -2.4087, 2.2298],
        dtype=np.float64,
    )
    long_angle_link6 = eef_pose_to_link6(long_angle_eef, eef_offset)
    long_angle_link6_targets = relative_chunk_to_world(
        identity_actions,
        long_angle_link6,
    )
    long_angle_eef_targets = link6_actions_to_eef(
        long_angle_link6_targets,
        eef_offset,
    )
    np.testing.assert_allclose(
        long_angle_eef_targets[:, :3],
        np.tile(long_angle_eef[:3], (ACTION_HORIZON, 1)),
        atol=1e-9,
    )
    target_rotations = Rotation.from_rotvec(long_angle_eef_targets[:, 3:6])
    anchor_rotation = Rotation.from_rotvec(long_angle_eef[3:6])
    rotation_errors = (anchor_rotation.inv() * target_rotations).magnitude()
    np.testing.assert_allclose(rotation_errors, 0.0, atol=1e-9)

    origin = np.array([0.095, 0.001, 0.158], dtype=np.float64)
    direct_init = np.array([0.0, 0.0, 0.12, 0.2, -0.3, 0.4], dtype=np.float64)
    zmq_init = direct_link6_pose_to_zmq_eef(direct_init, origin, eef_offset)
    roundtrip = zmq_eef_pose_to_direct_link6(zmq_init, origin, eef_offset)
    np.testing.assert_allclose(roundtrip, direct_init, atol=1e-9)
    print("ShellGame OpenPI contract self-test passed")
