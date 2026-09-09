"""ShellGame zarr -> relative LeRobot conversion helpers for replay verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from utils.pose_util import mat_to_pose, mat_to_pose10d, pose10d_to_mat, pose_to_mat
from utils.shellgame_openpi import relative_actions_sequence_to_world

OBSERVE_FRAMES = 241
GRASP_OUTPUT_START = OBSERVE_FRAMES


@dataclass(frozen=True)
class ConvertedEpisode:
    episode_index: int
    source_indices: np.ndarray
    grasp_start: int
    relative_actions: np.ndarray
    anchor_poses: np.ndarray
    decoded_poses: np.ndarray
    decoded_grippers: np.ndarray


def _rolling_any(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    counts = np.convolve(signal.astype(np.int32), np.ones(window, dtype=np.int32), mode="same")
    return counts > 0


def _valid_absolute_action(action: np.ndarray) -> bool:
    return bool(
        action.shape == (7,)
        and np.all(np.isfinite(action))
        and np.linalg.norm(action[:3]) > 1e-6
    )


def _make_pose_matrices(positions: np.ndarray, rotvecs: np.ndarray) -> np.ndarray:
    poses = np.concatenate([positions, rotvecs], axis=-1)
    return np.asarray(pose_to_mat(poses), dtype=np.float64)


def detect_grasp_start(
    positions: np.ndarray,
    raw_actions: np.ndarray,
    *,
    search_start: int = OBSERVE_FRAMES,
    command_threshold_m: float = 0.002,
    motion_threshold_m: float = 0.001,
    persistence_frames: int = 5,
) -> tuple[int, bool]:
    episode_length = positions.shape[0]
    if episode_length <= search_start:
        raise ValueError(
            f"Episode has only {episode_length} frames, but {search_start} observation frames are required"
        )

    valid_action = np.all(np.isfinite(raw_actions[:, :3]), axis=-1) & (
        np.linalg.norm(raw_actions[:, :3], axis=-1) > 1e-6
    )
    command_error = np.linalg.norm(raw_actions[:, :3] - positions, axis=-1)
    commanded = valid_action & (command_error >= command_threshold_m)

    step_motion = np.zeros(episode_length, dtype=np.float64)
    step_motion[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    moving = step_motion >= motion_threshold_m

    active = _rolling_any(commanded | moving, persistence_frames)
    candidates = np.flatnonzero(active[search_start:])
    if candidates.size:
        return search_start + int(candidates[0]), False
    return search_start, True


def relative_episode(
    positions: np.ndarray,
    rotvecs: np.ndarray,
    gripper: np.ndarray,
    raw_actions: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measured_world = _make_pose_matrices(positions, rotvecs)
    start_inverse = np.linalg.inv(measured_world[source_indices[0]])

    selected_world = measured_world[source_indices]
    state_relative = start_inverse[None] @ selected_world
    state_pose9 = mat_to_pose10d(state_relative).astype(np.float32)
    state_gripper = gripper[source_indices].reshape(-1, 1).astype(np.float32)

    relative_actions = np.empty((source_indices.size, 10), dtype=np.float32)
    for output_index, source_index in enumerate(source_indices):
        next_output_index = min(output_index + 1, source_indices.size - 1)
        target_source_index = int(source_indices[next_output_index])
        raw_target = np.asarray(raw_actions[target_source_index], dtype=np.float64)

        if _valid_absolute_action(raw_target):
            target_world = pose_to_mat(raw_target[:6])
            target_gripper = float(raw_target[6])
        else:
            target_world = measured_world[target_source_index]
            target_gripper = float(gripper[target_source_index])

        current_inverse = np.linalg.inv(measured_world[source_index])
        target_relative = current_inverse @ target_world
        relative_actions[output_index, :9] = mat_to_pose10d(target_relative).astype(np.float32)
        relative_actions[output_index, 9] = np.float32(target_gripper)

    return (
        state_pose9[:, :3],
        state_pose9[:, 3:],
        state_gripper,
        relative_actions,
    )


def build_source_indices(
    source_length: int,
    *,
    grasp_start: int,
    grasp_stride: int = 1,
) -> np.ndarray:
    observe_indices = np.arange(OBSERVE_FRAMES, dtype=np.int64)
    grasp_indices = np.arange(grasp_start, source_length, grasp_stride, dtype=np.int64)
    return np.concatenate([observe_indices, grasp_indices])


def ground_truth_target_pose7(
    *,
    measured_world: np.ndarray,
    raw_actions: np.ndarray,
    gripper: np.ndarray,
    target_source_index: int,
) -> np.ndarray:
    raw_target = np.asarray(raw_actions[target_source_index], dtype=np.float64)
    if _valid_absolute_action(raw_target):
        pose6 = raw_target[:6]
        target_gripper = float(raw_target[6])
    else:
        pose6 = np.asarray(mat_to_pose(measured_world[target_source_index]), dtype=np.float64)
        target_gripper = float(gripper[target_source_index])
    return np.concatenate([pose6, [target_gripper]], dtype=np.float64)


def rotation_error_rad(pose_a6: np.ndarray, pose_b6: np.ndarray) -> float:
    rot_a = Rotation.from_rotvec(np.asarray(pose_a6[3:6], dtype=np.float64))
    rot_b = Rotation.from_rotvec(np.asarray(pose_b6[3:6], dtype=np.float64))
    return float((rot_a.inv() * rot_b).magnitude())


def convert_episode_from_arrays(
    *,
    episode_index: int,
    positions: np.ndarray,
    rotvecs: np.ndarray,
    gripper: np.ndarray,
    raw_actions: np.ndarray,
    grasp_stride: int = 1,
) -> ConvertedEpisode:
    grasp_start, _ = detect_grasp_start(positions, raw_actions)
    source_indices = build_source_indices(len(positions), grasp_start=grasp_start, grasp_stride=grasp_stride)
    _state_pos, _state_rot6d, _state_gripper, relative_actions = relative_episode(
        positions,
        rotvecs,
        gripper,
        raw_actions,
        source_indices,
    )
    measured_world = _make_pose_matrices(positions, rotvecs)
    anchor_poses = np.asarray(
        [mat_to_pose(measured_world[int(index)]) for index in source_indices],
        dtype=np.float64,
    )
    grasp_slice = slice(OBSERVE_FRAMES, None)
    grasp_actions = np.asarray(relative_actions[grasp_slice], dtype=np.float64)
    grasp_anchors = anchor_poses[grasp_slice]
    decoded = relative_actions_sequence_to_world(grasp_actions, grasp_anchors)
    return ConvertedEpisode(
        episode_index=episode_index,
        source_indices=source_indices,
        grasp_start=grasp_start,
        relative_actions=relative_actions,
        anchor_poses=anchor_poses,
        decoded_poses=decoded[:, :6],
        decoded_grippers=decoded[:, 6],
    )


def load_lerobot_episode_actions(
    lerobot_dir: Path,
    *,
    episode_index: int,
) -> tuple[np.ndarray, int]:
    """Load relative actions for one episode from a LeRobot parquet dataset."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "lerobot-relative requires pyarrow. Install it in your robot env, e.g. "
            "`pip install pyarrow`, or use --trajectory-source convert-inline instead."
        ) from exc

    lerobot_dir = lerobot_dir.expanduser().resolve()
    parquet_path = lerobot_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.is_file():
        candidates = sorted(lerobot_dir.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"episode parquet not found for episode {episode_index} under {lerobot_dir / 'data'}"
            )
        parquet_path = candidates[0]

    table = pq.read_table(parquet_path, columns=["actions", "frame_index", "episode_index"])
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row["frame_index"]))
    actions = np.asarray([row["actions"] for row in rows], dtype=np.float64)
    return actions, len(rows)


def build_grasp_trajectory_from_lerobot(
    demo_dir: Path,
    *,
    episode_index: int,
    lerobot_dir: Path | None = None,
    grasp_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ConvertedEpisode]:
    """Return decoded grasp trajectory and conversion metadata."""
    zarr_path = demo_dir / "replay_buffer.zarr"
    root = zarr.open(str(zarr_path), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if not 0 <= episode_index < len(ends):
        raise IndexError(f"episode {episode_index} out of range [0, {len(ends)})")
    start = 0 if episode_index == 0 else int(ends[episode_index - 1])
    end = int(ends[episode_index])
    data = root["data"]
    positions = np.asarray(data["robot0_eef_pos"][start:end], dtype=np.float64)
    rotvecs = np.asarray(data["robot0_eef_rot_axis_angle"][start:end], dtype=np.float64)
    gripper = np.asarray(data["robot0_gripper_width"][start:end], dtype=np.float64)
    raw_actions = np.asarray(data["action"][start:end], dtype=np.float64)

    converted = convert_episode_from_arrays(
        episode_index=episode_index,
        positions=positions,
        rotvecs=rotvecs,
        gripper=gripper,
        raw_actions=raw_actions,
        grasp_stride=grasp_stride,
    )

    if lerobot_dir is not None:
        lerobot_actions, _ = load_lerobot_episode_actions(lerobot_dir, episode_index=episode_index)
        if lerobot_actions.shape != converted.relative_actions.shape:
            raise ValueError(
                "LeRobot action count mismatch: "
                f"lerobot={lerobot_actions.shape}, inline={converted.relative_actions.shape}"
            )
        if not np.allclose(lerobot_actions, converted.relative_actions, atol=1e-5, rtol=1e-5):
            raise ValueError("LeRobot actions differ from inline conversion")

    return (
        converted.decoded_poses,
        converted.decoded_grippers,
        converted.anchor_poses[OBSERVE_FRAMES:],
        converted,
    )


def summarize_roundtrip_errors(
    *,
    converted: ConvertedEpisode,
    positions: np.ndarray,
    rotvecs: np.ndarray,
    gripper: np.ndarray,
    raw_actions: np.ndarray,
) -> dict[str, float]:
    measured_world = _make_pose_matrices(positions, rotvecs)
    pos_errors: list[float] = []
    rot_errors: list[float] = []
    grip_errors: list[float] = []

    source_indices = converted.source_indices
    relative_actions = converted.relative_actions
    for output_index in range(OBSERVE_FRAMES, len(source_indices) - 1):
        source_index = int(source_indices[output_index])
        next_source_index = int(source_indices[output_index + 1])
        anchor_pose6 = np.asarray(mat_to_pose(measured_world[source_index]), dtype=np.float64)
        gt = ground_truth_target_pose7(
            measured_world=measured_world,
            raw_actions=raw_actions,
            gripper=gripper,
            target_source_index=next_source_index,
        )
        decoded = relative_actions_sequence_to_world(
            np.asarray([relative_actions[output_index]], dtype=np.float64),
            np.asarray([anchor_pose6], dtype=np.float64),
        )[0]
        pos_errors.append(float(np.linalg.norm(decoded[:3] - gt[:3])))
        rot_errors.append(rotation_error_rad(decoded[:6], gt[:6]))
        grip_errors.append(abs(float(decoded[6]) - float(gt[6])))

    return {
        "max_pos_err_m": max(pos_errors) if pos_errors else 0.0,
        "mean_pos_err_m": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_rot_err_rad": max(rot_errors) if rot_errors else 0.0,
        "mean_rot_err_rad": float(np.mean(rot_errors)) if rot_errors else 0.0,
        "max_grip_err_m": max(grip_errors) if grip_errors else 0.0,
        "mean_grip_err_m": float(np.mean(grip_errors)) if grip_errors else 0.0,
    }
