"""Drag teleop + UMI zarr recording helpers for Kai0 DAgger eval.

Mirrors the collect_drag_umi_zmq_bimanual enter/exit and pose-as-action
recording path without importing that script.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from modules.arx5_env_with_images_bimanual_rs import Arx5EnvWithImages
from utils.interpolation_util import nearest_rotvec_equivalent


def action_from_obs(obs: dict, n_robots: int) -> np.ndarray:
    """Absolute TCP action: [xyz, rotvec, gripper] * n_robots."""
    action = np.zeros(7 * n_robots, dtype=np.float64)
    for robot_idx in range(n_robots):
        eef_pos = np.asarray(obs[f"robot{robot_idx}_eef_pos"][-1], dtype=np.float64)
        eef_rot = np.asarray(
            obs[f"robot{robot_idx}_eef_rot_axis_angle"][-1], dtype=np.float64
        )
        gripper = float(np.asarray(obs[f"robot{robot_idx}_gripper_width"][-1]).squeeze())
        action[7 * robot_idx : 7 * robot_idx + 3] = eef_pos
        action[7 * robot_idx + 3 : 7 * robot_idx + 6] = eef_rot
        action[7 * robot_idx + 6] = gripper
    return action


def record_pose_action(
    env: Arx5EnvWithImages,
    obs: dict,
    *,
    last_rotvecs: list[np.ndarray | None] | None = None,
) -> None:
    """Put current TCP pose into action_accumulator (no robot command)."""
    if env.action_accumulator is None:
        return
    n_robots = len(env.robots)
    action = action_from_obs(obs, n_robots)
    if last_rotvecs is not None:
        for robot_idx in range(n_robots):
            sl = slice(7 * robot_idx + 3, 7 * robot_idx + 6)
            rv = action[sl]
            prev = last_rotvecs[robot_idx]
            if prev is not None:
                rv = nearest_rotvec_equivalent(rv, prev)
                action[sl] = rv
            last_rotvecs[robot_idx] = np.asarray(action[sl], dtype=np.float64).copy()
    t = float(np.asarray(obs["timestamp"][-1]))
    env.action_accumulator.put(action[None, :], np.array([t], dtype=np.float64))


def enter_drag_mode(env: Arx5EnvWithImages) -> None:
    print("Entering drag mode...")
    env.set_gravity_compensation()
    time.sleep(0.25)
    print(
        "Drag mode ON (gravity if supported, else damping). "
        "Drag arms freely; squeeze grippers by hand."
    )


def leave_drag_stay_pose(env: Arx5EnvWithImages) -> None:
    """Exit zero-force / drag; stay at current pose and restore gains."""
    print("Leaving drag (stay at current pose)...")
    if hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
        except Exception as exc:  # noqa: BLE001
            print(f"set_end_control warning: {exc}")
    if hasattr(env, "restore_gains_after_drag"):
        try:
            env.restore_gains_after_drag()
        except Exception as exc:  # noqa: BLE001
            print(f"restore_gains_after_drag warning: {exc}")


def abort_episode(env: Arx5EnvWithImages) -> None:
    """Stop recording without writing zarr (do not drop a previously saved ep)."""
    try:
        env.camera.stop_recording()
    except Exception as exc:  # noqa: BLE001
        print(f"stop_recording warning: {exc}")

    episode_id = env.replay_buffer.n_episodes
    video_dir = env.video_dir.joinpath(str(episode_id))
    if video_dir.exists():
        shutil.rmtree(str(video_dir))

    env.obs_accumulator = None
    env.action_accumulator = None
    if getattr(env, "image_accumulator", None) is not None:
        env.image_accumulator = None
    print(f"Episode {episode_id} aborted (not saved).")


def save_intervention_sidecar(
    output_dir: Path,
    episode_id: int,
    labels: list[int] | np.ndarray,
) -> Path:
    """Write 0=policy / 1=drag labels next to the zarr root."""
    out_dir = Path(output_dir) / "intervention"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{episode_id}.npy"
    arr = np.asarray(labels, dtype=np.uint8).reshape(-1)
    np.save(path, arr)
    print(f"Intervention sidecar saved: {path} (T={len(arr)}, human={int(arr.sum())})")
    return path
