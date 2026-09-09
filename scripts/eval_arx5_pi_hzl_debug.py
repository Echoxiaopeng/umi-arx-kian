#!/usr/bin/env python3
"""Diagnostic wrapper for ``eval_arx5_pi_hzl.py``.

The robot-control behaviour is intentionally unchanged: this script delegates
to the original entrypoint and preserves its common-anchor action decoder.  It
adds persistent, layered logs for policy inputs/outputs, both possible relative
chunk interpretations, safety filtering, command scheduling, and measured
tracking on the next observation.
"""

from __future__ import annotations

import atexit
import dataclasses
import datetime as dt
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np

import eval_arx5_pi_hzl as _v1
from modules.arx5_env import Arx5Env
from utils.pose_util import mat_to_pose
from utils.pose_util import pose10d_to_mat
from utils.pose_util import pose_to_mat


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "data_local" / "shellgame_eval" / "debug_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"eval_arx5_pi_hzl_debug_{dt.datetime.now():%Y%m%d_%H%M%S}.log"


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(bool(getattr(stream, "isatty", lambda: False)()) for stream in self._streams)


_log_stream = LOG_PATH.open("a", encoding="utf-8", buffering=1)
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = _Tee(_original_stdout, _log_stream)
sys.stderr = _Tee(_original_stderr, _log_stream)


def _close_debug_log() -> None:
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    _log_stream.flush()
    _log_stream.close()


atexit.register(_close_debug_log)


@dataclasses.dataclass
class _DebugState:
    inference_index: int = 0
    previous_current_frame: np.ndarray | None = None
    last_obs_pose6: np.ndarray | None = None
    last_obs_gripper: float | None = None
    expect_policy_exec: bool = False
    pending_command: dict[str, Any] | None = None


_debug = _DebugState()


def _array_text(value: np.ndarray, precision: int = 6) -> str:
    return np.array2string(
        np.asarray(value),
        precision=precision,
        suppress_small=False,
        separator=", ",
        max_line_width=240,
    )


def _pose_error(pose_a6: np.ndarray, pose_b6: np.ndarray) -> tuple[float, float]:
    a = np.asarray(pose_a6, dtype=np.float64).reshape(6)
    b = np.asarray(pose_b6, dtype=np.float64).reshape(6)
    pos_mm = float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)
    ra = _v1.Rotation.from_rotvec(a[3:6])
    rb = _v1.Rotation.from_rotvec(b[3:6])
    rot_deg = float(np.degrees((ra.inv() * rb).magnitude()))
    return pos_mm, rot_deg


def _image_stats(frame: np.ndarray) -> str:
    image = np.asarray(frame)
    finite = bool(np.all(np.isfinite(image)))
    if image.ndim == 3 and image.shape[-1] == 3:
        channel_mean = image.astype(np.float64).mean(axis=(0, 1))
    else:
        channel_mean = np.asarray([], dtype=np.float64)
    return (
        f"shape={image.shape} dtype={image.dtype} finite={finite} "
        f"min={float(np.nanmin(image)):.3f} max={float(np.nanmax(image)):.3f} "
        f"mean={float(np.nanmean(image)):.3f} std={float(np.nanstd(image)):.3f} "
        f"channel_mean={_array_text(channel_mean, 3)}"
    )


def _chain_incremental_actions(actions: np.ndarray, anchor_pose6: np.ndarray) -> np.ndarray:
    """Diagnostic-only sequential composition; it is not returned to the controller."""
    action_array = np.asarray(actions, dtype=np.float64)
    current = pose_to_mat(np.asarray(anchor_pose6, dtype=np.float64).reshape(6))
    world = np.empty((len(action_array), 7), dtype=np.float64)
    for index, action in enumerate(action_array):
        current = current @ pose10d_to_mat(action[:9])
        world[index, :6] = np.asarray(mat_to_pose(current), dtype=np.float64)
        world[index, 6] = float(action[9])
    return world


def _log_policy_input(
    history,
    *,
    current_frame: np.ndarray,
    current_pose6: np.ndarray,
    gripper_width: float,
    prompt: str,
) -> None:
    _debug.inference_index += 1
    current = np.asarray(current_pose6, dtype=np.float64).reshape(6)
    relative = _v1.relative_pose10(history.episode_start_pose6, current)
    frame = np.asarray(current_frame)
    print(f"[TRACE][INFER:{_debug.inference_index}] prompt={prompt!r}")
    print(f"[TRACE][INFER:{_debug.inference_index}] episode_start_link6={_array_text(history.episode_start_pose6)}")
    print(f"[TRACE][INFER:{_debug.inference_index}] current_link6={_array_text(current)}")
    print(f"[TRACE][INFER:{_debug.inference_index}] state10={_array_text(np.r_[relative, gripper_width])}")
    print(f"[TRACE][INFER:{_debug.inference_index}] current_image {_image_stats(frame)}")
    if _debug.previous_current_frame is not None and _debug.previous_current_frame.shape == frame.shape:
        difference = np.abs(frame.astype(np.float32) - _debug.previous_current_frame.astype(np.float32))
        print(
            f"[TRACE][INFER:{_debug.inference_index}] current_image_vs_previous "
            f"mae={float(difference.mean()):.4f} max={float(difference.max()):.1f} "
            f"changed_fraction={float(np.mean(difference > 1.0)):.6f}"
        )
    _debug.previous_current_frame = frame.copy()


_original_infer_step = _v1.ShellGamePolicyClient.infer_step


def _debug_infer_step(
    self,
    history,
    *,
    current_frame: np.ndarray,
    current_pose6: np.ndarray,
    gripper_width: float,
    prompt: str,
):
    _log_policy_input(
        history,
        current_frame=current_frame,
        current_pose6=current_pose6,
        gripper_width=gripper_width,
        prompt=prompt,
    )
    started = time.monotonic()
    result = _original_infer_step(
        self,
        history,
        current_frame=current_frame,
        current_pose6=current_pose6,
        gripper_width=gripper_width,
        prompt=prompt,
    )
    elapsed = time.monotonic() - started
    actions = np.asarray(result.get("actions"), dtype=np.float64)
    print(f"[TRACE][INFER:{_debug.inference_index}] response_time={elapsed:.6f}s shape={actions.shape}")
    if actions.ndim == 2 and actions.shape[1] >= 10:
        xyz = actions[:, :3]
        xyz_norm = np.linalg.norm(xyz, axis=-1)
        print(f"[TRACE][INFER:{_debug.inference_index}] raw_actions_all=\n{_array_text(actions)}")
        print(
            f"[TRACE][INFER:{_debug.inference_index}] raw_xyz_norm_mm="
            f"{_array_text(xyz_norm * 1000.0, 3)} min/mean/max="
            f"{float(xyz_norm.min()*1000):.3f}/"
            f"{float(xyz_norm.mean()*1000):.3f}/"
            f"{float(xyz_norm.max()*1000):.3f}"
        )
        print(
            f"[TRACE][INFER:{_debug.inference_index}] raw_xyz_sum_local="
            f"{_array_text(xyz.sum(axis=0))} gripper="
            f"{_array_text(actions[:, 9])}"
        )
    if "memory" in result:
        print(f"[TRACE][INFER:{_debug.inference_index}] memory={result['memory']}")
    return result


_v1.ShellGamePolicyClient.infer_step = _debug_infer_step


_original_upload_history = _v1.ShellGamePolicyClient.upload_history


def _debug_upload_history(self, history, *, prompt: str):
    frames = np.stack(history.frames(), axis=0)
    temporal_mae = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
    print(f"[TRACE][HISTORY] frames={frames.shape}/{frames.dtype} prompt={prompt!r}")
    print(f"[TRACE][HISTORY] frame0 {_image_stats(frames[0])}")
    print(f"[TRACE][HISTORY] frame240 {_image_stats(frames[-1])}")
    print(
        f"[TRACE][HISTORY] adjacent_frame_mae min/mean/max="
        f"{float(temporal_mae.min()):.4f}/"
        f"{float(temporal_mae.mean()):.4f}/"
        f"{float(temporal_mae.max()):.4f} "
        f"near_duplicate_fraction={float(np.mean(temporal_mae < 0.05)):.4f}"
    )
    return _original_upload_history(self, history, prompt=prompt)


_v1.ShellGamePolicyClient.upload_history = _debug_upload_history


_original_relative_chunk_to_world = _v1.relative_chunk_to_world


def _debug_relative_chunk_to_world(action_chunk: np.ndarray, anchor_pose6: np.ndarray) -> np.ndarray:
    """Compare both semantics, but return the original common-anchor result."""
    actions = np.asarray(action_chunk, dtype=np.float64)
    anchor = np.asarray(anchor_pose6, dtype=np.float64).reshape(6)
    common = _original_relative_chunk_to_world(actions, anchor)
    chained = _chain_incremental_actions(actions, anchor)
    common_delta = common[:, :3] - anchor[:3]
    chained_delta = chained[:, :3] - anchor[:3]
    common_steps = np.linalg.norm(np.diff(np.vstack([anchor[:3], common[:, :3]]), axis=0), axis=-1)
    chained_steps = np.linalg.norm(np.diff(np.vstack([anchor[:3], chained[:, :3]]), axis=0), axis=-1)
    axes_world = _v1.Rotation.from_rotvec(anchor[3:6]).as_matrix()
    print(f"[TRACE][DECODE:{_debug.inference_index}] anchor_pose6={_array_text(anchor)}")
    print(f"[TRACE][DECODE:{_debug.inference_index}] anchor_rotation_matrix(local_axes_columns)=\n{_array_text(axes_world)}")
    print(f"[TRACE][DECODE:{_debug.inference_index}] common_anchor_delta_xyz=\n{_array_text(common_delta)}")
    print(f"[TRACE][DECODE:{_debug.inference_index}] chained_delta_xyz_DIAGNOSTIC_ONLY=\n{_array_text(chained_delta)}")
    print(
        f"[TRACE][DECODE:{_debug.inference_index}] common_step_norm_mm="
        f"{_array_text(common_steps*1000.0, 3)} endpoint_mm="
        f"{float(np.linalg.norm(common_delta[-1])*1000.0):.3f}"
    )
    print(
        f"[TRACE][DECODE:{_debug.inference_index}] chained_step_norm_mm_DIAGNOSTIC_ONLY="
        f"{_array_text(chained_steps*1000.0, 3)} endpoint_mm="
        f"{float(np.linalg.norm(chained_delta[-1])*1000.0):.3f}"
    )
    print("[TRACE][DECODE] controller_return=COMMON_ANCHOR (original eval_arx5_pi_hzl.py behavior)")
    return common


_v1.relative_chunk_to_world = _debug_relative_chunk_to_world


_original_select_safe_waypoint_count = _v1.select_safe_waypoint_count


def _debug_select_safe_waypoint_count(
    world_actions: np.ndarray,
    *,
    anchor_pose6: np.ndarray,
    dt: float,
    max_pos_speed: float,
    max_rot_speed: float,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    max_waypoints: int,
):
    actions = np.asarray(world_actions, dtype=np.float64)
    anchor = np.asarray(anchor_pose6, dtype=np.float64).reshape(6)
    selected = actions[:max_waypoints]
    poses = np.vstack([anchor, selected[:, :6]])
    pos_steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=-1)
    rotations = _v1.Rotation.from_rotvec(poses[:, 3:6])
    rot_steps = np.asarray(
        [(rotations[i].inv() * rotations[i + 1]).magnitude() for i in range(len(rotations) - 1)]
    )
    print(
        f"[TRACE][SAFETY:{_debug.inference_index}] requested={max_waypoints} dt={dt:.6f}s "
        f"pos_limit={max_pos_speed*dt:.6f}m rot_limit={max_rot_speed*dt:.6f}rad"
    )
    print(f"[TRACE][SAFETY:{_debug.inference_index}] position_steps_mm={_array_text(pos_steps*1000.0, 3)}")
    print(f"[TRACE][SAFETY:{_debug.inference_index}] rotation_steps_deg={_array_text(np.degrees(rot_steps), 3)}")
    print(
        f"[TRACE][SAFETY:{_debug.inference_index}] xyz_min={_array_text(selected[:,:3].min(axis=0))} "
        f"xyz_max={_array_text(selected[:,:3].max(axis=0))} "
        f"workspace=[{_array_text(workspace_min)}, {_array_text(workspace_max)}]"
    )
    result = _original_select_safe_waypoint_count(
        world_actions,
        anchor_pose6=anchor_pose6,
        dt=dt,
        max_pos_speed=max_pos_speed,
        max_rot_speed=max_rot_speed,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        max_waypoints=max_waypoints,
    )
    _debug.expect_policy_exec = result[0] > 0
    print(f"[TRACE][SAFETY:{_debug.inference_index}] safe_count={result[0]} reason={result[1]!r}")
    return result


_v1.select_safe_waypoint_count = _debug_select_safe_waypoint_count


_original_get_obs = Arx5Env.get_obs


def _debug_get_obs(self, *args, **kwargs):
    obs = _original_get_obs(self, *args, **kwargs)
    try:
        pose6 = np.concatenate(
            [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]], axis=-1
        ).astype(np.float64)
        gripper = float(np.asarray(obs["robot0_gripper_width"][-1]).reshape(-1)[0])
        _debug.last_obs_pose6 = pose6
        _debug.last_obs_gripper = gripper
        pending = _debug.pending_command
        if pending is not None and time.time() >= pending["end_timestamp"] - 0.005:
            start_pose = pending["start_pose6"]
            target_pose = pending["target_pose6"]
            moved_mm, moved_rot_deg = _pose_error(pose6, start_pose)
            target_err_mm, target_err_rot_deg = _pose_error(pose6, target_pose)
            command_mm, command_rot_deg = _pose_error(target_pose, start_pose)
            print(
                f"[TRACE][TRACK:{pending['inference_index']}] commanded="
                f"{command_mm:.3f}mm/{command_rot_deg:.3f}deg actual="
                f"{moved_mm:.3f}mm/{moved_rot_deg:.3f}deg final_error="
                f"{target_err_mm:.3f}mm/{target_err_rot_deg:.3f}deg"
            )
            print(f"[TRACE][TRACK:{pending['inference_index']}] start_pose6={_array_text(start_pose)}")
            print(f"[TRACE][TRACK:{pending['inference_index']}] target_pose6={_array_text(target_pose)}")
            print(f"[TRACE][TRACK:{pending['inference_index']}] measured_pose6={_array_text(pose6)}")
            print(
                f"[TRACE][TRACK:{pending['inference_index']}] gripper target/measured="
                f"{pending['target_gripper']:.6f}/{gripper:.6f}"
            )
            _debug.pending_command = None
    except Exception as exc:  # noqa: BLE001
        print(f"[TRACE][TRACK][ERROR] failed to inspect observation: {exc!r}")
    return obs


Arx5Env.get_obs = _debug_get_obs


_original_exec_actions = Arx5Env.exec_actions


def _debug_exec_actions(self, *args, **kwargs):
    actions_arg = kwargs.get("actions", args[0] if args else None)
    timestamps_arg = kwargs.get("timestamps", args[1] if len(args) > 1 else None)
    actions = np.asarray(actions_arg, dtype=np.float64)
    timestamps = np.asarray(timestamps_arg, dtype=np.float64)
    is_policy = bool(_debug.expect_policy_exec)
    if is_policy:
        now = time.time()
        print(f"[TRACE][SUBMIT:{_debug.inference_index}] actions=\n{_array_text(actions)}")
        print(
            f"[TRACE][SUBMIT:{_debug.inference_index}] now={now:.6f} timestamps="
            f"{_array_text(timestamps)} lead_ms={_array_text((timestamps-now)*1000.0, 3)} "
            f"fresh={_array_text(timestamps > now)}"
        )
        if len(timestamps) > 1:
            print(
                f"[TRACE][SUBMIT:{_debug.inference_index}] timestamp_dt_ms="
                f"{_array_text(np.diff(timestamps)*1000.0, 3)}"
            )
        if _debug.last_obs_pose6 is not None and len(actions):
            target_pose = actions[-1, :6].copy()
            command_mm, command_rot_deg = _pose_error(target_pose, _debug.last_obs_pose6)
            print(
                f"[TRACE][SUBMIT:{_debug.inference_index}] endpoint_command="
                f"{command_mm:.3f}mm/{command_rot_deg:.3f}deg"
            )
            _debug.pending_command = {
                "inference_index": _debug.inference_index,
                "start_pose6": _debug.last_obs_pose6.copy(),
                "target_pose6": target_pose,
                "target_gripper": float(actions[-1, 6]),
                "end_timestamp": float(timestamps[-1]),
            }
        _debug.expect_policy_exec = False
    return _original_exec_actions(self, *args, **kwargs)


Arx5Env.exec_actions = _debug_exec_actions


def _print_debug_header() -> None:
    print("=" * 88)
    print(f"[TRACE][START] log_path={LOG_PATH}")
    print(f"[TRACE][START] timestamp={dt.datetime.now().isoformat()}")
    print(f"[TRACE][START] host={socket.gethostname()} pid={os.getpid()}")
    print(f"[TRACE][START] argv={sys.argv!r}")
    print(
        f"[TRACE][CONTRACT] history={_v1.HISTORY_FRAMES} total_frames={_v1.TOTAL_MODEL_FRAMES} "
        f"action_horizon={_v1.ACTION_HORIZON} action_dim={_v1.ACTION_DIM}"
    )
    print("[TRACE][CONTRACT] CONTROL BEHAVIOR: original COMMON-ANCHOR decoder")
    print("[TRACE][CONTRACT] CHAINED decoder is logged for comparison only and is never submitted")
    print("=" * 88)


if __name__ == "__main__":
    _print_debug_header()
    _v1.main()
