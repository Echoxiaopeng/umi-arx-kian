#!/usr/bin/env python3
"""Kai0 ARX5 bimanual eval — RTC v2 (official async + StreamActionBuffer).

Aligns client scheduling with kai0
``train_deploy_alignment/inference/arx/inference/arx_openpi_inference_rtc.py``:

- background inference thread at ``--inference-rate`` (default 4 Hz);
- control loop at ``--frequency`` pops **one** action per tick from a buffer;
- new chunks are latency-trimmed and temporally blended into the buffer;
- RTC payload uses ``execute_horizon`` default = chunk size (not soft_min pad)
  and ``inference_delay`` from median RTT × control frequency.

Still uses this repo's ZMQ/camera stack (not ROS2).

Example (policy server must be started with ``--rtc``)::

# 效果不错，很平滑。
# C→S → data_local/kai0_eval/model_traj/replay_buffer.zarr（采集同款，无 videos）
conda activate umi-arx
python scripts/eval_arx5_kai0_bimanual_rtc_v2.py \
  --policy-ip 127.0.0.1 --policy-port 8000 \
  --robot-ports 8765,8766 --camera-reorder 0,1 \
  --frequency 20 --inference-rate 4 \
  --rtc-max-guidance-weight 0.5 \
  --min-smooth-steps 10 --latency-k 8 \
  --dispatch tick \
  --gripper-close-threshold 0.03 \
  --chunk-size 50 \
  --output data_local/kai0_eval

``--dispatch schedule`` (default): timed ZMQ waypoints via interpolator.
``--dispatch tick``: ROS-style immediate SET_JOINT_POSITION each control tick.

Controls: C=start, S=stop/hold, Enter=return to init (after stop), Q=home then quit.

DAgger (``--dagger``)::

    python scripts/eval_arx5_kai0_bimanual_rtc_v2.py \
      --policy-ip 127.0.0.1 --policy-port 8000 \
      --robot-ports 8765,8766 --camera-reorder 0,1 \
      --frequency 20 --inference-rate 4 --dispatch tick \
      --dagger --output /media/lenovo/data/kian/kai0_dagger

    # traj also under data_local/kai0_dagger/model_traj/

Controls (--dagger): C=start episode+policy, D=drag intervene, R=resume policy,
  S=save episode then return to init, B=abort episode, Q=quit (aborts open episode).

Live dashboard (default on): http://127.0.0.1:8090/  (--no-web to disable).

Model trajectory dump (default on): each C→S episode appends to
``{output}/model_traj/replay_buffer.zarr`` in collect-style layout
(``timestamp``, ``action``, ``camera*_rgb``, robot joints, plus
``buffer_act`` / ``ctrl_mode``). Raw infer chunks go to
``model_traj/infer/episode_XXXX.npz``. No ``videos/`` (RGB is in zarr).
Disable with ``--no-save-model-traj``; skip images with ``--no-save-traj-images``.
"""

from __future__ import annotations

from collections import deque
import importlib.util
import json
from multiprocessing.managers import SharedMemoryManager
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import click
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_LOG = ROOT / "data_local" / "kai0_eval" / "last_run_rtc_v2.log"
DEFAULT_DAGGER_OUTPUT = ROOT / "data_local" / "kai0_dagger"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
if CLIENT_SRC.is_dir():
    sys.path.insert(0, str(CLIENT_SRC))

from modules.arx5_env_with_images_bimanual_rs import Arx5EnvWithImages  # noqa: E402
from modules.eval_kai0_web_ui import (  # noqa: E402
    CommandQueue,
    SharedEvalState,
    make_progress_cb,
    obs_to_jpeg,
    start_eval_web_ui,
)
from modules.replay_buffer import ReplayBuffer  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from utils.kai0_dagger_drag import abort_episode  # noqa: E402
from utils.kai0_dagger_drag import enter_drag_mode  # noqa: E402
from utils.kai0_dagger_drag import leave_drag_stay_pose  # noqa: E402
from utils.kai0_dagger_drag import record_pose_action  # noqa: E402
from utils.kai0_dagger_drag import save_intervention_sidecar  # noqa: E402
from utils.kai0_rtc import StreamActionBuffer  # noqa: E402
from utils.other_util import precise_wait  # noqa: E402

# Reuse helpers / logging / init from the v1 eval module without running its CLI.
_V1_PATH = ROOT / "scripts" / "eval_arx5_kai0_bimanual.py"
_spec = importlib.util.spec_from_file_location("kai0_bimanual_eval_v1", _V1_PATH)
assert _spec is not None and _spec.loader is not None
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

ACTION_DIM = 14
DEFAULT_PROMPT = _v1.DEFAULT_PROMPT
DEFAULT_INIT_JOINTS = _v1.DEFAULT_INIT_JOINTS
# Joint-space home (ARX --joint): arm q → 0, grippers open-ish at 0.
DEFAULT_HOME_JOINTS14 = np.zeros(ACTION_DIM, dtype=np.float64)


class _DelayTracker:
    """Official-style median RTT → delay steps."""

    def __init__(self, maxlen: int = 20) -> None:
        self._buf: deque[float] = deque(maxlen=int(maxlen))
        self.pred_delay_steps = 0
        self.last_rtt_s = 0.0

    def update(self, rtt_s: float, control_hz: float) -> int:
        if rtt_s is None or not np.isfinite(rtt_s):
            return self.pred_delay_steps
        self.last_rtt_s = float(rtt_s)
        self._buf.append(float(rtt_s))
        median_rtt = float(np.median(np.asarray(self._buf, dtype=np.float64)))
        self.pred_delay_steps = int(max(0, round(median_rtt * float(control_hz))))
        return self.pred_delay_steps


_CTRL_MODE_TO_U8 = {
    "POLICY": np.uint8(0),
    "DRAG": np.uint8(1),
    "HOLD": np.uint8(2),
    "EMPTY": np.uint8(3),
}


def _obs_cam_rgb(obs: dict, cam_index: int) -> np.ndarray:
    """Latest RGB frame for camera{i} as uint8 HxWx3 (RGB)."""
    key = f"camera{cam_index}_rgb"
    if key not in obs:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    frame = np.asarray(obs[key][-1])
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and float(np.nanmax(frame)) <= 1.5:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return np.ascontiguousarray(frame[..., :3])


def _obs_timestamp(obs: dict | None) -> float:
    if obs is None or "timestamp" not in obs:
        return float(time.time())
    ts = np.asarray(obs["timestamp"], dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return float(time.time())
    return float(ts[-1])


class ModelTrajRecorder:
    """Collect-style zarr dump (no videos/) + infer-chunk sidecar npz."""

    def __init__(
        self,
        out_dir: Path,
        *,
        meta: dict[str, Any] | None = None,
        enabled: bool = True,
        save_images: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.save_images = bool(save_images) and self.enabled
        self.out_dir = Path(out_dir)
        self.infer_dir = self.out_dir / "infer"
        self.zarr_path = self.out_dir / "replay_buffer.zarr"
        self.meta: dict[str, Any] = dict(meta or {})
        self._lock = threading.Lock()
        self._active = False
        self._episode_id: int | None = None
        self._t0 = 0.0
        self._replay: ReplayBuffer | None = None
        self._clear_buffers()
        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.infer_dir.mkdir(parents=True, exist_ok=True)
            self._replay = ReplayBuffer.create_from_path(
                zarr_path=str(self.zarr_path.resolve()), mode="a"
            )
            run_meta_path = self.out_dir / "run.meta.json"
            if not run_meta_path.is_file():
                run_meta_path.write_text(
                    json.dumps(self.meta, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    def _clear_buffers(self) -> None:
        self._infer_t: list[float] = []
        self._infer_id: list[int] = []
        self._infer_rtt: list[float] = []
        self._infer_delay: list[int] = []
        self._infer_exec_h: list[int] = []
        self._raw_chunks: list[np.ndarray] = []
        self._timestamp: list[float] = []
        self._ctrl_step: list[int] = []
        self._state: list[np.ndarray] = []
        self._buffer_act: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._ctrl_mode: list[np.uint8] = []
        self._cam0: list[np.ndarray] = []
        self._cam1: list[np.ndarray] = []
        self._cam2: list[np.ndarray] = []

    @property
    def active(self) -> bool:
        return self._active

    def start_episode(self, episode_id: int | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._clear_buffers()
            self._active = True
            self._t0 = time.time()
            # Prefer next id in the traj zarr so numbering stays continuous.
            if self._replay is not None:
                self._episode_id = int(self._replay.n_episodes)
            elif episode_id is None:
                self._episode_id = 0
            else:
                self._episode_id = int(episode_id)

    def record_infer(
        self,
        *,
        actions: np.ndarray,
        infer_id: int,
        rtt_s: float,
        delay_steps: int,
        exec_h: int,
        obs: dict | None = None,
        t_mono: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        chunk = np.asarray(actions, dtype=np.float32)
        t_wall = _obs_timestamp(obs) if obs is not None else float(time.time())
        with self._lock:
            if not self._active:
                return
            # Prefer wall/obs time for alignment with ctrl timestamps.
            self._infer_t.append(
                float(t_wall if t_mono is None else t_mono)
            )
            self._infer_id.append(int(infer_id))
            self._infer_rtt.append(float(rtt_s))
            self._infer_delay.append(int(delay_steps))
            self._infer_exec_h.append(int(exec_h))
            self._raw_chunks.append(chunk.copy())

    def record_ctrl(
        self,
        *,
        step: int,
        state14: np.ndarray,
        buffer_act14: np.ndarray,
        guarded14: np.ndarray,
        mode: str = "POLICY",
        obs: dict | None = None,
        t_mono: float | None = None,  # noqa: ARG002
    ) -> None:
        if not self.enabled:
            return
        # Copy image buffers outside the lock (get_obs may reuse arrays).
        cam0 = cam1 = cam2 = None
        if self.save_images and obs is not None:
            cam0 = _obs_cam_rgb(obs, 0).copy()
            cam1 = _obs_cam_rgb(obs, 1).copy()
            cam2 = _obs_cam_rgb(obs, 2).copy()
        ts = _obs_timestamp(obs)
        with self._lock:
            if not self._active:
                return
            self._timestamp.append(float(ts))
            self._ctrl_step.append(int(step))
            self._state.append(
                np.asarray(state14, dtype=np.float32).reshape(ACTION_DIM).copy()
            )
            self._buffer_act.append(
                np.asarray(buffer_act14, dtype=np.float32)
                .reshape(ACTION_DIM)
                .copy()
            )
            self._action.append(
                np.asarray(guarded14, dtype=np.float32)
                .reshape(ACTION_DIM)
                .copy()
            )
            self._ctrl_mode.append(
                _CTRL_MODE_TO_U8.get(str(mode).upper(), np.uint8(255))
            )
            if cam0 is not None and cam1 is not None and cam2 is not None:
                self._cam0.append(cam0)
                self._cam1.append(cam1)
                self._cam2.append(cam2)

    def discard_episode(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active = False
            self._clear_buffers()
            self._episode_id = None

    def finish_episode(self) -> Path | None:
        """Flush to collect-style zarr (+ infer sidecar). Returns zarr path."""
        if not self.enabled or self._replay is None:
            return None
        with self._lock:
            if not self._active:
                return None
            n_infer = len(self._raw_chunks)
            n_ctrl = len(self._action)
            if n_infer == 0 and n_ctrl == 0:
                self._active = False
                self._clear_buffers()
                self._episode_id = None
                return None

            ep_id = int(
                self._replay.n_episodes
                if self._episode_id is None
                else self._episode_id
            )
            meta = dict(self.meta)
            meta.update(
                {
                    "episode_id": ep_id,
                    "n_infer": n_infer,
                    "n_ctrl": n_ctrl,
                    "save_images": bool(self.save_images),
                    "t_start": float(self._t0),
                    "t_end": float(time.time()),
                    "layout": "collect_zarr_no_video",
                    "action_space": "joint14_guarded",
                    "ctrl_mode": {
                        "0": "POLICY",
                        "1": "DRAG",
                        "2": "HOLD",
                        "3": "EMPTY",
                    },
                }
            )

            if n_ctrl > 0:
                state = np.stack(self._state, axis=0)
                episode: dict[str, np.ndarray] = {
                    "timestamp": np.asarray(self._timestamp, dtype=np.float64),
                    "action": np.stack(self._action, axis=0).astype(np.float32),
                    "buffer_act": np.stack(self._buffer_act, axis=0).astype(
                        np.float32
                    ),
                    "state": state,
                    "ctrl_step": np.asarray(self._ctrl_step, dtype=np.int32),
                    "ctrl_mode": np.asarray(self._ctrl_mode, dtype=np.uint8),
                    # Collect-compatible joint views (from measured state14).
                    "robot0_joint_pos": state[:, 0:6].astype(np.float32),
                    "robot0_gripper_width": state[:, 6:7].astype(np.float32),
                    "robot1_joint_pos": state[:, 7:13].astype(np.float32),
                    "robot1_gripper_width": state[:, 13:14].astype(np.float32),
                }
                if self.save_images and len(self._cam0) == n_ctrl:
                    episode["camera0_rgb"] = np.stack(self._cam0, axis=0)
                    episode["camera1_rgb"] = np.stack(self._cam1, axis=0)
                    episode["camera2_rgb"] = np.stack(self._cam2, axis=0)
                self._replay.add_episode(episode, compressors="disk")
                ep_id = int(self._replay.n_episodes - 1)
                meta["episode_id"] = ep_id

            if n_infer > 0:
                raw_chunks = np.empty(n_infer, dtype=object)
                for i, chunk in enumerate(self._raw_chunks):
                    raw_chunks[i] = chunk
                infer_path = self.infer_dir / f"episode_{ep_id:04d}.npz"
                np.savez_compressed(
                    infer_path,
                    raw_chunks=raw_chunks,
                    infer_t=np.asarray(self._infer_t, dtype=np.float64),
                    infer_id=np.asarray(self._infer_id, dtype=np.int32),
                    infer_rtt=np.asarray(self._infer_rtt, dtype=np.float64),
                    infer_delay=np.asarray(self._infer_delay, dtype=np.int32),
                    infer_exec_h=np.asarray(self._infer_exec_h, dtype=np.int32),
                    meta_json=json.dumps(meta, ensure_ascii=False),
                )
                meta["infer_sidecar"] = str(infer_path.name)

            meta_path = self.out_dir / f"episode_{ep_id:04d}.meta.json"
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            self._active = False
            self._clear_buffers()
            self._episode_id = None
            return self.zarr_path


def _tighten_grippers(
    action: np.ndarray,
    *,
    gripper_min: float,
    gripper_max: float,
    close_threshold: float,
    grasp_width: float,
    enabled: bool,
) -> np.ndarray:
    """Kai0/ZMQ: larger = more open. Soft policy closes → snap to grasp_width."""
    out = np.asarray(action, dtype=np.float64).copy()
    if not enabled:
        out[6] = float(np.clip(out[6], gripper_min, gripper_max))
        out[13] = float(np.clip(out[13], gripper_min, gripper_max))
        return out
    thr = float(close_threshold)
    target = float(grasp_width)
    exec_min = max(0.0, min(float(gripper_min), target))
    for idx in (6, 13):
        g = float(out[idx])
        if g < thr:
            g = target
        out[idx] = float(np.clip(g, exec_min, gripper_max))
    return out


def _policy_anchor(
    last_commanded: np.ndarray | None,
    measured: np.ndarray,
    *,
    reanchor_thresh: float = 0.08,
) -> tuple[np.ndarray, bool, float, float]:
    """Choose guard anchor; re-pin to measured if tracking lag is large.

    Using only ``last_commanded`` lets the client open-loop ratchet away from a
    stuck/slow arm (cmd-meas → 0.2+), then the servo catches up in one rush.
    """
    measured = np.asarray(measured, dtype=np.float64).reshape(ACTION_DIM)
    if last_commanded is None:
        return measured, False, 0.0, 0.0
    last = np.asarray(last_commanded, dtype=np.float64).reshape(ACTION_DIM)
    lag_l = float(np.max(np.abs(last[0:6] - measured[0:6])))
    lag_r = float(np.max(np.abs(last[7:13] - measured[7:13])))
    if max(lag_l, lag_r) >= float(reanchor_thresh):
        return measured, True, lag_l, lag_r
    return last, False, lag_l, lag_r


def _guard_one(
    action: np.ndarray,
    current: np.ndarray,
    *,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
    gripper_close_threshold: float = 0.04,
    gripper_grasp_width: float = 0.0,
    gripper_tighten: bool = True,
    max_joint_step_per_arm: tuple[float, float] | None = None,
) -> np.ndarray:
    # Allow grasp_width below gripper_min during the joint-rate guard clip.
    clip_lo = (
        min(float(gripper_min), float(gripper_grasp_width))
        if gripper_tighten
        else float(gripper_min)
    )
    clip_lo = max(0.0, clip_lo)
    out = _v1.guarded_chunk(
        np.asarray(action, dtype=np.float64).reshape(1, -1),
        current,
        count=1,
        max_joint_step=max_joint_step,
        gripper_min=clip_lo,
        gripper_max=gripper_max,
        max_joint_step_per_arm=max_joint_step_per_arm,
    )
    return _tighten_grippers(
        np.asarray(out[0], dtype=np.float64),
        gripper_min=gripper_min,
        gripper_max=gripper_max,
        close_threshold=gripper_close_threshold,
        grasp_width=gripper_grasp_width,
        enabled=gripper_tighten,
    )


def _reset_both_to_home(
    env: Arx5EnvWithImages,
    *,
    frequency: float,
    duration: float,
    gripper_min: float,
    gripper_max: float,
    on_progress=None,
) -> None:
    """Slow joint-space ease to home (zeros), not firmware RESET_TO_HOME.

    Firmware home is often too abrupt; reuse the same interpolator path as init.
    """
    print(
        f"[v2] Q: slow joint home over {float(duration):.1f}s "
        "(not firmware RESET_TO_HOME)...",
        flush=True,
    )
    target = DEFAULT_HOME_JOINTS14.copy()
    target[6] = float(np.clip(target[6], gripper_min, gripper_max))
    target[13] = float(np.clip(target[13], gripper_min, gripper_max))

    _v1.move_to_init_joints(
        env,
        target,
        frequency=frequency,
        duration=float(duration),
        gripper_min=gripper_min,
        gripper_max=gripper_max,
        on_progress=on_progress,
        progress_label="home",
    )


def _web_keys_from_cmds(
    cmd_queue: CommandQueue | None,
    *,
    dagger: bool = False,
) -> list[str]:
    if cmd_queue is None:
        return []
    out: list[str] = []
    for item in cmd_queue.drain():
        action = str(item.get("action", "")).lower()
        if action in {"c", "start"}:
            out.append("c")
        elif dagger and action in {"s", "save"}:
            out.append("s")
        elif (not dagger) and action in {"s", "stop"}:
            out.append("s")
        elif dagger and action in {"d", "drag"}:
            out.append("d")
        elif dagger and action in {"r", "resume"}:
            out.append("r")
        elif dagger and action in {"b", "abort"}:
            out.append("b")
        elif (not dagger) and action in {"init", "enter", "return_init"}:
            out.append("\n")
        elif action in {"q", "home", "quit"}:
            out.append("q")
    return out


def run_rtc_v2(
    *,
    env: Arx5EnvWithImages,
    policy: WebsocketClientPolicy,
    keys: _v1.TerminalKeys,
    prompt: str,
    frequency: float,
    inference_rate: float,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
    observe_only: bool,
    preview_enabled: bool,
    chunk_size: int,
    rtc_execute_horizon: int | None,
    rtc_max_guidance_weight: float,
    rtc_mask_prefix_delay: bool,
    latency_k: int,
    min_smooth_steps: int,
    disable_smoothing: bool,
    gripper_close_threshold: float = 0.04,
    gripper_grasp_width: float = 0.0,
    gripper_tighten: bool = True,
    dispatch: str = "schedule",
    skip_server_warmup: bool = False,
    init_q14: np.ndarray | None = None,
    init_duration: float = 2.5,
    home_duration: float = 5.0,
    web_state: SharedEvalState | None = None,
    web_cmds: CommandQueue | None = None,
    traj_recorder: ModelTrajRecorder | None = None,
) -> None:
    """Official-style: infer thread fills buffer; control pops 1 action / tick."""
    dt = 1.0 / float(frequency)
    infer_period = 1.0 / max(1e-6, float(inference_rate))
    exec_h = int(chunk_size if rtc_execute_horizon is None else rtc_execute_horizon)
    exec_h = max(1, min(exec_h, int(chunk_size)))
    smooth = "raw" if disable_smoothing else "temporal"
    use_immediate = str(dispatch).lower() == "tick"
    buffer = StreamActionBuffer(state_dim=ACTION_DIM, smooth_method=smooth)
    delay_tracker = _DelayTracker()
    prev_chunk_lock = threading.Lock()
    rtc_prev_chunk: np.ndarray | None = None
    stop_event = threading.Event()
    executing = False
    last_commanded: np.ndarray | None = None
    infer_count = 0
    step_count = 0
    empty_streak = 0
    last_infer_log = 0.0
    recorder = traj_recorder or ModelTrajRecorder(
        Path("."), enabled=False
    )

    def _hold() -> None:
        _v1.hold_current(env, immediate=use_immediate)

    def _get_obs_state() -> tuple[dict, np.ndarray]:
        obs = env.get_obs()
        state = _v1.measured_state(env)
        return obs, state

    def _infer_loop() -> None:
        nonlocal rtc_prev_chunk, infer_count, last_infer_log
        next_t = time.monotonic()
        while not stop_event.is_set():
            if not executing:
                time.sleep(0.02)
                next_t = time.monotonic()
                continue
            try:
                obs, state = _get_obs_state()
                with prev_chunk_lock:
                    prev = (
                        None
                        if rtc_prev_chunk is None
                        else np.asarray(rtc_prev_chunk, dtype=np.float32)
                    )
                delay = int(delay_tracker.pred_delay_steps)
                payload = _v1.rtc_policy_observation(
                    obs,
                    state,
                    prompt,
                    prev_action_chunk=prev,
                    inference_delay=delay,
                    execute_horizon=exec_h,
                    max_guidance_weight=rtc_max_guidance_weight,
                    enable_rtc=True,
                )
                # Official also exposes mask_prefix_delay on the wire when set.
                payload["mask_prefix_delay"] = bool(rtc_mask_prefix_delay)

                t0 = time.monotonic()
                result = policy.infer(payload)
                rtt = time.monotonic() - t0
                delay_tracker.update(rtt, frequency)
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                    raise RuntimeError(f"bad actions shape {actions.shape}")

                with prev_chunk_lock:
                    rtc_prev_chunk = np.asarray(actions, dtype=np.float32)
                buffer.integrate_new_chunk(
                    actions,
                    max_k=int(latency_k),
                    min_m=int(min_smooth_steps),
                )
                infer_count += 1
                recorder.record_infer(
                    actions=actions,
                    infer_id=infer_count,
                    rtt_s=rtt,
                    delay_steps=int(delay_tracker.pred_delay_steps),
                    exec_h=exec_h,
                    obs=obs,
                )
                now = time.monotonic()
                if now - last_infer_log > 1.0:
                    last_infer_log = now
                    print(
                        f"[v2/infer] #{infer_count} rtt={rtt:.3f}s "
                        f"delay={delay_tracker.pred_delay_steps} "
                        f"exec_h={exec_h} buf={len(buffer)} "
                        f"H={actions.shape[0]} first={np.round(actions[0], 3)}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[v2/infer] error: {exc}")

            next_t += infer_period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()

    # Same warmup as v1: plain + RTC guidance compile (unless skipped).
    obs, state = _get_obs_state()
    if skip_server_warmup:
        print(
            "[warmup] skipped (--skip-server-warmup); "
            "first C may be slow while the server compiles"
        )
        warm = policy.infer(_v1.policy_observation(obs, state, prompt))
        action_horizon = int(np.asarray(warm["actions"]).shape[0])
    else:
        # Warm RTC path with the same execute_horizon v2 will use at runtime.
        warm_exec_h = int(
            chunk_size if rtc_execute_horizon is None else rtc_execute_horizon
        )
        warm_exec_h = max(1, min(warm_exec_h, int(chunk_size)))
        action_horizon = _v1.warmup_policy_server(
            env=env,
            policy=policy,
            prompt=prompt,
            rtc=True,
            rtc_s_min=warm_exec_h,
            rtc_max_guidance_weight=rtc_max_guidance_weight,
        )
        obs, state = _get_obs_state()

    if chunk_size != action_horizon:
        print(
            f"[v2] NOTE: --chunk-size={chunk_size} vs model H={action_horizon}; "
            f"using model H for buffer/RTC horizon"
        )
        chunk_size = action_horizon
        exec_h = int(chunk_size if rtc_execute_horizon is None else rtc_execute_horizon)
        exec_h = max(1, min(exec_h, chunk_size))
    recorder.meta["chunk_size"] = int(chunk_size)
    recorder.meta["exec_h"] = int(exec_h)

    with prev_chunk_lock:
        rtc_prev_chunk = np.tile(
            state.astype(np.float32)[None, :], (chunk_size, 1)
        )

    soft_len = max(0, exec_h - delay_tracker.pred_delay_steps)
    print(
        f"[v2] RTC async+buffer: ctrl={frequency}Hz infer={inference_rate}Hz "
        f"H={chunk_size} exec_h={exec_h} soft≈{soft_len}+(grows with delay) "
        f"max_gw={rtc_max_guidance_weight} smooth={smooth} "
        f"min_m={min_smooth_steps} latency_k={latency_k} "
        f"dispatch={'tick/immediate' if use_immediate else 'schedule'}"
    )
    print(
        "Controls: C=start, S=stop/hold, "
        "Enter=return to init (after S), Q=home then quit"
    )
    progress_cb = make_progress_cb(web_state)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if web_state is not None:
            web_state.append_log(msg)

    def _publish_web(obs: dict, *, mode: str) -> None:
        if web_state is None:
            return
        try:
            joints = _v1.measured_state(env)
        except Exception:  # noqa: BLE001
            joints = np.zeros(ACTION_DIM, dtype=np.float64)
        web_state.update(
            mode=mode,
            step=step_count,
            infer_count=infer_count,
            buf_len=len(buffer),
            delay_steps=int(delay_tracker.pred_delay_steps),
            exec_h=int(exec_h),
            frequency=float(frequency),
            inference_rate=float(inference_rate),
            dispatch="tick" if use_immediate else "schedule",
            rtt_s=float(delay_tracker.last_rtt_s),
            joints14=joints,
            prompt=prompt,
            jpeg=obs_to_jpeg(obs),
        )

    def _reseed_prev_from_state() -> None:
        nonlocal rtc_prev_chunk
        state_now = _v1.measured_state(env)
        with prev_chunk_lock:
            rtc_prev_chunk = np.tile(
                state_now.astype(np.float32)[None, :], (chunk_size, 1)
            )

    if web_state is not None:
        web_state.update(
            mode="HOLD",
            frequency=float(frequency),
            inference_rate=float(inference_rate),
            dispatch="tick" if use_immediate else "schedule",
            exec_h=int(exec_h),
            prompt=prompt,
        )
        web_state.append_log("control loop ready")

    infer_thread = threading.Thread(
        target=_infer_loop, name="kai0-rtc-v2-infer", daemon=True
    )
    infer_thread.start()
    t_loop = time.monotonic()

    try:
        while True:
            obs = env.get_obs()
            for key in list(keys.read()) + _web_keys_from_cmds(web_cmds):
                if key == "q":
                    executing = False
                    stop_event.set()
                    buffer.clear()
                    if recorder.active:
                        saved = recorder.finish_episode()
                        if saved is not None:
                            _log(f"[traj] saved {saved}")
                    if web_state is not None:
                        web_state.update(mode="HOME")
                    try:
                        _reset_both_to_home(
                            env,
                            frequency=frequency,
                            duration=float(home_duration),
                            gripper_min=gripper_min,
                            gripper_max=gripper_max,
                            on_progress=progress_cb,
                        )
                    except Exception as exc:  # noqa: BLE001
                        _log(f"[v2] home on Q failed: {exc}; holding instead")
                        _hold()
                    if web_state is not None:
                        web_state.clear_progress()
                    return
                if key == "s":
                    executing = False
                    buffer.clear()
                    last_commanded = None
                    if recorder.active:
                        saved = recorder.finish_episode()
                        if saved is not None:
                            _log(f"[traj] saved {saved}")
                        else:
                            _log("[traj] empty episode; nothing saved")
                    _hold()
                    _log(
                        "Policy stopped; holding measured joints "
                        "(Enter → init, C → resume, Q → home+quit)"
                    )
                elif key in {"\n", "\r"}:
                    if executing:
                        _log("Ignore Enter while executing; press S first")
                    elif observe_only:
                        _log("Ignoring Enter: --observe-only")
                    elif init_q14 is None:
                        _log(
                            "No init joints configured "
                            "(started with --no-init); cannot return to init"
                        )
                    else:
                        _log("Returning to init joints...")
                        if web_state is not None:
                            web_state.update(mode="INIT")
                        try:
                            _v1.move_to_init_joints(
                                env,
                                init_q14,
                                frequency=frequency,
                                duration=float(init_duration),
                                gripper_min=gripper_min,
                                gripper_max=gripper_max,
                                on_progress=progress_cb,
                            )
                            last_commanded = None
                            buffer.clear()
                            _reseed_prev_from_state()
                            _log("At init; press C to start, Q to home+quit")
                        except Exception as exc:  # noqa: BLE001
                            _log(f"[v2] return-to-init failed: {exc}")
                            _hold()
                        if web_state is not None:
                            web_state.clear_progress()
                            web_state.update(mode="HOLD")
                elif key == "c":
                    if observe_only:
                        _log("Ignoring C: --observe-only")
                    else:
                        executing = True
                        empty_streak = 0
                        if not recorder.active:
                            recorder.start_episode()
                        mode = "tick" if use_immediate else "schedule"
                        _log(
                            f"Policy execution started (RTC v2 async+buffer, "
                            f"dispatch={mode})"
                        )

            mode_now = "EXEC" if executing else "HOLD"
            _publish_web(obs, mode=mode_now)
            _v1.show_preview(
                obs,
                f"V2 RTC {mode_now}  buf={len(buffer)}  "
                f"S:stop Enter:init Q:home",
                enabled=preview_enabled,
            )

            if not executing or observe_only:
                t_loop += 0.05
                precise_wait(t_loop)
                continue

            act = buffer.pop_next_action()
            if act is None:
                empty_streak += 1
                if empty_streak in {1, 20, 50} or empty_streak % 100 == 0:
                    print(f"[v2] buffer empty x{empty_streak} (waiting for infer)")
                t_loop += dt
                precise_wait(t_loop)
                continue

            empty_streak = 0
            try:
                measured = _v1.measured_state(env)
            except Exception:  # noqa: BLE001
                measured = (
                    last_commanded
                    if last_commanded is not None
                    else np.zeros(ACTION_DIM, dtype=np.float64)
                )
            anchor, reanchored, lag_l, lag_r = _policy_anchor(
                last_commanded, measured, reanchor_thresh=0.08
            )
            if reanchored:
                print(
                    f"[v2/WARN] reanchor to measured step={step_count} "
                    f"lag L={lag_l:.4f} R={lag_r:.4f}",
                    flush=True,
                )
            guarded = _guard_one(
                act,
                anchor,
                max_joint_step=max_joint_step,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                gripper_close_threshold=gripper_close_threshold,
                gripper_grasp_width=gripper_grasp_width,
                gripper_tighten=gripper_tighten,
            )
            if use_immediate:
                # Blend over one control period on the ~200 Hz interpolator.
                # Hard set_joint_position every 20 Hz tick causes step jitter.
                _v1.command_joint_immediate(env, guarded, duration_s=dt)
            else:
                target_t = max(time.time() + 0.02, t_loop + 0.02)
                _v1.command_joint_actions(
                    env,
                    guarded.reshape(1, -1),
                    np.asarray([target_t], dtype=np.float64),
                )
            last_commanded = guarded
            step_count += 1
            recorder.record_ctrl(
                step=step_count,
                state14=measured,
                buffer_act14=act,
                guarded14=guarded,
                mode="POLICY",
                obs=obs,
            )
            if step_count % 40 == 0:
                print(
                    f"[v2/ctrl] step={step_count} buf={len(buffer)} "
                    f"delay={delay_tracker.pred_delay_steps} "
                    f"infers={infer_count}"
                )

            t_loop += dt
            precise_wait(t_loop)
    finally:
        stop_event.set()
        infer_thread.join(timeout=2.0)
        if recorder.active:
            saved = recorder.finish_episode()
            if saved is not None:
                print(f"[traj] flushed on exit: {saved}", flush=True)
        print(
            f"[v2] session: steps={step_count} infers={infer_count} "
            f"last_delay={delay_tracker.pred_delay_steps}"
        )


def run_dagger_v2(
    *,
    env: Arx5EnvWithImages,
    policy: WebsocketClientPolicy,
    keys: _v1.TerminalKeys,
    prompt: str,
    frequency: float,
    inference_rate: float,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
    preview_enabled: bool,
    chunk_size: int,
    rtc_execute_horizon: int | None,
    rtc_max_guidance_weight: float,
    rtc_mask_prefix_delay: bool,
    latency_k: int,
    min_smooth_steps: int,
    disable_smoothing: bool,
    output_dir: Path,
    gripper_close_threshold: float = 0.04,
    gripper_grasp_width: float = 0.0,
    gripper_tighten: bool = True,
    dispatch: str = "schedule",
    skip_server_warmup: bool = False,
    init_q14: np.ndarray | None = None,
    init_duration: float = 2.5,
    web_state: SharedEvalState | None = None,
    web_cmds: CommandQueue | None = None,
    traj_recorder: ModelTrajRecorder | None = None,
) -> None:
    """RTC v2 async buffer + DAgger episode / drag intervene / UMI zarr."""
    dt = 1.0 / float(frequency)
    infer_period = 1.0 / max(1e-6, float(inference_rate))
    exec_h = int(chunk_size if rtc_execute_horizon is None else rtc_execute_horizon)
    exec_h = max(1, min(exec_h, int(chunk_size)))
    smooth = "raw" if disable_smoothing else "temporal"
    use_immediate = str(dispatch).lower() == "tick"
    buffer = StreamActionBuffer(state_dim=ACTION_DIM, smooth_method=smooth)
    delay_tracker = _DelayTracker()
    prev_chunk_lock = threading.Lock()
    rtc_prev_chunk: np.ndarray | None = None
    stop_event = threading.Event()
    executing = False
    recording = False
    drag_mode = False
    motion_busy = False  # True while return-to-init (ignore C/D/R/S/B)
    # Bump on every policy stop so in-flight infer results are discarded.
    action_epoch = 0
    intervention_labels: list[int] = []
    last_rotvecs: list[np.ndarray | None] = [None, None]
    last_commanded: np.ndarray | None = None
    infer_count = 0
    step_count = 0
    empty_streak = 0
    last_infer_log = 0.0
    # Soft resume after drag→policy (R) / after return-to-init→C.
    # Longer / gentler than before: logs showed soft-hold itself was fine, but
    # right after exit the first chunk chased mid-trajectory targets at nearly
    # full max_joint_step → felt like a rush on C.
    resume_soft_left = 0
    resume_soft_total = 40  # ~2.0s at 20 Hz
    resume_hold_ticks = 16  # pure hold portion of soft window
    resume_post_soft_left = 0
    resume_post_soft_total = 40  # ~2.0s taper (caps at ~0.55× max step)
    # True while pure-hold ticks run; on exit we flush buffer so we do not
    # start mid-chunk (popping during hold used to cause a C/R rush).
    soft_pure_hold_active = False
    # Jump / tracking warn thresholds (rad); tuned for overcurrent forensics.
    _warn_cmd_step = max(0.12, float(max_joint_step) * 1.5)
    _warn_track_err = 0.20
    _warn_guard_sat = max(1e-4, float(max_joint_step) * 0.9)
    _last_jump_warn_t = 0.0
    recorder = traj_recorder or ModelTrajRecorder(
        Path("."), enabled=False
    )

    def _hold() -> None:
        _v1.hold_current(env, immediate=use_immediate)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if web_state is not None:
            web_state.append_log(msg)

    def _safe_measured() -> np.ndarray | None:
        try:
            return np.asarray(_v1.measured_state(env), dtype=np.float64)
        except Exception:  # noqa: BLE001
            return None

    def _seed_buffer_anchor(state: np.ndarray | None = None) -> None:
        """After clear(), keep last_action so the next chunk blends from pose.

        Empty-buffer integrate without last_action installs the chunk raw and
        the first executable step can be far from measured (C-onset rush).
        """
        q = state if state is not None else _safe_measured()
        if q is None:
            return
        buffer.last_action = np.asarray(q, dtype=np.float64).copy()

    def _post_soft_step_limit(batches_left: int) -> float:
        """Gentler than v1: 0.05→0.55 of max_joint_step (not 0.20→1.00)."""
        total = max(1, int(resume_post_soft_total))
        left = max(0, int(batches_left))
        done = total - left
        t = float(np.clip(done / float(total), 0.0, 1.0))
        return float(max_joint_step) * (0.05 + 0.50 * t)

    def _arm_max_abs(delta: np.ndarray) -> tuple[float, float]:
        d = np.asarray(delta, dtype=np.float64).reshape(-1)
        if d.size < ACTION_DIM:
            return float(np.max(np.abs(d))) if d.size else 0.0, 0.0
        return float(np.max(np.abs(d[0:6]))), float(np.max(np.abs(d[7:13])))

    def _diag_snapshot(tag: str, **named: np.ndarray | None) -> None:
        """One-line pose dump for mode transitions (measured / cmd / act)."""
        parts = [f"[DAgger/diag] {tag} step={step_count} buf={len(buffer)}"]
        measured = named.get("measured")
        if measured is None:
            measured = _safe_measured()
        if measured is not None:
            parts.append(f"meas={_v1._fmt14(measured)}")
        for key in ("last_cmd", "act", "guarded", "anchor"):
            arr = named.get(key)
            if arr is None:
                continue
            parts.append(f"{key}={_v1._fmt14(arr)}")
        # Pairwise arm-max |Δ| for the most useful contrasts.
        pairs = (
            ("cmd-meas", named.get("last_cmd"), measured),
            ("act-meas", named.get("act"), measured),
            ("grd-meas", named.get("guarded"), measured),
            ("grd-cmd", named.get("guarded"), named.get("last_cmd")),
            ("act-grd", named.get("act"), named.get("guarded")),
            ("act-anc", named.get("act"), named.get("anchor")),
        )
        for label, a, b in pairs:
            if a is None or b is None:
                continue
            l_m, r_m = _arm_max_abs(np.asarray(a) - np.asarray(b))
            parts.append(f"|{label}| L={l_m:.4f} R={r_m:.4f}")
        _log(" ".join(parts))

    def _warn_if_jump(
        tag: str,
        *,
        prev_cmd: np.ndarray | None,
        new_cmd: np.ndarray,
        measured: np.ndarray | None,
        raw_act: np.ndarray | None = None,
        min_interval_s: float = 0.2,
    ) -> None:
        """Rate-limited WARN when cmd step or tracking looks dangerous."""
        nonlocal _last_jump_warn_t
        new_cmd = np.asarray(new_cmd, dtype=np.float64)
        reasons: list[str] = []
        if prev_cmd is not None:
            l_m, r_m = _arm_max_abs(new_cmd - np.asarray(prev_cmd))
            step = max(l_m, r_m)
            if step >= _warn_cmd_step:
                reasons.append(f"cmd_step L={l_m:.4f} R={r_m:.4f}")
        if measured is not None:
            l_m, r_m = _arm_max_abs(new_cmd - np.asarray(measured))
            track = max(l_m, r_m)
            if track >= _warn_track_err:
                reasons.append(f"cmd-meas L={l_m:.4f} R={r_m:.4f}")
            if prev_cmd is not None:
                l_m, r_m = _arm_max_abs(
                    np.asarray(prev_cmd) - np.asarray(measured)
                )
                if max(l_m, r_m) >= _warn_track_err:
                    reasons.append(f"track_lag L={l_m:.4f} R={r_m:.4f}")
        if raw_act is not None and prev_cmd is not None:
            # How hard the guard clipped this tick.
            guarded_delta = new_cmd - np.asarray(prev_cmd)
            raw_delta = np.asarray(raw_act) - np.asarray(prev_cmd)
            # Compare requested vs allowed step magnitude per arm.
            for arm, sl in (("L", slice(0, 6)), ("R", slice(7, 13))):
                req = float(np.max(np.abs(raw_delta[sl])))
                got = float(np.max(np.abs(guarded_delta[sl])))
                if req >= _warn_guard_sat and (req - got) > 1e-4:
                    reasons.append(f"guard_sat_{arm} req={req:.4f} got={got:.4f}")
        if not reasons:
            return
        now = time.monotonic()
        if now - _last_jump_warn_t < float(min_interval_s):
            return
        _last_jump_warn_t = now
        _log(
            f"[DAgger/WARN] {tag} step={step_count} "
            + " ".join(reasons)
            + f" lim_step={_warn_cmd_step:.3f} lim_track={_warn_track_err:.3f}"
        )
        _diag_snapshot(
            f"WARN:{tag}",
            measured=measured,
            last_cmd=prev_cmd,
            act=raw_act,
            guarded=new_cmd,
        )

    def _flush_keys() -> None:
        """Drop buffered terminal/web keys (e.g. C pressed during return-to-init)."""
        try:
            keys.read()
        except Exception:  # noqa: BLE001
            pass
        if web_cmds is not None:
            try:
                web_cmds.drain()
            except Exception:  # noqa: BLE001
                pass

    def _invalidate_policy_stream() -> None:
        """Stop accepting new cmds; drop buffer + any in-flight infer chunk."""
        nonlocal executing, action_epoch, last_commanded, soft_pure_hold_active
        nonlocal resume_soft_left, resume_post_soft_left
        was_exec = bool(executing)
        prev = (
            None
            if last_commanded is None
            else np.asarray(last_commanded, dtype=np.float64).copy()
        )
        executing = False
        action_epoch += 1
        buffer.clear()
        last_commanded = None
        soft_pure_hold_active = False
        resume_soft_left = 0
        resume_post_soft_left = 0
        if was_exec:
            _diag_snapshot(
                f"invalidate_policy epoch={action_epoch}",
                measured=_safe_measured(),
                last_cmd=prev,
            )

    def _get_obs_state() -> tuple[dict, np.ndarray]:
        obs = env.get_obs()
        state = _v1.measured_state(env)
        return obs, state

    def _reseed_prev_from_state() -> None:
        nonlocal rtc_prev_chunk
        state_now = _v1.measured_state(env)
        with prev_chunk_lock:
            rtc_prev_chunk = np.tile(
                state_now.astype(np.float32)[None, :], (chunk_size, 1)
            )

    def _record_tick(obs: dict) -> None:
        if not recording:
            return
        record_pose_action(env, obs, last_rotvecs=last_rotvecs)
        _v1._sync_intervention_labels(
            env, intervention_labels, 1 if drag_mode else 0
        )

    def _status_text() -> str:
        if motion_busy:
            return "INIT  (wait… ignore C until done)"
        if not recording:
            return "IDLE  C:start D:n/a S:n/a Q:quit"
        if drag_mode:
            return "DRAG  R:policy S:save B:abort Q:quit"
        if executing:
            return f"POLICY  buf={len(buffer)}  D:drag S:save B:abort Q:quit"
        return "HOLD  C:policy D:drag S:save B:abort Q:quit"

    def _mode_name() -> str:
        if motion_busy:
            return "INIT"
        if not recording:
            return "IDLE"
        if drag_mode:
            return "DRAG"
        if executing:
            return "POLICY"
        return "HOLD"

    def _stop_policy_motion() -> None:
        _invalidate_policy_stream()
        try:
            _hold()
        except Exception as exc:  # noqa: BLE001
            _log(f"hold warning: {exc}")

    def _infer_loop() -> None:
        nonlocal rtc_prev_chunk, infer_count, last_infer_log
        next_t = time.monotonic()
        while not stop_event.is_set():
            if not executing or drag_mode or motion_busy:
                time.sleep(0.02)
                next_t = time.monotonic()
                continue
            epoch_at_start = int(action_epoch)
            try:
                obs, state = _get_obs_state()
                with prev_chunk_lock:
                    prev = (
                        None
                        if rtc_prev_chunk is None
                        else np.asarray(rtc_prev_chunk, dtype=np.float32)
                    )
                delay = int(delay_tracker.pred_delay_steps)
                payload = _v1.rtc_policy_observation(
                    obs,
                    state,
                    prompt,
                    prev_action_chunk=prev,
                    inference_delay=delay,
                    execute_horizon=exec_h,
                    max_guidance_weight=rtc_max_guidance_weight,
                    enable_rtc=True,
                )
                payload["mask_prefix_delay"] = bool(rtc_mask_prefix_delay)
                t0 = time.monotonic()
                result = policy.infer(payload)
                rtt = time.monotonic() - t0
                # Drop result if policy was stopped / epoch advanced mid-infer.
                if (
                    int(action_epoch) != epoch_at_start
                    or not executing
                    or drag_mode
                    or motion_busy
                ):
                    continue
                delay_tracker.update(rtt, frequency)
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                    raise RuntimeError(f"bad actions shape {actions.shape}")
                with prev_chunk_lock:
                    rtc_prev_chunk = np.asarray(actions, dtype=np.float32)
                buffer.integrate_new_chunk(
                    actions,
                    max_k=int(latency_k),
                    min_m=int(min_smooth_steps),
                )
                infer_count += 1
                recorder.record_infer(
                    actions=actions,
                    infer_id=infer_count,
                    rtt_s=rtt,
                    delay_steps=int(delay_tracker.pred_delay_steps),
                    exec_h=exec_h,
                    obs=obs,
                )
                now = time.monotonic()
                if now - last_infer_log > 1.0:
                    last_infer_log = now
                    first = np.asarray(actions[0], dtype=np.float64)
                    d_l, d_r = _arm_max_abs(first - np.asarray(state))
                    print(
                        f"[dagger/infer] #{infer_count} rtt={rtt:.3f}s "
                        f"delay={delay_tracker.pred_delay_steps} "
                        f"buf={len(buffer)} "
                        f"|first-state| L={d_l:.4f} R={d_r:.4f} "
                        f"first={_v1._fmt14(first)}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[dagger/infer] error: {exc}")
            next_t += infer_period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()

    def _start_episode() -> None:
        nonlocal recording, drag_mode, executing, intervention_labels, last_rotvecs
        nonlocal last_commanded, empty_streak, resume_soft_left, resume_post_soft_left
        nonlocal soft_pure_hold_active
        if motion_busy:
            _log("[DAgger] Ignore C: still returning to init")
            return
        if recording:
            if drag_mode:
                _log("Already recording in drag; press R to resume policy")
            elif executing:
                # Mid-episode C while already in POLICY used to re-arm soft-hold
                # and snap cmd→measured (caused ~1 rad / 50 ms overcurrent).
                _log("Already in POLICY; ignore C (D=drag, S=save)")
                _diag_snapshot(
                    "ignore_C_while_POLICY",
                    measured=_safe_measured(),
                    last_cmd=last_commanded,
                )
            else:
                prev_cmd = (
                    None
                    if last_commanded is None
                    else np.asarray(last_commanded, dtype=np.float64).copy()
                )
                executing = True
                empty_streak = 0
                buffer.clear()
                _reseed_prev_from_state()
                # Same as non-dagger: no soft-hold on C (standstill onset caused
                # chase-of-future-waypoints rush). Soft remains for R after drag.
                resume_soft_left = 0
                resume_post_soft_left = 0
                soft_pure_hold_active = False
                try:
                    last_commanded = np.asarray(
                        _v1.measured_state(env), dtype=np.float64
                    )
                except Exception:  # noqa: BLE001
                    last_commanded = None
                _seed_buffer_anchor(last_commanded)
                _log("Policy execution resumed from HOLD (no soft; like eval C)")
                _diag_snapshot(
                    "C_resume_from_HOLD",
                    measured=last_commanded,
                    last_cmd=prev_cmd,
                )
            return
        episode_id = env.replay_buffer.n_episodes
        env.start_episode(time.time() + 0.5)
        recording = True
        drag_mode = False
        intervention_labels = []
        last_rotvecs = [None, None]
        empty_streak = 0
        buffer.clear()
        _reseed_prev_from_state()
        try:
            last_commanded = np.asarray(_v1.measured_state(env), dtype=np.float64)
        except Exception:  # noqa: BLE001
            last_commanded = None
        _seed_buffer_anchor(last_commanded)
        # Match non-dagger eval C: start from actions[0] immediately.
        # Soft-hold on C made us stand still then chase a future waypoint.
        resume_soft_left = 0
        resume_post_soft_left = 0
        soft_pure_hold_active = False
        executing = True
        recorder.start_episode(episode_id=int(episode_id))
        _log(
            f"[DAgger] Episode {episode_id} started (policy, no soft on C; "
            f"soft only after R)"
        )
        _diag_snapshot(
            f"C_start_ep{episode_id}",
            measured=last_commanded,
            last_cmd=last_commanded,
        )

    def _enter_drag() -> None:
        nonlocal drag_mode
        if motion_busy:
            _log("[DAgger] Ignore D: still returning to init")
            return
        if not recording:
            _log("Press C to start an episode before intervening")
            return
        if drag_mode:
            _log("Already in drag mode")
            return
        _diag_snapshot(
            "D_enter_before_stop",
            measured=_safe_measured(),
            last_cmd=last_commanded,
        )
        _stop_policy_motion()
        enter_drag_mode(env)
        drag_mode = True
        _log("[DAgger] Drag intervention ON (R=resume, S=save, B=abort)")
        _diag_snapshot("D_enter_after", measured=_safe_measured())

    def _resume_policy() -> None:
        nonlocal drag_mode, executing, last_commanded
        nonlocal resume_soft_left, resume_post_soft_left, empty_streak
        nonlocal soft_pure_hold_active
        if motion_busy:
            _log("[DAgger] Ignore R: still returning to init")
            return
        if not recording:
            _log("No active episode")
            return
        if not drag_mode:
            _log("Not in drag mode")
            return
        _diag_snapshot("R_before_leave_drag", measured=_safe_measured())
        leave_drag_stay_pose(env)
        drag_mode = False
        try:
            _hold()
        except Exception as exc:  # noqa: BLE001
            _log(f"post-drag hold warning: {exc}")
        time.sleep(0.25)
        try:
            _hold()
        except Exception as exc:  # noqa: BLE001
            _log(f"post-settle hold warning: {exc}")
        state = _v1.measured_state(env)
        last_commanded = np.asarray(state, dtype=np.float64).copy()
        buffer.clear()
        _seed_buffer_anchor(last_commanded)
        _reseed_prev_from_state()
        resume_soft_left = int(resume_soft_total)
        resume_post_soft_left = int(resume_post_soft_total)
        soft_pure_hold_active = False
        empty_streak = 0
        executing = True
        _log(
            f"[DAgger] Resumed policy (soft={resume_soft_left} "
            f"hold={resume_hold_ticks} post={resume_post_soft_left})"
        )
        _diag_snapshot(
            "R_resume_policy",
            measured=last_commanded,
            last_cmd=last_commanded,
        )

    def _return_to_init(*, reason: str) -> None:
        nonlocal last_commanded, motion_busy
        if init_q14 is None:
            _log(
                f"[DAgger] skip return-to-init after {reason} "
                "(no init joints; started with --no-init)"
            )
            return
        motion_busy = True
        _invalidate_policy_stream()
        _flush_keys()
        _log(
            f"[DAgger] Returning to init after {reason}... "
            "(ignore C until finished)"
        )
        _diag_snapshot(
            f"return_init_before:{reason}",
            measured=_safe_measured(),
            last_cmd=(
                None
                if init_q14 is None
                else np.asarray(init_q14, dtype=np.float64)
            ),
        )
        if web_state is not None:
            web_state.update(mode="INIT")
        try:
            # Let any in-flight infer finish and get discarded before moving.
            time.sleep(0.15)
            buffer.clear()
            _v1.move_to_init_joints(
                env,
                init_q14,
                frequency=frequency,
                duration=float(init_duration),
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                on_progress=make_progress_cb(web_state),
            )
            try:
                last_commanded = np.asarray(
                    _v1.measured_state(env), dtype=np.float64
                )
            except Exception:  # noqa: BLE001
                last_commanded = None
            buffer.clear()
            _reseed_prev_from_state()
            _log("[DAgger] At init; press C to start next episode")
            _diag_snapshot(
                f"return_init_after:{reason}",
                measured=last_commanded,
                last_cmd=np.asarray(init_q14, dtype=np.float64),
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"[DAgger] return-to-init failed: {exc}")
            _diag_snapshot(
                f"return_init_FAILED:{reason}",
                measured=_safe_measured(),
            )
            try:
                _hold()
            except Exception:  # noqa: BLE001
                pass
        finally:
            _flush_keys()
            motion_busy = False
            if web_state is not None:
                web_state.clear_progress()
                web_state.update(mode="IDLE")

    def _save_episode() -> None:
        nonlocal recording, drag_mode, intervention_labels, last_rotvecs
        if motion_busy:
            _log("[DAgger] Ignore S: still returning to init")
            return
        if not recording:
            _log("No active episode to save")
            return
        _invalidate_policy_stream()
        if drag_mode:
            leave_drag_stay_pose(env)
            drag_mode = False
        try:
            _hold()
        except Exception as exc:  # noqa: BLE001
            _log(f"pre-save hold warning: {exc}")
        n_acc = (
            len(env.action_accumulator) if env.action_accumulator is not None else 0
        )
        labels_arr = np.asarray(intervention_labels[:n_acc], dtype=np.uint8)
        if n_acc == 0:
            _log("[DAgger] Empty episode; aborting instead of save")
            abort_episode(env)
            recorder.discard_episode()
        else:
            env.end_episode()
            if env.replay_buffer.n_episodes > 0:
                ep_id = env.replay_buffer.n_episodes - 1
                try:
                    ep = env.replay_buffer.get_episode(ep_id, copy=False)
                    labels_arr = labels_arr[: int(len(ep["action"]))]
                except Exception as exc:  # noqa: BLE001
                    _log(f"intervention align warning: {exc}")
                save_intervention_sidecar(output_dir, ep_id, labels_arr)
                _log(f"[DAgger] Saved episode {ep_id}")
            else:
                _log("[DAgger] end_episode wrote nothing")
            saved = recorder.finish_episode()
            if saved is not None:
                _log(f"[traj] saved {saved}")
        recording = False
        intervention_labels = []
        last_rotvecs = [None, None]
        _return_to_init(reason="save")

    def _abort() -> None:
        nonlocal recording, drag_mode, intervention_labels, last_rotvecs
        if motion_busy:
            _log("[DAgger] Ignore B: still returning to init")
            return
        if not recording:
            _log("No active episode to abort")
            return
        _invalidate_policy_stream()
        if drag_mode:
            leave_drag_stay_pose(env)
            drag_mode = False
        abort_episode(env)
        recorder.discard_episode()
        recording = False
        intervention_labels = []
        last_rotvecs = [None, None]
        try:
            _hold()
        except Exception as exc:  # noqa: BLE001
            _log(f"post-abort hold warning: {exc}")
        _log("[DAgger] Episode aborted")

    def _quit() -> None:
        nonlocal drag_mode, recording, motion_busy
        motion_busy = False
        _invalidate_policy_stream()
        stop_event.set()
        if drag_mode:
            leave_drag_stay_pose(env)
            drag_mode = False
        if recording:
            _log("[DAgger] Quit with active episode — aborting")
            abort_episode(env)
            recorder.discard_episode()
            recording = False
        try:
            _hold()
        except Exception as exc:  # noqa: BLE001
            _log(f"quit hold warning: {exc}")

    def _publish_web(obs: dict) -> None:
        if web_state is None:
            return
        try:
            joints = _v1.measured_state(env)
        except Exception:  # noqa: BLE001
            joints = np.zeros(ACTION_DIM, dtype=np.float64)
        web_state.update(
            mode=_mode_name(),
            step=step_count,
            infer_count=infer_count,
            buf_len=len(buffer),
            delay_steps=int(delay_tracker.pred_delay_steps),
            exec_h=int(exec_h),
            frequency=float(frequency),
            inference_rate=float(inference_rate),
            dispatch="tick" if use_immediate else "schedule",
            rtt_s=float(delay_tracker.last_rtt_s),
            joints14=joints,
            prompt=prompt,
            jpeg=obs_to_jpeg(obs),
        )

    def _dispatch_cmd(
        guarded: np.ndarray, *, duration_s: float | None = None
    ) -> None:
        if use_immediate:
            # duration_s=0 → snap-hold (soft-hold / freeze). Default = one ctrl dt.
            dur = float(dt if duration_s is None else duration_s)
            _v1.command_joint_immediate(env, guarded, duration_s=dur)
        else:
            target_t = max(time.time() + 0.02, t_loop + 0.02)
            _v1.command_joint_actions(
                env,
                guarded.reshape(1, -1),
                np.asarray([target_t], dtype=np.float64),
            )

    # Warmup (same as eval path).
    obs, state = _get_obs_state()
    if skip_server_warmup:
        _log(
            "[warmup] skipped (--skip-server-warmup); "
            "first C may be slow while the server compiles"
        )
        warm = policy.infer(_v1.policy_observation(obs, state, prompt))
        action_horizon = int(np.asarray(warm["actions"]).shape[0])
    else:
        warm_exec_h = int(
            chunk_size if rtc_execute_horizon is None else rtc_execute_horizon
        )
        warm_exec_h = max(1, min(warm_exec_h, int(chunk_size)))
        action_horizon = _v1.warmup_policy_server(
            env=env,
            policy=policy,
            prompt=prompt,
            rtc=True,
            rtc_s_min=warm_exec_h,
            rtc_max_guidance_weight=rtc_max_guidance_weight,
        )
        obs, state = _get_obs_state()

    if chunk_size != action_horizon:
        _log(
            f"[dagger] NOTE: --chunk-size={chunk_size} vs model H={action_horizon}; "
            "using model H"
        )
        chunk_size = action_horizon
        exec_h = int(chunk_size if rtc_execute_horizon is None else rtc_execute_horizon)
        exec_h = max(1, min(exec_h, chunk_size))
    recorder.meta["chunk_size"] = int(chunk_size)
    recorder.meta["exec_h"] = int(exec_h)

    with prev_chunk_lock:
        rtc_prev_chunk = np.tile(state.astype(np.float32)[None, :], (chunk_size, 1))

    _log(
        f"[dagger] RTC v2 async+buffer: ctrl={frequency}Hz infer={inference_rate}Hz "
        f"H={chunk_size} exec_h={exec_h} smooth={smooth} "
        f"dispatch={'tick' if use_immediate else 'schedule'} "
        f"out={output_dir}"
    )
    _log(
        "[DAgger] ready. C=start episode+policy, D=drag, R=resume, "
        "S=save, B=abort, Q=quit"
    )
    if web_state is not None:
        web_state.update(
            mode="IDLE",
            frequency=float(frequency),
            inference_rate=float(inference_rate),
            dispatch="tick" if use_immediate else "schedule",
            exec_h=int(exec_h),
            prompt=prompt,
        )
        web_state.append_log("dagger control loop ready")

    infer_thread = threading.Thread(
        target=_infer_loop, name="kai0-rtc-v2-dagger-infer", daemon=True
    )
    infer_thread.start()
    t_loop = time.monotonic()

    try:
        while True:
            obs = env.get_obs()
            for key in list(keys.read()) + _web_keys_from_cmds(
                web_cmds, dagger=True
            ):
                if key == "q":
                    _quit()
                    return
                if key == "c":
                    _start_episode()
                elif key == "d":
                    _enter_drag()
                elif key == "r":
                    _resume_policy()
                elif key == "s":
                    _save_episode()
                elif key == "b":
                    _abort()

            _publish_web(obs)
            _v1.show_preview(obs, _status_text(), enabled=preview_enabled)

            if motion_busy:
                t_loop += dt
                precise_wait(t_loop)
                continue

            # Always record while episode is open (policy / drag / hold).
            if recording and (drag_mode or not executing):
                _record_tick(obs)
                try:
                    measured = _v1.measured_state(env)
                except Exception:  # noqa: BLE001
                    measured = np.zeros(ACTION_DIM, dtype=np.float64)
                step_count += 1
                recorder.record_ctrl(
                    step=step_count,
                    state14=measured,
                    buffer_act14=measured,
                    guarded14=measured,
                    mode="DRAG" if drag_mode else "HOLD",
                    obs=obs,
                )
                t_loop += dt
                precise_wait(t_loop)
                continue

            if not executing:
                t_loop += 0.05
                precise_wait(t_loop)
                continue

            soft = resume_soft_left > 0
            post_soft = (not soft) and resume_post_soft_left > 0
            per_arm = None
            step_lim = float(max_joint_step)
            if soft:
                per_arm = _v1.resume_soft_step_limits(
                    max_joint_step,
                    resume_soft_left,
                    total_batches=resume_soft_total,
                    hold_batches=resume_hold_ticks,
                )
                step_lim = max(per_arm) if max(per_arm) > 0 else 1e-6
            elif post_soft:
                step_lim = _post_soft_step_limit(resume_post_soft_left)
                per_arm = (step_lim, step_lim)

            pure_hold = soft and per_arm is not None and max(per_arm) <= 0.0
            ctrl_mode = "POLICY"
            # Avoid gripper snap-shut during onset (felt like a jerk with arm move).
            tighten_now = bool(gripper_tighten) and not (soft or post_soft)
            prev_cmd_for_warn = (
                None
                if last_commanded is None
                else np.asarray(last_commanded, dtype=np.float64).copy()
            )
            dispatch_duration: float | None = None

            if pure_hold:
                # Hold still — do NOT pop the buffer. Popping here discarded
                # actions[0..N] near the current pose and left mid-chunk
                # targets, which caused a rush when soft-hold ended.
                entering_soft_hold = not soft_pure_hold_active
                soft_pure_hold_active = True
                try:
                    guarded = _v1.measured_state(env)
                except Exception:  # noqa: BLE001
                    guarded = (
                        last_commanded
                        if last_commanded is not None
                        else np.zeros(ACTION_DIM, dtype=np.float64)
                    )
                act = np.asarray(guarded, dtype=np.float64)
                ctrl_mode = "HOLD"
                # Snap-hold: freeze interpolator at measured (no 50ms drive).
                dispatch_duration = 0.0
                if entering_soft_hold:
                    _diag_snapshot(
                        "soft_hold_enter",
                        measured=act,
                        last_cmd=prev_cmd_for_warn,
                        guarded=act,
                    )
                    _warn_if_jump(
                        "soft_hold_enter",
                        prev_cmd=prev_cmd_for_warn,
                        new_cmd=act,
                        measured=act,
                        min_interval_s=0.0,
                    )
                resume_soft_left -= 1
            else:
                if soft_pure_hold_active:
                    # Just left pure hold: drop any chunk that aged while we
                    # stood still, reseed RTC, discard in-flight infer.
                    soft_pure_hold_active = False
                    action_epoch += 1
                    meas_flush = _safe_measured()
                    buffer.clear()
                    _seed_buffer_anchor(meas_flush)
                    _reseed_prev_from_state()
                    empty_streak = 0
                    _log(
                        "[DAgger] soft-hold done; flushed buffer + reseed "
                        "+ anchor (blend next chunk from measured)"
                    )
                    _diag_snapshot(
                        "soft_hold_exit_flush",
                        measured=meas_flush,
                        last_cmd=prev_cmd_for_warn,
                    )
                act = buffer.pop_next_action()
                if act is None:
                    empty_streak += 1
                    if empty_streak in {1, 20, 50} or empty_streak % 100 == 0:
                        print(f"[dagger] buffer empty x{empty_streak}")
                    # Keep recording / holding pose while waiting for infer.
                    _record_tick(obs)
                    try:
                        measured = _v1.measured_state(env)
                    except Exception:  # noqa: BLE001
                        measured = (
                            last_commanded
                            if last_commanded is not None
                            else np.zeros(ACTION_DIM, dtype=np.float64)
                        )
                    if last_commanded is not None:
                        _warn_if_jump(
                            "empty_hold",
                            prev_cmd=prev_cmd_for_warn,
                            new_cmd=last_commanded,
                            measured=measured,
                        )
                        _dispatch_cmd(last_commanded, duration_s=0.0)
                    step_count += 1
                    recorder.record_ctrl(
                        step=step_count,
                        state14=measured,
                        buffer_act14=measured,
                        guarded14=(
                            last_commanded
                            if last_commanded is not None
                            else measured
                        ),
                        mode="EMPTY",
                        obs=obs,
                    )
                    t_loop += dt
                    precise_wait(t_loop)
                    continue

                empty_streak = 0
                try:
                    measured = _v1.measured_state(env)
                except Exception:  # noqa: BLE001
                    measured = (
                        last_commanded
                        if last_commanded is not None
                        else np.zeros(ACTION_DIM, dtype=np.float64)
                    )
                anchor, reanchored, lag_l, lag_r = _policy_anchor(
                    last_commanded, measured, reanchor_thresh=0.08
                )
                if soft and per_arm is not None and max(per_arm) > 0.0:
                    # Soft ease-in: always prefer measured if we drifted.
                    try:
                        if last_commanded is None or float(
                            np.max(np.abs(measured - anchor))
                        ) > 1e-3:
                            anchor = measured
                            last_commanded = np.asarray(
                                measured, dtype=np.float64
                            )
                            reanchored = False
                    except Exception:  # noqa: BLE001
                        pass
                if reanchored:
                    now_ra = time.monotonic()
                    if now_ra - _last_jump_warn_t >= 0.2:
                        _last_jump_warn_t = now_ra
                        _log(
                            f"[DAgger/WARN] reanchor to measured "
                            f"step={step_count} lag L={lag_l:.4f} "
                            f"R={lag_r:.4f} (prevent open-loop cmd ratchet)"
                        )
                        _diag_snapshot(
                            "reanchor",
                            measured=measured,
                            last_cmd=prev_cmd_for_warn,
                            act=act,
                            anchor=anchor,
                        )
                guarded = _guard_one(
                    act,
                    anchor,
                    max_joint_step=step_lim,
                    gripper_min=gripper_min,
                    gripper_max=gripper_max,
                    gripper_close_threshold=gripper_close_threshold,
                    gripper_grasp_width=gripper_grasp_width,
                    gripper_tighten=tighten_now,
                    max_joint_step_per_arm=per_arm,
                )
                if soft:
                    resume_soft_left -= 1
                elif post_soft:
                    resume_post_soft_left -= 1

            try:
                measured = _v1.measured_state(env)
            except Exception:  # noqa: BLE001
                measured = np.asarray(guarded, dtype=np.float64)

            _warn_if_jump(
                f"dispatch:{ctrl_mode}",
                prev_cmd=prev_cmd_for_warn,
                new_cmd=guarded,
                measured=measured,
                raw_act=None if ctrl_mode == "HOLD" else act,
            )
            _dispatch_cmd(guarded, duration_s=dispatch_duration)
            last_commanded = np.asarray(guarded, dtype=np.float64)
            _record_tick(obs)
            step_count += 1
            recorder.record_ctrl(
                step=step_count,
                state14=measured,
                buffer_act14=act,
                guarded14=guarded,
                mode=ctrl_mode,
                obs=obs,
            )
            if step_count % 40 == 0:
                track_l = track_r = step_l = step_r = float("nan")
                if measured is not None and last_commanded is not None:
                    track_l, track_r = _arm_max_abs(last_commanded - measured)
                if prev_cmd_for_warn is not None:
                    step_l, step_r = _arm_max_abs(
                        np.asarray(guarded) - prev_cmd_for_warn
                    )
                print(
                    f"[dagger/ctrl] step={step_count} buf={len(buffer)} "
                    f"delay={delay_tracker.pred_delay_steps} "
                    f"infers={infer_count} soft={resume_soft_left} "
                    f"mode={ctrl_mode} "
                    f"dcmd L={step_l:.4f} R={step_r:.4f} "
                    f"track L={track_l:.4f} R={track_r:.4f} "
                    f"cmd={_v1._fmt14(guarded)}",
                    flush=True,
                )
            t_loop += dt
            precise_wait(t_loop)
    finally:
        stop_event.set()
        infer_thread.join(timeout=2.0)
        if drag_mode:
            try:
                leave_drag_stay_pose(env)
            except Exception:  # noqa: BLE001
                pass
        if recording:
            try:
                abort_episode(env)
            except Exception:  # noqa: BLE001
                pass
            recorder.discard_episode()
        print(
            f"[dagger] session: steps={step_count} infers={infer_count} "
            f"last_delay={delay_tracker.pred_delay_steps}"
        )


@click.command()
@click.option("--policy-ip", default="127.0.0.1", show_default=True)
@click.option("--policy-port", default=8000, type=int, show_default=True)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot-ports", default="8765,8766", show_default=True)
@click.option("--camera-reorder", default="0,1", show_default=True)
@click.option("--rs-serial", default=None)
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=True)
@click.option(
    "--frequency",
    default=20.0,
    type=float,
    show_default=True,
    help="Control-loop Hz (official ARX default 30; keep 20 if matching your data).",
)
@click.option(
    "--inference-rate",
    default=4.0,
    type=float,
    show_default=True,
    help="Background RTC infer Hz (official default 4).",
)
@click.option("--max-joint-speed", default=1.5, type=float, show_default=True)
@click.option("--gripper-min", default=0.0, type=float, show_default=True)
@click.option("--gripper-max", default=0.088, type=float, show_default=True)
@click.option(
    "--gripper-tighten/--no-gripper-tighten",
    default=True,
    show_default=True,
    help="When policy width < threshold, force grasp_width (tighter close).",
)
@click.option(
    "--gripper-close-threshold",
    default=0.04,
    type=float,
    show_default=True,
    help="ZMQ open-width below this is treated as grasp → snap to grasp_width.",
)
@click.option(
    "--gripper-grasp-width",
    default=0.0,
    type=float,
    show_default=True,
    help="Forced open-width (m) while grasping; 0 = fully closed.",
)
@click.option("--camera-obs-latency", default=0.125, type=float, show_default=True)
@click.option("--robot-obs-latency", default=0.0001, type=float, show_default=True)
@click.option("--server-timeout", default=30.0, type=float, show_default=True)
@click.option("--observe-only", is_flag=True)
@click.option("--no-preview", is_flag=True)
@click.option("--no-init", is_flag=True)
@click.option("--init-joints", default=DEFAULT_INIT_JOINTS, show_default=False)
@click.option("--init-duration", default=2.5, type=float, show_default=True)
@click.option(
    "--home-duration",
    default=5.0,
    type=float,
    show_default=True,
    help="Seconds for slow joint-space return to home on Q (ease-in-out).",
)
@click.option(
    "--chunk-size",
    default=50,
    type=int,
    show_default=True,
    help="Expected action horizon; overridden by warmup model H if different.",
)
@click.option(
    "--rtc-execute-horizon",
    default=None,
    type=int,
    help="RTC execute_horizon (default=chunk/model H, same as official).",
)
@click.option(
    "--rtc-max-guidance-weight",
    default=0.5,
    type=float,
    show_default=True,
)
@click.option(
    "--rtc-mask-prefix-delay",
    is_flag=True,
    help="Pass mask_prefix_delay=true like official --rtc_mask_prefix_delay.",
)
@click.option(
    "--latency-k",
    default=8,
    type=int,
    show_default=True,
    help="Max steps to drop from new chunk (= buffer.k cap), official default 8.",
)
@click.option(
    "--min-smooth-steps",
    default=10,
    type=int,
    show_default=True,
    help="Temporal overlap pad length (official min_smooth_steps default 10).",
)
@click.option(
    "--rtc-disable-smoothing",
    is_flag=True,
    help="Official --rtc_disable_smoothing: latency trim only, no temporal blend.",
)
@click.option(
    "--dispatch",
    type=click.Choice(["schedule", "tick"], case_sensitive=False),
    default="schedule",
    show_default=True,
    help=(
        "schedule=timed ZMQ waypoints (interpolator); "
        "tick=ROS-style immediate joint cmd each control step."
    ),
)
@click.option(
    "--skip-server-warmup",
    is_flag=True,
    help="Skip plain+RTC throw-away infer (first C may wait on JAX compile).",
)
@click.option(
    "--dagger",
    is_flag=True,
    help=(
        "DAgger mode: C start episode+policy, D gravity-comp drag intervene, "
        "R resume policy, S save UMI zarr episode, B abort. "
        "Writes intervention sidecars under {output}/intervention/."
    ),
)
@click.option(
    "--output",
    default=None,
    help=(
        "UMI zarr output dir. Default: data_local/kai0_eval (eval) or "
        "data_local/kai0_dagger (--dagger)."
    ),
)
@click.option(
    "--drag-profile",
    type=click.Choice(["light", "stable"], case_sensitive=False),
    default="stable",
    show_default=True,
    help=(
        "Soft-hold drag feel for DAgger intervene (D): stable=anti-sag, "
        "light=same as collect. Ignored with true gravity-comp server."
    ),
)
@click.option(
    "--web/--no-web",
    default=True,
    show_default=True,
    help="Serve live camera/status dashboard (see --web-port).",
)
@click.option("--web-host", default="0.0.0.0", show_default=True)
@click.option("--web-port", default=8090, type=int, show_default=True)
@click.option(
    "--log-file",
    default=str(DEFAULT_RUN_LOG),
    show_default=True,
)
@click.option(
    "--save-model-traj/--no-save-model-traj",
    default=True,
    show_default=True,
    help=(
        "Save collect-style traj zarr (timestamp/action/cameras/joints) under "
        "{output}/model_traj/replay_buffer.zarr; infer chunks in model_traj/infer/."
    ),
)
@click.option(
    "--save-traj-images/--no-save-traj-images",
    default=True,
    show_default=True,
    help=(
        "With --save-model-traj, store per-tick camera0/1/2_rgb in the zarr "
        "(no videos/ folder)."
    ),
)
def main(
    policy_ip: str,
    policy_port: int,
    robot_ip: str,
    robot_ports: str,
    camera_reorder: str,
    rs_serial: str | None,
    prompt: str,
    frequency: float,
    inference_rate: float,
    max_joint_speed: float,
    gripper_min: float,
    gripper_max: float,
    gripper_tighten: bool,  # noqa: FBT001
    gripper_close_threshold: float,
    gripper_grasp_width: float,
    camera_obs_latency: float,
    robot_obs_latency: float,
    server_timeout: float,
    observe_only: bool,  # noqa: FBT001
    no_preview: bool,  # noqa: FBT001
    no_init: bool,  # noqa: FBT001
    init_joints: str,
    init_duration: float,
    home_duration: float,
    chunk_size: int,
    rtc_execute_horizon: int | None,
    rtc_max_guidance_weight: float,
    rtc_mask_prefix_delay: bool,  # noqa: FBT001
    latency_k: int,
    min_smooth_steps: int,
    rtc_disable_smoothing: bool,  # noqa: FBT001
    dispatch: str,
    skip_server_warmup: bool,  # noqa: FBT001
    dagger: bool,  # noqa: FBT001
    output: str | None,
    drag_profile: str,
    web: bool,  # noqa: FBT001
    web_host: str,
    web_port: int,
    log_file: str,
    save_model_traj: bool,  # noqa: FBT001
    save_traj_images: bool,  # noqa: FBT001
) -> None:
    if frequency <= 0 or inference_rate <= 0 or max_joint_speed <= 0:
        raise click.BadParameter("frequency/inference-rate/max-joint-speed must be > 0")
    if not 0 <= gripper_min < gripper_max:
        raise click.BadParameter("invalid gripper range")
    if gripper_close_threshold <= 0:
        raise click.BadParameter("--gripper-close-threshold must be > 0")
    if gripper_grasp_width < 0:
        raise click.BadParameter("--gripper-grasp-width must be >= 0")
    if gripper_grasp_width >= gripper_max:
        raise click.BadParameter("--gripper-grasp-width must be < --gripper-max")
    if rtc_max_guidance_weight <= 0:
        raise click.BadParameter("--rtc-max-guidance-weight must be positive")
    if home_duration <= 0 or init_duration <= 0:
        raise click.BadParameter("--home-duration/--init-duration must be > 0")
    if dagger and observe_only:
        raise click.ClickException("--dagger cannot be combined with --observe-only")

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    _v1.install_run_log(log_path)

    ports = _v1.parse_pair(robot_ports, "--robot-ports")
    camera_order = _v1.parse_pair(camera_reorder, "--camera-reorder")
    max_joint_step = float(max_joint_speed) / float(frequency)

    print(f"Connecting to policy server ws://{policy_ip}:{policy_port}")
    policy = WebsocketClientPolicy(
        host=policy_ip,
        port=policy_port,
        ping_timeout=server_timeout,
        close_timeout=server_timeout,
    )
    metadata = policy.get_server_metadata()
    print(f"Policy connected; metadata={metadata}")
    if not metadata.get("supports_rtc"):
        raise click.ClickException(
            "RTC v2 requires server metadata supports_rtc=true. "
            "Restart Kai0 with: uv run scripts/serve_kai0_policy.py "
            "--checkpoint <jax_ckpt> --rtc --port 8000"
        )

    soft_hint = (
        chunk_size
        if rtc_execute_horizon is None
        else min(int(rtc_execute_horizon), int(chunk_size))
    )
    grip_note = (
        f"tighten<{gripper_close_threshold:.3f}→{gripper_grasp_width:.4f}"
        if gripper_tighten
        else "tighten=off"
    )
    print(
        f"[v2] enabled: ctrl={frequency}Hz infer={inference_rate}Hz "
        f"exec_h≈{soft_hint} max_gw={rtc_max_guidance_weight} "
        f"smooth={'raw' if rtc_disable_smoothing else 'temporal'} "
        f"min_m={min_smooth_steps} latency_k={latency_k} "
        f"dispatch={dispatch} grip={grip_note}"
    )
    if dagger:
        print(
            "Controls (DAgger): C=start episode+policy, D=drag intervene, "
            "R=resume policy, S=save then return to init, B=abort, Q=quit"
        )
    else:
        print(
            "Controls: C=start, S=stop/hold, "
            "Enter=return to init (after S), Q=home then quit"
        )
    if observe_only:
        print("OBSERVE ONLY: motor commands are permanently disabled")

    # Before spawning controllers so soft-hold DAgger drag uses the chosen feel.
    _v1.apply_drag_profile(drag_profile)
    if dagger:
        print(
            f"[drag] profile={_v1.get_drag_profile()} "
            "(soft-hold intervene; true GC ignores)"
        )

    preview_enabled = (not no_preview) and bool(os.environ.get("DISPLAY"))
    if no_preview:
        print("Preview disabled (--no-preview)")
    elif not preview_enabled:
        print(
            "Preview disabled: DISPLAY is unset. "
            "Use a local desktop or pass --no-preview."
        )

    init_q14 = None if no_init else _v1.parse_init_joints(init_joints)
    do_home = not no_init
    if init_q14 is not None:
        print(
            f"Will joint-init after env start ({init_duration:.1f}s) "
            "from collect_bimanual_10_22 ep0 median"
        )

    if output is None:
        output_path = (
            DEFAULT_DAGGER_OUTPUT.resolve()
            if dagger
            else (ROOT / "data_local" / "kai0_eval").resolve()
        )
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = (ROOT / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if dagger:
        print(f"[DAgger] UMI zarr output -> {output_path}")

    traj_dir = output_path / "model_traj"
    traj_meta = {
        "prompt": prompt,
        "frequency": float(frequency),
        "inference_rate": float(inference_rate),
        "chunk_size": int(chunk_size),
        "rtc_execute_horizon": rtc_execute_horizon,
        "latency_k": int(latency_k),
        "min_smooth_steps": int(min_smooth_steps),
        "smooth": "raw" if rtc_disable_smoothing else "temporal",
        "dispatch": str(dispatch),
        "rtc_max_guidance_weight": float(rtc_max_guidance_weight),
        "rtc_mask_prefix_delay": bool(rtc_mask_prefix_delay),
        "gripper_min": float(gripper_min),
        "gripper_max": float(gripper_max),
        "gripper_tighten": bool(gripper_tighten),
        "gripper_close_threshold": float(gripper_close_threshold),
        "gripper_grasp_width": float(gripper_grasp_width),
        "max_joint_speed": float(max_joint_speed),
        "dagger": bool(dagger),
        "policy_ip": policy_ip,
        "policy_port": int(policy_port),
        "policy_metadata": {
            k: (v if isinstance(v, (bool, int, float, str, type(None))) else str(v))
            for k, v in dict(metadata).items()
        },
    }
    traj_recorder = ModelTrajRecorder(
        traj_dir,
        meta=traj_meta,
        enabled=bool(save_model_traj),
        save_images=bool(save_traj_images),
    )
    if save_model_traj:
        img_note = (
            "rgb in zarr, no videos/"
            if save_traj_images
            else "images=off"
        )
        print(
            f"[traj] model traj -> {traj_dir}/replay_buffer.zarr "
            f"(C→S; {img_note}; infer sidecars in traj/infer/)"
        )
    else:
        print("[traj] model traj disabled (--no-save-model-traj)")

    web_state: SharedEvalState | None = None
    web_cmds: CommandQueue | None = None
    web_server = None
    if web:
        web_state = SharedEvalState()
        web_cmds = CommandQueue()
        web_server = start_eval_web_ui(
            web_state, web_cmds, host=web_host, port=int(web_port)
        )
        web_state.append_log(
            f"dashboard http://{web_host}:{web_port}/ (cameras + status)"
        )
        print(f"[v2] open dashboard: http://127.0.0.1:{web_port}/")

    try:
        with SharedMemoryManager() as shm_manager, Arx5EnvWithImages(
            output_dir=output_path,
            robots_config=_v1.robot_config(
                robot_ip,
                ports,
                robot_obs_latency,
                reset_to_home_on_start=do_home,
            ),
            frequency=frequency,
            obs_image_resolution=(224, 224),
            max_obs_buffer_size=120,
            obs_float32=False,
            camera_reorder=camera_order,
            camera_obs_latency=camera_obs_latency,
            camera_obs_horizon=1,
            robot_obs_horizon=1,
            init_joints=False,
            enable_multi_cam_vis=False,
            shm_manager=shm_manager,
            rs_serial=rs_serial,
            enable_depth=False,
        ) as env, _v1.TerminalKeys() as keys:
            print("Waiting for cameras and both ARX5 ZMQ controllers...")
            if web_state is not None:
                web_state.append_log("waiting for cameras / ARX5 controllers...")
            while not env.is_ready:
                time.sleep(0.1)
            print("Environment ready")
            if web_state is not None:
                web_state.append_log("environment ready")

            if init_q14 is not None:
                time.sleep(0.5)
                if web_state is not None:
                    web_state.update(mode="INIT")
                _v1.move_to_init_joints(
                    env,
                    init_q14,
                    frequency=frequency,
                    duration=float(init_duration),
                    gripper_min=gripper_min,
                    gripper_max=gripper_max,
                    on_progress=make_progress_cb(web_state),
                )
                if web_state is not None:
                    web_state.clear_progress()
                    web_state.update(mode="HOLD")
                    try:
                        web_state.update(
                            jpeg=obs_to_jpeg(env.get_obs()),
                            joints14=_v1.measured_state(env),
                        )
                    except Exception:  # noqa: BLE001
                        pass
            try:
                common_rtc = dict(
                    env=env,
                    policy=policy,
                    keys=keys,
                    prompt=prompt,
                    frequency=frequency,
                    inference_rate=inference_rate,
                    max_joint_step=max_joint_step,
                    gripper_min=gripper_min,
                    gripper_max=gripper_max,
                    gripper_close_threshold=float(gripper_close_threshold),
                    gripper_grasp_width=float(gripper_grasp_width),
                    gripper_tighten=bool(gripper_tighten),
                    preview_enabled=preview_enabled,
                    chunk_size=int(chunk_size),
                    rtc_execute_horizon=rtc_execute_horizon,
                    rtc_max_guidance_weight=rtc_max_guidance_weight,
                    rtc_mask_prefix_delay=rtc_mask_prefix_delay,
                    latency_k=int(latency_k),
                    min_smooth_steps=int(min_smooth_steps),
                    disable_smoothing=rtc_disable_smoothing,
                    dispatch=dispatch,
                    skip_server_warmup=skip_server_warmup,
                    web_state=web_state,
                    web_cmds=web_cmds,
                )
                if dagger:
                    run_dagger_v2(
                        **common_rtc,
                        output_dir=output_path,
                        init_q14=init_q14,
                        init_duration=float(init_duration),
                        traj_recorder=traj_recorder,
                    )
                else:
                    run_rtc_v2(
                        **common_rtc,
                        observe_only=observe_only,
                        init_q14=init_q14,
                        init_duration=float(init_duration),
                        home_duration=float(home_duration),
                        traj_recorder=traj_recorder,
                    )
            finally:
                try:
                    _v1.hold_current(
                        env, immediate=(str(dispatch).lower() == "tick")
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"Final hold warning: {exc}")
                if preview_enabled:
                    cv2.destroyAllWindows()
    finally:
        if web_server is not None:
            try:
                web_server.stop()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
