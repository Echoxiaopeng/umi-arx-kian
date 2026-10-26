#!/usr/bin/env python3
"""Run a Kai0 dual-arm joint policy on the ARX5 ZMQ/camera stack.

The policy server runs separately from the Kai0 repository. This client:

1. on start: ``reset_to_home`` then joint-space init (median episode-0 joints
   from ``collect_bimanual_10_22``);
2. reads left wrist + center D435 + right wrist images;
3. reads both arms' six joints and gripper widths;
4. sends the Kai0 observation over websocket;
5. receives ``[T, 14]`` absolute joint actions;
6. schedules a guarded prefix via ``schedule_joint_waypoint``.

With ``--rtc``, overlaps inference with execution and sends
``prev_action_chunk`` / ``inference_delay`` / ``execute_horizon`` so the
server (``serve_kai0_policy.py --rtc``) can run Pi0RTC guidance.

ZMQ servers should start with ``--joint`` (Kai0 is joint throughout).
Example:

    python python/communication/zmq_server.py X5_umi can0 --port 8765 --joint --no-idle-reset-home --no_cmd_timeout 0
    python python/communication/zmq_server.py X5_umi can1 --port 8766 --joint --no-idle-reset-home --no_cmd_timeout 0

    # Kai0 server with RTC:
    #   uv run scripts/serve_kai0_policy.py --checkpoint <jax_ckpt> --rtc --port 8000

    conda activate umi-arx
    python scripts/eval_arx5_kai0_bimanual.py \
      --policy-ip 127.0.0.1 --policy-port 8000 \
      --robot-ports 8765,8766 --camera-reorder 0,1 --no-preview \
      --steps-per-inference 32

    python scripts/eval_arx5_kai0_bimanual.py \
      --policy-ip 127.0.0.1 --policy-port 8000 \
      --robot-ports 8765,8766 --camera-reorder 0,1 --no-preview \
      --rtc --frequency 20 --steps-per-inference 8

Controls: C=start execution, S=stop/hold, I=infer once without motion, Q=quit.
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import os
import select
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import sys
import termios
import time
import tty

import click
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_LOG = ROOT / "data_local" / "kai0_eval" / "last_run.log"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
if CLIENT_SRC.is_dir():
    sys.path.insert(0, str(CLIENT_SRC))

from modules.arx5_env_with_images_bimanual_rs import Arx5EnvWithImages  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from utils.kai0_rtc import JointRTCChunk  # noqa: E402
from utils.kai0_rtc import JointRTCInferenceWorker  # noqa: E402
from utils.kai0_rtc import build_joint_chunk  # noqa: E402
from utils.kai0_rtc import estimate_delay_steps  # noqa: E402
from utils.kai0_rtc import stitch_chunk_to_state  # noqa: E402

DEFAULT_PROMPT = "Flatten and fold the cloth."
ACTION_DIM = 14
# Client C1 blend length. Longer than soft(~10) so catch-up after swap is gradual.
RTC_BLEND_STEPS = 24
# Request at least this many soft-guidance steps (extends into hold-last pad).
RTC_SOFT_MIN = 20


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _fmt14(x: np.ndarray, *, prec: int = 3) -> str:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < ACTION_DIM:
        return np.array2string(arr, precision=prec, suppress_small=True)
    # left q0-2 + grip | right q0-2 + grip (compact boundary view)
    return (
        f"L[{arr[0]:.{prec}f},{arr[1]:.{prec}f},{arr[2]:.{prec}f}|g{arr[6]:.{prec}f}] "
        f"R[{arr[7]:.{prec}f},{arr[8]:.{prec}f},{arr[9]:.{prec}f}|g{arr[13]:.{prec}f}]"
    )


def _chunk_step_stats(actions: np.ndarray, n: int = 8) -> str:
    """Max |Δq| over first n successive steps (joints only, both arms)."""
    a = np.asarray(actions, dtype=np.float64)
    if len(a) < 2:
        return "n/a"
    m = min(int(n), len(a) - 1)
    deltas = []
    for i in range(m):
        d = a[i + 1] - a[i]
        deltas.append(max(float(np.max(np.abs(d[0:6]))), float(np.max(np.abs(d[7:13])))))
    arr = np.asarray(deltas)
    return f"p50={np.median(arr):.4f} max={arr.max():.4f} sat@0.04={(arr >= 0.039).mean()*100:.0f}%"


def _log_rtc_swap(
    *,
    tag: str,
    swap_at: int,
    trim_steps: int,
    s_used: int,
    raw_chunk: JointRTCChunk,
    stitched: JointRTCChunk,
    last_commanded: np.ndarray | None,
    latency_s: float,
    swap_index: int,
    wall_dt_s: float | None,
    max_joint_step: float,
) -> None:
    soft = max(0, int(s_used) - int(trim_steps))
    last = (
        np.asarray(last_commanded, dtype=np.float64)
        if last_commanded is not None
        else None
    )
    raw0 = raw_chunk.actions[0] if raw_chunk.horizon else None
    st0 = stitched.actions[0] if stitched.horizon else None
    jump_raw = _max_abs(raw0, last) if raw0 is not None and last is not None else float("nan")
    jump_st = _max_abs(st0, last) if st0 is not None and last is not None else float("nan")
    # Direction change: compare pre-swap velocity proxy vs first post-stitch step.
    step1 = (
        _max_abs(stitched.actions[1], stitched.actions[0])
        if stitched.horizon >= 2
        else float("nan")
    )
    wall = f" dt_swap={wall_dt_s:.3f}s" if wall_dt_s is not None else ""
    print(
        f"[RTC] {tag} #{swap_index}: cursor={swap_at} trim={trim_steps} "
        f"s={s_used} soft={soft} H={stitched.horizon} lat={latency_s:.3f}s{wall}"
    )
    print(
        f"[RTC]   jump_raw0={jump_raw:.4f} jump_stitch0={jump_st:.4f} "
        f"step1={step1:.4f} (limit={max_joint_step:.4f}) "
        f"raw_steps[{_chunk_step_stats(raw_chunk.actions)}] "
        f"st_steps[{_chunk_step_stats(stitched.actions)}]"
    )
    if last is not None and st0 is not None:
        print(f"[RTC]   last={_fmt14(last)}")
        print(f"[RTC]   raw0={_fmt14(raw0)}")
        print(f"[RTC]   st0 ={_fmt14(st0)}")


def _log_rtc_infer_start(
    *,
    cursor: int,
    chunk_h: int,
    delay: int,
    switch_s: int,
    start_trigger: int,
    commit_end: int | None,
    leftover: int,
    max_gw: float,
    state: np.ndarray | None,
) -> None:
    soft = max(0, leftover - delay)  # after hard mask, leftover soft capacity
    print(
        f"[RTC] infer start: cursor={cursor}/{chunk_h} "
        f"trigger={start_trigger} switch_s={switch_s} "
        f"delay={delay} commit_end={commit_end} "
        f"leftover={leftover} soft_cap={soft} max_gw={max_gw}"
    )
    if state is not None:
        print(f"[RTC]   proprio={_fmt14(state)}")
# Median first-frame joints from collect_bimanual_10_22 (91/100 inliers).
# Order: left q6, left grip, right q6, right grip.
DEFAULT_INIT_JOINTS = (
    "0.024986,0.637255,0.694095,-0.620852,0.026380,-0.052849,0.083843,"
    "0.051690,0.633059,0.692569,-0.632296,0.068475,0.029945,0.067379"
)


class _TeeIO:
    """Mirror writes to the terminal and a log file (line-buffered)."""

    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        self._secondary.write(data)
        self._primary.flush()
        self._secondary.flush()
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def install_run_log(log_path: Path) -> Path:
    """Overwrite ``log_path`` and tee stdout/stderr into it for this process."""
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _TeeIO(real_out, log_file)
    sys.stderr = _TeeIO(real_err, log_file)

    def _restore() -> None:
        sys.stdout = real_out
        sys.stderr = real_err
        try:
            log_file.flush()
            log_file.close()
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_restore)
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[log] writing run log (overwrite) -> {log_path}")
    print(f"[log] started_at={stamp}")
    print(f"[log] argv={' '.join(sys.argv)}")
    return log_path


class TerminalKeys:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("An interactive TTY is required for C/S/I/Q controls.")
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args):
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self) -> list[str]:
        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if char == "\x03":
                raise KeyboardInterrupt
            if char:
                keys.append(char.lower())
        return keys


def parse_pair(text: str, name: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 2:
        raise click.BadParameter(f"{name} requires two comma-separated values")
    return values


def robot_config(
    robot_ip: str,
    ports: list[int],
    obs_latency: float,
    *,
    reset_to_home_on_start: bool,
) -> list[dict]:
    return [
        {
            "robot_type": "arx5",
            "robot_ip": robot_ip,
            "robot_port": port,
            "robot_obs_latency": obs_latency,
            "robot_action_latency": 0.01,
            "tcp_offset": 0.0,
            "height_threshold": 0.0,
            "sphere_radius": 0.1,
            "sphere_center": [0, -0.06, -0.185],
            # Home first, then joint-space move to dataset init.
            "reset_to_home_on_start": bool(reset_to_home_on_start),
        }
        for port in ports
    ]


def ease_in_out_quad(t: float) -> float:
    t *= 2.0
    if t < 1.0:
        return 0.5 * t * t
    t -= 1.0
    return -0.5 * (t * (t - 2.0) - 1.0)


def parse_init_joints(text: str) -> np.ndarray:
    vals = [float(v) for v in str(text).split(",") if str(v).strip() != ""]
    if len(vals) != ACTION_DIM:
        raise click.BadParameter(
            f"--init-joints needs {ACTION_DIM} values "
            f"(left q6+grip, right q6+grip), got {len(vals)}"
        )
    return np.asarray(vals, dtype=np.float64)


def move_to_init_joints(
    env: Arx5EnvWithImages,
    target14: np.ndarray,
    *,
    frequency: float,
    duration: float,
    gripper_min: float,
    gripper_max: float,
) -> None:
    """Interpolate both arms in joint space to dataset init (needs --joint ZMQ)."""
    target14 = np.asarray(target14, dtype=np.float64).reshape(ACTION_DIM)
    target14[6] = float(np.clip(target14[6], gripper_min, gripper_max))
    target14[13] = float(np.clip(target14[13], gripper_min, gripper_max))

    current = measured_state(env).astype(np.float64)
    print(
        "Joint init (collect_bimanual_10_22 ep0 median):\n"
        f"  from {np.array2string(current, precision=4, separator=',')}\n"
        f"    to {np.array2string(target14, precision=4, separator=',')}"
    )

    dt = 1.0 / float(frequency)
    steps = max(2, int(float(duration) * float(frequency)))
    actions = np.zeros((steps, ACTION_DIM), dtype=np.float64)
    for k in range(steps):
        a = ease_in_out_quad(k / (steps - 1))
        actions[k] = current + a * (target14 - current)
        actions[k, 6] = float(np.clip(actions[k, 6], gripper_min, gripper_max))
        actions[k, 13] = float(np.clip(actions[k, 13], gripper_min, gripper_max))
    timestamps = np.arange(steps, dtype=np.float64) * dt + (time.time() + 0.2)
    try:
        command_joint_actions(env, actions, timestamps)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Joint init failed (needs ZMQ --joint / SET_JOINT_POSITION). "
            "Restart zmq_server WITH --joint, e.g.\n"
            "  python python/communication/zmq_server.py X5_umi can0 "
            "--port 8765 --joint --no-idle-reset-home --no_cmd_timeout 0\n"
            f"Underlying error: {exc}"
        ) from exc
    time.sleep(float(duration) + 0.3)
    after = measured_state(env).astype(np.float64)
    err = np.abs(after - target14)
    print(
        f"Joint init done. |after-target| max={err.max():.4f} "
        f"joints_max={err[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].max():.4f}"
    )


def measured_state(env: Arx5EnvWithImages) -> np.ndarray:
    """Kai0 order: left q6, left gripper, right q6, right gripper."""
    result: list[float] = []
    for state in env.get_robot_state():
        joints = np.asarray(state["ActualQ"], dtype=np.float32).reshape(6)
        gripper = float(state["gripper_position"])
        result.extend(joints.tolist())
        result.append(gripper)
    value = np.asarray(result, dtype=np.float32)
    if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"Invalid robot state: shape={value.shape}, value={value}")
    return value


def policy_observation(obs: dict, state: np.ndarray, prompt: str) -> dict:
    """Map logical cameras to the keys used by LerobotAgilexDataConfig."""
    return {
        "observation.images.top_head": np.asarray(obs["camera1_rgb"][-1], dtype=np.uint8),
        "observation.images.hand_left": np.asarray(obs["camera0_rgb"][-1], dtype=np.uint8),
        "observation.images.hand_right": np.asarray(obs["camera2_rgb"][-1], dtype=np.uint8),
        "observation.state": state,
        "prompt": prompt,
    }


def rtc_policy_observation(
    obs: dict,
    state: np.ndarray,
    prompt: str,
    *,
    prev_action_chunk: np.ndarray | None,
    inference_delay: int,
    execute_horizon: int,
    max_guidance_weight: float,
    enable_rtc: bool = True,
) -> dict:
    """Kai0 obs plus RTC fields for Pi0RTC serve."""
    payload = policy_observation(obs, state, prompt)
    payload["enable_rtc"] = bool(enable_rtc)
    payload["inference_delay"] = int(inference_delay)
    payload["execute_horizon"] = int(execute_horizon)
    payload["max_guidance_weight"] = float(max_guidance_weight)
    if prev_action_chunk is not None:
        payload["prev_action_chunk"] = np.asarray(prev_action_chunk, dtype=np.float32)
    return payload


def guarded_chunk(
    actions: np.ndarray,
    current: np.ndarray,
    *,
    count: int,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(f"Policy returned {actions.shape}, expected [T, {ACTION_DIM}]")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Policy returned NaN/Inf actions")

    chunk = actions[: min(count, len(actions))].copy()
    previous = np.asarray(current, dtype=np.float64).copy()
    for index in range(len(chunk)):
        for offset in (0, 7):
            delta = chunk[index, offset : offset + 6] - previous[offset : offset + 6]
            chunk[index, offset : offset + 6] = (
                previous[offset : offset + 6]
                + np.clip(delta, -max_joint_step, max_joint_step)
            )
            chunk[index, offset + 6] = np.clip(
                chunk[index, offset + 6], gripper_min, gripper_max
            )
        previous = chunk[index]
    return chunk


def command_joint_actions(
    env: Arx5EnvWithImages,
    actions: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """Schedule absolute joint waypoints with wall-clock timestamps (lingbot-style)."""
    actions = np.asarray(actions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(f"Expected actions [T, {ACTION_DIM}], got {actions.shape}")
    if timestamps.shape != (len(actions),):
        raise RuntimeError(
            f"timestamps shape {timestamps.shape} != ({len(actions)},)"
        )
    receive_time = time.time()
    is_new = timestamps > receive_time
    if not np.any(is_new):
        return
    for action, target_time in zip(actions[is_new], timestamps[is_new]):
        for robot_index, robot in enumerate(env.robots):
            target = np.asarray(
                action[7 * robot_index : 7 * robot_index + 7], dtype=np.float64
            )
            robot.schedule_joint_waypoint(target, target_time=float(target_time))


def hold_current(env: Arx5EnvWithImages) -> None:
    # Near-future stamp so the controller inserts a hold waypoint immediately.
    command_joint_actions(
        env,
        measured_state(env).reshape(1, -1),
        np.asarray([time.time() + 0.05], dtype=np.float64),
    )


def show_preview(obs: dict, status: str, *, enabled: bool) -> None:
    if not enabled:
        return
    views = [
        np.asarray(obs[f"camera{index}_rgb"][-1], dtype=np.uint8)
        for index in (0, 1, 2)
    ]
    canvas = np.concatenate(views, axis=1)
    cv2.putText(
        canvas,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Kai0 ARX5 bimanual", canvas[..., ::-1])
    cv2.pollKey()


def run_open_loop(
    *,
    env: Arx5EnvWithImages,
    policy: WebsocketClientPolicy,
    keys: TerminalKeys,
    prompt: str,
    frequency: float,
    steps_per_inference: int,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
    observe_only: bool,
    preview_enabled: bool,
) -> None:
    """Original blocking infer → execute chunk → wait loop."""
    dt = 1.0 / frequency
    executing = False
    while True:
        obs = env.get_obs()
        requested_infer = False
        for key in keys.read():
            if key == "q":
                hold_current(env)
                return
            if key == "s":
                executing = False
                hold_current(env)
                print("Policy stopped; holding measured joints")
            elif key == "c":
                if observe_only:
                    print("Ignoring C: --observe-only")
                else:
                    executing = True
                    print("Policy execution started")
            elif key == "i":
                requested_infer = True

        show_preview(
            obs,
            "EXECUTING  S:stop Q:quit"
            if executing
            else "HOLD  C:start I:infer Q:quit",
            enabled=preview_enabled,
        )
        if not executing and not requested_infer:
            time.sleep(0.02)
            continue

        current = measured_state(env)
        started = time.monotonic()
        result = policy.infer(policy_observation(obs, current, prompt))
        latency = time.monotonic() - started
        actions = guarded_chunk(
            result["actions"],
            current,
            count=steps_per_inference,
            max_joint_step=max_joint_step,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        )
        print(
            f"inference={latency:.3f}s actions={np.asarray(result['actions']).shape} "
            f"execute={executing} first={np.round(actions[0], 3)}"
        )
        if not executing:
            continue

        command_start_time = time.time() + 0.05
        action_timestamps = (
            np.arange(len(actions), dtype=np.float64) * dt + command_start_time
        )
        command_joint_actions(env, actions, action_timestamps)

        deadline = float(action_timestamps[-1])
        while time.time() < deadline:
            for key in keys.read():
                if key in {"s", "q"}:
                    executing = False
                    hold_current(env)
                    if key == "q":
                        return
                    break
            if not executing:
                break
            time.sleep(0.01)


def run_rtc(
    *,
    env: Arx5EnvWithImages,
    policy: WebsocketClientPolicy,
    keys: TerminalKeys,
    prompt: str,
    frequency: float,
    steps_per_inference: int,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
    observe_only: bool,
    preview_enabled: bool,
    rtc_s_min: int,
    rtc_delay_buffer_size: int,
    rtc_max_guidance_weight: float,
) -> None:
    """Overlapping RTC loop: execute while next infer runs with prev_chunk guidance."""
    dt = 1.0 / frequency
    executing = False
    worker: JointRTCInferenceWorker | None = None
    current_chunk = None
    cursor = 0
    schedule_end = 0.0  # wall time of last scheduled waypoint
    last_commanded: np.ndarray | None = None
    prev_commanded: np.ndarray | None = None  # for C1 stitch velocity
    swap_count = 0
    last_swap_mono: float | None = None
    batch_sat_count = 0
    batch_step_count = 0

    def _apply_swap(
        tag: str,
        *,
        swap_at: int,
        polled: tuple,
    ) -> None:
        nonlocal current_chunk, cursor, swap_count, last_swap_mono
        raw_chunk, trim_steps, s_used = polled
        prev_vel = None
        if last_commanded is not None and prev_commanded is not None:
            prev_vel = (
                np.asarray(last_commanded, dtype=np.float32)
                - np.asarray(prev_commanded, dtype=np.float32)
            )
        stitched = stitch_chunk_to_state(
            raw_chunk,
            last_commanded,
            blend_steps=RTC_BLEND_STEPS,
            prev_velocity=prev_vel,
        )
        now = time.monotonic()
        wall_dt = (now - last_swap_mono) if last_swap_mono is not None else None
        swap_count += 1
        last_swap_mono = now
        _log_rtc_swap(
            tag=tag,
            swap_at=swap_at,
            trim_steps=int(trim_steps),
            s_used=int(s_used),
            raw_chunk=raw_chunk,
            stitched=stitched,
            last_commanded=last_commanded,
            latency_s=float(worker.last_latency_s) if worker is not None else 0.0,
            swap_index=swap_count,
            wall_dt_s=wall_dt,
            max_joint_step=max_joint_step,
        )
        current_chunk = stitched
        cursor = 0

    def _get_obs_bundle() -> dict:
        obs = env.get_obs()
        state = measured_state(env)
        return {"obs": obs, "state": state}

    def _rtc_infer_fn(
        *,
        obs_bundle: dict,
        rtc_prev_chunk: np.ndarray,
        rtc_inference_delay: int,
        rtc_execution_horizon: int,
    ) -> tuple[np.ndarray, float]:
        started = time.monotonic()
        payload = rtc_policy_observation(
            obs_bundle["obs"],
            obs_bundle["state"],
            prompt,
            prev_action_chunk=rtc_prev_chunk,
            inference_delay=rtc_inference_delay,
            execute_horizon=rtc_execution_horizon,
            max_guidance_weight=rtc_max_guidance_weight,
            enable_rtc=True,
        )
        result = policy.infer(payload)
        latency = time.monotonic() - started
        actions = np.asarray(result["actions"], dtype=np.float32)
        return actions, latency

    def _stop_worker() -> None:
        nonlocal worker, current_chunk, cursor, schedule_end, last_commanded
        nonlocal prev_commanded
        nonlocal swap_count, last_swap_mono, batch_sat_count, batch_step_count
        if worker is not None:
            if batch_step_count > 0:
                print(
                    f"[RTC] session stats: swaps={swap_count} "
                    f"guard_sat={batch_sat_count}/{batch_step_count} "
                    f"({100.0 * batch_sat_count / batch_step_count:.1f}%)"
                )
            worker.close(wait=True, timeout_s=5.0)
            worker = None
        current_chunk = None
        cursor = 0
        schedule_end = 0.0
        last_commanded = None
        prev_commanded = None
        swap_count = 0
        last_swap_mono = None
        batch_sat_count = 0
        batch_step_count = 0

    def _seed_and_start() -> None:
        """Warmup + first RTC chunk; create worker."""
        nonlocal worker, current_chunk, cursor, schedule_end, last_commanded
        nonlocal last_swap_mono
        _stop_worker()
        obs = env.get_obs()
        state = measured_state(env)
        # First call without prev to learn action_horizon, then seed like kai0 ARX.
        started = time.monotonic()
        warm = policy.infer(policy_observation(obs, state, prompt))
        warm_latency = time.monotonic() - started
        raw = np.asarray(warm["actions"], dtype=np.float32)
        action_horizon = int(raw.shape[0])
        seed_prev = np.tile(state.astype(np.float32)[None, :], (action_horizon, 1))
        started = time.monotonic()
        result = policy.infer(
            rtc_policy_observation(
                obs,
                state,
                prompt,
                prev_action_chunk=seed_prev,
                inference_delay=0,
                execute_horizon=min(rtc_s_min, action_horizon),
                max_guidance_weight=rtc_max_guidance_weight,
                enable_rtc=True,
            )
        )
        latency = time.monotonic() - started
        raw = np.asarray(result["actions"], dtype=np.float32)
        current_chunk = build_joint_chunk(raw)
        cursor = 0
        init_delay = max(
            1, estimate_delay_steps(max(latency, warm_latency), frequency)
        )
        # RTC must schedule in small batches (≤ delay_align). A huge
        # --steps-per-inference (e.g. 50) would dump the whole chunk before
        # maybe_start, leaving an empty leftover and a jerky first swap.
        rtc_batch = min(int(steps_per_inference), 8)
        worker = JointRTCInferenceWorker(
            _rtc_infer_fn,
            frequency=frequency,
            s_min=rtc_s_min,
            delay_buffer_size=rtc_delay_buffer_size,
            initial_delay_steps=init_delay,
            action_dim=ACTION_DIM,
            action_horizon=action_horizon,
            schedule_batch_size=rtc_batch,
            soft_min=RTC_SOFT_MIN,
        )
        d0, switch0, trig0 = worker.plan_horizons(current_chunk.horizon)
        print(
            f"[RTC] initial chunk H={current_chunk.horizon} "
            f"(model_H={action_horizon}) latency={latency:.3f}s warm={warm_latency:.3f}s "
            f"delay~{worker.capped_delay()} s_min={worker.s_min} max_gw={rtc_max_guidance_weight} "
            f"schedule_batch={rtc_batch} (cli steps_per_inference={steps_per_inference}) "
            f"delay_align={worker.delay_align} blend={RTC_BLEND_STEPS} "
            f"soft_min={RTC_SOFT_MIN} stitch=C1+skip0"
        )
        print(
            f"[RTC]   plan: infer_at={trig0} switch_at={switch0} delay={d0} "
            f"first={_fmt14(raw[0])} steps[{_chunk_step_stats(raw)}]"
        )
        print(f"[RTC]   seed_state={_fmt14(state)}")
        last_swap_mono = time.monotonic()
        if steps_per_inference > 8:
            print(
                f"[RTC] NOTE: clamping schedule batch {steps_per_inference}->8; "
                "use --steps-per-inference 8 for RTC"
            )

    try:
        while True:
            obs = env.get_obs()
            requested_infer = False
            for key in keys.read():
                if key == "q":
                    _stop_worker()
                    hold_current(env)
                    return
                if key == "s":
                    executing = False
                    _stop_worker()
                    hold_current(env)
                    print("Policy stopped; holding measured joints")
                elif key == "c":
                    if observe_only:
                        print("Ignoring C: --observe-only")
                    else:
                        executing = True
                        print("Policy execution started (RTC)")
                        _seed_and_start()
                elif key == "i":
                    requested_infer = True

            show_preview(
                obs,
                "RTC EXEC  S:stop Q:quit"
                if executing
                else "HOLD  C:start I:infer Q:quit",
                enabled=preview_enabled,
            )

            if requested_infer and not executing:
                state = measured_state(env)
                started = time.monotonic()
                result = policy.infer(
                    rtc_policy_observation(
                        obs,
                        state,
                        prompt,
                        prev_action_chunk=np.tile(
                            state.astype(np.float32)[None, :], (50, 1)
                        ),
                        inference_delay=0,
                        execute_horizon=steps_per_inference,
                        max_guidance_weight=rtc_max_guidance_weight,
                        enable_rtc=True,
                    )
                )
                latency = time.monotonic() - started
                actions = guarded_chunk(
                    result["actions"],
                    state,
                    count=min(8, steps_per_inference),
                    max_joint_step=max_joint_step,
                    gripper_min=gripper_min,
                    gripper_max=gripper_max,
                )
                print(
                    f"[RTC][infer-only] latency={latency:.3f}s "
                    f"actions={np.asarray(result['actions']).shape} "
                    f"first={np.round(actions[0], 3)}"
                )
                continue

            if not executing:
                time.sleep(0.02)
                continue

            if worker is None or current_chunk is None:
                _seed_and_start()
                continue

            polled = worker.poll(executed_since_chunk=cursor)
            if polled is not None:
                _apply_swap("swap", swap_at=cursor, polled=polled)

            if cursor >= current_chunk.horizon:
                if worker.awaiting_swap:
                    # Chunk ended before commit_end (trimmed shorter than
                    # start+delay). Release the held infer so we don't freeze.
                    release_at = max(cursor, worker.commit_end or cursor)
                    done = worker.poll(executed_since_chunk=release_at)
                    if done is not None:
                        _apply_swap("swap(horizon_flush)", swap_at=cursor, polled=done)
                        continue
                    time.sleep(0.005)
                    continue
                print(
                    f"[RTC] exhausted chunk H={current_chunk.horizon} "
                    f"cursor={cursor}; forcing replan"
                )
                worker.maybe_start(
                    executed_since_chunk=max(cursor, 1),
                    current_chunk=current_chunk,
                    get_obs_bundle=_get_obs_bundle,
                    force=True,
                )
                time.sleep(0.01)
                continue

            # While next infer is pending (in flight or held), only execute the
            # RTC-committed prefix (start + delay). Early swap before that
            # prefix runs snaps the arm; poll holds until cursor >= commit_end.
            if worker.commit_end is not None:
                if cursor >= worker.commit_end:
                    while executing and worker.awaiting_swap:
                        for key in keys.read():
                            if key in {"s", "q"}:
                                executing = False
                                _stop_worker()
                                hold_current(env)
                                if key == "q":
                                    return
                                print("Policy stopped; holding measured joints")
                                break
                        if not executing:
                            break
                        done = worker.poll(executed_since_chunk=cursor)
                        if done is not None:
                            _apply_swap("swap(commit_end)", swap_at=cursor, polled=done)
                            break
                        time.sleep(0.005)
                    continue
                remaining_commit = worker.commit_end - cursor
            else:
                # Infer at switch_s - delay; execute through switch_s before swap.
                _d, _switch_s, s_trigger = worker.plan_horizons(current_chunk.horizon)
                if cursor < s_trigger:
                    remaining_commit = s_trigger - cursor
                else:
                    remaining_commit = worker.delay_align

            remaining = current_chunk.horizon - cursor
            schedule_batch = min(steps_per_inference, worker.delay_align)
            n = min(schedule_batch, remaining, remaining_commit)
            if n <= 0:
                time.sleep(0.005)
                continue
            # Continuity anchor: last commanded pose (not measured), so swap
            # does not yank toward lagged encoder readings.
            anchor = (
                last_commanded
                if last_commanded is not None
                else measured_state(env)
            )
            raw_batch = np.asarray(
                current_chunk.actions[cursor : cursor + n], dtype=np.float64
            )
            batch_actions = guarded_chunk(
                raw_batch,
                anchor,
                count=n,
                max_joint_step=max_joint_step,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )
            # Guard saturation + tracking vs raw policy (diagnose jerk source).
            prev = np.asarray(anchor, dtype=np.float64)
            for i in range(len(batch_actions)):
                batch_step_count += 1
                d_cmd = batch_actions[i] - prev
                joint_max = max(
                    float(np.max(np.abs(d_cmd[0:6]))),
                    float(np.max(np.abs(d_cmd[7:13]))),
                )
                if joint_max >= max_joint_step * 0.975:
                    batch_sat_count += 1
                prev = batch_actions[i]
            raw_vs_guard = float(
                np.max(np.abs(raw_batch[: len(batch_actions)] - batch_actions))
            )
            if raw_vs_guard > max_joint_step * 0.5 or cursor == 0 or (cursor % 16 == 0):
                meas = measured_state(env)
                track = _max_abs(meas, anchor) if last_commanded is not None else float("nan")
                print(
                    f"[RTC] sched cursor={cursor}->{cursor+n}/{current_chunk.horizon} "
                    f"n={n} raw_vs_guard={raw_vs_guard:.4f} "
                    f"track_err={track:.4f} "
                    f"sat_cum={batch_sat_count}/{batch_step_count}"
                )

            t0 = max(time.time() + 0.05, schedule_end)
            action_timestamps = (
                np.arange(len(batch_actions), dtype=np.float64) * dt + t0
            )
            command_joint_actions(env, batch_actions, action_timestamps)
            schedule_end = float(action_timestamps[-1])
            # Keep last two commanded poses for C1 velocity at the next swap.
            if len(batch_actions) >= 2:
                prev_commanded = np.asarray(batch_actions[-2], dtype=np.float64)
            elif last_commanded is not None:
                prev_commanded = np.asarray(last_commanded, dtype=np.float64)
            last_commanded = np.asarray(batch_actions[-1], dtype=np.float64)
            cursor += n
            started = worker.busy or worker.awaiting_swap
            worker.maybe_start(
                executed_since_chunk=cursor,
                current_chunk=current_chunk,
                get_obs_bundle=_get_obs_bundle,
            )
            if not started and (worker.busy or worker.awaiting_swap):
                _d, _switch_s, _st = worker.plan_horizons(current_chunk.horizon)
                _log_rtc_infer_start(
                    cursor=cursor,
                    chunk_h=current_chunk.horizon,
                    delay=_d,
                    switch_s=_switch_s,
                    start_trigger=_st,
                    commit_end=worker.commit_end,
                    leftover=current_chunk.horizon - cursor,
                    max_gw=rtc_max_guidance_weight,
                    state=last_commanded,
                )

            deadline = schedule_end
            while time.time() < deadline:
                for key in keys.read():
                    if key in {"s", "q"}:
                        executing = False
                        _stop_worker()
                        hold_current(env)
                        if key == "q":
                            return
                        print("Policy stopped; holding measured joints")
                        break
                if not executing:
                    break
                time.sleep(0.005)
    finally:
        _stop_worker()


@click.command()
@click.option("--policy-ip", default="127.0.0.1", show_default=True)
@click.option("--policy-port", default=8000, type=int, show_default=True)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot-ports", default="8765,8766", show_default=True)
@click.option("--camera-reorder", default="0,1", show_default=True)
@click.option("--rs-serial", default=None)
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=True)
@click.option("--frequency", default=20.0, type=float, show_default=True)
@click.option("--steps-per-inference", default=8, type=click.IntRange(1, 50), show_default=True)
@click.option("--max-joint-speed", default=1.5, type=float, show_default=True, help="rad/s")
@click.option("--gripper-min", default=0.0, type=float, show_default=True)
@click.option("--gripper-max", default=0.085, type=float, show_default=True)
@click.option("--camera-obs-latency", default=0.17, type=float, show_default=True)
@click.option("--robot-obs-latency", default=0.1867, type=float, show_default=True)
@click.option("--server-timeout", default=600.0, type=float, show_default=True)
@click.option("--observe-only", is_flag=True, help="Never command either arm.")
@click.option(
    "--no-preview",
    is_flag=True,
    help="Disable OpenCV window (required over SSH without X11/DISPLAY).",
)
@click.option(
    "--no-init",
    is_flag=True,
    help="Skip reset_to_home and joint init (stay at current pose).",
)
@click.option(
    "--init-joints",
    default=DEFAULT_INIT_JOINTS,
    show_default=False,
    help=(
        "Absolute init joints: left q6,left grip,right q6,right grip. "
        "Default = median episode-0 from collect_bimanual_10_22."
    ),
)
@click.option(
    "--init-duration",
    default=2.5,
    type=float,
    show_default=True,
    help="Joint init move duration (seconds)",
)
@click.option(
    "--rtc/--no-rtc",
    default=False,
    show_default=True,
    help="Overlapping RTC with prev_action_chunk (server must use --rtc).",
)
@click.option(
    "--rtc-s-min",
    default=8,
    type=click.IntRange(1, 48),
    show_default=True,
    help=(
        "Steps before swapping to the next chunk. Infer starts at s-delay. "
        "NOTE: soft length ≈ H-s_min after trim — raising s_min swaps less often "
        "but shortens server soft-guidance; client C1 blend + soft_min pad compensate."
    ),
)
@click.option(
    "--rtc-delay-buffer-size",
    default=16,
    type=click.IntRange(1, 64),
    show_default=True,
)
@click.option(
    "--rtc-max-guidance-weight",
    default=2,
    type=float,
    show_default=True,
    help="RTC server soft-guidance weight (kai0 default 0.5). Raise carefully.",
)
@click.option("--output", default="data_local/kai0_eval", show_default=True)
@click.option(
    "--log-file",
    default=str(DEFAULT_RUN_LOG),
    show_default=True,
    help="Overwrite this file each run with teed stdout/stderr (for post-run debug).",
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
    steps_per_inference: int,
    max_joint_speed: float,
    gripper_min: float,
    gripper_max: float,
    camera_obs_latency: float,
    robot_obs_latency: float,
    server_timeout: float,
    observe_only: bool,  # noqa: FBT001
    no_preview: bool,  # noqa: FBT001
    no_init: bool,  # noqa: FBT001
    init_joints: str,
    init_duration: float,
    rtc: bool,  # noqa: FBT001
    rtc_s_min: int,
    rtc_delay_buffer_size: int,
    rtc_max_guidance_weight: float,
    output: str,
    log_file: str,
) -> None:
    if frequency <= 0 or max_joint_speed <= 0:
        raise click.BadParameter("frequency and max-joint-speed must be positive")
    if not 0 <= gripper_min < gripper_max:
        raise click.BadParameter("invalid gripper range")
    if rtc_max_guidance_weight <= 0:
        raise click.BadParameter("--rtc-max-guidance-weight must be positive")

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    install_run_log(log_path)

    ports = parse_pair(robot_ports, "--robot-ports")
    camera_order = parse_pair(camera_reorder, "--camera-reorder")
    output_path = (ROOT / output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to policy server ws://{policy_ip}:{policy_port}")
    policy = WebsocketClientPolicy(
        host=policy_ip,
        port=policy_port,
        ping_timeout=server_timeout,
        close_timeout=server_timeout,
    )
    metadata = policy.get_server_metadata()
    print(f"Policy connected; metadata={metadata}")
    if rtc:
        if not metadata.get("supports_rtc"):
            raise click.ClickException(
                "Client --rtc requires server metadata supports_rtc=true. "
                "Restart Kai0 with: uv run scripts/serve_kai0_policy.py "
                "--checkpoint <jax_ckpt> --rtc --port 8000"
            )
        print(
            f"[RTC] enabled: s_min={rtc_s_min}, delay_buf={rtc_delay_buffer_size}, "
            f"max_gw={rtc_max_guidance_weight}, steps_per_inference={steps_per_inference}"
        )
    print("Controls: C=start execution, S=stop/hold, I=infer once, Q=quit")
    if observe_only:
        print("OBSERVE ONLY: motor commands are permanently disabled")

    dt = 1.0 / frequency
    max_joint_step = max_joint_speed * dt
    preview_enabled = (not no_preview) and bool(os.environ.get("DISPLAY"))
    if no_preview:
        print("Preview disabled (--no-preview)")
    elif not preview_enabled:
        print(
            "Preview disabled: DISPLAY is unset (SSH/headless). "
            "Use a local desktop, export DISPLAY, or pass --no-preview."
        )

    init_q14 = None if no_init else parse_init_joints(init_joints)
    do_home = not no_init
    if no_init:
        print("Init skipped (--no-init): arms stay at current pose")
    else:
        print(
            f"Startup init: joint reset_to_home + joint init "
            f"({init_duration:.1f}s) from collect_bimanual_10_22 ep0 median"
        )
        print("NOTE: ZMQ must be --joint (joint-space home + SET_JOINT_*).")

    with SharedMemoryManager() as shm_manager, Arx5EnvWithImages(
        output_dir=output_path,
        robots_config=robot_config(
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
    ) as env, TerminalKeys() as keys:
        print("Waiting for cameras and both ARX5 ZMQ controllers...")
        while not env.is_ready:
            time.sleep(0.1)
        print("Environment ready")

        if init_q14 is not None:
            time.sleep(0.5)  # let joint reset_to_home settle
            move_to_init_joints(
                env,
                init_q14,
                frequency=frequency,
                duration=float(init_duration),
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )

        try:
            common = dict(
                env=env,
                policy=policy,
                keys=keys,
                prompt=prompt,
                frequency=frequency,
                steps_per_inference=steps_per_inference,
                max_joint_step=max_joint_step,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                observe_only=observe_only,
                preview_enabled=preview_enabled,
            )
            if rtc:
                run_rtc(
                    **common,
                    rtc_s_min=rtc_s_min,
                    rtc_delay_buffer_size=rtc_delay_buffer_size,
                    rtc_max_guidance_weight=rtc_max_guidance_weight,
                )
            else:
                run_open_loop(**common)
        finally:
            try:
                hold_current(env)
            except Exception as exc:  # noqa: BLE001
                print(f"Final hold warning: {exc}")
            if preview_enabled:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
