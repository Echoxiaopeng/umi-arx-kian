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

With ``--dagger``, policy rollout can be interrupted for gravity-comp drag
teleop; episodes are saved as UMI ``replay_buffer.zarr`` (TCP action schema,
same as ``collect_drag_umi_zmq_bimanual.py``) plus an intervention sidecar.

ZMQ servers should start with ``--joint`` (Kai0 is joint throughout).
Example:

    python python/communication/zmq_server.py X5_umi can0 --port 8765 --joint --no-idle-reset-home --no_cmd_timeout 0
    python python/communication/zmq_server.py X5_umi can1 --port 8766 --joint --no-idle-reset-home --no_cmd_timeout 0

    # Kai0 server with RTC:
    #   uv run scripts/serve_kai0_policy.py --checkpoint <jax_ckpt> --rtc --port 8000

# ssh 远程运行方式：
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
conda activate umi-arx
python scripts/eval_arx5_kai0_bimanual.py \
    --policy-ip 127.0.0.1 --policy-port 8000 \
    --robot-ports 8765,8766 --camera-reorder 0,1 \
    --steps-per-inference 50



# Gentler RTC fade (smoothstep handoff + longer soft/blend):
python scripts/eval_arx5_kai0_bimanual.py \
        --policy-ip 127.0.0.1 --policy-port 8000  \
        --robot-ports 8765,8766 --camera-reorder 0,1 \
        --rtc --frequency 20 \
        --steps-per-inference 4 --rtc-s-min 32 \
        --rtc-max-guidance-weight 0.7 \
        --rtc-smooth-method temporal \
        --rtc-blend-steps 16 --rtc-fade smoothstep

    # DAgger (policy + drag intervention, UMI zarr):
    python scripts/eval_arx5_kai0_bimanual.py \
      --policy-ip 127.0.0.1 --policy-port 8000 \
      --robot-ports 8765,8766 --camera-reorder 0,1 --no-preview \
      --dagger --output data_local/kai0_dagger \
      --steps-per-inference 32

Controls (default): C=start execution, S=stop/hold, I=infer once without motion, Q=quit.
Controls (--dagger): C=start episode+policy, D=drag intervene, R=resume policy,
  S=save episode, B=abort episode, Q=quit.

On start (after cameras/arms are ready), a throw-away inference warms the policy
server (JAX compile) so the first C is not delayed; use --skip-server-warmup
to disable.
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

from modules.arx5_controller import apply_drag_profile  # noqa: E402
from modules.arx5_controller import get_drag_profile  # noqa: E402
from modules.arx5_env_with_images_bimanual_rs import Arx5EnvWithImages  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from utils.kai0_dagger_drag import abort_episode  # noqa: E402
from utils.kai0_dagger_drag import enter_drag_mode  # noqa: E402
from utils.kai0_dagger_drag import leave_drag_stay_pose  # noqa: E402
from utils.kai0_dagger_drag import record_pose_action  # noqa: E402
from utils.kai0_dagger_drag import save_intervention_sidecar  # noqa: E402
from utils.kai0_rtc import JointRTCChunk  # noqa: E402
from utils.kai0_rtc import JointRTCInferenceWorker  # noqa: E402
from utils.kai0_rtc import apply_rtc_client_smooth  # noqa: E402
from utils.kai0_rtc import build_joint_chunk  # noqa: E402
from utils.kai0_rtc import estimate_delay_steps  # noqa: E402
from utils.kai0_rtc import stitch_chunk_to_state  # noqa: E402
from utils.other_util import precise_wait  # noqa: E402

DEFAULT_PROMPT = "Flatten and fold the cloth."
ACTION_DIM = 14
# Client smooth window: temporal=min_smooth_steps (kai0 ARX default 10);
# c1=blend length. Prefer temporal to match official ARX RTC client.
# 16 + smoothstep keeps leftover influence longer and avoids a C1 kink when
# the blend hands off to the raw suffix (mid-chunk jerk).
RTC_BLEND_STEPS = 16
RTC_SMOOTH_METHOD = "temporal"
RTC_FADE = "smoothstep"  # linear | smoothstep | ease_out
# Request at least this many soft-guidance steps (extends into hold-last pad).
RTC_SOFT_MIN = 32
# Command-space acceleration limit for RTC. At 20 Hz, 8 rad/s² allows the
# per-step velocity (Δq) to change by at most 0.02 rad.
RTC_MAX_JOINT_ACCEL = 8.0



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
    """Mirror writes to the terminal and one or more log files (line-buffered)."""

    def __init__(self, primary, *secondaries) -> None:
        self._primary = primary
        self._secondaries = secondaries

    def write(self, data):
        self._primary.write(data)
        self._primary.flush()
        for secondary in self._secondaries:
            secondary.write(data)
            secondary.flush()
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        for secondary in self._secondaries:
            secondary.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def install_run_log(log_path: Path) -> Path:
    """Tee stdout/stderr to a timestamped ``run_*.log`` and overwrite ``log_path``.

    ``log_path`` is the stable last-run alias (default ``.../last_run.log``).
    A sibling ``run_YYYYMMDD_HHMMSS.log`` keeps each execution for later debug.
    """
    last_path = Path(log_path).resolve()
    log_dir = last_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    local_now = datetime.now(timezone.utc).astimezone()
    timed_path = log_dir / f"run_{local_now.strftime('%Y%m%d_%H%M%S')}.log"
    # Kept open for the process lifetime; closed in atexit.
    timed_file = open(timed_path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    last_file = open(last_path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _TeeIO(real_out, timed_file, last_file)
    sys.stderr = _TeeIO(real_err, timed_file, last_file)

    def _restore() -> None:
        sys.stdout = real_out
        sys.stderr = real_err
        for handle in (timed_file, last_file):
            try:
                handle.flush()
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    atexit.register(_restore)
    print(f"[log] writing run log -> {timed_path}")
    print(f"[log] also updating last log -> {last_path}")
    print(f"[log] started_at={local_now.isoformat(timespec='seconds')}")
    print(f"[log] argv={' '.join(sys.argv)}")
    return timed_path


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


def warmup_policy_server(
    *,
    env: Arx5EnvWithImages,
    policy: WebsocketClientPolicy,
    prompt: str,
    rtc: bool,
    rtc_s_min: int,
    rtc_max_guidance_weight: float,
) -> None:
    """Throw-away infer(s) so the policy server compiles before the first C.

    Arms are not commanded. First server-side run may take minutes (JAX).
    With ``rtc``, also warms the RTC guidance path used by ``_seed_and_start``.
    """
    obs = env.get_obs()
    state = measured_state(env)
    print(
        "[warmup] sending throw-away inference "
        "(first run may take 1-3 min while the model compiles; arms will not move)..."
    )
    sys.stdout.flush()
    started = time.monotonic()
    warm = policy.infer(policy_observation(obs, state, prompt))
    warm_latency = time.monotonic() - started
    raw = np.asarray(warm["actions"], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != ACTION_DIM:
        raise RuntimeError(
            f"Warmup returned {raw.shape}, expected [T, {ACTION_DIM}]"
        )
    action_horizon = int(raw.shape[0])
    print(
        f"[warmup] plain infer ready in {warm_latency:.1f}s "
        f"actions={raw.shape}"
    )
    sys.stdout.flush()

    if not rtc:
        print("[warmup] policy server ready (press C when ready to execute)")
        return

    seed_prev = np.tile(state.astype(np.float32)[None, :], (action_horizon, 1))
    print(
        "[warmup] sending RTC throw-away inference "
        "(compiles guidance path; arms will not move)..."
    )
    sys.stdout.flush()
    started = time.monotonic()
    result = policy.infer(
        rtc_policy_observation(
            obs,
            state,
            prompt,
            prev_action_chunk=seed_prev,
            inference_delay=0,
            execute_horizon=min(int(rtc_s_min), action_horizon),
            max_guidance_weight=float(rtc_max_guidance_weight),
            enable_rtc=True,
        )
    )
    rtc_latency = time.monotonic() - started
    rtc_raw = np.asarray(result["actions"], dtype=np.float32)
    print(
        f"[warmup] RTC infer ready in {rtc_latency:.1f}s "
        f"actions={rtc_raw.shape} "
        f"total={warm_latency + rtc_latency:.1f}s"
    )
    print("[warmup] policy server ready (press C when ready to execute)")
    sys.stdout.flush()


def guarded_chunk(
    actions: np.ndarray,
    current: np.ndarray,
    *,
    count: int,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
    max_joint_step_per_arm: tuple[float, float] | None = None,
    previous_before_current: np.ndarray | None = None,
    max_joint_accel_step: float | None = None,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(f"Policy returned {actions.shape}, expected [T, {ACTION_DIM}]")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Policy returned NaN/Inf actions")

    chunk = actions[: min(count, len(actions))].copy()
    previous = np.asarray(current, dtype=np.float64).copy()
    previous_before = (
        np.asarray(previous_before_current, dtype=np.float64).copy()
        if previous_before_current is not None
        else None
    )
    for index in range(len(chunk)):
        for arm_i, offset in enumerate((0, 7)):
            lim = (
                float(max_joint_step_per_arm[arm_i])
                if max_joint_step_per_arm is not None
                else float(max_joint_step)
            )
            delta = chunk[index, offset : offset + 6] - previous[offset : offset + 6]
            if (
                previous_before is not None
                and max_joint_accel_step is not None
                and max_joint_accel_step > 0
            ):
                prev_delta = (
                    previous[offset : offset + 6]
                    - previous_before[offset : offset + 6]
                )
                accel_step = float(max_joint_accel_step)
                delta = np.clip(
                    delta,
                    prev_delta - accel_step,
                    prev_delta + accel_step,
                )
            # Velocity safety limit remains authoritative if the incoming
            # velocity was already outside the acceleration-feasible range.
            delta = np.clip(delta, -lim, lim)
            chunk[index, offset : offset + 6] = (
                previous[offset : offset + 6] + delta
            )
            chunk[index, offset + 6] = np.clip(
                chunk[index, offset + 6], gripper_min, gripper_max
            )
        previous_before = previous
        previous = chunk[index]
    return chunk


def resume_soft_step_limits(
    max_joint_step: float,
    batches_left: int,
    *,
    total_batches: int = 10,
    hold_batches: int = 6,
) -> tuple[float, float]:
    """Δq caps after drag→policy (R).

    First ``hold_batches``: lim=0 (pure hold). Then ease-in equally on both
    arms up to ~0.18 * max_joint_step (keeps the hold→move onset gentle).
    """
    total = max(1, int(total_batches))
    left = max(0, int(batches_left))
    done = total - left
    if done < int(hold_batches):
        return 0.0, 0.0
    move_total = max(1, total - int(hold_batches))
    move_done = done - int(hold_batches)
    t = float(np.clip(move_done / float(move_total), 0.0, 1.0))
    # cubic ease-in: starts near 0, ends ~0.18 of max_joint_step
    base = 0.02 + 0.16 * (t * t * t)
    lim = float(max_joint_step) * base
    return lim, lim


def resume_post_soft_step_limit(
    max_joint_step: float,
    batches_left: int,
    *,
    total_batches: int = 6,
) -> float:
    """Taper from soft cap → full max_joint_step after the soft window."""
    total = max(1, int(total_batches))
    left = max(0, int(batches_left))
    done = total - left
    t = float(np.clip(done / float(total), 0.0, 1.0))
    # 0.20 → 1.00 of max_joint_step
    return float(max_joint_step) * (0.20 + 0.80 * t)


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


def command_joint_immediate(
    env: Arx5EnvWithImages,
    action: np.ndarray,
    *,
    duration_s: float | None = None,
) -> None:
    """Tick command: blend one absolute dual-arm pose over duration_s.

    No wall-clock stamps. ``duration_s=None`` → each arm uses ``1/frequency``.
    Pass ``duration_s=0`` to snap-hold.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise RuntimeError(f"Expected action [{ACTION_DIM}], got {action.shape}")
    for robot_index, robot in enumerate(env.robots):
        target = np.asarray(
            action[7 * robot_index : 7 * robot_index + 7], dtype=np.float64
        )
        robot.set_joint_immediate(target, duration_s=duration_s)


def hold_current(
    env: Arx5EnvWithImages,
    *,
    horizon_s: float = 0.05,
    immediate: bool = False,
) -> float:
    """Hold measured joints. Returns wall-clock target time of the hold waypoint."""
    q = measured_state(env).reshape(1, -1)
    if immediate:
        # Snap-hold: freeze at measured joints.
        command_joint_immediate(env, q[0], duration_s=0.0)
        return float(time.time())
    target_t = time.time() + float(horizon_s)
    command_joint_actions(
        env,
        q,
        np.asarray([target_t], dtype=np.float64),
    )
    return float(target_t)


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
    debug_timing: bool,
) -> None:
    """Original blocking infer → execute chunk → wait loop."""
    dt = 1.0 / frequency
    executing = False
    prev_chunk_end_wall: float | None = None
    chunk_index = 0
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

        infer_start_wall = time.time()
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

        chunk_index += 1
        boundary_gap_ms = (
            (infer_start_wall - prev_chunk_end_wall) * 1000.0
            if prev_chunk_end_wall is not None
            else float("nan")
        )
        command_start_time = time.time() + 0.05
        action_timestamps = (
            np.arange(len(actions), dtype=np.float64) * dt + command_start_time
        )
        if debug_timing:
            exec_window_ms = (len(actions) * dt) * 1000.0
            bridge_ms = (
                command_start_time - infer_start_wall
            ) * 1000.0  # infer-done -> first waypoint target
            print(
                f"[DBG][OL] chunk#{chunk_index} "
                f"infer={latency*1000.0:.1f}ms boundary_gap={boundary_gap_ms:.1f}ms "
                f"bridge={bridge_ms:.1f}ms exec_window={exec_window_ms:.1f}ms "
                f"H={len(actions)}"
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
        prev_chunk_end_wall = deadline


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
    rtc_blend_steps: int,
    rtc_smooth_method: str,
    rtc_max_joint_accel: float,
    rtc_fade: str = RTC_FADE,
    debug_timing: bool = False,
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
    underrun_count = 0
    max_underrun_ms = 0.0
    blend_steps = max(1, int(rtc_blend_steps))
    smooth_method = str(rtc_smooth_method).strip().lower()
    fade = str(rtc_fade).strip().lower()
    max_joint_accel_step = float(rtc_max_joint_accel) * dt * dt

    def _apply_swap(
        tag: str,
        *,
        swap_at: int,
        polled: tuple,
    ) -> None:
        nonlocal current_chunk, cursor, swap_count, last_swap_mono
        raw_chunk, trim_steps, s_used = polled
        stitched = apply_rtc_client_smooth(
            raw_chunk,
            method=smooth_method,
            old_chunk=current_chunk,
            old_cursor=swap_at,
            last_commanded=last_commanded,
            prev_commanded=prev_commanded,
            blend_steps=blend_steps,
            fade=fade,
        )
        if last_commanded is not None and max_joint_accel_step > 0:
            smoothed_actions = guarded_chunk(
                stitched.actions,
                last_commanded,
                count=stitched.horizon,
                max_joint_step=max_joint_step,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                previous_before_current=prev_commanded,
                max_joint_accel_step=max_joint_accel_step,
            )
            stitched = JointRTCChunk(
                actions=smoothed_actions.astype(np.float32),
                created_monotonic=stitched.created_monotonic,
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
        nonlocal underrun_count, max_underrun_ms
        if worker is not None:
            if batch_step_count > 0:
                print(
                    f"[RTC] session stats: swaps={swap_count} "
                    f"guard_sat={batch_sat_count}/{batch_step_count} "
                    f"({100.0 * batch_sat_count / batch_step_count:.1f}%)"
                )
            if debug_timing:
                print(
                    f"[DBG][RTC] underrun_count={underrun_count} "
                    f"max_underrun={max_underrun_ms:.1f}ms"
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
        underrun_count = 0
        max_underrun_ms = 0.0

    def _seed_and_start() -> None:
        """Warmup + first RTC chunk; create worker."""
        nonlocal worker, current_chunk, cursor, schedule_end, last_commanded
        nonlocal prev_commanded, last_swap_mono
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
        initial_actions = guarded_chunk(
            raw,
            state,
            count=len(raw),
            max_joint_step=max_joint_step,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            previous_before_current=state,
            max_joint_accel_step=max_joint_accel_step,
        )
        current_chunk = build_joint_chunk(initial_actions)
        cursor = 0
        # Seed zero command velocity so the first RTC chunk ramps from rest.
        last_commanded = np.asarray(state, dtype=np.float64).copy()
        prev_commanded = np.asarray(state, dtype=np.float64).copy()
        init_delay = max(
            1, estimate_delay_steps(max(latency, warm_latency), frequency)
        )
        # RTC must schedule in small batches (≤ delay_align). A huge
        # --steps-per-inference (e.g. 50) would dump the whole chunk before
        # maybe_start, leaving an empty leftover and a jerky first swap.
        # Cap at 4 so delay/trim tracks ~0.12s RTT (~3 steps@20Hz), not 8.
        rtc_batch = min(int(steps_per_inference), 4)
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
            f"delay_align={worker.delay_align} smooth={smooth_method} "
            f"fade={fade} blend/min_m={blend_steps} soft_min={RTC_SOFT_MIN} "
            f"max_accel={rtc_max_joint_accel:.2f}rad/s^2 "
            f"(ddq_step={max_joint_accel_step:.4f}rad)"
        )
        print(
            f"[RTC]   plan: infer_at={trig0} switch_at={switch0} delay={d0} "
            f"first={_fmt14(raw[0])} steps[{_chunk_step_stats(raw)}]"
        )
        print(f"[RTC]   seed_state={_fmt14(state)}")
        last_swap_mono = time.monotonic()
        if steps_per_inference > 4:
            print(
                f"[RTC] NOTE: clamping schedule batch {steps_per_inference}->4; "
                "use --steps-per-inference 4 for RTC"
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
                if debug_timing and current_chunk is not None:
                    remain = max(0, current_chunk.horizon - cursor)
                    print(
                        f"[DBG][RTC] swap_ready cursor={cursor}/{current_chunk.horizon} "
                        f"leftover={remain}"
                    )
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
                            if debug_timing and current_chunk is not None:
                                remain = max(0, current_chunk.horizon - cursor)
                                print(
                                    f"[DBG][RTC] swap_commit cursor={cursor}/"
                                    f"{current_chunk.horizon} leftover={remain}"
                                )
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
                if debug_timing and current_chunk is not None and worker.commit_end is not None:
                    print(
                        f"[DBG][RTC] wait_commit cursor={cursor}/{current_chunk.horizon} "
                        f"commit_end={worker.commit_end}"
                    )
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

            now_wall = time.time()
            if schedule_end > 0.0:
                # Continue on one fixed control grid across 4-step batches and
                # RTC swaps. Re-anchoring every batch at ``now + dt`` adds the
                # Python loop overhead to the boundary interval (typically
                # 60--65 ms instead of 50 ms at 20 Hz), which is felt as a
                # periodic 5 Hz speed ripple.
                grid_t0 = schedule_end + dt
                if grid_t0 > now_wall:
                    t0 = grid_t0
                else:
                    # We genuinely missed the next slot. Do not submit a stale
                    # waypoint; restart one period ahead and make the underrun
                    # visible in the log.
                    late_s = now_wall - grid_t0
                    late_ms = late_s * 1000.0
                    underrun_count += 1
                    if late_ms > max_underrun_ms:
                        max_underrun_ms = late_ms
                    t0 = now_wall + dt
                    print(
                        f"[RTC] schedule underrun={late_ms:.1f}ms; "
                        "re-anchoring control grid"
                    )
            else:
                t0 = now_wall + dt
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
                if debug_timing:
                    remain = current_chunk.horizon - cursor
                    print(
                        f"[DBG][RTC] infer_launched cursor={cursor}/{current_chunk.horizon} "
                        f"remain={remain} commit_end={worker.commit_end} "
                        f"delay_align={worker.delay_align}"
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


def _sync_intervention_labels(
    env: Arx5EnvWithImages,
    labels: list[int],
    flag: int,
) -> None:
    """Grow labels to match action_accumulator grid after a record put."""
    if env.action_accumulator is None:
        return
    n = len(env.action_accumulator)
    while len(labels) < n:
        labels.append(int(flag))


def run_dagger(
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
    output_dir: Path,
    use_rtc: bool,
    rtc_s_min: int,
    rtc_delay_buffer_size: int,
    rtc_max_guidance_weight: float,
    rtc_blend_steps: int,
    rtc_smooth_method: str,
    rtc_fade: str = RTC_FADE,
    debug_timing: bool = False,  # reserved; keep signature aligned with common args
) -> None:
    """Policy-in-the-loop DAgger with gravity-comp drag and UMI zarr episodes."""
    dt = 1.0 / frequency
    blend_steps = max(1, int(rtc_blend_steps))
    smooth_method = str(rtc_smooth_method).strip().lower()
    fade = str(rtc_fade).strip().lower()
    recording = False
    drag_mode = False
    executing = False
    intervention_labels: list[int] = []
    last_rotvecs: list[np.ndarray | None] = [None, None]
    # Open-loop chunk bookkeeping
    ol_actions: np.ndarray | None = None
    ol_schedule_end = 0.0
    # After R: soft-start so first chunks don't dump a long jerking traj.
    resume_soft_batches_left = 0
    resume_post_soft_batches_left = 0
    resume_soft_total = 14
    resume_hold_batches = 6
    resume_post_soft_total = 6
    # RTC bookkeeping
    worker: JointRTCInferenceWorker | None = None
    current_chunk = None
    cursor = 0
    schedule_end = 0.0
    last_commanded: np.ndarray | None = None
    prev_commanded: np.ndarray | None = None

    def _record_tick(obs: dict) -> None:
        if not recording:
            return
        record_pose_action(env, obs, last_rotvecs=last_rotvecs)
        _sync_intervention_labels(env, intervention_labels, 1 if drag_mode else 0)

    def _stop_rtc() -> None:
        nonlocal worker, current_chunk, cursor, schedule_end
        nonlocal last_commanded, prev_commanded
        if worker is not None:
            worker.close(wait=True, timeout_s=5.0)
            worker = None
        current_chunk = None
        cursor = 0
        schedule_end = 0.0
        last_commanded = None
        prev_commanded = None

    def _clear_ol_chunk() -> None:
        nonlocal ol_actions, ol_schedule_end
        ol_actions = None
        ol_schedule_end = 0.0

    def _stop_policy_motion() -> None:
        _stop_rtc()
        _clear_ol_chunk()
        try:
            hold_current(env)
        except Exception as exc:  # noqa: BLE001
            print(f"hold_current warning: {exc}")

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
        return np.asarray(result["actions"], dtype=np.float32), latency

    def _seed_rtc(*, resume_blend: bool = False) -> None:
        nonlocal worker, current_chunk, cursor, schedule_end
        nonlocal last_commanded, prev_commanded
        _stop_rtc()

        def _hold_pulse(tag: str, horizon_s: float = 0.35) -> None:
            nonlocal schedule_end
            if not resume_blend:
                return
            try:
                t_hold = hold_current(env, horizon_s=horizon_s)
                schedule_end = max(float(schedule_end), t_hold)
            except Exception as exc:  # noqa: BLE001
                print(f"{tag} warning: {exc}")

        # One hold covering warm+RTC infer; track its target in schedule_end.
        _hold_pulse("pre-seed hold", horizon_s=0.40)
        obs = env.get_obs()
        state = measured_state(env)
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
        raw_chunk = build_joint_chunk(raw)
        # Match plain run_rtc on C: raw chunk. Only stitch hard after drag (R).
        if resume_blend:
            # Fresh hold AFTER infer, then policy must schedule after this time.
            _hold_pulse("post-infer hold", horizon_s=0.15)
            state = measured_state(env)
            blend = blend_steps * 2
            current_chunk = stitch_chunk_to_state(
                raw_chunk,
                state,
                blend_steps=blend,
                prev_velocity=None,
                keep_anchor=True,  # from rest: first cmd must be hold, not skip0
            )
            # ~0.4s pure hold in the chunk before any policy motion.
            hold_n = max(6, min(10, int(round(0.40 * frequency))))
            hold_rows = np.tile(
                np.asarray(state, dtype=np.float32)[None, :], (hold_n, 1)
            )
            current_chunk = JointRTCChunk(
                actions=np.concatenate(
                    [hold_rows, np.asarray(current_chunk.actions, dtype=np.float32)],
                    axis=0,
                ),
                created_monotonic=current_chunk.created_monotonic,
            )
            last_commanded = np.asarray(state, dtype=np.float64).copy()
            jump_l = float(
                np.max(np.abs(raw[0, 0:6] - state[0:6]))
            ) if len(raw) else 0.0
            jump_r = float(
                np.max(np.abs(raw[0, 7:13] - state[7:13]))
            ) if len(raw) else 0.0
            print(
                f"[DAgger/RTC] resume stitch keep_anchor hold_n={hold_n} "
                f"raw0_jump L={jump_l:.4f} R={jump_r:.4f} "
                f"schedule_end={schedule_end:.3f}"
            )
        else:
            blend = 0
            current_chunk = raw_chunk
            last_commanded = None
        cursor = 0
        if resume_blend:
            # Strictly after the last hold waypoint (never behind).
            schedule_end = max(float(schedule_end), time.time() + 0.05) + 0.02
        else:
            schedule_end = 0.0
        prev_commanded = None
        init_delay = max(
            1, estimate_delay_steps(max(latency, warm_latency), frequency)
        )
        rtc_batch = min(int(steps_per_inference), 4)
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
        print(
            f"[DAgger/RTC] seeded H={current_chunk.horizon} "
            f"latency={latency:.3f}s blend={blend} "
            f"schedule_batch={rtc_batch} delay_align={worker.delay_align} "
            f"state={_fmt14(state)}"
        )

    def _start_episode() -> None:
        nonlocal recording, drag_mode, executing, intervention_labels, last_rotvecs
        if observe_only:
            print("Ignoring C: --observe-only")
            return
        if recording:
            if drag_mode:
                print("Already recording in drag; press R to resume policy")
            else:
                executing = True
                print("Policy execution resumed")
                if use_rtc:
                    _seed_rtc()
                else:
                    _clear_ol_chunk()
            return
        episode_id = env.replay_buffer.n_episodes
        eval_t_start = time.time() + 0.5
        env.start_episode(eval_t_start)
        recording = True
        drag_mode = False
        executing = True
        intervention_labels = []
        last_rotvecs = [None, None]
        _clear_ol_chunk()
        print(f"[DAgger] Episode {episode_id} started (policy)")
        if use_rtc:
            _seed_rtc()

    def _enter_drag() -> None:
        nonlocal drag_mode, executing
        if not recording:
            print("Press C to start an episode before intervening")
            return
        if drag_mode:
            print("Already in drag mode")
            return
        executing = False
        _stop_policy_motion()
        enter_drag_mode(env)
        drag_mode = True
        print("[DAgger] Drag intervention ON (R=resume policy, S=save, B=abort)")

    def _resume_policy() -> None:
        nonlocal drag_mode, executing, resume_soft_batches_left
        nonlocal resume_post_soft_batches_left, schedule_end
        if not recording:
            print("No active episode")
            return
        if not drag_mode:
            print("Not in drag mode")
            return
        leave_drag_stay_pose(env)
        drag_mode = False
        # One short hold to arm joint_control; seed tracks schedule_end itself.
        try:
            schedule_end = max(float(schedule_end), hold_current(env, horizon_s=0.20))
        except Exception as exc:  # noqa: BLE001
            print(f"post-drag hold warning: {exc}")
        time.sleep(0.20)
        try:
            schedule_end = max(float(schedule_end), hold_current(env, horizon_s=0.25))
        except Exception as exc:  # noqa: BLE001
            print(f"post-settle hold warning: {exc}")
        executing = True
        _clear_ol_chunk()
        # Soft-start after R: pure hold → ease-in → post-soft taper to full step.
        if use_rtc:
            resume_soft_batches_left = int(resume_soft_total)
            resume_post_soft_batches_left = int(resume_post_soft_total)
            _seed_rtc(resume_blend=True)
        else:
            resume_soft_batches_left = 8
            resume_post_soft_batches_left = 4
        print(
            "[DAgger] Resumed policy from current pose "
            f"(soft={resume_soft_batches_left} post={resume_post_soft_batches_left})"
        )

    def _save_episode() -> None:
        nonlocal recording, drag_mode, executing, intervention_labels, last_rotvecs
        if not recording:
            print("No active episode to save")
            return
        executing = False
        _stop_rtc()
        _clear_ol_chunk()
        if drag_mode:
            leave_drag_stay_pose(env)
            drag_mode = False
        try:
            hold_current(env)
        except Exception as exc:  # noqa: BLE001
            print(f"pre-save hold warning: {exc}")

        n_acc = (
            len(env.action_accumulator) if env.action_accumulator is not None else 0
        )
        labels_arr = np.asarray(intervention_labels[:n_acc], dtype=np.uint8)
        if n_acc == 0:
            print("[DAgger] Empty episode; aborting instead of save")
            abort_episode(env)
        else:
            env.end_episode()
            if env.replay_buffer.n_episodes > 0:
                ep_id = env.replay_buffer.n_episodes - 1
                try:
                    ep = env.replay_buffer.get_episode(ep_id, copy=False)
                    t_ep = int(len(ep["action"]))
                    labels_arr = labels_arr[:t_ep]
                except Exception as exc:  # noqa: BLE001
                    print(f"intervention align warning: {exc}")
                save_intervention_sidecar(output_dir, ep_id, labels_arr)
                print(f"[DAgger] Saved episode {ep_id}")
            else:
                print("[DAgger] end_episode wrote nothing")

        recording = False
        intervention_labels = []
        last_rotvecs = [None, None]

    def _abort() -> None:
        nonlocal recording, drag_mode, executing, intervention_labels, last_rotvecs
        if not recording:
            print("No active episode to abort")
            return
        executing = False
        _stop_rtc()
        _clear_ol_chunk()
        if drag_mode:
            leave_drag_stay_pose(env)
            drag_mode = False
        abort_episode(env)
        recording = False
        intervention_labels = []
        last_rotvecs = [None, None]
        try:
            hold_current(env)
        except Exception as exc:  # noqa: BLE001
            print(f"post-abort hold warning: {exc}")

    def _status_text() -> str:
        if not recording:
            return "IDLE  C:start D:n/a S:n/a Q:quit"
        if drag_mode:
            return "DRAG  R:policy S:save B:abort Q:quit"
        if executing:
            tag = "RTC" if use_rtc else "OL"
            return f"POLICY({tag})  D:drag S:save B:abort Q:quit"
        return "HOLD  C:policy D:drag S:save B:abort Q:quit"

    def _schedule_ol_batch() -> None:
        nonlocal ol_actions, ol_schedule_end, resume_soft_batches_left
        obs = env.get_obs()
        current = measured_state(env)
        started = time.monotonic()
        result = policy.infer(policy_observation(obs, current, prompt))
        latency = time.monotonic() - started
        soft = resume_soft_batches_left > 0
        count = min(8, int(steps_per_inference)) if soft else int(steps_per_inference)
        per_arm = None
        step_lim = float(max_joint_step)
        if soft:
            per_arm = resume_soft_step_limits(
                max_joint_step,
                resume_soft_batches_left,
                total_batches=6,
                hold_batches=2,
            )
            step_lim = max(per_arm) if max(per_arm) > 0 else 1e-6
        if soft and max(per_arm) <= 0.0:
            ol_actions = np.tile(
                np.asarray(current, dtype=np.float64).reshape(1, -1), (count, 1)
            )
        else:
            ol_actions = guarded_chunk(
                result["actions"],
                current,
                count=count,
                max_joint_step=step_lim,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                max_joint_step_per_arm=per_arm,
            )
        if soft:
            resume_soft_batches_left -= 1
            d0 = ol_actions[0] - np.asarray(current, dtype=np.float64)
            print(
                f"[DAgger/OL] soft lim L={per_arm[0]:.4f} R={per_arm[1]:.4f} "
                f"d0 L={float(np.max(np.abs(d0[0:6]))):.4f} "
                f"R={float(np.max(np.abs(d0[7:13]))):.4f}"
            )
        t0 = max(time.time() + 0.05, float(ol_schedule_end) + 0.01)
        stamps = np.arange(len(ol_actions), dtype=np.float64) * dt + t0
        command_joint_actions(env, ol_actions, stamps)
        ol_schedule_end = float(stamps[-1])
        soft_tag = " soft" if soft else ""
        print(
            f"[DAgger/OL] infer={latency:.3f}s exec={len(ol_actions)}{soft_tag} "
            f"first={_fmt14(ol_actions[0])}"
        )

    def _schedule_rtc_batch() -> bool:
        """Schedule next RTC batch; return False if nothing to do this tick."""
        nonlocal current_chunk, cursor, schedule_end, last_commanded, prev_commanded
        nonlocal resume_soft_batches_left, resume_post_soft_batches_left
        if worker is None or current_chunk is None:
            _seed_rtc()
            return False

        soft = resume_soft_batches_left > 0
        post_soft = (not soft) and resume_post_soft_batches_left > 0
        polled = worker.poll(executed_since_chunk=cursor)
        if polled is not None:
            # Soft resume from standstill: defer RTC swaps (reinits jerked arms).
            # Post-soft: allow temporal swaps only (no c1).
            if soft:
                print(
                    f"[DAgger/RTC] soft: defer swap "
                    f"(left={resume_soft_batches_left})"
                )
            else:
                raw_chunk, trim_steps, s_used = polled
                blend = blend_steps * 2 if post_soft else blend_steps
                method = "temporal" if post_soft else smooth_method
                if method == "c1":
                    method = "temporal"
                stitched = apply_rtc_client_smooth(
                    raw_chunk,
                    method=method,
                    old_chunk=current_chunk,
                    old_cursor=cursor,
                    last_commanded=last_commanded,
                    prev_commanded=(
                        None if post_soft else prev_commanded
                    ),
                    blend_steps=blend,
                    fade=fade,
                )
                print(
                    f"[DAgger/RTC] swap cursor={cursor} trim={trim_steps} "
                    f"s={s_used} H={stitched.horizon} smooth={method} "
                    f"fade={fade} blend={blend}"
                )
                current_chunk = stitched
                cursor = 0

        if cursor >= current_chunk.horizon:
            if worker.awaiting_swap:
                return False
            worker.maybe_start(
                executed_since_chunk=max(cursor, 1),
                current_chunk=current_chunk,
                get_obs_bundle=_get_obs_bundle,
                force=True,
            )
            return False

        if worker.commit_end is not None:
            if cursor >= worker.commit_end:
                # Hold until the pending infer commits, then swap next tick.
                if soft:
                    return False
                if worker.awaiting_swap:
                    done = worker.poll(executed_since_chunk=cursor)
                    if done is not None:
                        raw_chunk, trim_steps, s_used = done
                        blend = blend_steps * 2 if post_soft else blend_steps
                        method = "temporal" if post_soft else smooth_method
                        if method == "c1":
                            method = "temporal"
                        current_chunk = apply_rtc_client_smooth(
                            raw_chunk,
                            method=method,
                            old_chunk=current_chunk,
                            old_cursor=cursor,
                            last_commanded=last_commanded,
                            prev_commanded=(
                                None if post_soft else prev_commanded
                            ),
                            blend_steps=blend,
                            fade=fade,
                        )
                        print(
                            f"[DAgger/RTC] swap(commit_end) trim={trim_steps} "
                            f"s={s_used} H={current_chunk.horizon} "
                            f"smooth={method} fade={fade}"
                        )
                        cursor = 0
                        return True
                return False
            remaining_commit = worker.commit_end - cursor
        else:
            _d, _switch_s, s_trigger = worker.plan_horizons(current_chunk.horizon)
            remaining_commit = (
                s_trigger - cursor if cursor < s_trigger else worker.delay_align
            )

        remaining = current_chunk.horizon - cursor
        schedule_batch = min(steps_per_inference, worker.delay_align)
        if soft or post_soft:
            schedule_batch = min(schedule_batch, 4)
        n = min(schedule_batch, remaining, remaining_commit)
        if n <= 0:
            return False

        anchor = (
            last_commanded if last_commanded is not None else measured_state(env)
        )
        per_arm = None
        step_lim = float(max_joint_step)
        if soft:
            per_arm = resume_soft_step_limits(
                max_joint_step,
                resume_soft_batches_left,
                total_batches=resume_soft_total,
                hold_batches=resume_hold_batches,
            )
            step_lim = max(per_arm) if max(per_arm) > 0 else 1e-6
        elif post_soft:
            step_lim = resume_post_soft_step_limit(
                max_joint_step,
                resume_post_soft_batches_left,
                total_batches=resume_post_soft_total,
            )
            per_arm = (step_lim, step_lim)

        if soft and max(per_arm) <= 0.0:
            # Pure hold batches: ignore policy targets entirely.
            batch_actions = np.tile(
                np.asarray(anchor, dtype=np.float64).reshape(1, -1), (n, 1)
            )
        else:
            # Hold→move onset: re-anchor to measured so we don't shove from a
            # stale last_commanded that drifted during the hold window.
            if soft and max(per_arm) > 0.0:
                try:
                    measured = measured_state(env)
                    if last_commanded is None or float(
                        np.max(np.abs(measured - anchor))
                    ) > 1e-3:
                        anchor = measured
                        last_commanded = np.asarray(measured, dtype=np.float64)
                except Exception:  # noqa: BLE001
                    pass
            batch_actions = guarded_chunk(
                current_chunk.actions[cursor : cursor + n],
                anchor,
                count=n,
                max_joint_step=step_lim,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                max_joint_step_per_arm=per_arm,
            )
        if soft:
            resume_soft_batches_left -= 1
            d0 = batch_actions[0] - np.asarray(anchor, dtype=np.float64)
            jump_l = float(np.max(np.abs(d0[0:6])))
            jump_r = float(np.max(np.abs(d0[7:13])))
            print(
                f"[DAgger/RTC] sched soft n={n} "
                f"lim L={per_arm[0]:.4f} R={per_arm[1]:.4f} "
                f"left={resume_soft_batches_left} "
                f"d0 L={jump_l:.4f} R={jump_r:.4f} "
                f"t0_ref={schedule_end:.3f}"
            )
        elif post_soft:
            resume_post_soft_batches_left -= 1
            d0 = batch_actions[0] - np.asarray(anchor, dtype=np.float64)
            print(
                f"[DAgger/RTC] sched post-soft n={n} lim={step_lim:.4f} "
                f"left={resume_post_soft_batches_left} "
                f"d0 L={float(np.max(np.abs(d0[0:6]))):.4f} "
                f"R={float(np.max(np.abs(d0[7:13]))):.4f}"
            )
        # Never schedule behind an outstanding hold / prior batch.
        t0 = max(time.time() + 0.05, float(schedule_end) + 0.01)
        stamps = np.arange(len(batch_actions), dtype=np.float64) * dt + t0
        command_joint_actions(env, batch_actions, stamps)
        schedule_end = float(stamps[-1])
        if len(batch_actions) >= 2:
            prev_commanded = np.asarray(batch_actions[-2], dtype=np.float64)
        elif last_commanded is not None:
            prev_commanded = np.asarray(last_commanded, dtype=np.float64)
        last_commanded = np.asarray(batch_actions[-1], dtype=np.float64)
        cursor += n
        worker.maybe_start(
            executed_since_chunk=cursor,
            current_chunk=current_chunk,
            get_obs_bundle=_get_obs_bundle,
        )
        return True

    def _quit_from_keys() -> None:
        nonlocal drag_mode, recording
        if drag_mode:
            leave_drag_stay_pose(env)
            drag_mode = False
        if recording:
            print("[DAgger] Quit with active episode — aborting")
            abort_episode(env)
            recording = False
        _stop_rtc()
        try:
            hold_current(env)
        except Exception as exc:  # noqa: BLE001
            print(f"quit hold warning: {exc}")

    def _process_keys(key_list: list[str]) -> str:
        """Handle keys. Returns '' | 'quit' | 'break' (leave wait/exec path)."""
        for key in key_list:
            if key == "q":
                _quit_from_keys()
                return "quit"
            if key == "c":
                _start_episode()
            elif key == "d":
                _enter_drag()
                return "break"
            elif key == "r":
                _resume_policy()
                return "break"
            elif key == "s":
                _save_episode()
                return "break"
            elif key == "b":
                _abort()
                return "break"
        return ""

    t_start = time.monotonic()
    iter_idx = 0
    print(
        "[DAgger] ready. C=start episode+policy, D=drag, R=resume, "
        "S=save, B=abort, Q=quit"
    )
    try:
        while True:
            t_cycle = t_start + (iter_idx + 1) * dt
            action = _process_keys(keys.read())
            if action == "quit":
                return

            if drag_mode or not executing:
                obs = env.get_obs()
                _record_tick(obs)
                show_preview(obs, _status_text(), enabled=preview_enabled)
                precise_wait(t_cycle)
                iter_idx += 1
                continue

            if use_rtc:
                # Match plain run_rtc: schedule then wait until schedule_end so
                # the next batch chains without a wall-clock gap (chunk hitch).
                _schedule_rtc_batch()
                deadline = (
                    float(schedule_end) if schedule_end > 0.0 else time.time() + dt
                )
                next_rec = time.time()
                aborted = False
                while time.time() < deadline:
                    action = _process_keys(keys.read())
                    if action == "quit":
                        return
                    if action == "break" or drag_mode or not executing:
                        aborted = True
                        break
                    now = time.time()
                    if now >= next_rec:
                        obs = env.get_obs()
                        _record_tick(obs)
                        show_preview(
                            obs, _status_text(), enabled=preview_enabled
                        )
                        next_rec = now + dt
                    time.sleep(0.005)
                if aborted:
                    precise_wait(t_cycle)
                iter_idx += 1
                continue

            obs = env.get_obs()
            _record_tick(obs)
            show_preview(obs, _status_text(), enabled=preview_enabled)
            now = time.time()
            if ol_actions is None or ol_schedule_end <= now + dt:
                _schedule_ol_batch()
            precise_wait(t_cycle)
            iter_idx += 1
    finally:
        if drag_mode:
            try:
                leave_drag_stay_pose(env)
            except Exception as exc:  # noqa: BLE001
                print(f"final leave_drag warning: {exc}")
        _stop_rtc()


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
@click.option("--max-joint-speed", default=2.5, type=float, show_default=True, help="rad/s")
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
    "--dagger",
    is_flag=True,
    help=(
        "DAgger mode: C start episode+policy, D gravity-comp drag intervene, "
        "R resume policy, S save UMI zarr episode, B abort. "
        "Writes intervention sidecars under {output}/intervention/."
    ),
)
@click.option(
    "--rtc-s-min",
    default=16,
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
    default=0.5,
    type=float,
    show_default=True,
    help="RTC server soft-guidance weight (kai0 default 0.5). Raise carefully.",
)
@click.option(
    "--rtc-smooth-method",
    type=click.Choice(["temporal", "c1", "raw"], case_sensitive=False),
    default=RTC_SMOOTH_METHOD,
    show_default=True,
    help=(
        "Client boundary smooth: temporal=official kai0 ARX StreamActionBuffer "
        "overlap blend; c1=local C1+skip0 stitch; raw=no client blend."
    ),
)
@click.option(
    "--rtc-blend-steps",
    default=RTC_BLEND_STEPS,
    type=click.IntRange(1, 48),
    show_default=True,
    help=(
        "temporal: min_smooth_steps (kai0 ARX default 10). "
        "c1: C1 blend length after each swap."
    ),
)
@click.option(
    "--rtc-fade",
    type=click.Choice(["smoothstep", "linear", "ease_out"], case_sensitive=False),
    default=RTC_FADE,
    show_default=True,
    help=(
        "temporal old→new weight curve. smoothstep=C1-friendly handoff; "
        "ease_out=stay on leftover longer; linear=official kai0."
    ),
)
@click.option(
    "--rtc-max-joint-accel",
    default=RTC_MAX_JOINT_ACCEL,
    type=click.FloatRange(min=0.0),
    show_default=True,
    help=(
        "RTC command-space joint acceleration limit in rad/s^2. "
        "0 disables it; 8 gives max Δ(Δq)=0.02 rad at 20 Hz."
    ),
)
@click.option("--output", default="data_local/kai0_eval", show_default=True)
@click.option(
    "--log-file",
    default=str(DEFAULT_RUN_LOG),
    show_default=True,
    help=(
        "Last-run log path (overwritten each run). A timestamped sibling "
        "run_YYYYMMDD_HHMMSS.log is also written in the same directory."
    ),
)
@click.option(
    "--skip-server-warmup",
    is_flag=True,
    help=(
        "Do not run throw-away inference at startup. First C/I may then take "
        "1-3 min while the policy server compiles, or hit the WS timeout."
    ),
)
@click.option(
    "--debug-timing",
    is_flag=True,
    help="Verbose timing diagnostics for chunk boundaries / RTC swap behavior.",
)
@click.option(
    "--drag-profile",
    type=click.Choice(["light", "stable"], case_sensitive=False),
    default="stable",
    show_default=True,
    help=(
        "Soft-hold drag feel for DAgger intervene (D): stable=anti-sag (heavier), "
        "light=same as collect. Ignored when true gravity-comp server is used."
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
    dagger: bool,  # noqa: FBT001
    rtc_s_min: int,
    rtc_delay_buffer_size: int,
    rtc_max_guidance_weight: float,
    rtc_smooth_method: str,
    rtc_blend_steps: int,
    rtc_fade: str,
    rtc_max_joint_accel: float,
    output: str,
    log_file: str,
    skip_server_warmup: bool,  # noqa: FBT001
    debug_timing: bool,  # noqa: FBT001
    drag_profile: str,
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
    if debug_timing:
        print("[DBG] timing diagnostics enabled")
    if rtc:
        if not metadata.get("supports_rtc"):
            raise click.ClickException(
                "Client --rtc requires server metadata supports_rtc=true. "
                "Restart Kai0 with: uv run scripts/serve_kai0_policy.py "
                "--checkpoint <jax_ckpt> --rtc --port 8000"
            )
        print(
            f"[RTC] enabled: s_min={rtc_s_min}, delay_buf={rtc_delay_buffer_size}, "
            f"max_gw={rtc_max_guidance_weight}, smooth={rtc_smooth_method}, "
            f"fade={rtc_fade}, blend/min_m={rtc_blend_steps}, "
            f"soft_min={RTC_SOFT_MIN}, max_accel={rtc_max_joint_accel:.2f}rad/s^2, "
            f"steps_per_inference={steps_per_inference}"
        )
    if dagger:
        print(
            "Controls (DAgger): C=start episode+policy, D=drag intervene, "
            "R=resume policy, S=save episode, B=abort, Q=quit"
        )
    else:
        print("Controls: C=start execution, S=stop/hold, I=infer once, Q=quit")
    if observe_only:
        print("OBSERVE ONLY: motor commands are permanently disabled")
    if dagger and observe_only:
        raise click.ClickException("--dagger cannot be combined with --observe-only")

    # Before spawning controllers so soft-hold DAgger drag uses the chosen feel.
    apply_drag_profile(drag_profile)
    if dagger:
        print(
            f"[drag] profile={get_drag_profile()} "
            "(soft-hold intervene; true GC ignores)"
        )

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

        if skip_server_warmup:
            print(
                "[warmup] skipped (--skip-server-warmup); "
                "first C/I may be slow while the server compiles"
            )
        else:
            warmup_policy_server(
                env=env,
                policy=policy,
                prompt=prompt,
                rtc=rtc,
                rtc_s_min=rtc_s_min,
                rtc_max_guidance_weight=rtc_max_guidance_weight,
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
                debug_timing=debug_timing,
            )
            if dagger:
                print(
                    f"[DAgger] UMI zarr output -> {output_path} "
                    f"(rtc={'on' if rtc else 'off'})"
                )
                run_dagger(
                    **common,
                    output_dir=output_path,
                    use_rtc=rtc,
                    rtc_s_min=rtc_s_min,
                    rtc_delay_buffer_size=rtc_delay_buffer_size,
                    rtc_max_guidance_weight=rtc_max_guidance_weight,
                    rtc_blend_steps=rtc_blend_steps,
                    rtc_smooth_method=rtc_smooth_method,
                    rtc_fade=rtc_fade,
                )
            elif rtc:
                run_rtc(
                    **common,
                    rtc_s_min=rtc_s_min,
                    rtc_delay_buffer_size=rtc_delay_buffer_size,
                    rtc_max_guidance_weight=rtc_max_guidance_weight,
                    rtc_blend_steps=rtc_blend_steps,
                    rtc_smooth_method=rtc_smooth_method,
                    rtc_max_joint_accel=rtc_max_joint_accel,
                    rtc_fade=rtc_fade,
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
