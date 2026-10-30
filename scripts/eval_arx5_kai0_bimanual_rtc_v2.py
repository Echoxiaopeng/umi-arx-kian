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
python scripts/eval_arx5_kai0_bimanual_rtc_v2.py \
  --policy-ip 127.0.0.1 --policy-port 8000 \
  --robot-ports 8765,8766 --camera-reorder 0,1 \
  --frequency 20 --inference-rate 4 \
  --rtc-max-guidance-weight 0.5 \
  --min-smooth-steps 10 --latency-k 8 \
  --dispatch tick


``--dispatch schedule`` (default): timed ZMQ waypoints via interpolator.
``--dispatch tick``: ROS-style immediate SET_JOINT_POSITION each control tick.

Controls: C=start, S=stop/hold, Enter=return to init (after stop), Q=home then quit.

Live dashboard (default on): http://127.0.0.1:8090/  (--no-web to disable).
"""

from __future__ import annotations

from collections import deque
import importlib.util
from multiprocessing.managers import SharedMemoryManager
import os
from pathlib import Path
import sys
import threading
import time

import click
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_LOG = ROOT / "data_local" / "kai0_eval" / "last_run_rtc_v2.log"
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
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
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


def _guard_one(
    action: np.ndarray,
    current: np.ndarray,
    *,
    max_joint_step: float,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray:
    out = _v1.guarded_chunk(
        np.asarray(action, dtype=np.float64).reshape(1, -1),
        current,
        count=1,
        max_joint_step=max_joint_step,
        gripper_min=gripper_min,
        gripper_max=gripper_max,
    )
    return np.asarray(out[0], dtype=np.float64)


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


def _web_keys_from_cmds(cmd_queue: CommandQueue | None) -> list[str]:
    if cmd_queue is None:
        return []
    out: list[str] = []
    for item in cmd_queue.drain():
        action = str(item.get("action", "")).lower()
        if action in {"c", "start"}:
            out.append("c")
        elif action in {"s", "stop"}:
            out.append("s")
        elif action in {"init", "enter", "return_init"}:
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
    dispatch: str = "schedule",
    skip_server_warmup: bool = False,
    init_q14: np.ndarray | None = None,
    init_duration: float = 2.5,
    home_duration: float = 5.0,
    web_state: SharedEvalState | None = None,
    web_cmds: CommandQueue | None = None,
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
            anchor = (
                last_commanded
                if last_commanded is not None
                else _v1.measured_state(env)
            )
            guarded = _guard_one(
                act,
                anchor,
                max_joint_step=max_joint_step,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
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
        print(
            f"[v2] session: steps={step_count} infers={infer_count} "
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
    web: bool,  # noqa: FBT001
    web_host: str,
    web_port: int,
    log_file: str,
) -> None:
    if frequency <= 0 or inference_rate <= 0 or max_joint_speed <= 0:
        raise click.BadParameter("frequency/inference-rate/max-joint-speed must be > 0")
    if not 0 <= gripper_min < gripper_max:
        raise click.BadParameter("invalid gripper range")
    if rtc_max_guidance_weight <= 0:
        raise click.BadParameter("--rtc-max-guidance-weight must be positive")
    if home_duration <= 0 or init_duration <= 0:
        raise click.BadParameter("--home-duration/--init-duration must be > 0")

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
    print(
        f"[v2] enabled: ctrl={frequency}Hz infer={inference_rate}Hz "
        f"exec_h≈{soft_hint} max_gw={rtc_max_guidance_weight} "
        f"smooth={'raw' if rtc_disable_smoothing else 'temporal'} "
        f"min_m={min_smooth_steps} latency_k={latency_k} "
        f"dispatch={dispatch}"
    )
    print(
        "Controls: C=start, S=stop/hold, "
        "Enter=return to init (after S), Q=home then quit"
    )
    if observe_only:
        print("OBSERVE ONLY: motor commands are permanently disabled")

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

    output_path = (ROOT / "data_local" / "kai0_eval").resolve()
    output_path.mkdir(parents=True, exist_ok=True)

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
                run_rtc_v2(
                    env=env,
                    policy=policy,
                    keys=keys,
                    prompt=prompt,
                    frequency=frequency,
                    inference_rate=inference_rate,
                    max_joint_step=max_joint_step,
                    gripper_min=gripper_min,
                    gripper_max=gripper_max,
                    observe_only=observe_only,
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
                    init_q14=init_q14,
                    init_duration=float(init_duration),
                    home_duration=float(home_duration),
                    web_state=web_state,
                    web_cmds=web_cmds,
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
