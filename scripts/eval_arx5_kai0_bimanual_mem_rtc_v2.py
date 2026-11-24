#!/usr/bin/env python3
"""Kai0 ARX5 bimanual eval — MEM + RTC v2 middle layer.

Same async RTC control stack as ``eval_arx5_kai0_bimanual_rtc_v2.py``, plus
true short-horizon MEM episode handling:

- requires policy-server metadata ``supports_mem=true`` (and ``supports_rtc``);
- after warmup / on each episode start / on episode end, sends
  ``{"reset_memory": true}`` so server-side visual+proprio history is cleared;
- mid-episode resume (DAgger HOLD→C) does **not** reset MEM.

Policy server (kai0)::

    cd /media/lenovo/data/kian/kai0
    uv run scripts/serve_kai0_policy_mem.py \\
      --checkpoint /media/lenovo/data/kian/kai0/checkpoints/pi05_flatten_fold_clothes_mem/<step> \\
      --rtc --port 8000

Client::

    conda activate umi-arx
    python scripts/eval_arx5_kai0_bimanual_mem_rtc_v2.py \
      --policy-ip 127.0.0.1 --policy-port 8000 \
      --robot-ports 8765,8766 --camera-reorder 0,1 \
      --frequency 20 --inference-rate 4 \
      --rtc-max-guidance-weight 0.5 \
      --dispatch tick \
      --chunk-size 50 \
      --output data_local/kai0_eval_mem \
    --no-save-model-traj \
    --gripper-close-threshold 0.03 

Controls match RTC v2 (C/S/Enter/Q; DAgger: C/D/R/S/B/Q).
"""

from __future__ import annotations

import importlib.util
import os
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import sys
import time

import click
import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_LOG = ROOT / "data_local" / "kai0_eval_mem" / "last_run_mem_rtc_v2.log"
DEFAULT_EVAL_OUTPUT = ROOT / "data_local" / "kai0_eval_mem"
DEFAULT_DAGGER_OUTPUT = ROOT / "data_local" / "kai0_dagger_mem"
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

_V2_PATH = ROOT / "scripts" / "eval_arx5_kai0_bimanual_rtc_v2.py"
_spec = importlib.util.spec_from_file_location("kai0_bimanual_eval_rtc_v2", _V2_PATH)
assert _spec is not None and _spec.loader is not None
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)

_V1_PATH = ROOT / "scripts" / "eval_arx5_kai0_bimanual.py"
_spec_v1 = importlib.util.spec_from_file_location("kai0_bimanual_eval_v1_mem", _V1_PATH)
assert _spec_v1 is not None and _spec_v1.loader is not None
_v1 = importlib.util.module_from_spec(_spec_v1)
_spec_v1.loader.exec_module(_v1)

DEFAULT_PROMPT = _v1.DEFAULT_PROMPT
DEFAULT_INIT_JOINTS = _v1.DEFAULT_INIT_JOINTS
ModelTrajRecorder = _v2.ModelTrajRecorder
run_rtc_v2 = _v2.run_rtc_v2
run_dagger_v2 = _v2.run_dagger_v2


def _reset_mem(policy: WebsocketClientPolicy, *, reason: str) -> None:
    """Clear server-side MEM history; never raise into the control loop."""
    try:
        policy.reset()
        print(f"[mem] reset_memory ({reason})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[mem] reset_memory failed ({reason}): {exc}", flush=True)


def _require_mem_rtc_metadata(metadata: dict) -> None:
    if not metadata.get("supports_rtc"):
        raise click.ClickException(
            "MEM+RTC client requires server metadata supports_rtc=true.\n"
            "Start Kai0 with:\n"
            "  uv run scripts/serve_kai0_policy_mem.py --checkpoint <mem_ckpt> --rtc --port 8000"
        )
    if not metadata.get("supports_mem"):
        raise click.ClickException(
            "MEM client requires server metadata supports_mem=true.\n"
            "Use serve_kai0_policy_mem.py (not the plain serve_kai0_policy.py)."
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
    help="Control-loop Hz (keep 20 to match MEM memory_stride=20 @ fps=20).",
)
@click.option(
    "--inference-rate",
    default=4.0,
    type=float,
    show_default=True,
    help="Background RTC infer Hz.",
)
@click.option("--max-joint-speed", default=1.5, type=float, show_default=True)
@click.option("--gripper-min", default=0.0, type=float, show_default=True)
@click.option("--gripper-max", default=0.088, type=float, show_default=True)
@click.option(
    "--gripper-tighten/--no-gripper-tighten",
    default=True,
    show_default=True,
)
@click.option("--gripper-close-threshold", default=0.04, type=float, show_default=True)
@click.option("--gripper-grasp-width", default=0.0, type=float, show_default=True)
@click.option("--camera-obs-latency", default=0.125, type=float, show_default=True)
@click.option("--robot-obs-latency", default=0.0001, type=float, show_default=True)
@click.option("--server-timeout", default=30.0, type=float, show_default=True)
@click.option("--observe-only", is_flag=True)
@click.option("--no-preview", is_flag=True)
@click.option("--no-init", is_flag=True)
@click.option("--init-joints", default=DEFAULT_INIT_JOINTS, show_default=False)
@click.option("--init-duration", default=2.5, type=float, show_default=True)
@click.option("--home-duration", default=5.0, type=float, show_default=True)
@click.option("--chunk-size", default=50, type=int, show_default=True)
@click.option("--rtc-execute-horizon", default=None, type=int)
@click.option("--rtc-max-guidance-weight", default=0.5, type=float, show_default=True)
@click.option("--rtc-mask-prefix-delay", is_flag=True)
@click.option("--latency-k", default=8, type=int, show_default=True)
@click.option("--min-smooth-steps", default=10, type=int, show_default=True)
@click.option("--rtc-disable-smoothing", is_flag=True)
@click.option(
    "--dispatch",
    type=click.Choice(["schedule", "tick"], case_sensitive=False),
    default="schedule",
    show_default=True,
)
@click.option("--skip-server-warmup", is_flag=True)
@click.option("--dagger", is_flag=True)
@click.option("--output", default=None)
@click.option(
    "--drag-profile",
    type=click.Choice(["light", "stable"], case_sensitive=False),
    default="stable",
    show_default=True,
)
@click.option("--web/--no-web", default=True, show_default=True)
@click.option("--web-host", default="0.0.0.0", show_default=True)
@click.option("--web-port", default=8090, type=int, show_default=True)
@click.option("--log-file", default=str(DEFAULT_RUN_LOG), show_default=True)
@click.option("--save-model-traj/--no-save-model-traj", default=True, show_default=True)
@click.option("--save-traj-images/--no-save-traj-images", default=True, show_default=True)
@click.option(
    "--allow-non-mem-server",
    is_flag=True,
    help="Skip supports_mem check (debug only; still sends reset_memory).",
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
    allow_non_mem_server: bool,  # noqa: FBT001
) -> None:
    if frequency <= 0 or inference_rate <= 0 or max_joint_speed <= 0:
        raise click.BadParameter("frequency/inference-rate/max-joint-speed must be > 0")
    if not 0 <= gripper_min < gripper_max:
        raise click.BadParameter("invalid gripper range")
    if dagger and observe_only:
        raise click.ClickException("--dagger cannot be combined with --observe-only")

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    _v1.install_run_log(log_path)

    ports = _v1.parse_pair(robot_ports, "--robot-ports")
    camera_order = _v1.parse_pair(camera_reorder, "--camera-reorder")
    max_joint_step = float(max_joint_speed) / float(frequency)

    print(f"[mem] Connecting to MEM policy server ws://{policy_ip}:{policy_port}")
    policy = WebsocketClientPolicy(
        host=policy_ip,
        port=policy_port,
        ping_timeout=server_timeout,
        close_timeout=server_timeout,
    )
    metadata = policy.get_server_metadata()
    print(f"[mem] Policy connected; metadata={metadata}")
    if allow_non_mem_server:
        if not metadata.get("supports_rtc"):
            raise click.ClickException(
                "supports_rtc=true required even with --allow-non-mem-server"
            )
        print("[mem] WARNING: --allow-non-mem-server; skipping supports_mem check")
    else:
        _require_mem_rtc_metadata(metadata)

    mem_frames = metadata.get("memory_frames")
    mem_stride = metadata.get("memory_stride")
    print(
        f"[mem] server MEM: visual={metadata.get('use_visual_memory')} "
        f"proprio={metadata.get('use_proprioceptive_memory')} "
        f"frames={mem_frames} stride={mem_stride} "
        f"(control Hz={frequency}; stride should match training fps)"
    )

    def on_episode_start() -> None:
        _reset_mem(policy, reason="episode_start")

    def on_episode_end() -> None:
        _reset_mem(policy, reason="episode_end_or_warmup_clear")

    soft_hint = (
        chunk_size
        if rtc_execute_horizon is None
        else min(int(rtc_execute_horizon), int(chunk_size))
    )
    print(
        f"[mem/v2] enabled: ctrl={frequency}Hz infer={inference_rate}Hz "
        f"exec_h≈{soft_hint} max_gw={rtc_max_guidance_weight} "
        f"dispatch={dispatch}"
    )
    if dagger:
        print(
            "Controls (DAgger+MEM): C=start episode+policy (+reset MEM), "
            "D=drag, R=resume (keep MEM), S=save (+reset MEM), B=abort, Q=quit"
        )
    else:
        print(
            "Controls (MEM): C=start (+reset MEM), S=stop (+reset MEM), "
            "Enter=init, Q=home+quit"
        )

    _v1.apply_drag_profile(drag_profile)
    preview_enabled = (not no_preview) and bool(os.environ.get("DISPLAY"))
    init_q14 = None if no_init else _v1.parse_init_joints(init_joints)
    do_home = not no_init

    if output is None:
        output_path = (
            DEFAULT_DAGGER_OUTPUT.resolve()
            if dagger
            else DEFAULT_EVAL_OUTPUT.resolve()
        )
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = (ROOT / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    traj_dir = output_path / "model_traj"
    traj_meta = {
        "mode": "mem_rtc_v2",
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
        "dagger": bool(dagger),
        "policy_ip": policy_ip,
        "policy_port": int(policy_port),
        "memory_frames": mem_frames,
        "memory_stride": mem_stride,
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

    web_state: SharedEvalState | None = None
    web_cmds: CommandQueue | None = None
    web_server = None
    if web:
        web_state = SharedEvalState()
        web_cmds = CommandQueue()
        web_server = start_eval_web_ui(
            web_state, web_cmds, host=web_host, port=int(web_port)
        )
        print(f"[mem] open dashboard: http://127.0.0.1:{web_port}/")

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
            while not env.is_ready:
                time.sleep(0.1)
            print("Environment ready")

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
                on_episode_start=on_episode_start,
                on_episode_end=on_episode_end,
            )
            try:
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
