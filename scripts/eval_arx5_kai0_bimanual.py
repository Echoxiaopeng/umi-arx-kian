#!/usr/bin/env python3
"""Run a Kai0 dual-arm joint policy on the ARX5 ZMQ/camera stack.

The policy server runs separately from the Kai0 repository. This client:

1. reads left wrist + center D435 + right wrist images;
2. reads both arms' six joints and gripper widths;
3. sends the Kai0 observation over websocket;
4. receives ``[T, 14]`` absolute joint actions;
5. optionally schedules a guarded prefix on both arms via
   ``schedule_joint_waypoint`` (timestamped joint waypoints).

Example:

    conda activate umi-arx
    cd /home/lenovo/kian/umi-arx-kian
    python scripts/eval_arx5_kai0_bimanual.py \
      --policy-ip 127.0.0.1 --policy-port 8000 \
      --robot-ports 8765,8766 --camera-reorder 0,1

Controls: C=start execution, S=stop/hold, I=infer once without motion, Q=quit.
Execution is disabled at startup. Use ``--observe-only`` to permanently disable it.
"""

from __future__ import annotations

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
if CLIENT_SRC.is_dir():
    sys.path.insert(0, str(CLIENT_SRC))

from modules.arx5_env_with_images_bimanual_rs import Arx5EnvWithImages  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

DEFAULT_PROMPT = "Flatten and fold the cloth."
ACTION_DIM = 14


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


def robot_config(robot_ip: str, ports: list[int], obs_latency: float) -> list[dict]:
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
            # Starting this inference client must not move either arm.
            "reset_to_home_on_start": False,
        }
        for port in ports
    ]


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

def show_preview(obs: dict, status: str) -> None:
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
@click.option("--max-joint-speed", default=0.8, type=float, show_default=True, help="rad/s")
@click.option("--gripper-min", default=0.0, type=float, show_default=True)
@click.option("--gripper-max", default=0.085, type=float, show_default=True)
@click.option("--camera-obs-latency", default=0.17, type=float, show_default=True)
@click.option("--robot-obs-latency", default=0.1867, type=float, show_default=True)
@click.option("--server-timeout", default=600.0, type=float, show_default=True)
@click.option("--observe-only", is_flag=True, help="Never command either arm.")
@click.option("--output", default="data_local/kai0_eval", show_default=True)
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
    output: str,
) -> None:
    if frequency <= 0 or max_joint_speed <= 0:
        raise click.BadParameter("frequency and max-joint-speed must be positive")
    if not 0 <= gripper_min < gripper_max:
        raise click.BadParameter("invalid gripper range")

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
    print(f"Policy connected; metadata={policy.get_server_metadata()}")
    print("Controls: C=start execution, S=stop/hold, I=infer once, Q=quit")
    if observe_only:
        print("OBSERVE ONLY: motor commands are permanently disabled")

    dt = 1.0 / frequency
    max_joint_step = max_joint_speed * dt
    executing = False

    with SharedMemoryManager() as shm_manager, Arx5EnvWithImages(
        output_dir=output_path,
        robots_config=robot_config(robot_ip, ports, robot_obs_latency),
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

        try:
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

                # Wait out the scheduled chunk while polling stop/quit.
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
        finally:
            try:
                hold_current(env)
            except Exception as exc:  # noqa: BLE001
                print(f"Final hold warning: {exc}")
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
