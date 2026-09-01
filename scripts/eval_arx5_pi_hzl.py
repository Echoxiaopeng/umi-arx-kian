#!/usr/bin/env python3
"""Deploy the real-wrist Pi0 semantic-memory policy on one ARX5 arm.

Controls
--------
F: capture initial frame, then start each of three 80-frame swap recordings
C: start policy after the strict 241-frame history is ready
S: stop policy and return to human control
P: in human mode, return to init_pose and reset history for the next round
Q: quit
"""

from __future__ import annotations

from multiprocessing.managers import SharedMemoryManager
import os
from pathlib import Path
import sys
import time

import click
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
OPENPI_CLIENT_SRC = ROOT_DIR / "packages" / "openpi-client" / "src"
if OPENPI_CLIENT_SRC.is_dir():
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))
os.chdir(ROOT_DIR)

from utils.shellgame_openpi import ACTION_DIM  # noqa: E402
from utils.shellgame_openpi import ACTION_HORIZON  # noqa: E402
from utils.shellgame_openpi import SWAP_FRAMES  # noqa: E402
from utils.shellgame_openpi import TOTAL_MODEL_FRAMES  # noqa: E402
from utils.shellgame_openpi import ShellGameHistory  # noqa: E402
from utils.shellgame_openpi import direct_link6_pose_to_zmq_eef  # noqa: E402
from utils.shellgame_openpi import eef_pose_to_link6  # noqa: E402
from utils.shellgame_openpi import link6_actions_to_eef  # noqa: E402
from utils.shellgame_openpi import parse_csv_vector  # noqa: E402
from utils.shellgame_openpi import relative_chunk_to_world  # noqa: E402
from utils.shellgame_openpi import run_contract_self_test  # noqa: E402
from utils.shellgame_openpi import select_safe_waypoint_count  # noqa: E402
from utils.shellgame_openpi import validate_world_actions  # noqa: E402

DEFAULT_EEF_OFFSET = 0.145
DEFAULT_PROMPT = "Watch the shell game and lift the cup hiding the ball."


def ease_in_out_quad(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(value < 0.5, 2.0 * value * value, 1.0 - ((-2.0 * value + 2.0) ** 2) / 2.0)


def current_robot_pose_and_gripper(env) -> tuple[np.ndarray, float]:
    state = env.get_robot_state()[0]
    pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
    gripper = float(state["gripper_position"])
    return pose, gripper


def hold_current_pose(env) -> None:
    """Cancel queued ZMQ waypoints by driving each controller to its measured pose."""
    states = env.get_robot_state()
    for robot, state in zip(env.robots, states, strict=True):
        pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
        gripper = float(state["gripper_position"])
        robot.servoL(
            pose=pose,
            gripper_pos=gripper,
            duration=max(0.1, 1.0 / float(robot.frequency)),
        )


def move_to_pose(
    env,
    target_pose6: np.ndarray,
    *,
    duration: float,
    frequency: float,
    target_gripper: float | None,
    gripper_min: float,
    gripper_max: float,
) -> None:
    """Move to a fixed pose using smooth translation and shortest-path rotation."""
    start_pose, start_gripper = current_robot_pose_and_gripper(env)
    target_pose = np.asarray(target_pose6, dtype=np.float64).reshape(6)
    target_gripper_value = (
        start_gripper if target_gripper is None else float(np.clip(target_gripper, gripper_min, gripper_max))
    )
    steps = max(2, round(duration * frequency))
    linear = np.linspace(0.0, 1.0, steps)
    blend = ease_in_out_quad(linear)

    positions = start_pose[:3] + blend[:, None] * (target_pose[:3] - start_pose[:3])
    rotation_keyframes = Rotation.from_rotvec(np.stack([start_pose[3:], target_pose[3:]], axis=0))
    rotations = Slerp([0.0, 1.0], rotation_keyframes)(blend).as_rotvec()
    grippers = start_gripper + blend * (target_gripper_value - start_gripper)

    actions = np.concatenate([positions, rotations, grippers[:, None]], axis=-1)
    start_time = time.time() + 0.25
    timestamps = start_time + np.arange(steps, dtype=np.float64) / frequency
    env.exec_actions(actions=actions, timestamps=timestamps, compensate_latency=False)
    time.sleep(max(0.0, timestamps[-1] - time.time()) + 0.3)


def draw_status(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = np.asarray(frame).copy()
    for index, text in enumerate(lines):
        y = 24 + index * 24
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run_policy(
    *,
    env,
    client,
    history: ShellGameHistory,
    wrist_camera_index: int,
    prompt: str,
    frequency: float,
    steps_per_inference: int,
    action_lead_time: float,
    max_duration: float,
    max_inference_latency: float,
    max_pos_speed: float,
    max_rot_speed: float,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    gripper_min: float,
    gripper_max: float,
    eef_offset: float,
    key_counter,
    observe_only: bool,
    warmup_done: bool,
) -> tuple[bool, bool]:
    """Run one policy episode. Returns (quit_requested, warmup_done)."""
    from peripherals.keystroke_counter import KeyCode

    if not history.ready:
        print("History is incomplete; refusing to start policy")
        return False, warmup_done

    camera_key = f"camera{wrist_camera_index}_rgb"

    def infer_once(*, label: str = "infer") -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, float]:
        obs = env.get_obs()
        frame = obs[camera_key][-1]
        anchor_eef = np.concatenate([obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]], axis=-1).astype(
            np.float64
        )
        anchor_link6 = eef_pose_to_link6(anchor_eef, eef_offset)
        gripper_width = float(np.asarray(obs["robot0_gripper_width"][-1]).reshape(-1)[0])
        policy_obs = history.build_policy_observation(
            current_frame=frame,
            current_pose6=anchor_link6,
            gripper_width=gripper_width,
            prompt=prompt,
        )
        print(
            f"[policy] {label}: sending {TOTAL_MODEL_FRAMES} frames to policy server "
            f"(first call may take 1-3 min while the model compiles)..."
        )
        sys.stdout.flush()
        started = time.monotonic()
        result = client.infer(policy_obs)
        latency = time.monotonic() - started
        raw_actions = np.asarray(result["actions"], dtype=np.float64)
        if raw_actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Policy returned {raw_actions.shape}, expected {(ACTION_HORIZON, ACTION_DIM)}")
        return raw_actions, anchor_link6, anchor_eef, obs, latency

    if not warmup_done and not observe_only:
        print("Compiling/warming policy with a throw-away inference; no action will be sent...")
        _, _, _, _, warmup_latency = infer_once(label="warmup")
        warmup_done = True
        print(f"Policy warm-up complete in {warmup_latency:.3f}s; taking a fresh observation")

    if observe_only:
        actions, anchor_link6, anchor_eef, _, latency = infer_once(label="observe-only")
        world_link6 = relative_chunk_to_world(actions, anchor_link6)
        world_eef = link6_actions_to_eef(world_link6, eef_offset)
        print(f"[observe-only] inference={latency:.3f}s, raw={actions.shape}")
        print("[observe-only] anchor ZMQ eef:", np.array2string(anchor_eef, precision=5))
        print("[observe-only] anchor link6: ", np.array2string(anchor_link6, precision=5))
        print("[observe-only] first link6 target:", np.array2string(world_link6[0], precision=5))
        print("[observe-only] first ZMQ eef command:", np.array2string(world_eef[0], precision=5))
        return False, warmup_done

    episode_started = False
    quit_requested = False
    episode_start_time = time.monotonic()
    try:
        while True:
            for key in key_counter.get_press_events():
                if key == KeyCode(char="s"):
                    print("S pressed: stopping policy")
                    return False, warmup_done
                if key == KeyCode(char="q"):
                    quit_requested = True
                    return True, warmup_done

            actions, anchor_link6, anchor_eef, obs, inference_latency = infer_once(label="policy")
            print(f"Inference latency: {inference_latency:.3f}s")
            if inference_latency > max_inference_latency:
                raise RuntimeError(
                    f"Inference latency {inference_latency:.3f}s exceeds limit {max_inference_latency:.3f}s"
                )

            world_link6 = relative_chunk_to_world(actions, anchor_link6)
            world_actions = link6_actions_to_eef(world_link6, eef_offset)
            world_actions[:, 6] = np.clip(world_actions[:, 6], gripper_min, gripper_max)

            count = min(steps_per_inference, ACTION_HORIZON)
            safe_count, safety_reason = select_safe_waypoint_count(
                world_actions,
                anchor_pose6=anchor_eef,
                dt=1.0 / frequency,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                max_waypoints=count,
            )
            if safe_count == 0:
                print(f"[SAFETY] rejected chunk, holding pose: {safety_reason}")
                continue
            if safety_reason is not None:
                print(
                    f"[SAFETY] truncated {count}->{safe_count} waypoints: {safety_reason}"
                )
            selected = world_actions[:safe_count]
            first_timestamp = time.time() + action_lead_time
            timestamps = first_timestamp + np.arange(count, dtype=np.float64) / frequency
            fresh = timestamps > time.time()
            selected = selected[fresh]
            timestamps = timestamps[fresh]
            if not len(selected):
                raise RuntimeError("All action waypoints became stale before submission")

            if not episode_started:
                env.start_episode(time.time())
                episode_started = True
            env.exec_actions(
                actions=selected,
                timestamps=timestamps,
                compensate_latency=False,
                dynamic_latency=False,
            )
            print(f"Submitted {len(selected)} safe waypoints")

            view = draw_status(
                obs[f"camera{wrist_camera_index}_rgb"][-1],
                ["POLICY CONTROL", f"inference {inference_latency:.3f}s", "S: stop  Q: quit"],
            )
            cv2.imshow("ARX5 ShellGame", view[..., ::-1])
            cv2.pollKey()

            deadline = timestamps[-1]
            while time.time() < deadline:
                for key in key_counter.get_press_events():
                    if key == KeyCode(char="s"):
                        print("S pressed: stopping policy")
                        return False, warmup_done
                    if key == KeyCode(char="q"):
                        quit_requested = True
                        return True, warmup_done
                time.sleep(0.01)

            if time.monotonic() - episode_start_time >= max_duration:
                print("Maximum policy duration reached")
                break
    finally:
        # A fresh servoL target truncates the controller's queued
        # interpolation, so S does not continue executing the old chunk.
        hold_current_pose(env)
        time.sleep(0.15)
        if episode_started:
            env.end_episode()
        print("Policy stopped; human control restored. Restore the cups, then press P.")
    return quit_requested, warmup_done


@click.command()
@click.option("--output", "-o", default="data_local/shellgame_eval", show_default=True)
@click.option("--policy-ip", default="127.0.0.1", show_default=True)
@click.option("--policy-port", default=8000, show_default=True, type=int)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot-port", default=8765, show_default=True, type=int)
@click.option("--camera-reorder", default="0", show_default=True)
@click.option("--wrist-camera-index", default=0, show_default=True, type=int)
@click.option("--frequency", default=20.0, show_default=True, type=float)
@click.option("--steps-per-inference", default=8, show_default=True, type=click.IntRange(1, ACTION_HORIZON))
@click.option("--action-lead-time", default=0.10, show_default=True, type=float)
@click.option("--max-duration", default=30.0, show_default=True, type=float)
@click.option("--max-inference-latency", default=3.0, show_default=True, type=float)
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=True)
@click.option(
    "--init-pose",
    default=None,
    type=str,
    help="初始 TCP 位姿 x,y,z,rx,ry,rz。省略时保持当前位置。",
)
@click.option(
    "--init-pose-frame",
    type=click.Choice(["zmq", "direct"], case_sensitive=False),
    default="zmq",
    show_default=True,
    help="init-pose 坐标系: zmq=eef 绝对位姿; direct=Direct 采集 link6 位姿。",
)
@click.option(
    "--direct-zmq-origin",
    default=None,
    type=str,
    help=(
        "Direct→ZMQ 的 link6 原点平移 x,y,z (米)。"
        "init-pose-frame=direct 时必填，除非使用 --calibrate-direct-origin。"
    ),
)
@click.option(
    "--calibrate-direct-origin",
    is_flag=True,
    help=(
        "启动时用当前 ZMQ 位姿标定 direct-zmq-origin: "
        "origin = zmq_link6_now - direct_init_xyz。"
        "请先把臂放到与 --init-pose 对应的 Direct 物理构型。"
    ),
)
@click.option("--init-gripper", default=None, type=float)
@click.option("--init-duration", default=4.0, show_default=True, type=float)
@click.option("--gripper-min-width", default=0.0, show_default=True, type=float)
@click.option("--gripper-max-width", default=0.085, show_default=True, type=float)
@click.option("--workspace-min", default="-0.50,-0.50,0.03", show_default=True)
@click.option("--workspace-max", default="0.50,0.50,0.80", show_default=True)
@click.option("--max-pos-speed", default=0.40, show_default=True, type=float)
@click.option("--max-rot-speed", default=0.80, show_default=True, type=float)
@click.option("--camera-obs-latency", default=0.17, show_default=True, type=float)
@click.option("--robot-obs-latency", default=0.20, show_default=True, type=float)
@click.option(
    "--eef-offset",
    default=DEFAULT_EEF_OFFSET,
    show_default=True,
    type=float,
    help="ZMQ eef 相对 Direct link6 沿 TCP Z 的偏移（米）；设为 0 关闭。",
)
@click.option("--observe-only", is_flag=True, help="Run inference and print targets without commanding the robot.")
@click.option(
    "--observe-only-quick",
    is_flag=True,
    help="With --observe-only: seed dummy 241-frame history and infer once right after startup.",
)
@click.option("--self-test", is_flag=True, help="Validate transforms without opening cameras or robot hardware.")
def main(
    output: str,
    policy_ip: str,
    policy_port: int,
    robot_ip: str,
    robot_port: int,
    camera_reorder: str,
    wrist_camera_index: int,
    frequency: float,
    steps_per_inference: int,
    action_lead_time: float,
    max_duration: float,
    max_inference_latency: float,
    prompt: str,
    init_pose: str | None,
    init_pose_frame: str,
    direct_zmq_origin: str | None,
    calibrate_direct_origin: bool,  # noqa: FBT001
    init_gripper: float | None,
    init_duration: float,
    gripper_min_width: float,
    gripper_max_width: float,
    workspace_min: str,
    workspace_max: str,
    max_pos_speed: float,
    max_rot_speed: float,
    camera_obs_latency: float,
    robot_obs_latency: float,
    eef_offset: float,
    observe_only: bool,  # noqa: FBT001
    observe_only_quick: bool,  # noqa: FBT001
    self_test: bool,  # noqa: FBT001
) -> None:
    if self_test:
        run_contract_self_test()
        return
    if observe_only_quick and not observe_only:
        raise click.BadParameter("--observe-only-quick requires --observe-only")

    from modules.arx5_env import Arx5Env
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from peripherals.keystroke_counter import KeyCode
    from peripherals.keystroke_counter import KeystrokeCounter
    from utils.other_util import precise_wait

    if frequency <= 0 or init_duration <= 0:
        raise click.BadParameter("frequency and init-duration must be positive")
    if not 0 <= gripper_min_width < gripper_max_width:
        raise click.BadParameter("invalid gripper width range")
    if max_pos_speed <= 0 or max_rot_speed <= 0:
        raise click.BadParameter("speed limits must be positive")
    if eef_offset < 0:
        raise click.BadParameter("--eef-offset must be non-negative")
    init_pose_frame = init_pose_frame.lower()
    if calibrate_direct_origin and init_pose is None:
        raise click.BadParameter("--calibrate-direct-origin requires --init-pose")

    init_pose6_raw = (
        None
        if init_pose is None
        else parse_csv_vector(init_pose, size=6, name="--init-pose")
    )
    direct_origin3: np.ndarray | None = None
    if direct_zmq_origin is not None:
        direct_origin3 = parse_csv_vector(
            direct_zmq_origin, size=3, name="--direct-zmq-origin"
        )
    if init_pose6_raw is not None and init_pose_frame == "direct":
        if direct_origin3 is None and not calibrate_direct_origin:
            raise click.BadParameter(
                "init-pose-frame=direct requires --direct-zmq-origin "
                "or --calibrate-direct-origin"
            )
    elif init_pose_frame != "zmq":
        raise click.BadParameter(f"unsupported init-pose-frame: {init_pose_frame}")
    elif init_pose6_raw is not None and init_pose_frame == "zmq" and calibrate_direct_origin:
        raise click.BadParameter("--calibrate-direct-origin only works with init-pose-frame=direct")

    init_pose6 = init_pose6_raw
    if (
        init_pose6_raw is not None
        and init_pose_frame == "direct"
        and direct_origin3 is not None
    ):
        init_pose6 = direct_link6_pose_to_zmq_eef(
            init_pose6_raw,
            direct_origin3,
            eef_offset,
        )
        print(
            "[ALIGN] converted Direct init pose → ZMQ eef: "
            f"{np.array2string(init_pose6, precision=6, separator=',')}"
        )
        print(
            f"[ALIGN] direct-zmq-origin={np.array2string(direct_origin3, precision=6, separator=',')}"
        )
    workspace_min3 = parse_csv_vector(workspace_min, size=3, name="--workspace-min")
    workspace_max3 = parse_csv_vector(workspace_max, size=3, name="--workspace-max")
    if np.any(workspace_min3 >= workspace_max3):
        raise click.BadParameter("workspace-min must be smaller than workspace-max")

    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    camera_order = [int(item.strip()) for item in camera_reorder.split(",")]

    print(f"Connecting to policy server ws://{policy_ip}:{policy_port}")
    client = WebsocketClientPolicy(host=policy_ip, port=policy_port)
    print("Policy server connected")
    print(
        "[ALIGN] Direct/ZMQ world-origin difference cancels in relative SE(3); "
        f"link6<->eef offset={eef_offset:.3f}m along TCP Z"
    )
    if observe_only:
        if observe_only_quick:
            print(
                "[observe-only] quick mode: will infer once immediately after env ready "
                "(dummy 241-frame history, no robot motion)."
            )
        else:
            print(
                "[observe-only] full mode: press F → capture 3×80 swap frames → C "
                "to print inference targets (no robot motion)."
            )

    robots_config = [
        {
            "robot_type": "arx5",
            "robot_ip": robot_ip,
            "robot_port": robot_port,
            "robot_obs_latency": robot_obs_latency,
            "robot_action_latency": 0.01,
        }
    ]

    history = ShellGameHistory()
    warmup_done = False
    dt = 1.0 / frequency

    with SharedMemoryManager() as shm_manager:  # noqa: SIM117
        with (
            KeystrokeCounter() as key_counter,
            Arx5Env(
                output_dir=output_path,
                robots_config=robots_config,
                frequency=frequency,
                obs_image_resolution=(224, 224),
                max_obs_buffer_size=120,
                obs_float32=False,
                camera_reorder=camera_order,
                init_joints=False,
                enable_multi_cam_vis=False,
                camera_obs_latency=camera_obs_latency,
                camera_obs_horizon=1,
                robot_obs_horizon=1,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                shm_manager=shm_manager,
            ) as env,
        ):
            cv2.setNumThreads(2)
            print(f"Waiting for camera and ARX5 ZMQ controller tcp://{robot_ip}:{robot_port}...")
            while not env.is_ready:
                time.sleep(0.1)

            if init_pose6_raw is not None and init_pose_frame == "direct" and calibrate_direct_origin:
                live_eef, _ = current_robot_pose_and_gripper(env)
                live_link6 = eef_pose_to_link6(live_eef, eef_offset)
                direct_origin3 = live_link6[:3] - init_pose6_raw[:3]
                init_pose6 = direct_link6_pose_to_zmq_eef(
                    init_pose6_raw,
                    direct_origin3,
                    eef_offset,
                )
                print(
                    "[ALIGN] calibrated direct-zmq-origin from current pose: "
                    f"{np.array2string(direct_origin3, precision=6, separator=',')}"
                )
                print(
                    "[ALIGN] converted Direct init pose → ZMQ eef: "
                    f"{np.array2string(init_pose6, precision=6, separator=',')}"
                )

            if init_pose6 is not None and (
                np.any(init_pose6[:3] < workspace_min3)
                or np.any(init_pose6[:3] > workspace_max3)
            ):
                raise click.BadParameter("init-pose is outside the configured workspace")

            if init_pose6 is None:
                init_pose6, startup_gripper = current_robot_pose_and_gripper(env)
                if init_gripper is None:
                    init_gripper = startup_gripper
                print(
                    "Environment ready; no --init-pose supplied. "
                    "Holding the current ZMQ eef pose as init_pose."
                )
            else:
                print("Environment ready; moving to explicit ZMQ eef init_pose")
                move_to_pose(
                    env,
                    init_pose6,
                    duration=init_duration,
                    frequency=frequency,
                    target_gripper=init_gripper,
                    gripper_min=gripper_min_width,
                    gripper_max=gripper_max_width,
                )
            if observe_only_quick:
                obs = env.get_obs()
                camera_key = f"camera{wrist_camera_index}_rgb"
                frame = obs[camera_key][-1]
                current_pose = np.concatenate(
                    [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]],
                    axis=-1,
                )
                current_link6_pose = eef_pose_to_link6(current_pose, eef_offset)
                history.seed_dummy_history(frame, current_link6_pose)
                print(
                    "[observe-only] dummy history seeded; running one alignment inference..."
                )
                run_policy(
                    env=env,
                    client=client,
                    history=history,
                    wrist_camera_index=wrist_camera_index,
                    prompt=prompt,
                    frequency=frequency,
                    steps_per_inference=steps_per_inference,
                    action_lead_time=action_lead_time,
                    max_duration=max_duration,
                    max_inference_latency=max_inference_latency,
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    workspace_min=workspace_min3,
                    workspace_max=workspace_max3,
                    gripper_min=gripper_min_width,
                    gripper_max=gripper_max_width,
                    eef_offset=eef_offset,
                    key_counter=key_counter,
                    observe_only=True,
                    warmup_done=True,
                )
                print("[observe-only] done.")
                cv2.destroyAllWindows()
                return

            print("At init_pose. Press F to capture the initial frame.")
            if observe_only:
                print(
                    "[observe-only] waiting for 241-frame history before inference; "
                    "or restart with --observe-only-quick"
                )

            quit_requested = False
            while not quit_requested:
                target_pose, target_gripper = current_robot_pose_and_gripper(env)
                start_policy = False
                iteration = 0
                loop_start = time.monotonic()

                while not start_policy and not quit_requested:
                    cycle_end = loop_start + (iteration + 1) * dt
                    obs = env.get_obs()
                    camera_key = f"camera{wrist_camera_index}_rgb"
                    frame = obs[camera_key][-1]
                    current_pose = np.concatenate(
                        [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]], axis=-1
                    )
                    current_link6_pose = eef_pose_to_link6(current_pose, eef_offset)
                    message = ""

                    for key in key_counter.get_press_events():
                        if key == KeyCode(char="q"):
                            quit_requested = True
                            break
                        if key == KeyCode(char="f"):
                            message = history.press_f(frame, current_link6_pose)
                            print(message)
                        elif key == KeyCode(char="c"):
                            if history.ready:
                                start_policy = True
                            else:
                                message = f"History incomplete: {history.frame_count}/241"
                                print(message)
                        elif key == KeyCode(char="p"):
                            print("P pressed: clearing history and returning to init_pose")
                            history.reset()
                            move_to_pose(
                                env,
                                init_pose6,
                                duration=init_duration,
                                frequency=frequency,
                                target_gripper=init_gripper,
                                gripper_min=gripper_min_width,
                                gripper_max=gripper_max_width,
                            )
                            target_pose, target_gripper = current_robot_pose_and_gripper(env)
                            loop_start = time.monotonic()
                            iteration = 0
                            message = "At init_pose; press F for the new initial frame"
                            print(message)

                    capture_message = history.capture_tick(frame)
                    if capture_message and (history.ready or history.frame_count % 10 == 0 or not history.collecting):
                        print(capture_message)
                    if start_policy or quit_requested:
                        break

                    # Hold pose before F locks the viewpoint; no SpaceMouse teleop.
                    if not history.locked:
                        command = np.concatenate([target_pose, [target_gripper]])
                        env.exec_actions(
                            actions=[command],
                            timestamps=[time.time() + dt],
                            compensate_latency=False,
                        )

                    lines = [
                        history.status(),
                        (
                            f"remaining {(SWAP_FRAMES - len(history.active_segment or [])) / frequency:.1f}s"
                            if history.collecting
                            else ""
                        ),
                        message,
                        "F: capture  C: policy  P: reset  Q: quit",
                    ]
                    view = draw_status(frame, [line for line in lines if line])
                    cv2.imshow("ARX5 ShellGame", view[..., ::-1])
                    cv2.pollKey()
                    precise_wait(cycle_end)
                    iteration += 1

                if quit_requested:
                    break
                quit_requested, warmup_done = run_policy(
                    env=env,
                    client=client,
                    history=history,
                    wrist_camera_index=wrist_camera_index,
                    prompt=prompt,
                    frequency=frequency,
                    steps_per_inference=steps_per_inference,
                    action_lead_time=action_lead_time,
                    max_duration=max_duration,
                    max_inference_latency=max_inference_latency,
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    workspace_min=workspace_min3,
                    workspace_max=workspace_max3,
                    gripper_min=gripper_min_width,
                    gripper_max=gripper_max_width,
                    eef_offset=eef_offset,
                    key_counter=key_counter,
                    observe_only=observe_only,
                    warmup_done=warmup_done,
                )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
