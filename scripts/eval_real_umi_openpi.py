#!/usr/bin/env python3
"""真实机器人 UMI 策略评估（expo-ft / OpenPI pi0.5）。

语义：
  state(10) = se3_to_vec(inv(T_0) @ T_t) + 当前 gripper
  action(10) = inv(T_curr) @ T_target_i + gripper target
  decode: T_i = T_curr @ A_i（所有 waypoint 相对同一当前 EEF，不链式）
  图像: wrist uint8 复制填 DROID 三槽
  夹爪: 训练侧大=夹紧；ZMQ 大=张开，默认双向 invert；
        默认夹取加紧（ZMQ 开度 < threshold → grasp_width）
        --no_binarize_gripper 可关掉

操作壳与 eval_arx5_fast_wam.py 对齐：
  Human: SpaceMouse；C 交给策略，Q 退出，B/Backspace 丢弃 episode
  Policy: S 停止并交还人工

推理：
  bash scripts/umi_replay/serve_umi_policy.sh
  python scripts/umi_replay/eval_real_umi_openpi.py \
    --umi_project_root=/path/to/umi-arx-kian -o ./umi_expo_eval \
    --policy_host=127.0.0.1 --policy_port=8000 --terminal_keys -eh 8 -f 10
"""

from __future__ import annotations

import contextlib
import os
import select
import subprocess
import sys
import termios
import time
import tty
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

SCRIPT_DIR = Path(__file__).resolve().parent
EXPO_FT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openpi_client import image_tools  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from umi_pose_utils import (  # noqa: E402
    episode_rel_state10,
    invert_gripper_width,
    link6_to_x5_eef,
    postprocess_gripper,
    rel_chunk_to_abs7,
    to_uint8_hwc,
    x5_eef_to_link6,
)

DEFAULT_TASK_PROMPT = "pick up the electronic component"
DEFAULT_GRIPPER_MAX_WIDTH = 0.085
X5_EEF_OFFSET_TCP_Z = 0.145
ACTION_DIM = 10
STATE_DIM = 10


class KeyCode:
    def __init__(self, char: str | None = None):
        self.char = char

    def __eq__(self, other):
        return isinstance(other, KeyCode) and self.char == other.char


class Key:
    backspace = object()


def _load_keystroke_counter():
    """仅在 X11/pynput 可用时导入全局按键监听。"""
    global Key, KeyCode
    from peripherals.keystroke_counter import (  # type: ignore
        Key as _Key,
        KeyCode as _KeyCode,
        KeystrokeCounter,
    )

    Key, KeyCode = _Key, _KeyCode
    return KeystrokeCounter


class _DummySpacemouse:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        print(
            "[WARN] SpaceMouse unavailable. Human teleop disabled; "
            "use keyboard C/S/Q. Preferred: Python 3.10 + spnav."
        )
        return self

    def __exit__(self, *args):
        del args
        return False

    def get_motion_state_transformed(self):
        return np.zeros(6, dtype=np.float32)

    def is_button_pressed(self, _idx: int) -> bool:
        return False


def _load_umi_runtime(umi_project_root: str):
    root = Path(umi_project_root).expanduser().resolve()
    if not root.is_dir():
        raise click.ClickException(f"UMI project root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from modules.arx5_env import Arx5Env  # type: ignore
        from utils.other_util import precise_wait  # type: ignore
    except ImportError as exc:
        raise click.ClickException(
            f"{root} must contain modules/arx5_env.py and utils/other_util.py: {exc}"
        ) from exc
    try:
        from peripherals.spacemouse_shared_memory import Spacemouse  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Spacemouse import failed: {exc}")
        Spacemouse = _DummySpacemouse
    return Arx5Env, precise_wait, Spacemouse


class TerminalKeyReader:
    """从 SSH TTY 非阻塞读取 c/q/s/b。"""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("--terminal_keys requires an interactive TTY (ssh -t)")
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        print("[terminal_keys] c=start, s=stop, q=quit, b=drop episode")
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def clear(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)

    def get_press_events(self):
        events = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            event = self._char_to_event(ch)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _char_to_event(ch: str):
        if ch.lower() in {"c", "q", "s", "1"}:
            return KeyCode(char=ch.lower())
        if ch in ("b", "B", "\x7f", "\b"):
            return Key.backspace
        return None


def _collect_key_events(key_counter, terminal_reader):
    events = []
    if key_counter is not None:
        events.extend(key_counter.get_press_events())
    if terminal_reader is not None:
        events.extend(terminal_reader.get_press_events())
    return events


def _clear_key_events(key_counter, terminal_reader):
    if key_counter is not None:
        key_counter.clear()
    if terminal_reader is not None:
        terminal_reader.clear()


def _imshow_available() -> bool:
    try:
        gui = cv2.getBuildInformation().split("GUI:", 1)[1].split("\n", 1)[0]
        if "NONE" in gui.upper():
            return False
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


def _show(win: str, bgr_img: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    try:
        cv2.imshow(win, bgr_img)
        cv2.pollKey()
    except cv2.error:
        pass


def ease_in_out_quad(t: float) -> float:
    t *= 2.0
    if t < 1.0:
        return 0.5 * t * t
    t -= 1.0
    return -0.5 * (t * (t - 2.0) - 1.0)


def _read_tcp(obs: dict) -> tuple[np.ndarray, np.ndarray, float]:
    pos = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
    rotvec = np.asarray(
        obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64
    ).reshape(3)
    grip = float(np.asarray(obs["robot0_gripper_width"][-1]).squeeze())
    return pos, rotvec, grip


def _read_live_tcp(env) -> tuple[np.ndarray, np.ndarray, float, float]:
    state = env.get_robot_state()[0]
    pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
    grip = float(state["gripper_position"])
    timestamp = float(state.get("robot_timestamp", time.time()))
    return pose[:3].copy(), pose[3:6].copy(), grip, timestamp


def _build_policy_obs(
    obs: dict,
    *,
    T0_pos: np.ndarray,
    T0_rotvec: np.ndarray,
    prompt: str,
    eef_offset: float,
    image_size: tuple[int, int],
    invert_state_gripper: bool,
    gripper_min_width: float,
    gripper_max_width: float,
) -> dict:
    wrist = to_uint8_hwc(obs["camera0_rgb"][-1])
    wrist = image_tools.resize_with_pad(wrist, image_size[0], image_size[1])
    exterior = wrist.copy()

    pos_robot, rotvec, grip_zmq = _read_tcp(obs)
    pos_link6 = x5_eef_to_link6(pos_robot, rotvec, eef_offset)
    ref_link6 = x5_eef_to_link6(T0_pos, T0_rotvec, eef_offset)

    # 训练数据大=夹紧；ZMQ 大=张开。state 与 action 必须使用同一训练极性。
    grip_policy = grip_zmq
    if invert_state_gripper:
        grip_policy = float(
            invert_gripper_width(
                grip_zmq,
                gripper_min_width=gripper_min_width,
                gripper_max_width=gripper_max_width,
            )
        )
    state = episode_rel_state10(
        pos_link6,
        rotvec,
        grip_policy,
        ref_pos=ref_link6,
        ref_rotvec=T0_rotvec,
    )
    return {
        "exterior_image_1_left": exterior,
        "exterior_image_2_left": exterior.copy(),
        "wrist_image_left": wrist,
        "state": state,
        "prompt": prompt,
    }


def _exec_abs(env, actions: np.ndarray, timestamps: np.ndarray) -> None:
    """兼容不同 Arx5Env 版本的延迟参数名。"""
    last_error: Exception | None = None
    for kwargs in (
        {"compensate_latency": False},
        {"dynamic_latency": False},
        {},
    ):
        try:
            env.exec_actions(actions=actions, timestamps=timestamps, **kwargs)
            return
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def _hand_off_online_rl(
    expo_ft_root: str, server_host: str, umi_project_root: str
) -> int:
    root = Path(expo_ft_root).expanduser().resolve()
    script = root / "scripts" / "umi_replay" / "run_client_rl.sh"
    if not server_host:
        raise click.ClickException("--server_host / SERVER_HOST required")
    if not script.is_file():
        raise click.ClickException(f"missing {script}")
    env = os.environ.copy()
    env.update(
        SERVER_HOST=server_host,
        UMI_PROJECT_ROOT=str(Path(umi_project_root).expanduser().resolve()),
        EXPO_FT_ROOT=str(root),
    )
    print(f"Online RL -> {script} (SERVER_HOST={server_host})")
    return subprocess.call(["bash", str(script)], cwd=str(root), env=env)


@click.command()
@click.option("--output", "-o", default="./umi_expo_eval_output")
@click.option("--policy_host", default="127.0.0.1")
@click.option("--policy_port", default=8000, type=int)
@click.option("--task_prompt", default=DEFAULT_TASK_PROMPT)
@click.option("--steps_per_inference", "-si", default=16, type=int, show_default=True)
@click.option("--exec_horizon", "-eh", default=8, type=int, show_default=True)
@click.option("--max_duration", "-md", default=2_000_000.0, type=float)
@click.option("--frequency", "-f", default=20.0, type=float, show_default=True)
@click.option("--command_latency", "-cl", default=0.01, type=float)
@click.option("--camera_reorder", "-cr", default="0")
@click.option("--no_mirror", "-nm", is_flag=True)
@click.option("--mirror_swap", is_flag=True)
@click.option("--init_joints", "-j", is_flag=True)
@click.option("--init_delta", default="0,0,0,0.1,-0.1,-1.0", show_default=True)
@click.option("--init_gripper_delta", type=float, default=0.11)
@click.option("--gripper_min_width", type=float, default=0.011)
@click.option("--gripper_max_width", type=float, default=DEFAULT_GRIPPER_MAX_WIDTH)
@click.option(
    "--no_binarize_gripper",
    is_flag=True,
    help="关掉夹取加紧：全程用策略原开度（仍会极性翻转并 clip）",
)
@click.option(
    "--gripper_close_threshold",
    type=float,
    default=0.04,
    show_default=True,
    help="ZMQ 开度低于该值视为夹取，下发 --gripper_grasp_width",
)
@click.option(
    "--gripper_grasp_width",
    type=float,
    default=0.0,
    show_default=True,
    help="夹取时强制下发的开度（米）；0 可压紧元器件",
)
@click.option(
    "--no_invert_gripper",
    is_flag=True,
    help="同时关闭 policy-state ZMQ→训练极性和 action 训练→ZMQ 极性转换",
)
@click.option("--terminal_keys", is_flag=True)
@click.option("--rebase_decode/--no_rebase_decode", default=True, show_default=True)
@click.option("--eef_offset", type=float, default=X5_EEF_OFFSET_TCP_Z)
@click.option("--online_rl", is_flag=True)
@click.option("--server_host", default=lambda: os.environ.get("SERVER_HOST", ""))
@click.option("--expo_ft_root", default=str(EXPO_FT_ROOT))
@click.option(
    "--umi_project_root",
    default=lambda: os.environ.get("UMI_PROJECT_ROOT", ""),
    help="包含 modules/arx5_env.py 的 UMI 工程",
)
def main(
    output,
    policy_host,
    policy_port,
    task_prompt,
    steps_per_inference,
    exec_horizon,
    max_duration,
    frequency,
    command_latency,
    camera_reorder,
    no_mirror,
    mirror_swap,
    init_joints,
    init_delta,
    init_gripper_delta,
    gripper_min_width,
    gripper_max_width,
    no_binarize_gripper,
    gripper_close_threshold,
    gripper_grasp_width,
    no_invert_gripper,
    terminal_keys,
    rebase_decode,
    eef_offset,
    online_rl,
    server_host,
    expo_ft_root,
    umi_project_root,
):
    if not umi_project_root:
        raise click.ClickException(
            "set --umi_project_root or UMI_PROJECT_ROOT "
            "(UMI project containing modules/arx5_env.py)"
        )
    if online_rl:
        raise SystemExit(
            _hand_off_online_rl(expo_ft_root, server_host, umi_project_root)
        )

    Arx5Env, precise_wait, Spacemouse = _load_umi_runtime(umi_project_root)
    try:
        os.sched_setaffinity(os.getpid(), [7])
    except Exception:
        pass

    gripper_min_width = max(0.0, float(gripper_min_width))
    gripper_max_width = float(gripper_max_width)
    if gripper_max_width <= gripper_min_width:
        raise click.BadParameter("must exceed min", param_hint="--gripper_max_width")
    gripper_close_threshold = float(gripper_close_threshold)
    gripper_grasp_width = float(gripper_grasp_width)
    if gripper_close_threshold <= 0.0:
        raise click.BadParameter(
            "must be > 0", param_hint="--gripper_close_threshold"
        )
    if gripper_grasp_width < 0.0:
        raise click.BadParameter("must be >= 0", param_hint="--gripper_grasp_width")
    if steps_per_inference <= 0 or exec_horizon <= 0:
        raise click.BadParameter("horizons must be > 0")
    if exec_horizon > steps_per_inference:
        raise click.BadParameter(
            "must be <= --steps_per_inference", param_hint="--exec_horizon"
        )
    if frequency <= 0:
        raise click.BadParameter("must be > 0", param_hint="--frequency")

    invert_gripper = not no_invert_gripper
    binarize_gripper = not no_binarize_gripper
    eef_offset = float(eef_offset)
    image_size = (224, 224)
    dt = 1.0 / frequency
    vis_ok = _imshow_available()
    if not vis_ok:
        print("[WARN] OpenCV GUI unavailable; disabling imshow.")
        if not terminal_keys:
            terminal_keys = True
            print("[WARN] Automatically enabling --terminal_keys.")

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output, "action"), exist_ok=True)

    policy = WebsocketClientPolicy(policy_host, policy_port)
    print(f"OpenPI policy service: ws://{policy_host}:{policy_port}")
    print(f"Task prompt: {task_prompt}")
    print(
        f"[ALIGN] state={STATE_DIM}D episode-rel; action={ACTION_DIM}D "
        f"current-EEF-rel; eef_offset={eef_offset:.3f}; "
        f"invert_state+action_gripper={invert_gripper}; "
        f"binarize={binarize_gripper}; rebase={rebase_decode}"
    )
    if binarize_gripper:
        print(
            f"[GRIP] tighten if exec < {gripper_close_threshold:.3f}m → "
            f"{gripper_grasp_width:.4f}m"
        )

    robots_config = [
        {
            "robot_type": "arx5",
            "robot_ip": "127.0.0.1",
            "robot_port": 8765,
            "robot_obs_latency": 0.20,
            "robot_action_latency": 0.01,
            "tcp_offset": 0,
            "height_threshold": 0.08,
            "sphere_radius": 0.1,
            "sphere_center": [0, -0.06, -0.185],
        }
    ]

    key_counter_ctx = contextlib.nullcontext(None)
    if not terminal_keys:
        try:
            key_counter_ctx = _load_keystroke_counter()()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] KeystrokeCounter unavailable ({exc}); using terminal.")
            terminal_keys = True
    terminal_reader_ctx = (
        TerminalKeyReader() if terminal_keys else contextlib.nullcontext(None)
    )

    with SharedMemoryManager() as shm_manager:
        with (
            Spacemouse(shm_manager=shm_manager, deadzone=0.1) as sm,
            key_counter_ctx as key_counter,
            terminal_reader_ctx as terminal_reader,
            Arx5Env(
                output_dir=output,
                robots_config=robots_config,
                frequency=frequency,
                obs_image_resolution=image_size,
                obs_float32=True,
                camera_reorder=[
                    int(x) for x in camera_reorder.split(",") if x != ""
                ],
                init_joints=init_joints,
                enable_multi_cam_vis=False,
                camera_obs_latency=0.17,
                camera_obs_horizon=2,
                robot_obs_horizon=2,
                no_mirror=no_mirror,
                mirror_swap=mirror_swap,
                max_pos_speed=4.0,
                max_rot_speed=10.0,
                shm_manager=shm_manager,
            ) as env,
        ):
            cv2.setNumThreads(2)
            print("Waiting for camera...")
            time.sleep(1.0)
            while not env.is_ready:
                time.sleep(0.1)
            print("Env is ready")

            if init_delta is not None or abs(init_gripper_delta) > 0:
                delta6 = np.asarray(
                    [float(v) for v in init_delta.split(",")], dtype=np.float64
                )
                if delta6.shape != (6,):
                    raise click.BadParameter("expected 6 comma-separated values")
                robot_state = env.get_robot_state()[0]
                start_pose = np.asarray(
                    robot_state["ActualTCPPose"], dtype=np.float64
                ).copy()
                start_grip = float(robot_state["gripper_position"])
                target_pose = start_pose + delta6
                target_grip = np.clip(
                    start_grip + init_gripper_delta,
                    gripper_min_width,
                    gripper_max_width,
                )
                warmup_sec = 2.5
                steps = max(2, int(warmup_sec * frequency))
                commands, timestamps = [], []
                start_time = time.time() + 0.3
                for k in range(steps):
                    alpha = ease_in_out_quad(k / (steps - 1))
                    command = np.zeros(7, dtype=np.float64)
                    command[:6] = start_pose + alpha * (target_pose - start_pose)
                    command[6] = np.clip(
                        start_grip + alpha * (target_grip - start_grip),
                        gripper_min_width,
                        gripper_max_width,
                    )
                    commands.append(command)
                    timestamps.append(start_time + k * dt)
                _exec_abs(env, np.asarray(commands), np.asarray(timestamps))
                time.sleep(warmup_sec + 0.2)
                print("Initialization adjustment done")

            print("Warming up policy inference...")
            obs = env.get_obs()
            pos0, rot0, _ = _read_tcp(obs)
            warm_obs = _build_policy_obs(
                obs,
                T0_pos=pos0,
                T0_rotvec=rot0,
                prompt=task_prompt,
                eef_offset=eef_offset,
                image_size=image_size,
                invert_state_gripper=invert_gripper,
                gripper_min_width=gripper_min_width,
                gripper_max_width=gripper_max_width,
            )
            warm_out = policy.infer(warm_obs)
            print(f"Warmup done: actions={np.asarray(warm_out['actions']).shape}")

            while True:
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack(
                    [state["ActualTCPPose"] for state in robot_states]
                )
                grip_target = np.asarray(
                    [state["gripper_position"] for state in robot_states]
                )
                control_indices = [0]
                loop_start = time.monotonic()
                iteration = 0
                episode_id = env.replay_buffer.n_episodes

                while True:
                    cycle_end = loop_start + (iteration + 1) * dt
                    sample_time = cycle_end - command_latency
                    command_time = cycle_end + dt
                    vis_img = obs["camera0_rgb"][-1]
                    obs = env.get_obs()
                    display = to_uint8_hwc(vis_img).copy()
                    cv2.putText(
                        display,
                        f"Episode: {episode_id}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    _show("default", display[..., ::-1], vis_ok)

                    start_policy = False
                    for event in _collect_key_events(
                        key_counter, terminal_reader
                    ):
                        if event == KeyCode(char="q"):
                            env.end_episode()
                            raise SystemExit(0)
                        if event == KeyCode(char="c"):
                            start_policy = True
                        elif event == Key.backspace:
                            if terminal_keys or click.confirm("Drop episode?"):
                                env.drop_episode()
                                _clear_key_events(key_counter, terminal_reader)
                        elif event == KeyCode(char="1"):
                            control_indices = [0]
                    if start_policy:
                        break

                    precise_wait(sample_time)
                    motion = sm.get_motion_state_transformed()
                    delta_pos = motion[:3] * (0.5 / frequency) * 0.4
                    delta_rot = st.Rotation.from_euler(
                        "xyz", motion[3:] * (1.5 / frequency) * 0.8
                    )
                    for robot_idx in control_indices:
                        target_pose[robot_idx, :3] += delta_pos
                        target_pose[robot_idx, 3:] = (
                            delta_rot
                            * st.Rotation.from_rotvec(
                                target_pose[robot_idx, 3:]
                            )
                        ).as_rotvec()

                    delta_grip = 0.0
                    if sm.is_button_pressed(0) and sm.is_button_pressed(1):
                        print("Reset to home...")
                        for robot_idx in control_indices:
                            env.robots[robot_idx].reset_to_home()
                        fresh = env.get_robot_state()
                        target_pose = np.stack(
                            [state["ActualTCPPose"] for state in fresh]
                        )
                        grip_target = np.asarray(
                            [state["gripper_position"] for state in fresh]
                        )
                    elif sm.is_button_pressed(0):
                        delta_grip = -0.02 / frequency
                    elif sm.is_button_pressed(1):
                        delta_grip = 0.02 / frequency
                    grip_target = np.clip(
                        grip_target + delta_grip,
                        gripper_min_width,
                        gripper_max_width,
                    )
                    command = np.zeros(7)
                    command[:6] = target_pose[0]
                    command[6] = grip_target[0]
                    _exec_abs(
                        env,
                        command[None],
                        np.asarray(
                            [command_time - time.monotonic() + time.time()]
                        ),
                    )
                    precise_wait(cycle_end)
                    iteration += 1

                episode_active = False
                try:
                    start_delay = 1.0
                    eval_start = time.time() + start_delay
                    monotonic_start = time.monotonic() + start_delay
                    env.start_episode(eval_start)
                    episode_active = True
                    obs = env.get_obs()
                    T0_pos, T0_rotvec, _ = _read_tcp(obs)
                    print(f"Episode T0 (robot eef): {T0_pos}")
                    frame_latency = 1.0 / 60.0
                    precise_wait(eval_start - frame_latency, time_func=time.time)
                    print("Policy in control!")
                    iteration = 0

                    while True:
                        obs = env.get_obs()
                        pos_robot, rotvec, _ = _read_tcp(obs)
                        pos_link6 = x5_eef_to_link6(
                            pos_robot, rotvec, eef_offset
                        )
                        infer_start = time.time()
                        request = _build_policy_obs(
                            obs,
                            T0_pos=T0_pos,
                            T0_rotvec=T0_rotvec,
                            prompt=task_prompt,
                            eef_offset=eef_offset,
                            image_size=image_size,
                            invert_state_gripper=invert_gripper,
                            gripper_min_width=gripper_min_width,
                            gripper_max_width=gripper_max_width,
                        )
                        result = policy.infer(request)
                        infer_seconds = time.time() - infer_start
                        actions = np.asarray(
                            result["actions"], dtype=np.float64
                        )
                        if actions.ndim == 1:
                            actions = actions[None]
                        if actions.shape[1] < ACTION_DIM:
                            raise ValueError(
                                f"Expected action dim >= 10, got {actions.shape}"
                            )
                        if len(actions) < exec_horizon:
                            raise ValueError(
                                f"chunk {len(actions)} < exec_horizon "
                                f"{exec_horizon}"
                            )
                        chunk = actions[:steps_per_inference, :ACTION_DIM][
                            :exec_horizon
                        ]

                        base_pos, base_rot = pos_link6, rotvec
                        if rebase_decode:
                            try:
                                live_pos, live_rot, _, _ = _read_live_tcp(env)
                                base_pos = x5_eef_to_link6(
                                    live_pos, live_rot, eef_offset
                                )
                                base_rot = live_rot
                            except Exception as exc:  # noqa: BLE001
                                print(f"[REBASE] fallback to observation: {exc}")

                        absolute_link6 = rel_chunk_to_abs7(
                            chunk, cur_pos=base_pos, cur_rotvec=base_rot
                        )
                        absolute_exec = absolute_link6.copy()
                        for i in range(len(absolute_exec)):
                            absolute_exec[i, :3] = link6_to_x5_eef(
                                absolute_exec[i, :3],
                                absolute_exec[i, 3:6],
                                eef_offset,
                            )
                        raw_grip = absolute_exec[:, 6].copy()
                        absolute_exec[:, 6] = postprocess_gripper(
                            raw_grip,
                            gripper_min_width=gripper_min_width,
                            gripper_max_width=gripper_max_width,
                            invert=invert_gripper,
                            binarize=binarize_gripper,
                            close_threshold=gripper_close_threshold,
                            grasp_width=gripper_grasp_width,
                        )

                        command_start = time.time() + 0.05
                        timestamps = (
                            np.arange(exec_horizon, dtype=np.float64) * dt
                            + command_start
                        )
                        episode_id = env.replay_buffer.n_episodes
                        print(
                            f"[CYCLE] ep={episode_id} iter={iteration} "
                            f"infer={infer_seconds*1000:.0f}ms "
                            f"state_grip={request['state'][-1]:.4f} "
                            f"raw_grip=[{raw_grip.min():.4f},"
                            f"{raw_grip.max():.4f}] "
                            f"exec_grip=[{absolute_exec[:, 6].min():.4f},"
                            f"{absolute_exec[:, 6].max():.4f}]"
                        )
                        _exec_abs(env, absolute_exec, timestamps)

                        display = to_uint8_hwc(
                            obs["camera0_rgb"][-1]
                        ).copy()
                        cv2.putText(
                            display,
                            (
                                f"Episode: {episode_id}, Time: "
                                f"{time.monotonic()-monotonic_start:.1f}s"
                            ),
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                        _show("default", display[..., ::-1], vis_ok)

                        stop = any(
                            event == KeyCode(char="s")
                            for event in _collect_key_events(
                                key_counter, terminal_reader
                            )
                        )
                        if time.time() - eval_start > max_duration:
                            print("Max duration reached.")
                            stop = True
                        if stop:
                            print("Stopped by user.")
                            break
                        precise_wait(
                            float(timestamps[-1]) + frame_latency,
                            time_func=time.time,
                        )
                        iteration += exec_horizon
                except KeyboardInterrupt:
                    print("Interrupted!")
                finally:
                    if episode_active:
                        env.end_episode()
                print("Stopped. Returning to human control.")


if __name__ == "__main__":
    main()
