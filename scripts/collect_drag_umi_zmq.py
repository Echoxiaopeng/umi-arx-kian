"""
ARX5 拖拽采数脚本 — ZMQ 版本（重力补偿 / 零力拖拽）。

依赖已启动的 ZMQ 机械臂服务（eef_link + ee2tcp，物理开口夹爪）:
    # 普通控制 / 采数（ac_one；P 键会自动回退到阻尼拖拽）:
    python python/communication/zmq_server_ac_one.py X5 can3 \\
        --gripper-open-readout -3.51 --gripper-width 0.085

    # 真零力（可选）:
    python python/communication/zmq_server_gravity_compensation.py X5 can3

启动微调与 eval_arx5_fast_wam.py 一致（相对 home 的 delta，非绝对位姿）:
    --init_delta 0,0,0,0.1,-0.1,-1.0
    --init_gripper_delta 0.11
    （也可用 --init_pose 指定绝对 TCP，会覆盖 delta）

与 scripts/collect_drag_umi.py（直连 SDK / link6）对比:
  - 本脚本走 Arx5Client → ZMQ server，采集的是 X5 eef_link 经 ee2tcp 后的 TCP
  - 夹爪为物理开口宽度（米），越大越张开（约 0~0.085）；SDK 版是电机行程映射、极性相反
  - 姿态已与 UMI 轴对齐；相对 SDK/link6 数据通常只需沿 TCP Z 做位置偏移（约 0.145m）

不依赖策略服务。打开重力补偿后手动拖动机械臂，按固定频率记录
相机 + 机器人状态；action 取当前末端位姿 + 夹爪开度。

同步策略:
  1. 每步 env.get_obs()：多相机最近邻对齐，机器人状态插值到相机时间
  2. 用 obs 相机时间戳把 pose-as-action 写入 action_accumulator
  3. env.end_episode()：以 action 时间栅格为轴，插值对齐 obs / 图像写入
     replay_buffer.zarr，并保存 videos/{episode_id}/*.mp4

用法:
    # 先起 ZMQ server，再网页入口（选择 数采 / 数据管理）:
    conda activate umi-arx-kian
    python scripts/collect_drag_umi_zmq.py -o data/collect_demo_zmq --web

    # 终端按键 + 网页（同上，进 hub 后再选手动数采）:
    python scripts/collect_drag_umi_zmq.py -o data/collect_demo_zmq --terminal_keys --web

    # 跳过选择页，直接数采（旧行为）:
    python scripts/collect_drag_umi_zmq.py -o data/collect_demo_zmq --terminal_keys --web --collect_direct

    # 跳过选择页，只管理已有数据（不启机械臂 / 相机）:
    python scripts/collect_drag_umi_zmq.py -o data/collect_demo_zmq --viewer_only

    # 与 fast_wam 相同的启动微调（默认已是这组 delta）:
    python scripts/collect_drag_umi_zmq.py -o data/collect_demo_zmq --terminal_keys --web --collect_direct \\
        --init_delta 0,0,0,0.1,-0.1,-1.0 --init_gripper_delta 0.11

    # 或指定绝对 TCP:
    python scripts/collect_drag_umi_zmq.py -o data/collect_demo_zmq \\
        --robot_ip 127.0.0.1 --robot_port 8765 \\
        --init_pose 0.4,0.0,0.25,0,1.57,0 --init_gripper 0.08

    python scripts/collect_drag_umi_zmq.py -o data/collect_demo_zmq --no_init_pose

键盘:
  P = 切换零力：首次进入；再按退出零力并缓动回启动微调后的位姿
  C = 开始采集（默认须先按 P；--no_zero_force 时可直接 C）
  S = 停止并保存
  D = 丢弃当前正在录的 episode（未保存）
  E = 删除已保存的指定 episode（空闲时；输入编号后 Y 确认）
  I = 开关位姿打印：开后约每 0.5s 输出一次；再按关闭
  Q = 退出（停留当前位置，不回 init/home）

启动流程: reset_to_home → 按 --init_delta / --init_gripper_delta 微调（与 fast_wam 相同；
--no_init_pose 跳过）→ 等 P 进零力（或 --no_zero_force 直接 C）。
默认不刷屏；按 I 开始/停止持续打印当前位姿（可复制为 --init_pose）。
不做 episode 级 init 移动；按 C 时臂在哪就从哪开录。

Author: Kian
"""

from __future__ import annotations

import contextlib
import os
import select
import shutil
import sys
import termios
import time
import tty
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import click
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from modules.arx5_env_with_images_zmq import Arx5EnvWithImages  # noqa: E402
from modules.replay_buffer import ReplayBuffer  # noqa: E402
from utils.other_util import precise_wait  # noqa: E402

# pynput 在无 DISPLAY 时会炸；viewer_only 不需要键盘钩子
try:
    from peripherals.keystroke_counter import Key, KeyCode  # noqa: E402
except Exception:  # noqa: BLE001

    class Key:  # type: ignore[no-redef]
        backspace = object()
        enter = object()
        esc = object()

    class KeyCode:  # type: ignore[no-redef]
        def __init__(self, char=None):
            self.char = char

        def __eq__(self, other):
            return isinstance(other, KeyCode) and self.char == getattr(
                other, "char", None
            )

        def __hash__(self):
            return hash(self.char)


# ---------------------------------------------------------------------------
# Terminal keys (SSH / no X11)
# ---------------------------------------------------------------------------
class TerminalKeyReader:
    """从 SSH 终端 stdin 非阻塞读单键（不依赖 pynput / X11）。"""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError(
                "--terminal_keys 需要交互式 TTY；请用 ssh -t 连接，"
                "或在本机终端运行。"
            )
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        print(
            "[terminal_keys] 在本终端按键: "
            "p=开关零力(再按回init), c=开始, s=保存, d=丢弃当前, "
            "e=删已保存, i=开关位姿打印, q=回home退出"
        )
        return self

    def __exit__(self, exc_type, exc, tb):
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
        if ch in ("p", "P"):
            return KeyCode(char="p")
        if ch in ("c", "C"):
            return KeyCode(char="c")
        if ch in ("q", "Q"):
            return KeyCode(char="q")
        if ch in ("s", "S"):
            return KeyCode(char="s")
        if ch in ("d", "D"):
            return KeyCode(char="d")
        if ch in ("e", "E"):
            return KeyCode(char="e")
        if ch in ("i", "I"):
            return KeyCode(char="i")
        if ch in ("y", "Y"):
            return KeyCode(char="y")
        if ch in ("n", "N"):
            return KeyCode(char="n")
        if ch in "0123456789":
            return KeyCode(char=ch)
        if ch in ("\n", "\r"):
            return Key.enter
        if ch == "\x1b":
            return Key.esc
        if ch in ("\x7f", "\b"):
            return Key.backspace
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_robots_config(robot_ip: str = "127.0.0.1", robot_port: int = 8765):
    return [
        {
            "robot_type": "arx5",
            "robot_ip": robot_ip,
            "robot_port": int(robot_port),
            "robot_obs_latency": 0.1867,
            "robot_action_latency": 0.01,
            "tcp_offset": 0.0,
            "height_threshold": 0.0,
            "sphere_radius": 0.1,
            "sphere_center": [0, -0.06, -0.185],
        }
    ]


def action_from_obs(obs: dict, n_robots: int) -> np.ndarray:
    """从对齐后的 obs 拼绝对 TCP action: [xyz, rotvec, gripper] * n_robots."""
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


def tcp_pose_grip_from_robot_state(
    env: Arx5EnvWithImages,
) -> list[tuple[np.ndarray, float]]:
    """
    从控制器 ActualTCPPose 取位姿（与 set_tcp_pose / exec_actions 同一套数）。

    注意: get_obs() 里的 eef_rot 经 PoseInterpolator 后常变成短角 rotvec，
    而 Arx5Client.tcp_pose 读数是长角等价形式；复制 init_pose 必须用这里。
    ZMQ 路径: eef_link + ee2tcp；夹爪为物理开口（越大越张开）。
    """
    out: list[tuple[np.ndarray, float]] = []
    for state in env.get_robot_state():
        pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6).copy()
        grip = float(state["gripper_position"])
        out.append((pose, grip))
    return out


def tcp_pose_grip_from_obs(
    obs: dict, n_robots: int
) -> list[tuple[np.ndarray, float]]:
    """每臂 (pose6 [xyz,rotvec], gripper_width)，来自 get_obs（仅供画面叠加）。"""
    out: list[tuple[np.ndarray, float]] = []
    for robot_idx in range(n_robots):
        pose = np.concatenate(
            [
                np.asarray(obs[f"robot{robot_idx}_eef_pos"][-1], dtype=np.float64),
                np.asarray(
                    obs[f"robot{robot_idx}_eef_rot_axis_angle"][-1], dtype=np.float64
                ),
            ],
            axis=-1,
        ).reshape(6)
        grip = float(np.asarray(obs[f"robot{robot_idx}_gripper_width"][-1]).squeeze())
        out.append((pose, grip))
    return out


def format_pose_cli(pose6: np.ndarray, grip: float | None = None) -> str:
    """生成可直接粘贴的 --init_pose / --init_gripper 参数串。"""
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    pose_s = ",".join(f"{v:.6f}" for v in pose6)
    if grip is None:
        return f"--init_pose {pose_s}"
    return f"--init_pose {pose_s} --init_gripper {float(grip):.6f}"


def print_current_pose(env: Arx5EnvWithImages, *, tag: str = "pose") -> None:
    """打印可复制的 init 位姿（命令坐标系 ActualTCPPose）。"""
    for robot_idx, (pose, grip) in enumerate(tcp_pose_grip_from_robot_state(env)):
        print(
            f"[{tag}] robot{robot_idx} "
            f"pos=[{pose[0]:.4f},{pose[1]:.4f},{pose[2]:.4f}] "
            f"rotvec=[{pose[3]:.4f},{pose[4]:.4f},{pose[5]:.4f}] "
            f"grip={grip:.4f}"
        )
        print(f"[{tag}] copy: {format_pose_cli(pose, grip)}")


def ease_in_out_quad(t: float) -> float:
    t *= 2.0
    if t < 1.0:
        return 0.5 * t * t
    t -= 1.0
    return -0.5 * (t * (t - 2.0) - 1.0)


def parse_pose6(text: str) -> np.ndarray:
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 6:
        raise click.BadParameter(
            f"--init_pose 需要 6 个数 x,y,z,rx,ry,rz，收到 {len(vals)} 个"
        )
    return np.asarray(vals, dtype=np.float64)


def capture_robot_pose_targets(
    env: Arx5EnvWithImages,
) -> list[tuple[np.ndarray, float]]:
    """记录各臂当前 TCP 位姿 + 夹爪（与 exec_actions 同源 ActualTCPPose）。"""
    out: list[tuple[np.ndarray, float]] = []
    for state in env.get_robot_state():
        pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6).copy()
        grip = float(state["gripper_position"])
        out.append((pose, grip))
    return out


def move_to_tcp_pose(
    env: Arx5EnvWithImages,
    targets: list[tuple[np.ndarray, float | None]],
    *,
    frequency: float,
    duration: float = 2.5,
    label: str = "target",
    ensure_end_control: bool = False,
) -> None:
    """
    末端控制模式下缓动到目标 TCP 位姿。

    targets: 每臂 (pose6, gripper)；gripper 为 None 则保持当前开度。
    与 fast_wam / init_delta 相同：全频率线性插值 pose6（避免 Slerp+降频导致抖/超距）。
    """
    if ensure_end_control and hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
        except Exception as exc:  # noqa: BLE001
            print(f"set_end_control warning: {exc}")

    robot_states = env.get_robot_state()
    n_robots = len(robot_states)
    if len(targets) != n_robots:
        raise ValueError(f"targets len {len(targets)} != n_robots {n_robots}")

    start_pose_all = []
    start_grip_all = []
    target_pose_all = []
    target_grip_all = []
    for ridx in range(n_robots):
        curr6 = robot_states[ridx]["ActualTCPPose"].astype(np.float64).copy()
        grip = float(robot_states[ridx]["gripper_position"])
        tgt6 = np.asarray(targets[ridx][0], dtype=np.float64).reshape(6)
        tgt_grip = targets[ridx][1]
        tgt_grip = grip if tgt_grip is None else float(tgt_grip)
        start_pose_all.append(curr6)
        target_pose_all.append(tgt6)
        start_grip_all.append(grip)
        target_grip_all.append(tgt_grip)

    start_pose_all = np.stack(start_pose_all, axis=0)
    target_pose_all = np.stack(target_pose_all, axis=0)
    start_grip_all = np.asarray(start_grip_all, dtype=np.float64)
    target_grip_all = np.asarray(target_grip_all, dtype=np.float64)

    print(f"Moving to {label} (end-effector control)...")
    for ridx in range(n_robots):
        print(
            f"  robot{ridx}  from "
            f"{np.array2string(start_pose_all[ridx], precision=4, separator=',')} "
            f"grip={start_grip_all[ridx]:.4f}"
        )
        print(
            f"  robot{ridx}    to "
            f"{np.array2string(target_pose_all[ridx], precision=4, separator=',')} "
            f"grip={target_grip_all[ridx]:.4f}"
        )

    dt = 1.0 / float(frequency)
    steps = max(2, int(float(duration) * float(frequency)))
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    t0 = time.time() + 0.3
    for k in range(steps):
        a = ease_in_out_quad(k / (steps - 1))
        cmd = np.zeros(7 * n_robots, dtype=np.float64)
        for ridx in range(n_robots):
            cmd[7 * ridx : 7 * ridx + 6] = start_pose_all[ridx] + a * (
                target_pose_all[ridx] - start_pose_all[ridx]
            )
            cmd[7 * ridx + 6] = start_grip_all[ridx] + a * (
                target_grip_all[ridx] - start_grip_all[ridx]
            )
        actions.append(cmd)
        timestamps.append(t0 + k * dt)

    env.exec_actions(
        actions=np.asarray(actions),
        timestamps=np.asarray(timestamps),
        compensate_latency=False,
    )
    time.sleep(float(duration) + 0.2)


def move_to_init_pose(
    env: Arx5EnvWithImages,
    target_pose6: np.ndarray,
    *,
    frequency: float,
    duration: float = 2.5,
    init_gripper: float | None = None,
) -> None:
    """缓动到给定绝对 TCP 位姿（启动时尚未进零力，不要调 set_end_control）。"""
    n_robots = len(env.robots)
    targets = [(target_pose6, init_gripper) for _ in range(n_robots)]
    move_to_tcp_pose(
        env,
        targets,
        frequency=frequency,
        duration=duration,
        label="init pose",
        ensure_end_control=False,
    )
    print("Init pose reached. Press P to enter zero-force, then C to record.")
    print_current_pose(env, tag="after-init")


def apply_init_delta_like_fast_wam(
    env: Arx5EnvWithImages,
    *,
    init_delta: str | None,
    init_gripper_delta: float,
    frequency: float,
    duration: float = 2.5,
    gripper_min_width: float = 0.011,
    gripper_max_width: float = 0.085,
) -> tuple[np.ndarray, float]:
    """
    与 eval_arx5_fast_wam.py 完全相同的启动微调实现:
      tgt_pose = ActualTCPPose + init_delta
      tgt_grip = clip(gripper + init_gripper_delta, min, max)
      线性插值 pose6（含 rotvec），全频率 steps，再 exec_actions

    注意: 不要用 Slerp + 降频；ZMQ server 要求相邻 EE 6D L2 < 0.1，
    大步长 / 表示跳变会触发 “far away from last command” 并弄死控制器。
    """
    delta6 = np.zeros(6, dtype=np.float64)
    if init_delta is not None and str(init_delta).strip() != "":
        vals = [float(v) for v in str(init_delta).split(",")]
        if len(vals) != 6:
            raise ValueError(
                f"--init_delta 需要 6 个数 dx,dy,dz,drx,dry,drz，收到 {len(vals)} 个"
            )
        delta6[:] = vals

    robot_states = env.get_robot_state()
    start_pose = robot_states[0]["ActualTCPPose"].astype(np.float64).copy()
    start_grip = float(robot_states[0]["gripper_position"])
    tgt_pose = start_pose + delta6
    tgt_grip = float(
        np.clip(start_grip + init_gripper_delta, gripper_min_width, gripper_max_width)
    )

    print(
        "Init fine-tune (same as eval_arx5_fast_wam): "
        f"delta={np.array2string(delta6, precision=4, separator=',')} "
        f"gripper_delta={init_gripper_delta:+.4f}"
    )
    print(
        f"  robot0  from {np.array2string(start_pose, precision=4, separator=',')} "
        f"grip={start_grip:.4f}"
    )
    print(
        f"  robot0    to {np.array2string(tgt_pose, precision=4, separator=',')} "
        f"grip={tgt_grip:.4f}"
    )

    # 与 fast_wam 一致：全频率 + 线性 6D（不用 Slerp、不降到 20Hz）
    dt = 1.0 / float(frequency)
    steps = max(2, int(float(duration) * float(frequency)))
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    t0 = time.time() + 0.3
    for k in range(steps):
        a = ease_in_out_quad(k / (steps - 1))
        cmd = np.zeros(7, dtype=np.float64)
        cmd[:6] = start_pose + a * (tgt_pose - start_pose)
        cmd[6] = float(
            np.clip(
                start_grip + a * (tgt_grip - start_grip),
                gripper_min_width,
                gripper_max_width,
            )
        )
        actions.append(cmd)
        timestamps.append(t0 + k * dt)

    env.exec_actions(
        actions=np.asarray(actions),
        timestamps=np.asarray(timestamps),
        compensate_latency=False,
    )
    time.sleep(float(duration) + 0.2)

    print("Init fine-tune done. Press P to enter zero-force, then C to record.")
    print_current_pose(env, tag="after-init")
    return tgt_pose, tgt_grip


def record_pose_action(env: Arx5EnvWithImages, obs: dict) -> None:
    """
    把当前位姿记入 action_accumulator（不向机械臂下发指令）。

    时间戳用 get_obs 对齐后的相机时间，供 end_episode 做 episode 级插值对齐。
    """
    if env.action_accumulator is None:
        return
    n_robots = len(env.robots)
    action = action_from_obs(obs, n_robots)
    t = float(np.asarray(obs["timestamp"][-1]))
    env.action_accumulator.put(action[None, :], np.array([t], dtype=np.float64))


def abort_episode(env: Arx5EnvWithImages) -> None:
    """停止录制且不写入 zarr（避免空 episode 时 drop 误删上一条）。"""
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


def check_episode_alignment(env) -> None:
    """启动采集前校验编号，并清理异常退出遗留的孤立视频目录。"""
    n_episodes = env.replay_buffer.n_episodes
    expected_ids = set(range(n_episodes))
    video_ids = {
        int(path.name)
        for path in env.video_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    }

    missing_video_ids = sorted(expected_ids - video_ids)
    if missing_video_ids:
        raise RuntimeError(
            "replay_buffer.zarr 中存在没有对应视频的 episode："
            f"{missing_video_ids}。请先修复输出目录，避免继续采集后编号错位。"
        )

    orphan_video_ids = sorted(video_ids - expected_ids)
    for episode_id in orphan_video_ids:
        video_dir = env.video_dir.joinpath(str(episode_id))
        shutil.rmtree(str(video_dir))
        print(
            f"Removed stale videos/{episode_id} "
            "(no corresponding zarr episode)."
        )


def delete_saved_episode(env, episode_id: int) -> bool:
    """删除已写入 zarr 的指定 episode，并重编号后续 videos/{id} 以保持连续。"""
    n_episodes = env.replay_buffer.n_episodes
    if n_episodes <= 0:
        print("No saved episode to delete.")
        return False
    if episode_id < 0 or episode_id >= n_episodes:
        print(
            f"Episode {episode_id} does not exist "
            f"(valid: 0..{n_episodes - 1})."
        )
        return False

    env.replay_buffer.delete_episode(episode_id)

    video_dir = env.video_dir.joinpath(str(episode_id))
    if video_dir.exists():
        shutil.rmtree(str(video_dir))

    for src_id in range(episode_id + 1, n_episodes):
        src = env.video_dir.joinpath(str(src_id))
        dst = env.video_dir.joinpath(str(src_id - 1))
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(str(dst))
        src.rename(dst)

    remaining = env.replay_buffer.n_episodes
    print(f"Deleted saved episode {episode_id}. Remaining: {remaining}.")
    if episode_id < n_episodes - 1:
        print(
            f"Renumbered videos {episode_id + 1}..{n_episodes - 1} "
            f"-> {episode_id}..{n_episodes - 2}."
        )
    check_episode_alignment(env)
    return True


def save_episode(env: Arx5EnvWithImages, episode_id: int) -> bool:
    """保存并验证 zarr/video 同时成功；任一侧失败则不保留该 episode。"""
    if env.replay_buffer.n_episodes != episode_id:
        raise RuntimeError(
            f"Episode ID changed before save: expected {episode_id}, "
            f"got {env.replay_buffer.n_episodes}."
        )

    video_dir = env.video_dir.joinpath(str(episode_id))
    try:
        env.end_episode()
    except Exception:
        # zarr 尚未提交时删除本次视频，避免下次从同一编号开始时错位。
        if env.replay_buffer.n_episodes == episode_id and video_dir.exists():
            shutil.rmtree(str(video_dir))
        raise

    n_episodes = env.replay_buffer.n_episodes
    if n_episodes == episode_id:
        # 没有有效时间步，end_episode 不会写 zarr。
        if video_dir.exists():
            shutil.rmtree(str(video_dir))
        print(
            f"Episode {episode_id} had no valid aligned steps; "
            "video removed and episode not saved."
        )
        return False

    if n_episodes != episode_id + 1:
        raise RuntimeError(
            f"Unexpected zarr episode count after save: {n_episodes} "
            f"(expected {episode_id + 1})."
        )

    video_files = list(video_dir.glob("*.mp4")) if video_dir.exists() else []
    if not video_files:
        # 当前调用刚写入的必定是最后一条，因此可以安全回滚。
        env.replay_buffer.drop_episode()
        if video_dir.exists():
            shutil.rmtree(str(video_dir))
        print(
            f"Episode {episode_id} video was not created; "
            "zarr episode rolled back."
        )
        return False

    return True


def leave_drag_mode(env: Arx5EnvWithImages) -> None:
    """退出拖拽：恢复末端控制再进阻尼，避免臂悬空在零力模式。"""
    # try:
    #     env.set_end_control()
    #     time.sleep(0.05)
    # except Exception as exc:  # noqa: BLE001
    #     print(f"set_end_control warning: {exc}")
    # try:
    #     env.set_to_damping()
    # except Exception as exc:  # noqa: BLE001
    #     print(f"set_to_damping warning: {exc}")


def leave_zero_force_to_init(
    env: Arx5EnvWithImages,
    init_pose6: np.ndarray | None,
    *,
    frequency: float,
    duration: float = 2.5,
    init_gripper: float | None = None,
) -> None:
    """
    退出零力后缓动回 init_pose。

    不在拖拽点硬锁满增益；先软退出并回位，到达后再恢复满增益。
    set_end_control(wait=True) 会阻塞到控制器完成增益爬升与位姿同步。
    """
    print("Zero-force OFF. Leaving drag mode...")
    if hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
        except Exception as exc:  # noqa: BLE001
            print(f"set_end_control warning: {exc}")

    if init_pose6 is None:
        print("No --init_pose given; staying at current pose.")
    else:
        move_to_init_pose(
            env,
            init_pose6,
            frequency=frequency,
            duration=duration,
            init_gripper=init_gripper,
        )

    if hasattr(env, "restore_gains_after_drag"):
        try:
            env.restore_gains_after_drag()
        except Exception as exc:  # noqa: BLE001
            print(f"restore_gains_after_drag warning: {exc}")


def leave_drag_before_quit(
    env: Arx5EnvWithImages,
    *,
    zero_force_active: bool = False,
) -> None:
    """按 Q 退出：若在拖拽中则软退出控制，停留在当前位置，不回 init/home。"""
    if not zero_force_active:
        print("Quit — stay at current pose (no home move).")
        return
    print("Quit — leave drag, stay at current pose (no home move).")
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


def make_vis_image(
    obs: dict,
    episode_id: int,
    t_elapsed: float,
    n_steps: int,
    *,
    zero_force_active: bool = False,
    no_zero_force: bool = False,
    recording: bool = False,
    delete_prompt: str | None = None,
    n_robots: int = 1,
) -> np.ndarray:
    vis = camera_strip_from_obs(obs).copy()
    if recording:
        status = "REC"
        help_text = "S save | D drop | I pose | Q quit"
    elif delete_prompt:
        status = "DEL"
        help_text = delete_prompt
    elif zero_force_active:
        status = "DRAG"
        help_text = "C start | P->init | I pose | Q quit"
    elif no_zero_force:
        status = "READY"
        help_text = "C start | P zero-force | I pose | Q quit"
    else:
        status = "IDLE"
        help_text = "P zero-force | I pose | E del ep | Q quit"
    text = f"{status} Ep:{episode_id}  t={t_elapsed:.1f}s  steps={n_steps}"
    cv2.putText(vis, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(
        vis,
        help_text,
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    try:
        for robot_idx, (pose, grip) in enumerate(
            tcp_pose_grip_from_obs(obs, n_robots)
        ):
            y0 = 84 + robot_idx * 44
            cv2.putText(
                vis,
                (
                    f"r{robot_idx} xyz "
                    f"{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}  "
                    f"grip {grip:.3f}"
                ),
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                vis,
                f"r{robot_idx} rot {pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f}",
                (10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
    except Exception:
        pass
    return vis


def camera_strip_from_obs(obs: dict) -> np.ndarray:
    """纯相机拼接图（无文字叠加），供网页预览。"""
    cam_keys = sorted(k for k in obs if k.startswith("camera") and k.endswith("_rgb"))
    if not cam_keys:
        raise RuntimeError("obs 中没有 camera*_rgb")
    frames = []
    for key in cam_keys:
        img = np.asarray(obs[key][-1])
        if img.dtype != np.uint8:
            if float(np.nanmax(img)) <= 1.5:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        frames.append(img)
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def drain_keys(key_source) -> list:
    if key_source is None:
        return []
    return list(key_source.get_press_events())


def is_drop_key(key_stroke) -> bool:
    return key_stroke in (Key.backspace, KeyCode(char="d"))


def key_char(key_stroke) -> str | None:
    char = getattr(key_stroke, "char", None)
    if isinstance(char, str) and len(char) == 1:
        return char.lower()
    return None


def parse_episode_id(text: str) -> int | None:
    stripped = text.strip()
    if not stripped.isdigit():
        return None
    return int(stripped)


def _compute_mode(
    *,
    recording: bool,
    zero_force_active: bool,
    no_zero_force: bool,
    delete_mode,
) -> str:
    if delete_mode is not None:
        return "DEL"
    if recording:
        return "REC"
    if zero_force_active:
        return "DRAG"
    if no_zero_force:
        return "READY"
    return "IDLE"


def publish_web_frame(
    web_state,
    *,
    env,
    vis_rgb: np.ndarray | None,
    recording: bool,
    zero_force_active: bool,
    no_zero_force: bool,
    delete_mode,
    delete_target,
    n_steps: int,
    t_elapsed: float,
    episode_id: int | None,
    frequency: float,
    message: str = "",
    clean_rgb: np.ndarray | None = None,
) -> None:
    if web_state is None:
        return
    from modules.collect_web_ui import publish_pose_from_env, vis_rgb_to_jpeg

    mode = _compute_mode(
        recording=recording,
        zero_force_active=zero_force_active,
        no_zero_force=no_zero_force,
        delete_mode=delete_mode,
    )
    publish_pose_from_env(web_state, env)
    jpeg = b""
    # 网页预览用无叠加文字的干净画面；OpenCV 窗仍可用 vis_rgb
    src = clean_rgb if clean_rgb is not None else vis_rgb
    if src is not None:
        try:
            jpeg = vis_rgb_to_jpeg(src, quality=75)
        except Exception:
            jpeg = b""
    web_state.update(
        mode=mode,
        zero_force_active=zero_force_active,
        recording=recording,
        no_zero_force=no_zero_force,
        n_steps=int(n_steps),
        t_elapsed=float(t_elapsed),
        episode_id=episode_id,
        delete_pending=delete_target if delete_mode == "confirm" else None,
        frequency=float(frequency),
        message=message,
        jpeg=jpeg,
        n_episodes=int(env.replay_buffer.n_episodes),
    )


def append_live_from_obs(web_state, obs: dict, *, step: int, t_elapsed: float) -> None:
    if web_state is None:
        return
    # 降采样：约每 3 步采一点
    if step > 0 and step % 3 != 0:
        return
    try:
        xyz = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
        rot = np.asarray(
            obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64
        ).reshape(3)
        grip = float(np.asarray(obs["robot0_gripper_width"][-1]).squeeze())
        action = action_from_obs(obs, 1)
        web_state.append_live_sample(
            step=step, t=t_elapsed, xyz=xyz, rot=rot, grip=grip, action=action
        )
    except Exception:
        pass


class OfflineDatasetEnv:
    """仅打开已有 replay_buffer.zarr + videos，不启臂 / 相机。"""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.video_dir = self.output_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = self.output_dir / "replay_buffer.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(
                f"未找到 {zarr_path}。请先采数，或检查 -o 输出目录。"
            )
        # r+：允许网页删除 episode
        self.replay_buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="r+")


def resolve_dataset_dir(path: str | Path) -> Path:
    """解析数据目录：绝对路径原样；相对路径相对仓库根（cwd）。"""
    p = Path(str(path).strip()).expanduser()
    if not p.is_absolute():
        p = (ROOT_DIR / p).resolve()
    else:
        p = p.resolve()
    return p


def open_offline_dataset(path: str | Path) -> OfflineDatasetEnv:
    """打开含 replay_buffer.zarr 的数据目录；失败抛 FileNotFoundError / OSError。"""
    out = resolve_dataset_dir(path)
    if not out.is_dir():
        raise FileNotFoundError(f"目录不存在: {out}")
    env = OfflineDatasetEnv(out)
    try:
        check_episode_alignment(env)
    except RuntimeError as exc:
        print(f"[dataset] alignment warning ({out}): {exc}")
    return env


def apply_manage_dataset(web_state, env: OfflineDatasetEnv, *, message: str = "") -> None:
    """把数据管理页切到给定 OfflineDatasetEnv。"""
    web_state.env = env
    web_state.output_dir = Path(env.output_dir)
    web_state.viewer_only = True
    if hasattr(web_state, "clear_live_series"):
        web_state.clear_live_series()
    n_ep = int(env.replay_buffer.n_episodes)
    web_state.update(
        mode="VIEW",
        viewer_only=True,
        recording=False,
        zero_force_active=False,
        delete_pending=None,
        n_episodes=n_ep,
        n_steps=0,
        t_elapsed=0.0,
        message=message or f"已加载 {env.output_dir} · {n_ep} episodes",
    )


def handle_load_dataset_cmd(web_state, cmd: dict, *, default_output: str) -> bool:
    """
    处理 load_dataset 命令。成功返回 True。
    path 为空时回退到启动时的 -o 目录。
    """
    raw = cmd.get("path", "")
    if raw is None or str(raw).strip() == "":
        raw = default_output
    try:
        env = open_offline_dataset(raw)
    except (FileNotFoundError, OSError, ValueError) as exc:
        web_state.update(message=f"加载失败: {exc}")
        print(f"[dataset] load failed: {exc}")
        return False
    apply_manage_dataset(web_state, env)
    print(f"[dataset] loaded {env.output_dir} ({env.replay_buffer.n_episodes} eps)")
    return True


def run_viewer_only(
    *,
    output: str,
    web_host: str,
    web_port: int,
) -> None:
    """离线数据管理：网页回看 / 删除，不连接机械臂。"""
    from modules.collect_web_ui import (
        CommandQueue,
        SharedCollectState,
        start_collect_web_ui,
    )

    web_state = SharedCollectState()
    web_cmds = CommandQueue()
    web_state.viewer_only = True
    web_state.output_dir = resolve_dataset_dir(output)

    env = None
    try:
        env = open_offline_dataset(output)
        apply_manage_dataset(
            web_state, env, message="viewer_only: no robot · 可切换数据目录"
        )
    except (FileNotFoundError, OSError) as exc:
        print(f"[viewer_only] 默认目录未就绪: {exc}")
        web_state.env = None
        web_state.update(
            mode="VIEW",
            viewer_only=True,
            recording=False,
            zero_force_active=False,
            n_episodes=0,
            message=f"请在上方填写含 replay_buffer.zarr 的目录后点「加载」· {exc}",
        )

    print("=" * 60)
    print("ARX5 Dataset Viewer (no robot)")
    print(f"  output     : {web_state.output_dir}")
    print(
        f"  episodes   : "
        f"{env.replay_buffer.n_episodes if env is not None else 0}"
    )
    print(f"  web        : http://{web_host}:{web_port}/")
    print("  controls   : browse / scrub / delete / load other folder")
    print("  quit       : Ctrl+C in this terminal, or Q on the page")
    print("=" * 60)

    server = start_collect_web_ui(web_state, web_cmds, host=web_host, port=int(web_port))
    print(f"Open viewer: http://127.0.0.1:{web_port}/")

    try:
        while True:
            for cmd in web_cmds.drain():
                action = cmd.get("action")
                if action == "quit":
                    print("Quit viewer.")
                    return
                if action == "back_hub":
                    # --viewer_only 无 hub；返回视为退出
                    print("Quit viewer (back).")
                    return
                if action == "load_dataset":
                    handle_load_dataset_cmd(
                        web_state, cmd, default_output=output
                    )
                    continue
                if action == "delete_cancel":
                    web_state.update(delete_pending=None, message="delete cancelled")
                    continue
                if action == "delete_request":
                    env = web_state.env
                    if env is None:
                        web_state.update(message="尚未加载数据目录")
                        continue
                    eid = cmd.get("episode_id")
                    n = int(env.replay_buffer.n_episodes)
                    if eid is None or int(eid) < 0 or int(eid) >= n:
                        web_state.update(message=f"invalid episode_id {eid}")
                    else:
                        web_state.update(
                            delete_pending=int(eid),
                            message=f"confirm delete ep {eid}",
                        )
                    continue
                if action == "delete_confirm":
                    env = web_state.env
                    if env is None:
                        continue
                    eid = cmd.get("episode_id", web_state.delete_pending)
                    if eid is not None:
                        ok = delete_saved_episode(env, int(eid))
                        web_state.update(
                            delete_pending=None,
                            n_episodes=int(env.replay_buffer.n_episodes),
                            message=(
                                f"deleted ep {eid}"
                                if ok
                                else f"delete ep {eid} failed"
                            ),
                        )
                    continue
                if action in ("drag_toggle", "start", "save", "drop"):
                    web_state.update(
                        message="viewer_only: collect controls disabled"
                    )
            env = web_state.env
            web_state.update(
                mode="VIEW",
                viewer_only=True,
                n_episodes=(
                    int(env.replay_buffer.n_episodes) if env is not None else 0
                ),
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted — closing viewer.")
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _hub_reset_state(web_state, output: str, message: str = "") -> None:
    """把 SharedCollectState 切回 HUB 选择页。"""
    web_state.env = None
    web_state.viewer_only = False
    if hasattr(web_state, "clear_live_series"):
        web_state.clear_live_series()
    n_ep = 0
    zarr_path = Path(output) / "replay_buffer.zarr"
    if zarr_path.exists():
        try:
            rb = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
            n_ep = int(rb.n_episodes)
        except Exception:
            n_ep = 0
    web_state.update(
        mode="HUB",
        viewer_only=False,
        recording=False,
        zero_force_active=False,
        delete_pending=None,
        n_episodes=n_ep,
        n_steps=0,
        t_elapsed=0.0,
        message=message
        or "选择模式：数据采集（需已启动 ZMQ server）或 数据管理",
    )


def _web_log(web_state, line: str) -> None:
    """终端 + 网页日志（有 web_state 时同步到 status.logs）。"""
    print(line)
    if web_state is not None and hasattr(web_state, "append_log"):
        try:
            web_state.append_log(line)
        except Exception:
            pass


def run_web_hub(
    *,
    output: str,
    web_host: str,
    web_port: int,
    collect_kwargs: dict,
) -> None:
    """统一网页入口：先选数采 / 数据管理；数采结束后回到入口。"""
    from modules.collect_web_ui import (
        CommandQueue,
        SharedCollectState,
        start_collect_web_ui,
    )

    os.makedirs(output, exist_ok=True)
    web_state = SharedCollectState()
    web_cmds = CommandQueue()
    web_state.output_dir = Path(output)
    _hub_reset_state(web_state, output)

    print("=" * 60)
    print("ARX5 Web Hub")
    print(f"  output     : {output}")
    print(f"  web        : http://{web_host}:{web_port}/")
    print("  collect    : needs ZMQ robot server running")
    print("  manage     : offline viewer / delete / load other folder (no robot)")
    print("  quit       : Ctrl+C or Q on hub page")
    print("=" * 60)

    server = start_collect_web_ui(
        web_state, web_cmds, host=web_host, port=int(web_port)
    )
    print(f"Open hub: http://127.0.0.1:{web_port}/")
    phase = "hub"  # hub | manage

    def return_to_hub(message: str = "", *, keep_logs: bool = False) -> None:
        nonlocal phase
        if not keep_logs and hasattr(web_state, "clear_logs"):
            web_state.clear_logs()
        _hub_reset_state(web_state, output, message=message)
        phase = "hub"
        print(f"Back to hub: http://127.0.0.1:{web_port}/")

    try:
        while True:
            for cmd in web_cmds.drain():
                action = cmd.get("action")
                if action == "quit":
                    print("Quit hub.")
                    return

                if phase == "hub":
                    if action == "enter_manage":
                        if hasattr(web_state, "clear_logs"):
                            web_state.clear_logs()
                        try:
                            env = open_offline_dataset(output)
                            apply_manage_dataset(
                                web_state,
                                env,
                                message="数据管理：回看 / 删除（无机械臂）· 可切换目录",
                            )
                        except (FileNotFoundError, OSError) as exc:
                            web_state.env = None
                            web_state.output_dir = resolve_dataset_dir(output)
                            web_state.viewer_only = True
                            web_state.update(
                                mode="VIEW",
                                viewer_only=True,
                                recording=False,
                                zero_force_active=False,
                                n_episodes=0,
                                message=(
                                    "默认目录无数据，请填写其它含 "
                                    f"replay_buffer.zarr 的路径后点「加载」· {exc}"
                                ),
                            )
                            print(f"[hub] manage: default missing, UI ready: {exc}")
                        phase = "manage"
                        print("[hub] entered manage")
                        continue
                    if action == "enter_collect":
                        print("[hub] starting collect session…")
                        if hasattr(web_state, "clear_logs"):
                            web_state.clear_logs()
                        _web_log(web_state, "收到「数据采集」请求")
                        _web_log(
                            web_state,
                            f"目标 ZMQ {collect_kwargs.get('robot_ip')}:"
                            f"{collect_kwargs.get('robot_port')}",
                        )
                        _web_log(web_state, f"输出目录: {output}")
                        _web_log(web_state, "保持网页服务，开始连接机械臂 / 相机…")
                        web_state.update(
                            mode="CONNECTING",
                            viewer_only=False,
                            recording=False,
                            message="正在启动数采…",
                        )
                        # 让浏览器先刷到 CONNECTING + 日志
                        time.sleep(0.25)
                        collect_err = ""
                        try:
                            run_collect_session(
                                **collect_kwargs,
                                web_state=web_state,
                                web_cmds=web_cmds,
                                web_server=server,
                            )
                            _web_log(web_state, "数采会话结束")
                        except KeyboardInterrupt:
                            print("\nCollect interrupted — back to hub.")
                            _web_log(web_state, "数采被中断（KeyboardInterrupt）")
                        except Exception as exc:  # noqa: BLE001
                            collect_err = f"数采启动失败: {exc}"
                            print(f"[hub] {collect_err}")
                            _web_log(web_state, collect_err)
                            import traceback

                            traceback.print_exc()
                        return_to_hub(
                            message=collect_err or "已返回入口（数采会话结束）",
                            keep_logs=bool(collect_err),
                        )
                        continue
                    if action == "back_hub":
                        continue
                    # ignore collect cmds on hub
                    if action in (
                        "drag_toggle",
                        "start",
                        "save",
                        "drop",
                        "delete_request",
                        "delete_confirm",
                        "delete_cancel",
                        "load_dataset",
                    ):
                        web_state.update(message="请先选择模式")
                    continue

                # phase == manage
                if action == "back_hub":
                    web_state.env = None
                    web_state.viewer_only = False
                    return_to_hub(message="已返回入口")
                    print("[hub] back from manage")
                    continue
                if action == "load_dataset":
                    handle_load_dataset_cmd(
                        web_state, cmd, default_output=output
                    )
                    continue
                if action == "delete_cancel":
                    web_state.update(delete_pending=None, message="delete cancelled")
                    continue
                if action == "delete_request":
                    env = web_state.env
                    if env is None:
                        web_state.update(message="尚未加载数据目录")
                        continue
                    eid = cmd.get("episode_id")
                    n = int(env.replay_buffer.n_episodes)
                    if eid is None or int(eid) < 0 or int(eid) >= n:
                        web_state.update(message=f"invalid episode_id {eid}")
                    else:
                        web_state.update(
                            delete_pending=int(eid),
                            message=f"confirm delete ep {eid}",
                        )
                    continue
                if action == "delete_confirm":
                    env = web_state.env
                    if env is None:
                        continue
                    eid = cmd.get("episode_id", web_state.delete_pending)
                    if eid is not None:
                        ok = delete_saved_episode(env, int(eid))
                        web_state.update(
                            delete_pending=None,
                            n_episodes=int(env.replay_buffer.n_episodes),
                            message=(
                                f"deleted ep {eid}"
                                if ok
                                else f"delete ep {eid} failed"
                            ),
                        )
                    continue
                if action in ("drag_toggle", "start", "save", "drop", "enter_collect"):
                    web_state.update(
                        message="viewer: collect controls disabled（返回入口后再选数采）"
                    )
                    continue
                if action == "enter_manage":
                    continue

            if phase == "manage":
                env = web_state.env
                web_state.update(
                    mode="VIEW",
                    viewer_only=True,
                    n_episodes=(
                        int(env.replay_buffer.n_episodes) if env is not None else 0
                    ),
                )
            elif phase == "hub":
                # keep HUB mode sticky（CONNECTING 由 enter_collect 同步占用）
                if getattr(web_state, "mode", None) not in ("HUB", "CONNECTING"):
                    # 若仍处在数采会话，mode 会是 IDLE/DRAG/REC — 此时不在 hub 循环
                    pass
                elif getattr(web_state, "mode", None) == "HUB":
                    web_state.update(mode="HUB", viewer_only=False)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted — closing hub.")
    finally:
        if server is not None:
            server.stop()


def run_collect_session(
    *,
    output,
    robot_ip,
    robot_port,
    frequency,
    camera_reorder,
    obs_width,
    obs_height,
    camera_obs_horizon,
    robot_obs_horizon,
    camera_obs_latency,
    no_mirror,
    mirror_swap,
    max_duration,
    terminal_keys,
    save_images,
    init_pose,
    no_init_pose,
    init_delta,
    init_gripper_delta,
    init_gripper,
    keep_gripper,
    gripper_min_width,
    gripper_max_width,
    init_duration,
    no_zero_force,
    quit_home_duration,
    web,
    web_host,
    web_port,
    no_cv,
    web_state=None,
    web_cmds=None,
    web_server=None,
) -> None:
    """连接机械臂 + 相机，跑一轮数采会话；Q / 网页 quit 后返回（不退出进程）。

    若传入 web_state/web_cmds/web_server（hub 复用），则不另起端口，并在连接阶段写 logs。
    """
    # 仅非 web、非 terminal_keys 时才需要 pynput；--web 用网页键，避免本机双触发
    KeystrokeCounter = None
    if not terminal_keys and not web:
        from peripherals.keystroke_counter import KeystrokeCounter as _KC

        KeystrokeCounter = _KC

    # --web 时默认不弹 OpenCV（无显示器 / SSH 更稳）；显式不加 --no_cv 且未 --web 则仍 imshow
    if web and not no_cv:
        # click 无法区分“用户显式传了 --no_cv=False”；约定：开 web 默认关 cv
        no_cv = True

    os.makedirs(output, exist_ok=True)
    dt = 1.0 / frequency
    robots_config = _default_robots_config(robot_ip=robot_ip, robot_port=robot_port)
    n_robots = len(robots_config)
    camera_reorder_list = [int(x) for x in camera_reorder.split(",") if x.strip() != ""]
    abs_init_pose6 = (
        None if (no_init_pose or init_pose is None) else parse_pose6(init_pose)
    )
    gripper_target: float | None
    if keep_gripper:
        gripper_target = None
    elif abs_init_pose6 is not None and init_gripper is not None:
        gripper_target = float(init_gripper)
    else:
        gripper_target = None  # delta 路径里再算；绝对路径可 keep

    print("=" * 60)
    print("ARX5 Drag Data Collection (ZMQ)")
    print(f"  output     : {output}")
    print(f"  zmq        : {robot_ip}:{robot_port}")
    print(f"  frequency  : {frequency} Hz")
    print(f"  n_robots   : {n_robots}")
    print(f"  save_images: {save_images}")
    print("  frame      : eef_link + ee2tcp (ZMQ)")
    print("  gripper    : physical opening (larger = more open)")
    if no_init_pose:
        print("  init       : (skip)")
    elif abs_init_pose6 is not None:
        print(
            f"  init_pose  : {np.array2string(abs_init_pose6, precision=4, separator=',')}"
        )
        print(
            f"  init_gripper: "
            f"{'keep current' if gripper_target is None else f'{gripper_target:.6f} m'}"
        )
    else:
        print(f"  init_delta : {init_delta}  (same as eval_arx5_fast_wam)")
        print(f"  init_gripper_delta: {init_gripper_delta:+.4f} m")
        print(f"  grip clip  : [{gripper_min_width:.4f}, {gripper_max_width:.4f}]")
    print(f"  init_duration: {init_duration:.1f}s")
    print(f"  no_zero_force: {no_zero_force}")
    if no_zero_force:
        print("  startup: reset_to_home → optional init → C to record (P optional)")
    else:
        print("  startup: reset_to_home → optional init → P then C")
    print(
        "  P=toggle zero-force (again -> init)  "
        "C=start  S=save  D=drop  E=del  I=pose  Q=quit(no home)"
    )
    if web:
        print(f"  web        : http://{web_host}:{web_port}/  (no_cv={no_cv})")
    print("=" * 60)

    own_web_server = False
    if web:
        from modules.collect_web_ui import (
            CommandQueue,
            SharedCollectState,
            start_collect_web_ui,
        )

        if web_state is None or web_cmds is None:
            web_state = SharedCollectState()
            web_cmds = CommandQueue()
            web_server = None
            own_web_server = True
        web_state.output_dir = Path(output)
        web_state.viewer_only = False
        web_state.update(
            frequency=float(frequency),
            no_zero_force=bool(no_zero_force),
            mode="CONNECTING",
            viewer_only=False,
            message="正在连接机械臂与相机…",
        )
        _web_log(web_state, f"配置: {frequency} Hz · cameras={camera_reorder}")
        _web_log(web_state, f"ZMQ 地址: {robot_ip}:{robot_port}")
        _web_log(web_state, "正在创建 Arx5Env（连接机械臂 + 相机）…")

    def _stop_web_if_owned() -> None:
        nonlocal web_server
        if own_web_server and web_server is not None:
            try:
                web_server.stop()
            except Exception:
                pass
            web_server = None

    # --web：只用网页快捷键，不启 pynput（本机开显示器时按 P 会网页+pynput 双触发，
    # 进拖拽又立刻退出）。SSH 终端键请加 --terminal_keys。
    if terminal_keys:
        key_ctx = contextlib.nullcontext(None)
        term_ctx = TerminalKeyReader()
        key_source_is_term = True
    elif web:
        key_ctx = contextlib.nullcontext(None)
        term_ctx = contextlib.nullcontext(None)
        key_source_is_term = False
    else:
        if KeystrokeCounter is None:
            raise RuntimeError(
                "无 --web / --terminal_keys 时需要 pynput 键盘钩子；"
                "请加 --web 或 --terminal_keys"
            )
        key_ctx = KeystrokeCounter()
        term_ctx = contextlib.nullcontext(None)
        key_source_is_term = False

    try:
        with SharedMemoryManager() as shm_manager:
            with key_ctx as key_counter, term_ctx as term_keys, Arx5EnvWithImages(
                output_dir=output,
                robots_config=robots_config,
                frequency=frequency,
                obs_image_resolution=(obs_width, obs_height),
                obs_float32=True,
                camera_reorder=camera_reorder_list,
                init_joints=False,
                enable_multi_cam_vis=False,
                camera_obs_latency=camera_obs_latency,
                camera_obs_horizon=camera_obs_horizon,
                robot_obs_horizon=robot_obs_horizon,
                no_mirror=no_mirror,
                mirror_swap=mirror_swap,
                max_pos_speed=4.0,
                max_rot_speed=10.0,
                shm_manager=shm_manager,
            ) as env:
                key_source = term_keys if key_source_is_term else key_counter

                cv2.setNumThreads(2)
                _web_log(web_state, "等待相机 / env ready…")
                time.sleep(1.0)
                wait_t0 = time.monotonic()
                last_wait_log = 0.0
                while not env.is_ready:
                    if web_cmds is not None:
                        for cmd in web_cmds.drain():
                            if cmd.get("action") == "quit":
                                _web_log(web_state, "用户取消连接")
                                raise RuntimeError("用户取消连接")
                    now = time.monotonic()
                    if now - last_wait_log >= 2.0:
                        _web_log(
                            web_state,
                            f"仍在等待 env ready… ({now - wait_t0:.0f}s)"
                            " — 请确认 ZMQ server / 相机已启动",
                        )
                        last_wait_log = now
                    time.sleep(0.1)
                check_episode_alignment(env)
                _web_log(web_state, "Env ready · 开始启动微调")

                if web and web_state is not None and web_cmds is not None:
                    web_state.env = env
                    web_state.output_dir = Path(output)
                    if web_server is None:
                        web_server = start_collect_web_ui(
                            web_state, web_cmds, host=web_host, port=int(web_port)
                        )
                        own_web_server = True
                        print(
                            f"Open collect UI: http://127.0.0.1:{web_port}/ "
                            f"(or host IP if remote)"
                        )
                    else:
                        _web_log(web_state, "复用 hub 网页服务，进入数采界面")
                    web_state.update(
                        mode="IDLE",
                        viewer_only=False,
                        n_episodes=int(env.replay_buffer.n_episodes),
                        message="数采就绪",
                    )

                # 与 fast_wam 对齐的启动微调；结果记为 init_pose6 / home（Q 退出回此处）
                init_pose6: np.ndarray | None = None
                if no_init_pose:
                    _web_log(web_state, "跳过 init 微调")
                    if no_zero_force:
                        print(
                            "No init fine-tune. Direct record mode: press C to start, "
                            "I to print pose string."
                        )
                    else:
                        print(
                            "No init fine-tune. Press P to zero-force, drag, "
                            "press I to print pose string."
                        )
                elif abs_init_pose6 is not None:
                    _web_log(web_state, "移动到绝对 init_pose…")
                    move_to_init_pose(
                        env,
                        abs_init_pose6,
                        frequency=frequency,
                        duration=init_duration,
                        init_gripper=gripper_target,
                    )
                    init_pose6 = abs_init_pose6.copy()
                    if gripper_target is None and not keep_gripper:
                        gripper_target = float(env.get_robot_state()[0]["gripper_position"])
                else:
                    _web_log(web_state, f"按 init_delta 微调（{init_duration:.1f}s）…")
                    init_pose6, grip_after = apply_init_delta_like_fast_wam(
                        env,
                        init_delta=init_delta,
                        init_gripper_delta=float(init_gripper_delta),
                        frequency=frequency,
                        duration=init_duration,
                        gripper_min_width=float(gripper_min_width),
                        gripper_max_width=float(gripper_max_width),
                    )
                    if not keep_gripper:
                        gripper_target = grip_after

                # Q 回的是微调后的位姿，不是 reset 后的裸 home（避免大姿态跳变）
                home_targets = capture_robot_pose_targets(env)
                _web_log(web_state, "启动完成 · 可开始采集（P/C）")
                print("Quit-home pose captured (post-init; used when pressing Q).")

                zero_force_active = False
                pose_print_enabled = False
                last_pose_print_t = 0.0
                pose_print_interval = 0.5
                delete_mode = None  # None | "input" | "confirm"
                delete_buf = ""
                delete_target = None

                if no_zero_force:
                    print("Direct record mode: press C to start (P optional for drag).")

                def handle_idle_web_cmds() -> tuple[bool, bool]:
                    """Returns (start_episode, should_quit)."""
                    nonlocal zero_force_active, delete_mode, delete_buf, delete_target
                    nonlocal pose_print_enabled, last_pose_print_t
                    if web_cmds is None:
                        return False, False
                    start = False
                    for cmd in web_cmds.drain():
                        action = cmd.get("action")
                        if action == "quit":
                            return False, True
                        if action == "delete_cancel":
                            print("Delete cancelled.")
                            delete_mode = None
                            delete_buf = ""
                            delete_target = None
                            continue
                        if action == "delete_confirm":
                            tid = cmd.get("episode_id", delete_target)
                            if tid is not None:
                                delete_saved_episode(env, int(tid))
                            delete_mode = None
                            delete_buf = ""
                            delete_target = None
                            continue
                        if action == "delete_request":
                            n_saved = env.replay_buffer.n_episodes
                            eid = cmd.get("episode_id")
                            if eid is None:
                                print("delete_request missing episode_id")
                            else:
                                eid = int(eid)
                                if 0 <= eid < n_saved:
                                    delete_target = eid
                                    delete_mode = "confirm"
                                    print(
                                        f"Confirm delete saved episode {eid}? "
                                        "(web Yes / keyboard Y)"
                                    )
                                else:
                                    print(f"Episode {eid} does not exist.")
                            continue
                        if delete_mode == "confirm" and action not in (
                            "delete_confirm",
                            "delete_cancel",
                        ):
                            continue
                        if action == "drag_toggle":
                            if not zero_force_active:
                                env.set_gravity_compensation()
                                zero_force_active = True
                                print(
                                    "Drag mode ON (gravity if supported, else damping). "
                                    "Drag freely."
                                )
                            else:
                                leave_zero_force_to_init(
                                    env,
                                    init_pose6,
                                    frequency=frequency,
                                    duration=init_duration,
                                    init_gripper=gripper_target,
                                )
                                zero_force_active = False
                        elif action == "start":
                            if zero_force_active or no_zero_force:
                                start = True
                            else:
                                print("Press P / Drag first to enter zero-force mode.")
                        elif action == "save":
                            print("Not recording — nothing to save.")
                        elif action == "drop":
                            print("No active episode to drop.")
                    return start, False

                while True:
                    # ---------- idle: optional P -> zero-force, then C -> start ----------
                    start_episode = False
                    while True:
                        web_start, web_quit = handle_idle_web_cmds()
                        if web_quit:
                            print("Quit.")
                            leave_drag_before_quit(
                                env,
                                zero_force_active=zero_force_active,
                            )
                            _stop_web_if_owned()
                            return
                        if web_start:
                            start_episode = True

                        for key_stroke in drain_keys(key_source):
                            if key_stroke == KeyCode(char="q"):
                                print("Quit.")
                                leave_drag_before_quit(
                                    env,
                                    zero_force_active=zero_force_active,
                                )
                                _stop_web_if_owned()
                                return

                            if delete_mode == "confirm":
                                ch = key_char(key_stroke)
                                if ch == "y":
                                    delete_saved_episode(env, delete_target)
                                    delete_mode = None
                                    delete_buf = ""
                                    delete_target = None
                                elif ch == "n" or key_stroke == Key.esc:
                                    print("Delete cancelled.")
                                    delete_mode = None
                                    delete_buf = ""
                                    delete_target = None
                                continue

                            if delete_mode == "input":
                                ch = key_char(key_stroke)
                                if key_stroke == Key.esc:
                                    print("Delete cancelled.")
                                    delete_mode = None
                                    delete_buf = ""
                                elif key_stroke == Key.backspace:
                                    delete_buf = delete_buf[:-1]
                                elif ch is not None and ch.isdigit():
                                    delete_buf += ch
                                elif key_stroke == Key.enter:
                                    n_saved = env.replay_buffer.n_episodes
                                    episode_id = parse_episode_id(delete_buf)
                                    if episode_id is None:
                                        print("Type an episode ID first (digits).")
                                    elif episode_id < 0 or episode_id >= n_saved:
                                        print(
                                            f"Episode {episode_id} does not exist "
                                            f"(valid: 0..{n_saved - 1})."
                                        )
                                    else:
                                        delete_target = episode_id
                                        delete_mode = "confirm"
                                        print(
                                            f"Confirm delete saved episode {episode_id}? "
                                            "Y=yes  N=no"
                                        )
                                continue

                            if key_stroke == KeyCode(char="p"):
                                if not zero_force_active:
                                    env.set_gravity_compensation()
                                    zero_force_active = True
                                    print(
                                        "Drag mode ON (gravity if supported, else damping). "
                                        "Drag freely. Press P again to exit and return to "
                                        "init, or C to record."
                                    )
                                else:
                                    leave_zero_force_to_init(
                                        env,
                                        init_pose6,
                                        frequency=frequency,
                                        duration=init_duration,
                                        init_gripper=gripper_target,
                                    )
                                    zero_force_active = False
                            if key_stroke == KeyCode(char="c"):
                                if zero_force_active or no_zero_force:
                                    start_episode = True
                                else:
                                    print("Press P first to enter zero-force mode.")
                            if key_stroke == KeyCode(char="s"):
                                print("Not recording — nothing to save.")
                            if is_drop_key(key_stroke):
                                print("No active episode to drop. Press E to delete a saved one.")
                            if key_stroke == KeyCode(char="i"):
                                pose_print_enabled = not pose_print_enabled
                                if pose_print_enabled:
                                    last_pose_print_t = 0.0
                                    print("Pose print ON (press I again to stop).")
                                    try:
                                        print_current_pose(env, tag="live")
                                    except Exception as exc:  # noqa: BLE001
                                        print(f"Failed to read pose: {exc}")
                                else:
                                    print("Pose print OFF.")
                            if key_stroke == KeyCode(char="e"):
                                n_saved = env.replay_buffer.n_episodes
                                if n_saved <= 0:
                                    print("No saved episode to delete.")
                                else:
                                    delete_mode = "input"
                                    delete_buf = ""
                                    delete_target = None
                                    print(
                                        f"Delete saved episode: type ID (0..{n_saved - 1}), "
                                        "Enter to confirm, Esc to cancel."
                                    )
                        if start_episode:
                            break

                        try:
                            obs = env.get_obs()
                            if pose_print_enabled:
                                now = time.monotonic()
                                if now - last_pose_print_t >= pose_print_interval:
                                    print_current_pose(env, tag="live")
                                    last_pose_print_t = now
                            n_saved = env.replay_buffer.n_episodes
                            delete_prompt = None
                            if delete_mode == "input":
                                delete_prompt = (
                                    f"ID:{delete_buf or '_'}  (0..{max(n_saved - 1, 0)}) "
                                    "Enter=ok Esc=cancel"
                                )
                            elif delete_mode == "confirm":
                                delete_prompt = (
                                    f"Delete Ep {delete_target}?  Y=yes N=no"
                                )
                            vis = make_vis_image(
                                obs,
                                n_saved,
                                0.0,
                                0,
                                zero_force_active=zero_force_active,
                                no_zero_force=no_zero_force,
                                recording=False,
                                delete_prompt=delete_prompt,
                                n_robots=n_robots,
                            )
                            if not no_cv:
                                cv2.imshow("drag_collect_zmq", vis[..., ::-1])
                                cv2.pollKey()
                            publish_web_frame(
                                web_state,
                                env=env,
                                vis_rgb=vis,
                                clean_rgb=camera_strip_from_obs(obs),
                                recording=False,
                                zero_force_active=zero_force_active,
                                no_zero_force=no_zero_force,
                                delete_mode=delete_mode,
                                delete_target=delete_target,
                                n_steps=0,
                                t_elapsed=0.0,
                                episode_id=None,
                                frequency=frequency,
                            )
                        except Exception:
                            pass
                        time.sleep(0.05)

                    # ---------- recording（零力可选；开录后不再重复 set_gravity） ----------
                    episode_id = env.replay_buffer.n_episodes
                    start_delay = 0.5
                    eval_t_start = time.time() + start_delay
                    print(f"\n>>> Episode {episode_id} START")
                    if web_state is not None:
                        web_state.clear_live_series()
                    env.start_episode(eval_t_start)
                    if not save_images:
                        env.image_accumulator = None

                    precise_wait(eval_t_start, time_func=time.time)

                    loop_t0 = time.monotonic()
                    last_tick = loop_t0
                    n_steps = 0
                    stop_save = False
                    stop_drop = False

                    try:
                        while True:
                            now = time.monotonic()
                            t_elapsed = now - loop_t0
                            t_cycle_end = last_tick + dt

                            if web_cmds is not None:
                                for cmd in web_cmds.drain():
                                    action = cmd.get("action")
                                    if action == "save":
                                        stop_save = True
                                        if web_state is not None:
                                            web_state.update(
                                                mode="SAVING",
                                                recording=True,
                                                message=f"正在保存 episode {episode_id}…请稍候",
                                            )
                                    elif action == "drop":
                                        stop_drop = True
                                        if web_state is not None:
                                            web_state.update(
                                                mode="SAVING",
                                                message="正在丢弃当前 episode…",
                                            )
                                    elif action == "quit":
                                        print(
                                            "Quit requested — abort current episode, "
                                            "stay at pose."
                                        )
                                        abort_episode(env)
                                        leave_drag_before_quit(
                                            env,
                                            zero_force_active=zero_force_active,
                                        )
                                        _stop_web_if_owned()
                                        return
                                    elif action in ("start", "drag_toggle"):
                                        print("Still recording — save/drop first.")
                                    elif action and action.startswith("delete"):
                                        print(
                                            "Still recording — save/drop first, then delete."
                                        )

                            for key_stroke in drain_keys(key_source):
                                if key_stroke == KeyCode(char="s"):
                                    stop_save = True
                                    if web_state is not None:
                                        web_state.update(
                                            mode="SAVING",
                                            recording=True,
                                            message=f"正在保存 episode {episode_id}…请稍候",
                                        )
                                elif is_drop_key(key_stroke):
                                    stop_drop = True
                                    if web_state is not None:
                                        web_state.update(
                                            mode="SAVING",
                                            message="正在丢弃当前 episode…",
                                        )
                                elif key_stroke == KeyCode(char="e"):
                                    print("Still recording — save/drop first, then press E.")
                                elif key_stroke == KeyCode(char="i"):
                                    pose_print_enabled = not pose_print_enabled
                                    if pose_print_enabled:
                                        last_pose_print_t = 0.0
                                        print("Pose print ON (press I again to stop).")
                                        print_current_pose(env, tag="rec")
                                    else:
                                        print("Pose print OFF.")
                                elif key_stroke == KeyCode(char="q"):
                                    print("Quit requested — abort current episode, stay at pose.")
                                    abort_episode(env)
                                    leave_drag_before_quit(
                                        env,
                                        zero_force_active=zero_force_active,
                                    )
                                    _stop_web_if_owned()
                                    return

                            obs = env.get_obs()
                            if pose_print_enabled:
                                if now - last_pose_print_t >= pose_print_interval:
                                    print_current_pose(env, tag="rec")
                                    last_pose_print_t = now
                            record_pose_action(env, obs)
                            n_steps += 1
                            append_live_from_obs(
                                web_state, obs, step=n_steps, t_elapsed=t_elapsed
                            )

                            vis = make_vis_image(
                                obs,
                                episode_id,
                                t_elapsed,
                                n_steps,
                                zero_force_active=zero_force_active,
                                no_zero_force=no_zero_force,
                                recording=True,
                                n_robots=n_robots,
                            )
                            if not no_cv:
                                cv2.imshow("drag_collect_zmq", vis[..., ::-1])
                                cv2.pollKey()
                            publish_web_frame(
                                web_state,
                                env=env,
                                vis_rgb=vis,
                                clean_rgb=camera_strip_from_obs(obs),
                                recording=True,
                                zero_force_active=zero_force_active,
                                no_zero_force=no_zero_force,
                                delete_mode=None,
                                delete_target=None,
                                n_steps=n_steps,
                                t_elapsed=t_elapsed,
                                episode_id=episode_id,
                                frequency=frequency,
                            )

                            if t_elapsed > max_duration:
                                print(f"Max duration {max_duration}s reached — auto save.")
                                stop_save = True
                                if web_state is not None:
                                    web_state.update(
                                        mode="SAVING",
                                        recording=True,
                                        message=f"超时自动保存 episode {episode_id}…",
                                    )

                            if stop_drop:
                                if web_state is not None:
                                    web_state.update(
                                        mode="SAVING",
                                        message="正在丢弃当前 episode…",
                                    )
                                abort_episode(env)
                                if web_state is not None:
                                    web_state.clear_live_series()
                                    web_state.update(
                                        mode="READY" if no_zero_force else (
                                            "DRAG" if zero_force_active else "IDLE"
                                        ),
                                        recording=False,
                                        message="已丢弃，可继续采集",
                                    )
                                # 丢弃后仍保持零力，可直接再按 C 录下一条
                                break

                            if stop_save:
                                if web_state is not None:
                                    web_state.update(
                                        mode="SAVING",
                                        recording=True,
                                        message=f"正在保存 episode {episode_id}…写盘中，请稍候",
                                    )
                                ok = save_episode(env, episode_id)
                                if ok:
                                    print(
                                        f"Episode {episode_id} saved "
                                        f"({n_steps} loop steps @ {frequency} Hz). "
                                        "Press E to delete a saved episode."
                                    )
                                if web_state is not None:
                                    web_state.clear_live_series()
                                    web_state.update(
                                        mode="READY" if no_zero_force else (
                                            "DRAG" if zero_force_active else "IDLE"
                                        ),
                                        recording=False,
                                        n_episodes=int(env.replay_buffer.n_episodes),
                                        n_steps=0,
                                        t_elapsed=0.0,
                                        message=(
                                            f"已保存 episode {episode_id}"
                                            if ok
                                            else f"保存 episode {episode_id} 失败（无有效帧）"
                                        ),
                                    )
                                # 保存后仍保持零力，可直接再按 C 录下一条
                                break

                            precise_wait(t_cycle_end, time_func=time.monotonic)
                            last_tick = t_cycle_end

                    except KeyboardInterrupt:
                        print("Interrupted — abort episode.")
                        abort_episode(env)
                        leave_drag_mode(env)
                        zero_force_active = False

                    if zero_force_active or no_zero_force:
                        print(
                            "Ready for next episode. Press C to start, "
                            "E to delete a saved episode, or Q to quit.\n"
                        )
                    else:
                        print(
                            "Ready. Press P for zero-force, "
                            "E to delete a saved episode, or C to start.\n"
                        )
    except Exception as exc:
        _web_log(web_state, f"连接/启动异常: {exc}")
        _stop_web_if_owned()
        raise
    finally:
        # hub 复用时不关 server；仅 --collect_direct 自建的才停
        pass


@click.command()
@click.option(
    "--output",
    "-o",
    default="data/collect_demo_zmq",
    show_default=True,
    help="输出目录（replay_buffer.zarr + videos）",
)
@click.option(
    "--robot_ip",
    default="127.0.0.1",
    show_default=True,
    help="ZMQ 机械臂服务 IP",
)
@click.option(
    "--robot_port",
    default=8765,
    type=int,
    show_default=True,
    help="ZMQ 机械臂服务端口",
)
@click.option(
    "--frequency",
    "-f",
    default=60.0,
    type=float,
    help="采数频率 Hz（默认 60；对齐用同一 dt）",
)
@click.option("--camera_reorder", "-cr", default="0", help="相机顺序，如 0 或 0,1")
@click.option("--obs_width", default=224, type=int)
@click.option("--obs_height", default=224, type=int)
@click.option("--camera_obs_horizon", default=2, type=int)
@click.option("--robot_obs_horizon", default=2, type=int)
@click.option("--camera_obs_latency", default=0.17, type=float)
@click.option("-nm", "--no_mirror", is_flag=True, default=False)
@click.option("--mirror_swap", is_flag=True, default=False)
@click.option(
    "--max_duration",
    "-md",
    default=120.0,
    type=float,
    help="单条 episode 最长秒数，超时自动保存",
)
@click.option(
    "--terminal_keys",
    is_flag=True,
    default=False,
    help="用终端 stdin 读键（SSH / 无 X11）；与 --web 联用时网页键与终端键都可用",
)
@click.option(
    "--save_images/--no_save_images",
    default=True,
    help="是否把 RGB 写入 zarr（默认开；视频始终保存）",
)
@click.option(
    "--init_pose",
    default=None,
    type=str,
    help="绝对 TCP 位姿 x,y,z,rx,ry,rz（若指定则覆盖 --init_delta）",
)
@click.option(
    "--no_init_pose",
    is_flag=True,
    default=False,
    help="启动时不做微调（保持 reset_to_home 后的位姿）",
)
@click.option(
    "--init_delta",
    default="0,0,0,0.1,-0.1,-1.0",
    show_default=True,
    type=str,
    help="与 eval_arx5_fast_wam 相同: 相对 home 的 dx,dy,dz,drx,dry,drz",
)
@click.option(
    "--init_gripper_delta",
    default=0.11,
    show_default=True,
    type=float,
    help="与 eval_arx5_fast_wam 相同: 夹爪相对 home 增量（米，ZMQ 物理开口）",
)
@click.option(
    "--init_gripper",
    default=None,
    type=float,
    help="绝对夹爪开口（米）；仅在与 --init_pose 一起用时生效",
)
@click.option(
    "--keep_gripper",
    is_flag=True,
    default=False,
    help="启动/回 init 时不改夹爪，保持当前开度",
)
@click.option(
    "--gripper_min_width",
    default=0.011,
    show_default=True,
    type=float,
    help="ZMQ 物理开口下界（米）",
)
@click.option(
    "--gripper_max_width",
    default=0.085,
    show_default=True,
    type=float,
    help="ZMQ 物理开口上界（米）",
)
@click.option(
    "--init_duration",
    default=2.5,
    type=float,
    show_default=True,
    help="启动微调时长（秒；与 fast_wam warmup_sec 一致）",
)
@click.option(
    "--no_zero_force",
    is_flag=True,
    default=True,
    help="不要求先进零力模式，可直接 C 开录 / S 保存（P 仍可选）",
)
@click.option(
    "--quit_home_duration",
    default=3.0,
    type=float,
    show_default=True,
    help="（已弃用）按 Q 不再回 home；保留参数仅兼容旧命令行",
)
@click.option(
    "--web",
    is_flag=True,
    default=False,
    help="启动网页入口（默认 hub：选择数采或数据管理）",
)
@click.option(
    "--web_host",
    default="0.0.0.0",
    show_default=True,
    help="网页绑定地址",
)
@click.option(
    "--web_port",
    default=8080,
    type=int,
    show_default=True,
    help="网页端口",
)
@click.option(
    "--no_cv",
    is_flag=True,
    default=False,
    help="不弹 OpenCV 窗口（--web 时默认开启）",
)
@click.option(
    "--collect_direct",
    is_flag=True,
    default=False,
    help="与 --web 联用时跳过 hub，直接连臂数采（旧行为）",
)
@click.option(
    "--viewer_only",
    is_flag=True,
    default=False,
    help="跳过 hub，直达数据管理（网页回看/删除，不启机械臂与相机）",
)
def main(
    output,
    robot_ip,
    robot_port,
    frequency,
    camera_reorder,
    obs_width,
    obs_height,
    camera_obs_horizon,
    robot_obs_horizon,
    camera_obs_latency,
    no_mirror,
    mirror_swap,
    max_duration,
    terminal_keys,
    save_images,
    init_pose,
    no_init_pose,
    init_delta,
    init_gripper_delta,
    init_gripper,
    keep_gripper,
    gripper_min_width,
    gripper_max_width,
    init_duration,
    no_zero_force,
    quit_home_duration,
    web,
    web_host,
    web_port,
    no_cv,
    viewer_only,
    collect_direct,
):
    if viewer_only:
        run_viewer_only(
            output=output,
            web_host=web_host,
            web_port=int(web_port),
        )
        return

    collect_kwargs = dict(
        output=output,
        robot_ip=robot_ip,
        robot_port=robot_port,
        frequency=frequency,
        camera_reorder=camera_reorder,
        obs_width=obs_width,
        obs_height=obs_height,
        camera_obs_horizon=camera_obs_horizon,
        robot_obs_horizon=robot_obs_horizon,
        camera_obs_latency=camera_obs_latency,
        no_mirror=no_mirror,
        mirror_swap=mirror_swap,
        max_duration=max_duration,
        terminal_keys=terminal_keys,
        save_images=save_images,
        init_pose=init_pose,
        no_init_pose=no_init_pose,
        init_delta=init_delta,
        init_gripper_delta=init_gripper_delta,
        init_gripper=init_gripper,
        keep_gripper=keep_gripper,
        gripper_min_width=gripper_min_width,
        gripper_max_width=gripper_max_width,
        init_duration=init_duration,
        no_zero_force=no_zero_force,
        quit_home_duration=quit_home_duration,
        web=web,
        web_host=web_host,
        web_port=web_port,
        no_cv=no_cv,
    )

    # --web 默认进 hub；--collect_direct 跳过选择页直连数采
    if web and not collect_direct:
        run_web_hub(
            output=output,
            web_host=web_host,
            web_port=int(web_port),
            collect_kwargs={**collect_kwargs, "web": True},
        )
        return

    run_collect_session(**collect_kwargs)



if __name__ == "__main__":
    main()
