"""
ARX5 双臂拖拽采数 — ZMQ + 双腕 GoPro（V4L）+ 中间 D435（pyrealsense2）。

硬件约定:
  robot0（左）  127.0.0.1:8765   例: can3
  robot1（右）  127.0.0.1:8766   例: can1
  camera0       左腕 GoPro（V4L / Elgato）
  camera1       中间 D435 RGB + depth（pyrealsense2）
  camera2       右腕 GoPro（V4L / Elgato）

---------------------------------------------------------------------------
D435 画面预处理（训练 / 推理必须一致）
---------------------------------------------------------------------------
作用对象: 仅 camera1（D435 RGB）。GoPro / depth 不做颜色处理。
写入范围: 预览、zarr 的 camera1_rgb、videos/*/1.mp4 均为调色后画面。

当前默认（可用 CLI 覆盖）:
  --rs_color_hue_shift     12.0   # 蓝→品红（OpenCV H 单位，0–180）
  --rs_color_temperature    0.0   # 关闭（不改冷暖）
  --rs_color_saturation     1.0   # 关闭（不改饱和度）
  --rs_color_contrast       1.0   # 关闭（不改对比度）

算法（与 peripherals.realsense_camera.grade_bgr_uint8 一致，输入 BGR uint8）:
  1) BGR → HSV（OpenCV）
  2) 仅对青蓝波段做加权色相偏移（不拖橙色夹爪）:
       center, half = 107.0, 28.0
       w = clip(1 - |H - center| / half, 0, 1)
       H <- (H + hue_shift * w) mod 180
  3) HSV → BGR
  4) temperature / saturation / contrast 仅在非默认时启用（当前默认全关）

推理脚本复用（推荐直接 import，勿手抄）:
  from peripherals.realsense_camera import grade_bgr_uint8
  bgr = grade_bgr_uint8(bgr, temperature=0.0, saturation=1.0,
                        contrast=1.0, hue_shift=12.0)
  # 若模型吃 RGB: rgb = bgr[..., ::-1]
---------------------------------------------------------------------------

依赖:
  - 两个已启动的 ZMQ 机械臂服务（eef_link + ee2tcp）
  - pip install pyrealsense2
  - 两路腕部相机在 /dev/v4l/by-id（脚本会过滤 RealSense 节点）

起服务示例:
    python python/communication/zmq_server_ac_one.py X5 can3 \\
        --gripper-open-readout -3.51 --gripper-width 0.085
    # 第二臂换端口（按你的 server 参数）:
    python python/communication/zmq_server_ac_one.py X5 can1 \\
        --gripper-open-readout -3.51 --gripper-width 0.085
    # 确保分别监听 8765 / 8766

用法:
 # 并确保: pip install pyrealsense2
    conda activate umi-arx  
    python scripts/collect_drag_umi_zmq_bimanual.py -o data/collect_bimanual --web

    # 拖拽手感默认 light（轻）；要防坠可 --drag_profile stable（与 DAgger 同档）
    python scripts/collect_drag_umi_zmq_bimanual.py \\
        -o /media/lenovo/data/kian/collect_bimanual_10_30 --web

    # 腕部 V4L 索引（过滤 RealSense 后的列表）；D435 固定为逻辑 camera1
    python scripts/collect_drag_umi_zmq_bimanual.py -o data/collect_bimanual \\
        --robot_ports 8765,8766 -cr 0,1 --web --collect_direct

    # 指定 D435 序列号 / 关 depth:
    python scripts/collect_drag_umi_zmq_bimanual.py -o data/collect_bimanual \\
        --rs_serial 123456789012 --no_save_depth --web

    # 关闭 D435 色相校正（存原图）:
    python scripts/collect_drag_umi_zmq_bimanual.py -o data/collect_bimanual \\
        --rs_color_hue_shift 0 --web

键盘: 与 collect_drag_umi_zmq.py 相同（P/C/S/D/E/I/Q），另加 R=回放已保存 episode。
拖拽手势: P 进零力后，连续两次开合 can1（默认 robot1/右臂）夹爪 → 开始采集；
          再连续两次开合 → 保存 episode（手势本身不写入 zarr）。
保存后（夹爪手势或键盘 S）: 缓动回 init，再重新进入拖拽模式。
左右臂共用同一组 --init_delta / --init_gripper_delta（各自相对 home）。

数据管理（网页 VIEW / --viewer_only / hub「数据管理」）:
  回看 scrub、删除 episode；可用「标起点 / 标终点 / 应用裁剪」自由裁切
  已保存 episode（改写 zarr + videos + 可选 intervention sidecar）。
  支持打开 LeRobot v2.1（meta/info.json）并原地裁剪 parquet/视频；
  zarr 页可「转换为 LeRobot」（默认旁路 {dir}_lerobot，带进度）。
  快捷键: [ = 标起点, ] = 标终点, T = 应用裁剪。
  若目录含 intervention/（DAgger），可一键拆出人工示教（去首尾静止）与纯模型
  子集到旁边的 {name}_human / {name}_policy（也可用 scripts/extract_dagger_split.py）。

输出:
  action (T, 14)；robot{0,1}_*；camera{0,1,2}_rgb；camera1_depth（uint16 mm）
  videos/{ep}/0.mp4, 1.mp4(D435 RGB 已调色), 2.mp4

Author: Kian
"""

from __future__ import annotations

import contextlib
import os
import select
import shutil
import sys
import termios
import threading
import time
import tty
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import click
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from modules.arx5_controller import apply_drag_profile  # noqa: E402
from modules.arx5_controller import get_drag_profile  # noqa: E402
from modules.arx5_env_with_images_bimanual_rs import Arx5EnvWithImages  # noqa: E402
from modules.replay_buffer import ReplayBuffer  # noqa: E402
from utils.interpolation_util import (  # noqa: E402
    continuize_action_rotvecs,
    nearest_rotvec_equivalent,
)
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
            "e=删已保存, r=整段回放, j=单步回放, "
            "0/1/2=回放臂(左/右/双), x=急停, i=位姿, q=退出"
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
        if ch in ("r", "R"):
            return KeyCode(char="r")
        if ch in ("j", "J"):
            return KeyCode(char="j")
        if ch in ("x", "X"):
            return KeyCode(char="x")
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
def _default_robots_config(
    robot_ip: str = "127.0.0.1",
    robot_ports: list[int] | None = None,
):
    """Dual-arm config: robot0=left, robot1=right."""
    if robot_ports is None:
        robot_ports = [8765, 8766]
    if len(robot_ports) != 2:
        raise ValueError(f"需要恰好 2 个端口（左,右），收到 {robot_ports}")
    cfg = []
    for port in robot_ports:
        cfg.append(
            {
                "robot_type": "arx5",
                "robot_ip": robot_ip,
                "robot_port": int(port),
                "robot_obs_latency": 0.1867,
                "robot_action_latency": 0.01,
                "tcp_offset": 0.0,
                "height_threshold": 0.0,
                "sphere_radius": 0.1,
                "sphere_center": [0, -0.06, -0.185],
            }
        )
    return cfg


def parse_robot_ports(text: str) -> list[int]:
    ports = [int(x.strip()) for x in str(text).split(",") if x.strip() != ""]
    if len(ports) != 2:
        raise click.BadParameter(
            f"--robot_ports 需要两个端口，如 8765,8766，收到 {ports}"
        )
    return ports


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


def move_to_tcp_pose_slerp(
    env: Arx5EnvWithImages,
    targets: list[tuple[np.ndarray, float | None]],
    *,
    frequency: float,
    duration: float = 2.5,
    label: str = "target",
    ensure_end_control: bool = False,
    poll_abort=None,
) -> bool:
    """缓动到目标：xyz/夹爪线性，姿态用 Slerp。返回 False 表示被急停打断。"""
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
        tgt6 = np.asarray(targets[ridx][0], dtype=np.float64).reshape(6).copy()
        # 把目标 rotvec 锚到当前表示，减少 as_rotvec 翻转造成的数值跳变
        tgt6[3:6] = nearest_rotvec_equivalent(tgt6[3:6], curr6[3:6])
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

    print(f"Moving to {label} (Slerp rotation)...")
    for ridx in range(n_robots):
        geo = float(
            (
                Rotation.from_rotvec(start_pose_all[ridx, 3:])
                .inv()
                * Rotation.from_rotvec(target_pose_all[ridx, 3:])
            ).magnitude()
        )
        print(
            f"  robot{ridx}  from "
            f"{np.array2string(start_pose_all[ridx], precision=4, separator=',')} "
            f"grip={start_grip_all[ridx]:.4f}"
        )
        print(
            f"  robot{ridx}    to "
            f"{np.array2string(target_pose_all[ridx], precision=4, separator=',')} "
            f"grip={target_grip_all[ridx]:.4f}  geo={geo:.3f}rad"
        )

    dt = 1.0 / float(frequency)
    steps = max(2, int(float(duration) * float(frequency)))
    blends = np.array(
        [ease_in_out_quad(k / (steps - 1)) for k in range(steps)], dtype=np.float64
    )
    rot_interps = []
    for ridx in range(n_robots):
        key = Rotation.from_rotvec(
            np.stack([start_pose_all[ridx, 3:], target_pose_all[ridx, 3:]], axis=0)
        )
        rot_interps.append(Slerp([0.0, 1.0], key))

    actions = np.zeros((steps, 7 * n_robots), dtype=np.float64)
    for k, a in enumerate(blends):
        for ridx in range(n_robots):
            actions[k, 7 * ridx : 7 * ridx + 3] = start_pose_all[ridx, :3] + a * (
                target_pose_all[ridx, :3] - start_pose_all[ridx, :3]
            )
            rv = np.asarray(rot_interps[ridx]([a])[0].as_rotvec(), dtype=np.float64)
            if k > 0:
                rv = nearest_rotvec_equivalent(
                    rv, actions[k - 1, 7 * ridx + 3 : 7 * ridx + 6]
                )
            else:
                rv = nearest_rotvec_equivalent(rv, start_pose_all[ridx, 3:])
            actions[k, 7 * ridx + 3 : 7 * ridx + 6] = rv
            actions[k, 7 * ridx + 6] = start_grip_all[ridx] + a * (
                target_grip_all[ridx] - start_grip_all[ridx]
            )

    # 分块下发，便于急停清轨迹（避免队列里堆满未执行 waypoint）
    chunk = max(2, int(0.4 * float(frequency)))
    for i0 in range(0, steps, chunk):
        if poll_abort is not None and poll_abort():
            emergency_stop(env)
            return False
        i1 = min(steps, i0 + chunk)
        t0 = time.time() + 0.12
        env.exec_actions(
            actions=actions[i0:i1],
            timestamps=t0 + np.arange(i1 - i0, dtype=np.float64) * dt,
            compensate_latency=False,
        )
        wait_s = max(0.0, float((i1 - i0 - 1) * dt) - 0.05)
        if interruptible_sleep(wait_s, poll_abort=poll_abort):
            emergency_stop(env)
            return False
    return True


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
) -> list[tuple[np.ndarray, float]]:
    """
    与 eval_arx5_fast_wam.py 相同的启动微调，左右臂各自相对 home 加同一 delta。

    返回每臂 (tgt_pose6, tgt_grip)，供退出零力时回位。
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
    n_robots = len(robot_states)
    starts: list[tuple[np.ndarray, float]] = []
    targets: list[tuple[np.ndarray, float]] = []
    for ridx, state in enumerate(robot_states):
        start_pose = state["ActualTCPPose"].astype(np.float64).copy()
        start_grip = float(state["gripper_position"])
        tgt_pose = start_pose + delta6
        tgt_grip = float(
            np.clip(
                start_grip + init_gripper_delta, gripper_min_width, gripper_max_width
            )
        )
        starts.append((start_pose, start_grip))
        targets.append((tgt_pose, tgt_grip))
        print(
            f"  robot{ridx}  from {np.array2string(start_pose, precision=4, separator=',')} "
            f"grip={start_grip:.4f}"
        )
        print(
            f"  robot{ridx}    to {np.array2string(tgt_pose, precision=4, separator=',')} "
            f"grip={tgt_grip:.4f}"
        )

    print(
        "Init fine-tune (same delta on both arms): "
        f"delta={np.array2string(delta6, precision=4, separator=',')} "
        f"gripper_delta={init_gripper_delta:+.4f}"
    )

    # 与 fast_wam 一致：全频率 + 线性 6D（不用 Slerp、不降到 20Hz）
    dt = 1.0 / float(frequency)
    steps = max(2, int(float(duration) * float(frequency)))
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    t0 = time.time() + 0.3
    for k in range(steps):
        a = ease_in_out_quad(k / (steps - 1))
        cmd = np.zeros(7 * n_robots, dtype=np.float64)
        for ridx in range(n_robots):
            start_pose, start_grip = starts[ridx]
            tgt_pose, tgt_grip = targets[ridx]
            base = 7 * ridx
            cmd[base : base + 6] = start_pose + a * (tgt_pose - start_pose)
            cmd[base + 6] = float(
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
    return targets


def record_pose_action(
    env: Arx5EnvWithImages,
    obs: dict,
    *,
    last_rotvecs: list[np.ndarray | None] | None = None,
) -> None:
    """
    把当前位姿记入 action_accumulator（不向机械臂下发指令）。

    时间戳用 get_obs 对齐后的相机时间，供 end_episode 做 episode 级插值对齐。
    若传入 last_rotvecs，会对每臂 rotvec 做 2π 等价连续化，避免 ±π 表示翻转。
    """
    if env.action_accumulator is None:
        return
    n_robots = len(env.robots)
    action = action_from_obs(obs, n_robots)
    if last_rotvecs is not None:
        for robot_idx in range(n_robots):
            sl = slice(7 * robot_idx + 3, 7 * robot_idx + 6)
            rv = action[sl]
            prev = last_rotvecs[robot_idx]
            if prev is not None:
                rv = nearest_rotvec_equivalent(rv, prev)
                action[sl] = rv
            last_rotvecs[robot_idx] = np.asarray(action[sl], dtype=np.float64).copy()
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


class GripperDoubleOpenCloseDetector:
    """检测单臂夹爪在短窗内连续两次「开合」（开↔关往返各一次算一合）。

    用于拖拽模式下无键盘开始/保存：第二次双开合触发时，
    ``last_gesture_start_t`` 指向本段双开合手势起点，便于截掉手势。

    注意：单次开合若拖很久（先开着夹爪干活再关上），起点可能很早；
    因此 cycle 起点与最终截断都会钳制在 ``window_s`` 量级内，避免误删大段轨迹。
    """

    def __init__(
        self,
        *,
        close_thresh: float,
        open_thresh: float,
        window_s: float = 1.5,
        cooldown_s: float = 1.0,
    ):
        if open_thresh <= close_thresh:
            raise ValueError(
                f"open_thresh ({open_thresh}) must be > close_thresh ({close_thresh})"
            )
        self.close_thresh = float(close_thresh)
        self.open_thresh = float(open_thresh)
        self.window_s = float(window_s)
        self.cooldown_s = float(cooldown_s)
        self.last_gesture_start_t: float | None = None
        self._state: str | None = None  # "open" | "closed"
        self._half: str | None = None  # 本轮开合已到达的第一端
        self._pending_start: float | None = None
        self._cycle_times: list[float] = []
        self._cycle_start_times: list[float] = []
        self._last_trigger_t = -1e9

    def reset(self) -> None:
        self.last_gesture_start_t = None
        self._state = None
        self._half = None
        self._pending_start = None
        self._cycle_times.clear()
        self._cycle_start_times.clear()

    def _expire(self, t: float) -> None:
        while self._cycle_times and (t - self._cycle_times[0]) > self.window_s:
            self._cycle_times.pop(0)
            self._cycle_start_times.pop(0)
        # 半程开合拖太久：作废，避免「很早打开、很晚关上」变成超长 cycle
        if (
            self._pending_start is not None
            and (t - float(self._pending_start)) > self.window_s
        ):
            self._half = None
            self._pending_start = None

    def update(self, width: float, t: float | None = None) -> bool:
        """喂入当前夹爪开口；若检测到双开合则返回 True。"""
        now = time.time() if t is None else float(t)
        w = float(width)
        self._expire(now)

        if w <= self.close_thresh:
            new_state = "closed"
        elif w >= self.open_thresh:
            new_state = "open"
        else:
            new_state = None

        if self._state is None:
            if new_state is not None:
                self._state = new_state
            return False

        in_cooldown = (now - self._last_trigger_t) < self.cooldown_s
        if new_state is None or new_state == self._state:
            return False

        # open ↔ closed 边沿
        if self._half is None:
            self._half = new_state
            self._pending_start = now
            self._state = new_state
            return False

        if self._half == new_state:
            self._state = new_state
            return False

        # 完成一次开合（往返）；起点钳制在 window 内，防止拖太久污染截断点
        raw_start = (
            self._pending_start if self._pending_start is not None else now
        )
        cycle_start = max(float(raw_start), now - self.window_s)
        self._cycle_times.append(now)
        self._cycle_start_times.append(cycle_start)
        self._half = None
        self._pending_start = None
        self._state = new_state
        self._expire(now)

        if in_cooldown:
            self._cycle_times.clear()
            self._cycle_start_times.clear()
            return False

        if len(self._cycle_times) >= 2:
            # 整段双开合手势起点：第二次触发时往前最多 window_s
            gesture_start = float(self._cycle_start_times[-2])
            self.last_gesture_start_t = max(gesture_start, now - self.window_s)
            self._last_trigger_t = now
            self._cycle_times.clear()
            self._cycle_start_times.clear()
            return True
        return False


def gripper_width_from_obs(obs: dict, robot_idx: int) -> float | None:
    key = f"robot{robot_idx}_gripper_width"
    if key not in obs:
        return None
    try:
        return float(np.asarray(obs[key][-1]).squeeze())
    except Exception:  # noqa: BLE001
        return None


def truncate_episode_before(env: Arx5EnvWithImages, cutoff_t: float) -> int:
    """丢掉 cutoff_t 及之后的累计样本，使结束手势不进入 zarr。返回保留的 action 步数。"""
    keep_actions = 0
    aa = env.action_accumulator
    if aa is not None and len(aa) > 0:
        actual = np.asarray(aa.actual_timestamps, dtype=np.float64)
        keep_actions = int(np.sum(actual < float(cutoff_t)))
        aa.size = keep_actions

    oa = env.obs_accumulator
    if oa is not None:
        for key in list(oa.timestamps.keys()):
            ts = oa.timestamps[key]
            n = sum(1 for t in ts if t < cutoff_t)
            oa.timestamps[key] = ts[:n]
            oa.data[key] = oa.data[key][:n]

    ia = getattr(env, "image_accumulator", None)
    if ia is not None:
        for key in list(ia.timestamps.keys()):
            ts = ia.timestamps[key]
            n = sum(1 for t in ts if t < cutoff_t)
            ia.timestamps[key] = ts[:n]
            ia.data[key] = ia.data[key][:n]

    return keep_actions


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


# 删除互斥：禁止重叠删 zarr（否则半截损坏）；后台线程避免卡住主循环导致掉臂
_DELETE_LOCK = threading.Lock()


def delete_saved_episode(env, episode_id: int, progress_cb=None) -> bool:
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

    def _cb(info):
        if progress_cb is not None:
            progress_cb(info)
        msg = info.get("message") if isinstance(info, dict) else None
        if msg:
            print(msg, flush=True)

    env.replay_buffer.delete_episode(episode_id, progress_cb=_cb)

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "videos",
                "key": "videos",
                "key_i": 0,
                "n_keys": 1,
                "frac": 0.98,
                "message": f"删除 ep{episode_id} · 重编号 videos…",
            }
        )

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


def is_delete_busy() -> bool:
    return _DELETE_LOCK.locked()


def schedule_delete_episode(
    env,
    episode_id: int,
    *,
    web_state=None,
    hold_robot: bool = False,
    done_mode: str = "READY",
) -> bool:
    """后台删除一条 episode；已有删除在进行时直接拒绝。

    hold_robot=True 时先 set_end_control 钉住位姿，主循环继续跑避免掉臂。
    """
    if env is None:
        if web_state is not None:
            web_state.update(message="尚未加载数据目录")
        return False
    if getattr(env, "dataset_format", "zarr") == "lerobot":
        msg = "LeRobot 模式暂不支持删除 episode（可用裁剪）"
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False
    if not _DELETE_LOCK.acquire(blocking=False):
        msg = "正在删除中，请等当前删除完成后再操作"
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False

    try:
        n_episodes = int(env.replay_buffer.n_episodes)
        eid = int(episode_id)
        if n_episodes <= 0 or eid < 0 or eid >= n_episodes:
            print(
                f"Episode {eid} does not exist "
                f"(valid: 0..{max(n_episodes - 1, 0)})."
            )
            if web_state is not None:
                web_state.update(
                    message=f"invalid episode_id {eid}",
                    delete_pending=None,
                )
            _DELETE_LOCK.release()
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"Delete pre-check failed: {exc}")
        if web_state is not None:
            web_state.update(message=f"delete failed: {exc}", delete_pending=None)
        _DELETE_LOCK.release()
        return False

    n_left_hint = n_episodes - 1
    is_tail = eid == n_episodes - 1
    hint = (
        "末条截断，通常很快"
        if is_tail
        else "需搬移后续相机数据，可能要数分钟，请看进度"
    )
    if web_state is not None:
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_pending=None,
            delete_progress={
                "episode_id": eid,
                "phase": "start",
                "key": "",
                "key_i": 0,
                "n_keys": 0,
                "frac": 0.0,
                "message": f"正在删除 ep{eid}…（{hint}）",
            },
            message=f"正在删除 ep{eid}…（{hint}）",
        )

    def _on_progress(info: dict) -> None:
        if web_state is None or not isinstance(info, dict):
            return
        payload = dict(info)
        payload.setdefault("episode_id", eid)
        msg = str(payload.get("message") or f"正在删除 ep{eid}…")
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_progress=payload,
            message=msg,
        )

    def _worker() -> None:
        ok = False
        err = ""
        try:
            if hold_robot and hasattr(env, "set_end_control"):
                try:
                    env.set_end_control()
                except Exception as exc:  # noqa: BLE001
                    print(f"delete hold set_end_control warning: {exc}")
            print(f"Deleting episode {eid} (background)… {hint}")
            ok = bool(delete_saved_episode(env, eid, progress_cb=_on_progress))
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            print(f"Delete episode {eid} failed: {exc}")
            ok = False
        finally:
            try:
                n_left = int(env.replay_buffer.n_episodes)
            except Exception:
                n_left = n_left_hint
            if web_state is not None:
                web_state.update(
                    mode=done_mode,
                    deleting=False,
                    delete_pending=None,
                    delete_progress=None,
                    n_episodes=n_left,
                    message=(
                        f"已删除 ep{eid} · 剩余 {n_left}"
                        if ok
                        else (f"删除 ep{eid} 失败: {err}" if err else f"删除 ep{eid} 失败")
                    ),
                )
            _DELETE_LOCK.release()

    threading.Thread(target=_worker, name=f"delete-ep{eid}", daemon=True).start()
    return True


def _map_steps_to_video_frames(
    start_step: int, end_step: int, n_steps: int, n_frames: int
) -> tuple[int, int]:
    """把 zarr 半开区间映射到视频帧半开区间。"""
    n_steps = max(int(n_steps), 1)
    n_frames = max(int(n_frames), 0)
    if n_frames <= 0:
        return 0, 0
    if n_frames != n_steps and n_steps > 1 and n_frames > 1:
        v0 = int(round(int(start_step) * (n_frames - 1) / (n_steps - 1)))
        v1 = int(round((int(end_step) - 1) * (n_frames - 1) / (n_steps - 1))) + 1
    else:
        v0 = int(start_step)
        v1 = int(end_step)
    v0 = int(np.clip(v0, 0, n_frames))
    v1 = int(np.clip(v1, v0 + 1 if v0 < n_frames else v0, n_frames))
    return v0, v1


def _trim_one_mp4_ffmpeg(
    mp4: Path, v0: int, v1: int, fps: float
) -> bool:
    """优先用 ffmpeg 按帧裁剪；失败返回 False。"""
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        return False
    tmp = mp4.with_suffix(".trim_tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    # trim 用帧号；再编码为浏览器友好的 yuv420p h264
    vf = f"trim=start_frame={v0}:end_frame={v1},setpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[trim] ffmpeg failed for {mp4.name}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(mp4)
        return True
    if tmp.exists():
        tmp.unlink()
    return False


def _trim_one_mp4_cv2(mp4: Path, v0: int, v1: int, fps: float) -> bool:
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        print(f"[trim] skip unreadable video {mp4}")
        return False
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            print(f"[trim] skip empty video {mp4}")
            return False
        if fps <= 1e-3:
            fps = 30.0
        tmp = mp4.with_suffix(".trim_tmp.mp4")
        if tmp.exists():
            tmp.unlink()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            print(f"[trim] VideoWriter failed for {mp4}")
            return False
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, v0)
            for _fi in range(v0, v1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
        finally:
            writer.release()
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(mp4)
            return True
        if tmp.exists():
            tmp.unlink()
            print(f"[trim] wrote empty video, kept original: {mp4}")
        return False
    finally:
        cap.release()


def _trim_episode_videos(
    video_dir: Path,
    *,
    start_step: int,
    end_step: int,
    n_steps: int,
    progress_cb=None,
) -> None:
    """按 zarr 步数比例裁剪 videos/{ep}/*.mp4，原地替换。"""
    if not video_dir.is_dir():
        return
    mp4s = sorted(video_dir.glob("*.mp4"))
    if not mp4s:
        return
    n_steps = max(int(n_steps), 1)
    start_step = int(start_step)
    end_step = int(end_step)

    for mi, mp4 in enumerate(mp4s):
        if progress_cb is not None:
            progress_cb(
                {
                    "phase": "video",
                    "key": mp4.name,
                    "key_i": mi + 1,
                    "n_keys": len(mp4s),
                    "frac": float(mi) / float(len(mp4s)),
                    "message": f"裁剪视频 {mp4.name} ({mi + 1}/{len(mp4s)})…",
                }
            )
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"[trim] skip unreadable video {mp4}")
            continue
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        if n_frames <= 0:
            print(f"[trim] skip empty video {mp4}")
            continue
        v0, v1 = _map_steps_to_video_frames(start_step, end_step, n_steps, n_frames)
        if v1 <= v0:
            print(f"[trim] empty frame range for {mp4.name}; skip")
            continue
        if not _trim_one_mp4_ffmpeg(mp4, v0, v1, fps):
            _trim_one_mp4_cv2(mp4, v0, v1, fps)

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "video",
                "key": "(done)",
                "key_i": len(mp4s),
                "n_keys": len(mp4s),
                "frac": 1.0,
                "message": f"视频裁剪完成（{len(mp4s)} 个）",
            }
        )


def _trim_intervention_sidecar(
    output_dir: Path, episode_id: int, start: int, end: int
) -> None:
    path = Path(output_dir) / "intervention" / f"episode_{episode_id}.npy"
    if not path.exists():
        return
    lab = np.load(path)
    if lab.ndim != 1:
        print(f"[trim] unexpected intervention shape {lab.shape} @ {path}")
        return
    if end > len(lab) or start < 0 or start >= end:
        print(
            f"[trim] intervention len={len(lab)} incompatible with "
            f"[{start},{end}); skip sidecar"
        )
        return
    np.save(path, np.asarray(lab[start:end], dtype=lab.dtype))
    print(f"[trim] intervention sidecar ep{episode_id}: {len(lab)} -> {end - start}")


def trim_saved_episode(
    env,
    episode_id: int,
    start: int,
    end: int,
    progress_cb=None,
) -> bool:
    """裁剪已保存 episode：保留本地半开区间 [start, end)。

    zarr：改 replay_buffer + videos + intervention。
    LeRobot：改 parquet + 三路 mp4 + meta。
    """
    fmt = getattr(env, "dataset_format", "zarr")
    if fmt == "lerobot":
        from modules.lerobot_dataset_io import trim_lerobot_episode

        rb = env.replay_buffer
        if not getattr(rb, "has_episode", lambda _e: False)(episode_id):
            print(f"Episode {episode_id} does not exist (lerobot).")
            return False
        sl = rb.get_episode_slice(episode_id)
        n_steps = int(sl.stop - sl.start)
        start = int(start)
        end = int(end)
        if start < 0 or end > n_steps or start >= end:
            print(
                f"Invalid trim range [{start}, {end}) for ep{episode_id} T={n_steps}"
            )
            return False
        print(
            f"Trimming lerobot episode {episode_id}: keep [{start}, {end}) "
            f"({end - start}/{n_steps} steps)…"
        )
        ok = bool(
            trim_lerobot_episode(
                Path(env.output_dir),
                int(episode_id),
                start,
                end,
                progress_cb=progress_cb,
            )
        )
        if ok and hasattr(rb, "refresh"):
            rb.refresh()
        return ok

    n_episodes = env.replay_buffer.n_episodes
    if n_episodes <= 0:
        print("No saved episode to trim.")
        return False
    if episode_id < 0 or episode_id >= n_episodes:
        print(
            f"Episode {episode_id} does not exist "
            f"(valid: 0..{n_episodes - 1})."
        )
        return False

    sl = env.replay_buffer.get_episode_slice(episode_id)
    n_steps = int(sl.stop - sl.start)
    start = int(start)
    end = int(end)
    if start < 0 or end > n_steps or start >= end:
        print(f"Invalid trim range [{start}, {end}) for ep{episode_id} T={n_steps}")
        return False
    if start == 0 and end == n_steps:
        print(f"Episode {episode_id} already covers [{start}, {end}); nothing to do.")
        return True

    def _cb(info):
        if progress_cb is not None and isinstance(info, dict):
            progress_cb(info)

    keep = end - start
    print(
        f"Trimming saved episode {episode_id}: keep [{start}, {end}) "
        f"({keep}/{n_steps} steps)…"
    )
    env.replay_buffer.trim_episode(episode_id, start, end, progress_cb=_cb)

    out_dir = Path(getattr(env, "output_dir", env.video_dir.parent))
    _trim_intervention_sidecar(out_dir, episode_id, start, end)

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "video",
                "key": "videos",
                "key_i": 0,
                "n_keys": 1,
                "frac": 0.95,
                "message": f"裁剪 ep{episode_id} · 视频…",
            }
        )
    video_dir = env.video_dir.joinpath(str(episode_id))
    _trim_episode_videos(
        video_dir,
        start_step=start,
        end_step=end,
        n_steps=n_steps,
        progress_cb=_cb,
    )

    new_sl = env.replay_buffer.get_episode_slice(episode_id)
    new_len = int(new_sl.stop - new_sl.start)
    print(
        f"Trimmed saved episode {episode_id}: {n_steps} -> {new_len} steps "
        f"(kept [{start}, {end}))."
    )
    return True


def schedule_trim_episode(
    env,
    episode_id: int,
    start: int,
    end: int,
    *,
    web_state=None,
    hold_robot: bool = False,
    done_mode: str = "READY",
) -> bool:
    """后台裁剪一条 episode；与删除共用锁，避免并发改 zarr。"""
    if env is None:
        if web_state is not None:
            web_state.update(message="尚未加载数据目录")
        return False
    if not _DELETE_LOCK.acquire(blocking=False):
        msg = "正在处理数据（删除/裁剪），请稍候…"
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False

    try:
        n_episodes = int(env.replay_buffer.n_episodes)
        eid = int(episode_id)
        fmt = getattr(env, "dataset_format", "zarr")
        if fmt == "lerobot":
            if n_episodes <= 0 or not env.replay_buffer.has_episode(eid):
                raise IndexError(f"invalid episode_id {eid}")
        elif n_episodes <= 0 or eid < 0 or eid >= n_episodes:
            raise IndexError(f"invalid episode_id {eid}")
        sl = env.replay_buffer.get_episode_slice(eid)
        n_steps = int(sl.stop - sl.start)
        start_i = int(start)
        end_i = int(end)
        if start_i < 0 or end_i > n_steps or start_i >= end_i:
            raise ValueError(
                f"invalid trim range [{start_i}, {end_i}) for T={n_steps}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Trim pre-check failed: {exc}")
        if web_state is not None:
            web_state.update(message=f"裁剪失败: {exc}", delete_pending=None)
        _DELETE_LOCK.release()
        return False

    keep = end_i - start_i
    if web_state is not None:
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_pending=None,
            delete_progress={
                "episode_id": eid,
                "phase": "start",
                "key": "",
                "key_i": 0,
                "n_keys": 0,
                "frac": 0.0,
                "message": (
                    f"正在裁剪 ep{eid} → 保留 [{start_i},{end_i}) "
                    f"共 {keep} 步…"
                ),
            },
            message=(
                f"正在裁剪 ep{eid} → 保留 [{start_i},{end_i}) 共 {keep} 步…"
            ),
        )

    def _on_progress(info: dict) -> None:
        if web_state is None or not isinstance(info, dict):
            return
        payload = dict(info)
        payload.setdefault("episode_id", eid)
        msg = str(payload.get("message") or f"正在裁剪 ep{eid}…")
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_progress=payload,
            message=msg,
        )

    def _worker() -> None:
        ok = False
        err = ""
        try:
            if hold_robot and hasattr(env, "set_end_control"):
                try:
                    env.set_end_control()
                except Exception as exc:  # noqa: BLE001
                    print(f"trim hold set_end_control warning: {exc}")
            print(
                f"Trimming episode {eid} [{start_i},{end_i}) (background)…"
            )
            ok = bool(
                trim_saved_episode(
                    env, eid, start_i, end_i, progress_cb=_on_progress
                )
            )
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            print(f"Trim episode {eid} failed: {exc}")
            ok = False
        finally:
            try:
                n_left = int(env.replay_buffer.n_episodes)
            except Exception:
                n_left = n_episodes
            if web_state is not None:
                web_state.update(
                    mode=done_mode,
                    deleting=False,
                    delete_pending=None,
                    delete_progress=None,
                    n_episodes=n_left,
                    message=(
                        f"已裁剪 ep{eid} → [{start_i},{end_i}) · {keep} 步"
                        if ok
                        else (
                            f"裁剪 ep{eid} 失败: {err}"
                            if err
                            else f"裁剪 ep{eid} 失败"
                        )
                    ),
                )
            _DELETE_LOCK.release()

    threading.Thread(
        target=_worker, name=f"trim-ep{eid}", daemon=True
    ).start()
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


def leave_drag_stay_pose(env: Arx5EnvWithImages) -> None:
    """退出零力/拖拽，停留在当前位姿并恢复末端控制（供回放前调用）。"""
    print("Leaving drag for replay (stay at current pose)...")
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


def emergency_stop(env: Arx5EnvWithImages) -> None:
    """急停：清轨迹并钉在当前实测位姿（set_end_control），两臂一起停。"""
    print("!!! E-STOP — hold current pose (clear trajectory)")
    if hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
        except Exception as exc:  # noqa: BLE001
            print(f"E-STOP set_end_control warning: {exc}")
    else:
        try:
            env.set_to_damping()
        except Exception as exc:  # noqa: BLE001
            print(f"E-STOP set_to_damping warning: {exc}")


def interruptible_sleep(
    seconds: float,
    *,
    poll_abort=None,
    dt: float = 0.05,
) -> bool:
    """可被 poll_abort 打断的 sleep；返回 True 表示被中止。"""
    deadline = time.time() + max(0.0, float(seconds))
    while True:
        if poll_abort is not None and poll_abort():
            return True
        now = time.time()
        if now >= deadline:
            return False
        time.sleep(min(float(dt), deadline - now))


def mask_actions_hold_other_arms(
    env: Arx5EnvWithImages,
    actions: np.ndarray,
    robot_indices: list[int] | None,
) -> np.ndarray:
    """只动选中的臂；未选中臂整段钉在当前实测 TCP+夹爪。None=双臂。"""
    out = np.asarray(actions, dtype=np.float64).copy()
    n_robots = len(env.robots)
    if robot_indices is None:
        return out
    selected = {int(i) for i in robot_indices}
    states = env.get_robot_state()
    for ridx in range(n_robots):
        if ridx in selected:
            continue
        pose6 = np.asarray(states[ridx]["ActualTCPPose"], dtype=np.float64).reshape(6)
        grip = float(states[ridx]["gripper_position"])
        out[:, 7 * ridx : 7 * ridx + 6] = pose6
        out[:, 7 * ridx + 6] = grip
    return out


def _format_replay_arms(robot_indices: list[int] | None, n_robots: int) -> str:
    if robot_indices is None or set(robot_indices) == set(range(n_robots)):
        return "both"
    labels = []
    for i in sorted(robot_indices):
        side = "L" if i == 0 else ("R" if i == 1 else f"r{i}")
        labels.append(f"{side}/robot{i}")
    return ",".join(labels) if labels else "none"


def load_episode_actions(
    env: Arx5EnvWithImages, episode_id: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """返回 (actions[T,7R], relative_timestamps[T])；失败返回 None。"""
    n_episodes = env.replay_buffer.n_episodes
    if n_episodes <= 0:
        print("No saved episode to replay.")
        return None
    if episode_id < 0 or episode_id >= n_episodes:
        print(
            f"Episode {episode_id} does not exist "
            f"(valid: 0..{n_episodes - 1})."
        )
        return None

    n_robots = len(env.robots)
    sl = env.replay_buffer.get_episode_slice(episode_id)
    actions = np.asarray(env.replay_buffer.data["action"][sl], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0:
        print(f"Episode {episode_id} has no action steps.")
        return None
    expected_dim = 7 * n_robots
    if actions.shape[1] != expected_dim:
        print(
            f"Episode {episode_id} action dim {actions.shape[1]} "
            f"!= 7*n_robots={expected_dim}."
        )
        return None

    actions = continuize_action_rotvecs(actions, n_robots=n_robots)
    frequency_fallback = 60.0
    timestamps = None
    if "timestamp" in env.replay_buffer.data:
        timestamps = np.asarray(
            env.replay_buffer.data["timestamp"][sl], dtype=np.float64
        )
    if timestamps is not None and len(timestamps) == len(actions):
        rel = timestamps - float(timestamps[0])
        duration = float(rel[-1]) if len(rel) > 1 else 0.0
        if duration <= 0.0 or not np.isfinite(duration):
            dt = 1.0 / frequency_fallback
            rel = np.arange(len(actions), dtype=np.float64) * dt
    else:
        dt = 1.0 / frequency_fallback
        rel = np.arange(len(actions), dtype=np.float64) * dt
    return actions, rel


def _anchor_actions_to_current_pose(
    env: Arx5EnvWithImages, actions: np.ndarray
) -> np.ndarray:
    """把整段 action 的 rotvec 锚到当前 ActualTCPPose，避免起点表示翻转。"""
    out = np.asarray(actions, dtype=np.float64).copy()
    n_robots = len(env.robots)
    states = env.get_robot_state()
    for ridx in range(n_robots):
        curr_rv = np.asarray(states[ridx]["ActualTCPPose"][3:6], dtype=np.float64)
        sl = slice(7 * ridx + 3, 7 * ridx + 6)
        out[0, sl] = nearest_rotvec_equivalent(out[0, sl], curr_rv)
        for t in range(1, len(out)):
            out[t, sl] = nearest_rotvec_equivalent(out[t, sl], out[t - 1, sl])
    return out


def _targets_from_action_row(
    action_row: np.ndarray, n_robots: int
) -> list[tuple[np.ndarray, float]]:
    targets: list[tuple[np.ndarray, float]] = []
    row = np.asarray(action_row, dtype=np.float64).reshape(-1)
    for ridx in range(n_robots):
        pose6 = row[7 * ridx : 7 * ridx + 6].copy()
        grip = float(row[7 * ridx + 6])
        targets.append((pose6, grip))
    return targets


def _exec_actions_chunked(
    env: Arx5EnvWithImages,
    actions: np.ndarray,
    rel: np.ndarray,
    *,
    poll_abort=None,
    chunk_s: float = 0.5,
    on_sample=None,
) -> bool:
    """按时长分块 schedule，便于急停；返回 False 表示被中止。

    on_sample(i, action_row, t_rel): 按时间推进回调，用于网页 LIVE 轨迹同步。
    """
    actions = np.asarray(actions, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)
    if len(actions) == 0:
        return True
    i0 = 0
    n = len(actions)
    while i0 < n:
        if poll_abort is not None and poll_abort():
            emergency_stop(env)
            return False
        t_cut = float(rel[i0]) + float(chunk_s)
        i1 = i0 + 1
        while i1 < n and float(rel[i1]) < t_cut:
            i1 += 1
        chunk = actions[i0:i1]
        chunk_rel = rel[i0:i1] - float(rel[i0])
        t0 = time.time() + 0.15
        env.exec_actions(
            actions=chunk,
            timestamps=t0 + chunk_rel,
            compensate_latency=False,
        )
        wait_s = max(0.0, float(chunk_rel[-1]) - 0.05)
        # 按相对时间推进推送 LIVE 点，与机械臂运动大致同步
        deadline = time.time() + wait_s
        next_i = i0
        wall0 = time.time()
        while True:
            if poll_abort is not None and poll_abort():
                emergency_stop(env)
                return False
            now = time.time()
            elapsed = now - wall0
            if on_sample is not None:
                while next_i < i1 and float(chunk_rel[next_i - i0]) <= elapsed + 0.02:
                    on_sample(next_i, actions[next_i], float(rel[next_i]))
                    next_i += 1
            if now >= deadline:
                break
            time.sleep(min(0.05, deadline - now))
        if on_sample is not None:
            while next_i < i1:
                on_sample(next_i, actions[next_i], float(rel[next_i]))
                next_i += 1
        i0 = i1
    if interruptible_sleep(0.2, poll_abort=poll_abort):
        emergency_stop(env)
        return False
    return True


def replay_saved_episode(
    env: Arx5EnvWithImages,
    episode_id: int,
    *,
    frequency: float,
    approach_duration: float = 2.5,
    robot_indices: list[int] | None = None,
    poll_abort=None,
    web_state=None,
) -> bool:
    """从 zarr ``action`` Slerp 到起点再分块回放。

    robot_indices: 要动的臂（如 [0]/[1]）；None=双臂。
    poll_abort: 返回 True 时急停并中止。
    web_state: 若给定，同步推送网页下方 LIVE 轨迹。
    """
    loaded = load_episode_actions(env, episode_id)
    if loaded is None:
        return False
    actions, rel = loaded
    n_robots = len(env.robots)
    actions = mask_actions_hold_other_arms(env, actions, robot_indices)
    duration = float(rel[-1]) if len(rel) > 1 else 0.0
    arms = _format_replay_arms(robot_indices, n_robots)

    print(
        f"\n>>> Replay episode {episode_id} [{arms}]: "
        f"{len(actions)} steps, ~{duration:.2f}s "
        f"(Slerp approach {approach_duration:.1f}s first)"
    )
    print("  (X / web E-Stop to abort; do not drag arms)")

    begin_replay_live_series(
        web_state,
        episode_id=episode_id,
        message=f"正在回放 ep{episode_id} [{arms}]…（X 急停）",
    )

    start_targets = _targets_from_action_row(actions[0], n_robots)
    for ridx, (pose6, grip) in enumerate(start_targets):
        moving = robot_indices is None or ridx in robot_indices
        tag = "MOVE" if moving else "HOLD"
        print(
            f"  robot{ridx}[{tag}] start "
            f"xyz=[{pose6[0]:.4f},{pose6[1]:.4f},{pose6[2]:.4f}] "
            f"rot=[{pose6[3]:.4f},{pose6[4]:.4f},{pose6[5]:.4f}] "
            f"grip={grip:.4f}"
        )

    try:
        ok_approach = move_to_tcp_pose_slerp(
            env,
            start_targets,
            frequency=frequency,
            duration=float(approach_duration),
            label=f"replay-ep{episode_id}-start",
            ensure_end_control=True,
            poll_abort=poll_abort,
        )
        if not ok_approach:
            print(f"<<< Replay episode {episode_id} aborted during approach.")
            return False
        actions = _anchor_actions_to_current_pose(env, actions)
        actions = mask_actions_hold_other_arms(env, actions, robot_indices)
        # 起点写入 LIVE；整段回放再按时间推进
        append_live_from_action_row(
            web_state,
            actions[0],
            step=0,
            t_elapsed=float(rel[0]) if len(rel) else 0.0,
            n_robots=n_robots,
            force=True,
        )
        if web_state is not None:
            try:
                from modules.collect_web_ui import publish_pose_from_env

                publish_pose_from_env(web_state, env)
            except Exception:
                pass
        if len(actions) <= 1:
            print(f"<<< Replay episode {episode_id} done (single frame).")
            return True
        play = actions[1:]
        play_rel = rel[1:] - float(rel[1])
        display_t0 = float(rel[1]) if len(rel) > 1 else 0.0

        def _on_sample(i: int, row: np.ndarray, t_rel: float) -> None:
            step = int(i) + 1  # play 从原 episode 第 1 帧起
            append_live_from_action_row(
                web_state,
                row,
                step=step,
                t_elapsed=display_t0 + float(t_rel),
                n_robots=n_robots,
            )
            if web_state is not None and step % 6 == 0:
                try:
                    from modules.collect_web_ui import publish_pose_from_env

                    publish_pose_from_env(web_state, env)
                except Exception:
                    pass

        ok_play = _exec_actions_chunked(
            env,
            play,
            play_rel,
            poll_abort=poll_abort,
            chunk_s=0.5,
            on_sample=_on_sample if web_state is not None else None,
        )
        if not ok_play:
            print(f"<<< Replay episode {episode_id} aborted (E-STOP).")
            return False
        # 末帧强制写入（降采样可能跳过）
        append_live_from_action_row(
            web_state,
            actions[-1],
            step=len(actions) - 1,
            t_elapsed=float(rel[-1]) if len(rel) else 0.0,
            n_robots=n_robots,
            force=True,
        )
        if web_state is not None:
            try:
                from modules.collect_web_ui import publish_pose_from_env

                publish_pose_from_env(web_state, env)
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"Replay episode {episode_id} failed: {exc}")
        return False

    print(f"<<< Replay episode {episode_id} done. Stay at final pose (end-control).")
    print("Press P for drag, C to record, J=step, R=full, X=E-Stop.\n")
    return True


def replay_step_to_index(
    env: Arx5EnvWithImages,
    actions: np.ndarray,
    step_idx: int,
    *,
    frequency: float,
    step_duration: float = 0.12,
    robot_indices: list[int] | None = None,
    poll_abort=None,
    web_state=None,
    t_elapsed: float | None = None,
) -> bool:
    """单步：Slerp 缓动到 ``actions[step_idx]``（可单臂）。"""
    n_robots = len(env.robots)
    if step_idx < 0 or step_idx >= len(actions):
        return False
    row = mask_actions_hold_other_arms(
        env, actions[step_idx][None, :], robot_indices
    )[0].copy()
    states = env.get_robot_state()
    for ridx in range(n_robots):
        curr_rv = np.asarray(states[ridx]["ActualTCPPose"][3:6], dtype=np.float64)
        sl = slice(7 * ridx + 3, 7 * ridx + 6)
        row[sl] = nearest_rotvec_equivalent(row[sl], curr_rv)
    targets = _targets_from_action_row(row, n_robots)
    try:
        ok = move_to_tcp_pose_slerp(
            env,
            targets,
            frequency=frequency,
            duration=float(step_duration),
            label=f"replay-step-{step_idx}",
            ensure_end_control=True,
            poll_abort=poll_abort,
        )
        if ok and web_state is not None:
            if t_elapsed is None:
                t_elapsed = float(step_idx) / max(float(frequency), 1e-6)
            append_live_from_action_row(
                web_state,
                row,
                step=int(step_idx),
                t_elapsed=float(t_elapsed),
                n_robots=n_robots,
                force=True,
            )
            try:
                from modules.collect_web_ui import publish_pose_from_env

                publish_pose_from_env(web_state, env)
            except Exception:
                pass
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"Replay step {step_idx} failed: {exc}")
        return False


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
    init_targets: list[tuple[np.ndarray, float | None]] | None,
    *,
    frequency: float,
    duration: float = 2.5,
) -> None:
    """
    退出零力后缓动回各臂 init 位姿。

    不在拖拽点硬锁满增益；先软退出并回位，到达后再恢复满增益。
    """
    print("Zero-force OFF. Leaving drag mode...")
    if hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
        except Exception as exc:  # noqa: BLE001
            print(f"set_end_control warning: {exc}")

    if init_targets is None:
        print("No init pose stored; staying at current pose.")
    else:
        move_to_tcp_pose(
            env,
            init_targets,
            frequency=frequency,
            duration=duration,
            label="return-to-init",
            ensure_end_control=False,
        )
        print_current_pose(env, tag="after-return-init")

    if hasattr(env, "restore_gains_after_drag"):
        try:
            env.restore_gains_after_drag()
        except Exception as exc:  # noqa: BLE001
            print(f"restore_gains_after_drag warning: {exc}")


def enter_drag_mode(env: Arx5EnvWithImages) -> None:
    """进入拖拽/零力，并等待控制器真正切完（保存回 init 后自动再进依赖此等待）。"""
    print("Entering drag mode...")
    env.set_gravity_compensation()
    # 伺服模式切换后再给一小段稳定时间
    time.sleep(0.25)
    print(
        "Drag mode ON (true GC if supported, else soft-hold damping). "
        "Drag arm freely; idle pose is held. Squeeze gripper by hand (torque assist)."
    )


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
    prompt_status: str = "DEL",
    n_robots: int = 1,
    camera_latency: list[dict] | None = None,
) -> np.ndarray:
    vis = camera_strip_from_obs(obs, latency=camera_latency).copy()
    if recording:
        status = "REC"
        help_text = "S save | D drop | 2x grip(can1) save | I | Q"
    elif delete_prompt:
        status = prompt_status
        help_text = delete_prompt
    elif zero_force_active:
        status = "DRAG"
        help_text = "C|R|J|0/1/2 arms|X stop|P->init|Q"
    elif no_zero_force:
        status = "READY"
        help_text = "C|R|J|0/1/2 arms|X E-Stop|P|Q"
    else:
        status = "IDLE"
        help_text = "P|R|J|0/1/2 arms|X E-Stop|E|Q"
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


# OpenCV Hershey 字体不支持中文，用 ASCII：左腕 / 中间 D435 / 右腕
_CAM_PANEL_LABELS = {
    0: "L  Left wrist",
    1: "M  Mid D435",
    2: "R  Right wrist",
}
_CAM_SHORT_LABELS = {0: "L", 1: "M", 2: "R"}


def compute_camera_latency(env, obs: dict | None = None) -> list[dict]:
    """
    当前预览帧相对墙钟的延迟（画面有多“旧”）。

    age_ms  = now - 选中帧 timestamp（越大越滞后）
    skew_ms = 选中帧 timestamp - 对齐基准时刻（相对其他相机的偏差）
    fresh_ms = now - 该路 ring buffer 最新帧（采集是否跟上）
    """
    cam_data = getattr(env, "last_camera_data", None)
    if not cam_data:
        return []
    now = time.time()
    if obs is not None and "timestamp" in obs:
        t_ref = float(np.asarray(obs["timestamp"][-1]))
    else:
        t_ref = float(getattr(env, "last_align_timestamp", None) or now)
    align_idx = getattr(env, "last_align_camera_idx", None)
    out: list[dict] = []
    for cam_idx in sorted(cam_data.keys()):
        value = cam_data[cam_idx]
        ts = np.asarray(value["timestamp"], dtype=np.float64)
        if ts.size == 0:
            continue
        nn = int(np.argmin(np.abs(ts - t_ref)))
        t_sel = float(ts[nn])
        t_latest = float(ts[-1])
        # Prefer raw receive time for "how old is the buffer" if present.
        recv = value.get("camera_receive_timestamp")
        if recv is not None:
            recv_arr = np.asarray(recv, dtype=np.float64)
            t_recv_latest = float(recv_arr[-1]) if recv_arr.size else t_latest
        else:
            t_recv_latest = t_latest
        item = {
            "idx": int(cam_idx),
            "label": _CAM_SHORT_LABELS.get(int(cam_idx), f"C{cam_idx}"),
            "age_ms": round((now - t_sel) * 1000.0, 1),
            "fresh_ms": round((now - t_recv_latest) * 1000.0, 1),
            "skew_ms": round((t_sel - t_ref) * 1000.0, 1),
            "is_ref": align_idx is not None and int(cam_idx) == int(align_idx),
        }
        out.append(item)
    return out


def _draw_cam_label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """在单路画面左上角画半透明底 + 多行标签。"""
    out = np.ascontiguousarray(img)
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.40, min(0.65, w / 360.0))
    thickness = 2
    pad = 5
    sizes = [cv2.getTextSize(t, font, scale, thickness) for t in lines]
    tw = max(s[0][0] for s in sizes)
    line_h = max(s[0][1] for s in sizes) + 4
    block_h = line_h * len(lines) + pad * 2
    x0, y0 = 4, 4
    x1 = min(w - 1, x0 + tw + pad * 2)
    y1 = min(h - 1, y0 + block_h)
    roi = out[y0:y1, x0:x1]
    if roi.size:
        overlay = roi.copy()
        overlay[:] = (0, 0, 0)
        cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    y = y0 + pad
    for text, ((_, th), _) in zip(lines, sizes):
        y += th
        # Green <80ms, yellow <160ms, red otherwise (age shown on line 2)
        color = (255, 255, 255)
        if text.startswith("lag "):
            try:
                ms = float(text.split()[1].replace("ms", ""))
                if ms < 80:
                    color = (120, 255, 120)
                elif ms < 160:
                    color = (80, 220, 255)
                else:
                    color = (80, 80, 255)
            except Exception:
                pass
        cv2.putText(out, text, (x0 + pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += 4
    return out


def camera_strip_from_obs(
    obs: dict, latency: list[dict] | None = None
) -> np.ndarray:
    """三相机横向拼接，标注左/中/右 + 画面延迟。"""
    cam_keys = sorted(k for k in obs if k.startswith("camera") and k.endswith("_rgb"))
    if not cam_keys:
        raise RuntimeError("obs 中没有 camera*_rgb")
    lat_by_idx = {int(d["idx"]): d for d in (latency or [])}
    frames = []
    for key in cam_keys:
        img = np.asarray(obs[key][-1])
        if img.dtype != np.uint8:
            if float(np.nanmax(img)) <= 1.5:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        try:
            cam_idx = int(key.replace("camera", "").replace("_rgb", ""))
        except ValueError:
            cam_idx = len(frames)
        name = _CAM_PANEL_LABELS.get(cam_idx, f"cam{cam_idx}")
        lat = lat_by_idx.get(cam_idx)
        if lat is not None:
            ref = "*" if lat.get("is_ref") else ""
            lines = [
                f"{name}{ref}",
                f"lag {lat['age_ms']:.0f}ms",
                f"skew {lat['skew_ms']:+.0f}ms",
            ]
        else:
            lines = [name]
        frames.append(_draw_cam_label(img.copy(), lines))
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
    camera_latency: list[dict] | None = None,
) -> None:
    if web_state is None:
        return
    from modules.collect_web_ui import publish_pose_from_env, vis_rgb_to_jpeg

    deleting = bool(getattr(web_state, "deleting", False)) or is_delete_busy()
    if deleting:
        mode = "DELETING"
    else:
        mode = _compute_mode(
            recording=recording,
            zero_force_active=zero_force_active,
            no_zero_force=no_zero_force,
            delete_mode=delete_mode,
        )
    # 删除线程正在改 zarr 时不要读 replay_buffer（避免竞态）
    if not deleting:
        publish_pose_from_env(web_state, env)
    jpeg = b""
    # 网页预览：带相机标签 + 延迟叠加的画面
    src = clean_rgb if clean_rgb is not None else vis_rgb
    if src is not None:
        try:
            jpeg = vis_rgb_to_jpeg(src, quality=75)
        except Exception:
            jpeg = b""
    payload = dict(
        mode=mode,
        deleting=deleting,
        zero_force_active=zero_force_active,
        recording=recording,
        no_zero_force=no_zero_force,
        n_steps=int(n_steps),
        t_elapsed=float(t_elapsed),
        episode_id=episode_id,
        delete_pending=(
            None if deleting else (delete_target if delete_mode == "confirm" else None)
        ),
        frequency=float(frequency),
        jpeg=jpeg,
        camera_latency=list(camera_latency or []),
    )
    if message:
        payload["message"] = message
    if not deleting:
        try:
            payload["n_episodes"] = int(env.replay_buffer.n_episodes)
        except Exception:
            pass
    web_state.update(**payload)


def append_live_from_obs(
    web_state, obs: dict, *, step: int, t_elapsed: float, n_robots: int = 2
) -> None:
    if web_state is None:
        return
    # 降采样：约每 3 步采一点
    if step > 0 and step % 3 != 0:
        return
    try:
        labels = ("L", "R", "A2", "A3")
        robots = []
        for ridx in range(n_robots):
            xyz_i = np.asarray(
                obs[f"robot{ridx}_eef_pos"][-1], dtype=np.float64
            ).reshape(3)
            rot_i = np.asarray(
                obs[f"robot{ridx}_eef_rot_axis_angle"][-1], dtype=np.float64
            ).reshape(3)
            grip_i = float(
                np.asarray(obs[f"robot{ridx}_gripper_width"][-1]).squeeze()
            )
            robots.append(
                {
                    "label": labels[ridx] if ridx < len(labels) else f"A{ridx}",
                    "xyz": [float(v) for v in xyz_i],
                    "rot": [float(v) for v in rot_i],
                    "grip": grip_i,
                }
            )
        action = action_from_obs(obs, n_robots)
        r0 = robots[0]
        web_state.append_live_sample(
            step=step,
            t=t_elapsed,
            xyz=np.asarray(r0["xyz"], dtype=np.float64),
            rot=np.asarray(r0["rot"], dtype=np.float64),
            grip=float(r0["grip"]),
            action=action,
            robots=robots,
        )
    except Exception:
        pass


def append_live_from_action_row(
    web_state,
    action_row: np.ndarray,
    *,
    step: int,
    t_elapsed: float,
    n_robots: int = 2,
    every: int = 3,
    force: bool = False,
) -> None:
    """把回放/单步的 action 行推到网页 LIVE 轨迹图。"""
    if web_state is None:
        return
    if not force and step > 0 and every > 1 and step % every != 0:
        return
    try:
        row = np.asarray(action_row, dtype=np.float64).reshape(-1)
        labels = ("L", "R", "A2", "A3")
        robots = []
        for ridx in range(n_robots):
            base = 7 * ridx
            if row.shape[0] < base + 7:
                break
            robots.append(
                {
                    "label": labels[ridx] if ridx < len(labels) else f"A{ridx}",
                    "xyz": [float(row[base]), float(row[base + 1]), float(row[base + 2])],
                    "rot": [
                        float(row[base + 3]),
                        float(row[base + 4]),
                        float(row[base + 5]),
                    ],
                    "grip": float(row[base + 6]),
                }
            )
        if not robots:
            return
        r0 = robots[0]
        web_state.append_live_sample(
            step=int(step),
            t=float(t_elapsed),
            xyz=np.asarray(r0["xyz"], dtype=np.float64),
            rot=np.asarray(r0["rot"], dtype=np.float64),
            grip=float(r0["grip"]),
            action=row,
            robots=robots,
        )
        web_state.update(n_steps=int(step) + 1, t_elapsed=float(t_elapsed))
    except Exception:
        pass


def begin_replay_live_series(
    web_state,
    *,
    episode_id: int,
    message: str = "",
) -> None:
    """回放开始：清空 LIVE 序列并切到 REPLAY，便于网页 Follow LIVE。"""
    if web_state is None:
        return
    if hasattr(web_state, "clear_live_series"):
        web_state.clear_live_series()
    web_state.update(
        mode="REPLAY",
        recording=False,
        zero_force_active=False,
        episode_id=int(episode_id),
        n_steps=0,
        t_elapsed=0.0,
        message=message or f"正在回放 ep{episode_id}…",
    )


class OfflineDatasetEnv:
    """仅打开已有 replay_buffer.zarr + videos，不启臂 / 相机。"""

    dataset_format = "zarr"

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


def open_offline_dataset(path: str | Path):
    """打开 zarr 或 LeRobot 数据目录；失败抛 FileNotFoundError / OSError。"""
    from modules.lerobot_dataset_io import (
        OfflineLerobotEnv,
        detect_dataset_format,
    )

    out = resolve_dataset_dir(path)
    if not out.is_dir():
        raise FileNotFoundError(f"目录不存在: {out}")
    fmt = detect_dataset_format(out)
    if fmt == "lerobot":
        env = OfflineLerobotEnv(out)
        print(f"[dataset] opened lerobot {out} ({env.replay_buffer.n_episodes} eps)")
        return env
    if fmt == "zarr":
        env = OfflineDatasetEnv(out)
        try:
            check_episode_alignment(env)
        except RuntimeError as exc:
            print(f"[dataset] alignment warning ({out}): {exc}")
        return env
    raise FileNotFoundError(
        f"未识别的数据目录（需要 replay_buffer.zarr 或 meta/info.json）: {out}"
    )


def apply_manage_dataset(web_state, env, *, message: str = "") -> None:
    """把数据管理页切到给定 OfflineDatasetEnv / OfflineLerobotEnv。"""
    from utils.dagger_extract import inspect_intervention

    web_state.env = env
    web_state.output_dir = Path(env.output_dir)
    web_state.viewer_only = True
    if hasattr(web_state, "clear_live_series"):
        web_state.clear_live_series()
    n_ep = int(env.replay_buffer.n_episodes)
    fmt = str(getattr(env, "dataset_format", "zarr") or "zarr")
    has_inter = False
    dagger_stats: dict = {"has_intervention": False}
    if fmt == "zarr":
        try:
            dagger_stats = inspect_intervention(env.output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[dataset] intervention inspect failed: {exc}")
            dagger_stats = {"has_intervention": False, "error": str(exc)}
        has_inter = bool(dagger_stats.get("has_intervention"))
    extra = ""
    if has_inter:
        extra = (
            f" · DAgger intervention"
            f"（human_eps={dagger_stats.get('n_human_eps', '?')}"
            f" policy={dagger_stats.get('n_policy_only', '?')}）"
        )
    fmt_tag = " · LeRobot" if fmt == "lerobot" else " · zarr"
    web_state.update(
        mode="VIEW",
        viewer_only=True,
        recording=False,
        zero_force_active=False,
        delete_pending=None,
        n_episodes=n_ep,
        n_steps=0,
        t_elapsed=0.0,
        has_intervention=has_inter,
        dagger_stats=dagger_stats,
        dataset_format=fmt,
        message=message
        or f"已加载 {env.output_dir} · {n_ep} episodes{fmt_tag}{extra}",
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
    print(
        f"[dataset] loaded {env.output_dir} "
        f"({env.replay_buffer.n_episodes} eps, "
        f"{getattr(env, 'dataset_format', 'zarr')})"
    )
    return True


def schedule_dagger_split(
    src_dir: str | Path,
    *,
    what: str = "both",
    human_out: str | Path | None = None,
    policy_out: str | Path | None = None,
    overwrite: bool = False,
    trim_still: bool = True,
    web_state=None,
    done_mode: str = "VIEW",
) -> bool:
    """后台把 DAgger 数据拆成人工示教 / 纯模型目录。"""
    from utils.dagger_extract import default_split_paths, extract_dagger_split

    src = Path(src_dir).expanduser().resolve()
    if not (src / "replay_buffer.zarr").exists():
        msg = f"缺少 replay_buffer.zarr: {src}"
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False
    if not (src / "intervention").is_dir():
        msg = f"缺少 intervention/: {src}（非 DAgger 数据？）"
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False

    if not _DELETE_LOCK.acquire(blocking=False):
        msg = "正在处理数据（删除/裁剪/拆分），请稍候…"
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False

    d_h, d_p = default_split_paths(src)
    human_path = Path(human_out).expanduser().resolve() if human_out else d_h
    policy_path = Path(policy_out).expanduser().resolve() if policy_out else d_p
    what_s = str(what).lower().strip() or "both"

    if web_state is not None:
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_pending=None,
            delete_progress={
                "phase": "start",
                "frac": 0.0,
                "message": f"DAgger 拆分（{what_s}）启动…",
            },
            message=f"DAgger 拆分（{what_s}）启动…",
        )

    def _on_progress(info: dict) -> None:
        if web_state is None or not isinstance(info, dict):
            return
        payload = dict(info)
        msg = str(payload.get("message") or "DAgger 拆分中…")
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_progress=payload,
            message=msg,
        )

    def _worker() -> None:
        ok = False
        err = ""
        result = None
        try:
            print(
                f"[dagger-split] {src} what={what_s} "
                f"human={human_path} policy={policy_path} "
                f"overwrite={overwrite} trim_still={trim_still}"
            )
            result = extract_dagger_split(
                src,
                dst_human=human_path,
                dst_policy=policy_path,
                what=what_s,
                overwrite=bool(overwrite),
                trim_still=bool(trim_still),
                progress_cb=_on_progress,
            )
            ok = True
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            print(f"[dagger-split] failed: {exc}")
            ok = False
        finally:
            if web_state is not None:
                if ok and isinstance(result, dict):
                    parts = []
                    if result.get("human"):
                        h = result["human"]
                        parts.append(
                            f"human {h['n_episodes']}ep/{h['n_steps']}steps → {h['dst']}"
                        )
                    if result.get("policy"):
                        p = result["policy"]
                        parts.append(
                            f"policy {p['n_episodes']}ep/{p['n_steps']}steps → {p['dst']}"
                        )
                    msg = "拆分完成 · " + (" · ".join(parts) if parts else "无输出")
                    if result.get("unlabeled_episodes"):
                        msg += f" · 跳过无标签 ep{result['unlabeled_episodes']}"
                else:
                    msg = f"拆分失败: {err}" if err else "拆分失败"
                web_state.update(
                    mode=done_mode,
                    deleting=False,
                    delete_pending=None,
                    delete_progress=None,
                    message=msg,
                )
            _DELETE_LOCK.release()

    threading.Thread(
        target=_worker, name=f"dagger-split-{what_s}", daemon=True
    ).start()
    return True


def handle_dagger_split_web_cmd(
    web_state,
    cmd: dict,
    *,
    done_mode: str = "VIEW",
) -> bool:
    """处理 dagger_split_request / dagger_split_confirm。"""
    action = cmd.get("action")
    env = web_state.env if web_state is not None else None
    if env is None:
        if web_state is not None:
            web_state.update(message="尚未加载数据目录")
        return False
    src = Path(env.output_dir)

    if action == "dagger_split_request":
        if is_delete_busy():
            if web_state is not None:
                web_state.update(message="正在处理数据，请稍候…")
            return False
        what = str(cmd.get("what", "both")).lower()
        if web_state is not None:
            web_state.update(
                message=f"confirm dagger split ({what}) from {src.name}",
            )
        print(f"Confirm DAgger split ({what}) from {src}? (web Yes)")
        return True

    if action == "dagger_split_confirm":
        if is_delete_busy():
            if web_state is not None:
                web_state.update(message="正在处理数据，请稍候…")
            return False
        what = str(cmd.get("what", "both")).lower()
        human_out = cmd.get("human_out") or None
        policy_out = cmd.get("policy_out") or None
        overwrite = bool(cmd.get("overwrite", False))
        trim_still = bool(cmd.get("trim_still", True))
        return bool(
            schedule_dagger_split(
                src,
                what=what,
                human_out=human_out,
                policy_out=policy_out,
                overwrite=overwrite,
                trim_still=trim_still,
                web_state=web_state,
                done_mode=done_mode,
            )
        )
    return False


def schedule_convert_to_lerobot(
    src_dir: str | Path,
    *,
    out_dir: str | Path,
    fps: int = 20,
    prompt: str = "Flatten and fold the cloth.",
    overwrite: bool = False,
    web_state=None,
    done_mode: str = "VIEW",
) -> bool:
    """后台把当前 zarr 数据集转成 LeRobot；完成后自动加载输出目录。"""
    from modules.convert_zarr_bimanual_to_lerobot import (
        convert_one_zarr,
        find_zarr,
    )

    src = Path(src_dir).expanduser().resolve()
    dst = Path(out_dir).expanduser().resolve()
    try:
        zpath = find_zarr(src)
    except FileNotFoundError as exc:
        msg = str(exc)
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False

    if not _DELETE_LOCK.acquire(blocking=False):
        msg = "正在处理数据（删除/裁剪/转换），请稍候…"
        print(msg)
        if web_state is not None:
            web_state.update(message=msg)
        return False

    if web_state is not None:
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_pending=None,
            delete_progress={
                "phase": "convert",
                "frac": 0.0,
                "message": f"转换为 LeRobot → {dst}…",
            },
            message=f"转换为 LeRobot → {dst}…",
        )

    def _on_progress(info: dict) -> None:
        if web_state is None or not isinstance(info, dict):
            return
        payload = dict(info)
        msg = str(payload.get("message") or "转换中…")
        web_state.update(
            mode="DELETING",
            deleting=True,
            delete_progress=payload,
            message=msg,
        )

    def _worker() -> None:
        ok = False
        err = ""
        n_eps = 0
        n_frames = 0
        try:
            print(
                f"[convert-lerobot] {zpath} → {dst} "
                f"fps={fps} overwrite={overwrite}"
            )
            n_eps, n_frames = convert_one_zarr(
                zpath,
                dst,
                prompt=str(prompt),
                fps=int(fps),
                overwrite=bool(overwrite),
                progress_cb=_on_progress,
            )
            ok = True
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            print(f"[convert-lerobot] failed: {exc}")
            ok = False
        finally:
            if web_state is not None:
                if ok:
                    try:
                        env = open_offline_dataset(dst)
                        apply_manage_dataset(
                            web_state,
                            env,
                            message=(
                                f"已转换为 LeRobot · {n_eps} ep / {n_frames} frames "
                                f"→ {dst}"
                            ),
                        )
                        web_state.update(
                            deleting=False,
                            delete_pending=None,
                            delete_progress=None,
                        )
                    except Exception as load_exc:  # noqa: BLE001
                        web_state.update(
                            mode=done_mode,
                            deleting=False,
                            delete_pending=None,
                            delete_progress=None,
                            message=(
                                f"转换完成但加载失败: {load_exc} · 输出在 {dst}"
                            ),
                        )
                else:
                    web_state.update(
                        mode=done_mode,
                        deleting=False,
                        delete_pending=None,
                        delete_progress=None,
                        message=f"转换失败: {err}" if err else "转换失败",
                    )
            _DELETE_LOCK.release()

    threading.Thread(
        target=_worker, name="convert-lerobot", daemon=True
    ).start()
    return True


def handle_convert_lerobot_web_cmd(
    web_state,
    cmd: dict,
    *,
    done_mode: str = "VIEW",
) -> bool:
    """处理 convert_lerobot_request / convert_lerobot_confirm。"""
    from modules.convert_zarr_bimanual_to_lerobot import default_lerobot_out_dir

    action = cmd.get("action")
    env = web_state.env if web_state is not None else None
    if env is None:
        if web_state is not None:
            web_state.update(message="尚未加载数据目录")
        return False
    if getattr(env, "dataset_format", "zarr") != "zarr":
        if web_state is not None:
            web_state.update(message="当前已是 LeRobot，无需再转换")
        return False
    src = Path(env.output_dir)

    if action == "convert_lerobot_request":
        if is_delete_busy():
            if web_state is not None:
                web_state.update(message="正在处理数据，请稍候…")
            return False
        out_raw = cmd.get("out_dir") or str(default_lerobot_out_dir(src))
        if web_state is not None:
            web_state.update(
                message=f"confirm convert to LeRobot → {out_raw}",
            )
        print(f"Confirm convert {src} → LeRobot {out_raw}? (web Yes)")
        return True

    if action == "convert_lerobot_confirm":
        if is_delete_busy():
            if web_state is not None:
                web_state.update(message="正在处理数据，请稍候…")
            return False
        out_raw = cmd.get("out_dir") or str(default_lerobot_out_dir(src))
        try:
            fps = int(cmd.get("fps", 20))
        except (TypeError, ValueError):
            fps = 20
        prompt = str(cmd.get("prompt") or "Flatten and fold the cloth.")
        overwrite = bool(cmd.get("overwrite", False))
        return bool(
            schedule_convert_to_lerobot(
                src,
                out_dir=out_raw,
                fps=fps,
                prompt=prompt,
                overwrite=overwrite,
                web_state=web_state,
                done_mode=done_mode,
            )
        )
    return False


def handle_trim_web_cmd(
    web_state,
    cmd: dict,
    *,
    done_mode: str = "VIEW",
    hold_robot: bool = False,
    prep_hold_fn=None,
) -> bool:
    """处理 trim_request / trim_confirm。成功调度或接受请求返回 True。"""
    action = cmd.get("action")
    env = web_state.env if web_state is not None else None

    if action == "trim_request":
        if is_delete_busy():
            if web_state is not None:
                web_state.update(message="正在处理数据，请稍候…")
            return False
        if env is None:
            if web_state is not None:
                web_state.update(message="尚未加载数据目录")
            return False
        try:
            eid = int(cmd.get("episode_id"))
            start = int(cmd.get("start"))
            end = int(cmd.get("end"))
        except (TypeError, ValueError):
            if web_state is not None:
                web_state.update(message="裁剪参数无效（需要 episode_id/start/end）")
            return False
        fmt = getattr(env, "dataset_format", "zarr")
        rb = env.replay_buffer
        if fmt == "lerobot":
            if not getattr(rb, "has_episode", lambda _e: False)(eid):
                if web_state is not None:
                    web_state.update(message=f"invalid episode_id {eid}")
                return False
        else:
            n = int(rb.n_episodes)
            if eid < 0 or eid >= n:
                if web_state is not None:
                    web_state.update(message=f"invalid episode_id {eid}")
                return False
        sl = rb.get_episode_slice(eid)
        T = int(sl.stop - sl.start)
        if start < 0 or end > T or start >= end:
            if web_state is not None:
                web_state.update(
                    message=f"裁剪区间无效 [{start},{end}) · ep{eid} 长度 {T}"
                )
            return False
        if web_state is not None:
            web_state.update(
                delete_pending=eid,
                message=(
                    f"confirm trim ep{eid} keep [{start},{end}) "
                    f"({end - start}/{T} steps)"
                ),
            )
        print(
            f"Confirm trim episode {eid} keep [{start}, {end}) "
            f"({end - start}/{T})? (web Yes)"
        )
        return True

    if action == "trim_confirm":
        if is_delete_busy():
            if web_state is not None:
                web_state.update(message="正在处理数据，请稍候…")
            return False
        if env is None:
            return False
        try:
            eid = int(cmd.get("episode_id", getattr(web_state, "delete_pending", -1)))
            start = int(cmd.get("start"))
            end = int(cmd.get("end"))
        except (TypeError, ValueError):
            if web_state is not None:
                web_state.update(message="裁剪确认参数无效")
            return False
        if prep_hold_fn is not None:
            try:
                prep_hold_fn()
            except Exception as exc:  # noqa: BLE001
                print(f"trim prep_hold warning: {exc}")
        return bool(
            schedule_trim_episode(
                env,
                eid,
                start,
                end,
                web_state=web_state,
                hold_robot=hold_robot,
                done_mode=done_mode,
            )
        )

    return False


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
    print("  controls   : browse / scrub / delete / trim / load other folder")
    print("  trim       : web Mark In/Out → Apply（[ ] T）")
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
                    if is_delete_busy():
                        web_state.update(message="正在删除中，无法取消")
                    else:
                        web_state.update(
                            delete_pending=None, message="delete cancelled"
                        )
                    continue
                if action == "delete_request":
                    if is_delete_busy():
                        web_state.update(message="正在删除中，请稍候…")
                        continue
                    env = web_state.env
                    if env is None:
                        web_state.update(message="尚未加载数据目录")
                        continue
                    if getattr(env, "dataset_format", "zarr") == "lerobot":
                        web_state.update(
                            message="LeRobot 模式暂不支持删除 episode（可用裁剪）"
                        )
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
                    if getattr(env, "dataset_format", "zarr") == "lerobot":
                        web_state.update(
                            message="LeRobot 模式暂不支持删除 episode"
                        )
                        continue
                    eid = cmd.get("episode_id", web_state.delete_pending)
                    if eid is not None:
                        schedule_delete_episode(
                            env,
                            int(eid),
                            web_state=web_state,
                            hold_robot=False,
                            done_mode="VIEW",
                        )
                    continue
                if action in ("trim_request", "trim_confirm"):
                    handle_trim_web_cmd(
                        web_state, cmd, done_mode="VIEW", hold_robot=False
                    )
                    continue
                if action in ("dagger_split_request", "dagger_split_confirm"):
                    handle_dagger_split_web_cmd(
                        web_state, cmd, done_mode="VIEW"
                    )
                    continue
                if action in (
                    "convert_lerobot_request",
                    "convert_lerobot_confirm",
                ):
                    handle_convert_lerobot_web_cmd(
                        web_state, cmd, done_mode="VIEW"
                    )
                    continue
                if action in ("drag_toggle", "start", "save", "drop"):
                    web_state.update(
                        message="viewer_only: collect controls disabled"
                    )
            # 删除/裁剪/拆分线程占用 zarr 时不要改 mode / 读 n_episodes
            if is_delete_busy() or bool(getattr(web_state, "deleting", False)):
                web_state.update(mode="DELETING", deleting=True, viewer_only=True)
            else:
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
        has_intervention=False,
        dagger_stats=None,
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
                            f"{collect_kwargs.get('robot_ports')}",
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
                        except RuntimeError as exc:
                            # 网页「取消连接」属于正常返回 hub，不要当启动失败刷栈
                            if "用户取消连接" in str(exc):
                                print("[hub] 用户取消连接 — back to hub.")
                                _web_log(web_state, "用户取消连接，已返回入口")
                            else:
                                collect_err = f"数采启动失败: {exc}"
                                print(f"[hub] {collect_err}")
                                _web_log(web_state, collect_err)
                                import traceback

                                traceback.print_exc()
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
                    if is_delete_busy():
                        web_state.update(message="正在删除中，无法取消")
                    else:
                        web_state.update(
                            delete_pending=None, message="delete cancelled"
                        )
                    continue
                if action == "delete_request":
                    if is_delete_busy():
                        web_state.update(message="正在删除中，请稍候…")
                        continue
                    env = web_state.env
                    if env is None:
                        web_state.update(message="尚未加载数据目录")
                        continue
                    if getattr(env, "dataset_format", "zarr") == "lerobot":
                        web_state.update(
                            message="LeRobot 模式暂不支持删除 episode（可用裁剪）"
                        )
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
                    if getattr(env, "dataset_format", "zarr") == "lerobot":
                        web_state.update(
                            message="LeRobot 模式暂不支持删除 episode"
                        )
                        continue
                    eid = cmd.get("episode_id", web_state.delete_pending)
                    if eid is not None:
                        schedule_delete_episode(
                            env,
                            int(eid),
                            web_state=web_state,
                            hold_robot=False,
                            done_mode="VIEW",
                        )
                    continue
                if action in ("trim_request", "trim_confirm"):
                    handle_trim_web_cmd(
                        web_state, cmd, done_mode="VIEW", hold_robot=False
                    )
                    continue
                if action in ("dagger_split_request", "dagger_split_confirm"):
                    handle_dagger_split_web_cmd(
                        web_state, cmd, done_mode="VIEW"
                    )
                    continue
                if action in (
                    "convert_lerobot_request",
                    "convert_lerobot_confirm",
                ):
                    handle_convert_lerobot_web_cmd(
                        web_state, cmd, done_mode="VIEW"
                    )
                    continue
                if is_delete_busy() and action in (
                    "load_dataset",
                    "back_hub",
                    "enter_collect",
                ):
                    web_state.update(message="正在处理数据，请稍候…")
                    continue
                if action in ("drag_toggle", "start", "save", "drop", "enter_collect"):
                    web_state.update(
                        message="viewer: collect controls disabled（返回入口后再选数采）"
                    )
                    continue
                if action == "enter_manage":
                    continue

            if phase == "manage":
                if is_delete_busy() or bool(getattr(web_state, "deleting", False)):
                    # 保持 DELETING，让网页能刷到 delete_progress / message
                    web_state.update(mode="DELETING", deleting=True, viewer_only=True)
                else:
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
    robot_ports,
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
    rs_serial=None,
    rs_width=640,
    rs_height=480,
    rs_fps=30,
    save_depth=True,
    rs_color_temperature=0.0,
    rs_color_saturation=1.0,
    rs_color_contrast=1.0,
    rs_color_hue_shift=12.0,
    web_state=None,
    web_cmds=None,
    web_server=None,
    grip_gesture_robot=1,
    grip_gesture_window=1.5,
    no_grip_gesture=False,
    replay_approach=2.5,
    drag_profile="light",
) -> None:
    """连接机械臂 + 相机，跑一轮数采会话；Q / 网页 quit 后返回（不退出进程）。

    若传入 web_state/web_cmds/web_server（hub 复用），则不另起端口，并在连接阶段写 logs。
    """
    # 须在 spawn 控制器进程前写入 env，子进程才能读到 soft-hold 手感。
    apply_drag_profile(drag_profile)

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
    if isinstance(robot_ports, str):
        robot_ports_list = parse_robot_ports(robot_ports)
    else:
        robot_ports_list = [int(p) for p in robot_ports]
    robots_config = _default_robots_config(
        robot_ip=robot_ip, robot_ports=robot_ports_list
    )
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
    print("ARX5 Bimanual Drag Collection (ZMQ + GoPro×2 + D435)")
    print(f"  output     : {output}")
    print(f"  zmq        : {robot_ip}:{robot_ports_list}  (left,right)")
    print(f"  frequency  : {frequency} Hz")
    print(f"  n_robots   : {n_robots}")
    print(f"  wrist cams : reorder={camera_reorder_list} -> camera0/camera2")
    print(
        f"  D435       : serial={rs_serial!r} "
        f"{rs_width}x{rs_height}@{rs_fps} depth={save_depth} -> camera1"
    )
    print(
        f"  D435 grade : temp={rs_color_temperature:.2f} "
        f"sat={rs_color_saturation:.2f} contrast={rs_color_contrast:.2f} "
        f"hue={rs_color_hue_shift:.1f}"
    )
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
    print(f"  drag_profile: {get_drag_profile()}  (soft-hold; true GC ignores)")
    print(f"  no_zero_force: {no_zero_force}")
    if no_zero_force:
        print("  startup: reset_to_home → optional init → C to record (P optional)")
    else:
        print("  startup: reset_to_home → optional init → P then C")
    print(
        "  P=toggle zero-force  C=start  S=save  D=drop  E=del  "
        "R=full-replay  J=step  0/1/2=replay-arm(L/R/both)  X=E-Stop  Q=quit"
    )
    print(f"  replay_approach: {float(replay_approach):.1f}s")
    if no_grip_gesture:
        print("  grip_gesture: OFF")
    else:
        print(
            f"  grip_gesture: robot{int(grip_gesture_robot)} (can1) "
            f"double open-close within {float(grip_gesture_window):.2f}s "
            "→ start / save (trimmed); S/gesture save → init then drag"
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
        _web_log(
            web_state,
            f"配置: {frequency} Hz · wrist={camera_reorder} · D435 depth={save_depth}",
        )
        _web_log(web_state, f"ZMQ 地址: {robot_ip}:{robot_ports_list}")
        _web_log(web_state, "正在创建 Arx5EnvBimanualRS（双臂 + GoPro×2 + D435）…")

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
                rs_serial=rs_serial,
                rs_resolution=(int(rs_width), int(rs_height)),
                rs_fps=int(rs_fps),
                enable_depth=bool(save_depth),
                rs_color_temperature=float(rs_color_temperature),
                rs_color_saturation=float(rs_color_saturation),
                rs_color_contrast=float(rs_color_contrast),
                rs_color_hue_shift=float(rs_color_hue_shift),
            ) as env:
                key_source = term_keys if key_source_is_term else key_counter

                cv2.setNumThreads(2)
                _web_log(web_state, "等待相机 / env ready…")
                time.sleep(1.0)
                wait_t0 = time.monotonic()
                last_wait_log = 0.0
                ready_timeout_s = 60.0

                def _ready_status() -> str:
                    """只读诊断：哪一块还没 ready（不发任何运动指令）。"""
                    parts = []
                    try:
                        cam_ok = bool(env.camera.is_ready)
                    except Exception as exc:  # noqa: BLE001
                        cam_ok = False
                        parts.append(f"camera=?({exc})")
                    else:
                        parts.append(f"camera={'OK' if cam_ok else 'WAIT'}")
                    for i, robot in enumerate(getattr(env, "robots", []) or []):
                        try:
                            ok = bool(robot.is_ready)
                            alive = bool(robot.is_alive()) if hasattr(robot, "is_alive") else True
                        except Exception as exc:  # noqa: BLE001
                            parts.append(f"robot{i}=?({exc})")
                            continue
                        port = "?"
                        try:
                            cfg = env.robots_config[i]
                            port = str(cfg.get("robot_port", "?"))
                        except Exception:
                            pass
                        if ok:
                            parts.append(f"robot{i}(:{port})=OK")
                        elif not alive:
                            parts.append(f"robot{i}(:{port})=DEAD")
                        else:
                            parts.append(
                                f"robot{i}(:{port})=WAIT"
                                "(卡在 GET_STATE/启动，不是相机)"
                            )
                    return ", ".join(parts)

                while not env.is_ready:
                    if web_cmds is not None:
                        for cmd in web_cmds.drain():
                            if cmd.get("action") == "quit":
                                _web_log(web_state, "用户取消连接")
                                raise RuntimeError("用户取消连接")
                    now = time.monotonic()
                    if now - wait_t0 >= ready_timeout_s:
                        detail = _ready_status()
                        raise RuntimeError(
                            f"等待 env ready 超时 ({ready_timeout_s:.0f}s): {detail}"
                        )
                    if now - last_wait_log >= 2.0:
                        detail = _ready_status()
                        _web_log(
                            web_state,
                            f"仍在等待 env ready… ({now - wait_t0:.0f}s) · {detail}",
                        )
                        print(f"[ready] {detail}", flush=True)
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

                # 与 fast_wam 对齐的启动微调；各臂目标记入 init_targets（P 退出零力回此处）
                init_targets: list[tuple[np.ndarray, float | None]] | None = None
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
                    _web_log(web_state, "移动到绝对 init_pose（两臂同一姿态）…")
                    move_to_init_pose(
                        env,
                        abs_init_pose6,
                        frequency=frequency,
                        duration=init_duration,
                        init_gripper=gripper_target,
                    )
                    if gripper_target is None and not keep_gripper:
                        gripper_target = float(
                            env.get_robot_state()[0]["gripper_position"]
                        )
                    init_targets = [
                        (abs_init_pose6.copy(), gripper_target)
                        for _ in range(n_robots)
                    ]
                else:
                    _web_log(web_state, f"按 init_delta 微调（{init_duration:.1f}s）…")
                    init_targets = apply_init_delta_like_fast_wam(
                        env,
                        init_delta=init_delta,
                        init_gripper_delta=float(init_gripper_delta),
                        frequency=frequency,
                        duration=init_duration,
                        gripper_min_width=float(gripper_min_width),
                        gripper_max_width=float(gripper_max_width),
                    )
                    if not keep_gripper and init_targets:
                        gripper_target = float(init_targets[0][1])

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
                replay_mode = None  # None | "input"
                replay_buf = ""
                # 单步回放会话：{'episode_id', 'actions', 'idx'}；idx=当前已到位的帧
                step_replay: dict | None = None
                # None=双臂；[0]=左；[1]=右
                replay_arm_sel: list[int] | None = None
                estop_requested = False

                grip_robot_idx = int(grip_gesture_robot)
                if grip_robot_idx < 0 or grip_robot_idx >= n_robots:
                    raise ValueError(
                        f"--grip_gesture_robot={grip_robot_idx} out of range "
                        f"(n_robots={n_robots})"
                    )
                grip_span = float(gripper_max_width) - float(gripper_min_width)
                grip_detector = None
                if not no_grip_gesture:
                    grip_detector = GripperDoubleOpenCloseDetector(
                        close_thresh=float(gripper_min_width) + 0.25 * grip_span,
                        open_thresh=float(gripper_min_width) + 0.75 * grip_span,
                        window_s=float(grip_gesture_window),
                        cooldown_s=1.0,
                    )
                    print(
                        f"Gripper gesture ON: robot{grip_robot_idx} "
                        f"(close≤{grip_detector.close_thresh:.3f} "
                        f"open≥{grip_detector.open_thresh:.3f} m). "
                        "In drag: double open-close to start; again to save."
                    )

                if no_zero_force:
                    print("Direct record mode: press C to start (P optional for drag).")

                def _ensure_end_control_for_replay() -> None:
                    nonlocal zero_force_active
                    if zero_force_active:
                        leave_drag_stay_pose(env)
                        zero_force_active = False
                        if grip_detector is not None:
                            grip_detector.reset()

                def _set_replay_arm(sel: list[int] | None) -> None:
                    nonlocal replay_arm_sel, step_replay
                    replay_arm_sel = sel
                    step_replay = None  # 换臂后重新从起点单步
                    arms = _format_replay_arms(sel, n_robots)
                    print(f"Replay arm select: {arms}")
                    if web_state is not None:
                        web_state.update(
                            message=f"回放臂: {arms}",
                            replay_arm=arms,
                        )

                def poll_abort() -> bool:
                    """回放过程中轮询急停（键盘 X / 网页 estop）。"""
                    nonlocal estop_requested, zero_force_active, step_replay
                    if estop_requested:
                        return True
                    if web_cmds is not None:
                        for cmd in web_cmds.drain():
                            if cmd.get("action") == "estop":
                                estop_requested = True
                                step_replay = None
                                zero_force_active = False
                                return True
                            if cmd.get("action") == "quit":
                                estop_requested = True
                                return True
                    for key_stroke in drain_keys(key_source):
                        if key_stroke == KeyCode(char="x"):
                            estop_requested = True
                            step_replay = None
                            zero_force_active = False
                            return True
                        if key_stroke == KeyCode(char="q"):
                            estop_requested = True
                            return True
                    return False

                def do_estop() -> None:
                    nonlocal estop_requested, step_replay, zero_force_active
                    estop_requested = True
                    step_replay = None
                    zero_force_active = False
                    emergency_stop(env)
                    if web_state is not None:
                        web_state.update(
                            mode="READY" if no_zero_force else "IDLE",
                            zero_force_active=False,
                            recording=False,
                            message="急停：已钉住当前位姿",
                        )

                def do_replay(episode_id: int | None = None) -> None:
                    """Exit drag if needed, then full-replay a saved episode (blocking)."""
                    nonlocal zero_force_active, replay_mode, replay_buf, step_replay
                    nonlocal estop_requested
                    n_saved = env.replay_buffer.n_episodes
                    if n_saved <= 0:
                        print("No saved episode to replay.")
                        return
                    if episode_id is None:
                        episode_id = n_saved - 1
                    episode_id = int(episode_id)
                    if episode_id < 0 or episode_id >= n_saved:
                        print(
                            f"Episode {episode_id} does not exist "
                            f"(valid: 0..{n_saved - 1})."
                        )
                        return
                    replay_mode = None
                    replay_buf = ""
                    step_replay = None
                    estop_requested = False
                    _ensure_end_control_for_replay()
                    arms = _format_replay_arms(replay_arm_sel, n_robots)
                    ok = replay_saved_episode(
                        env,
                        episode_id,
                        frequency=frequency,
                        approach_duration=float(replay_approach),
                        robot_indices=replay_arm_sel,
                        poll_abort=poll_abort,
                        web_state=web_state,
                    )
                    aborted = estop_requested
                    estop_requested = False
                    if web_state is not None:
                        web_state.update(
                            mode="READY" if no_zero_force else "IDLE",
                            zero_force_active=False,
                            recording=False,
                            n_episodes=int(env.replay_buffer.n_episodes),
                            message=(
                                f"回放 ep{episode_id} 急停中止"
                                if aborted
                                else (
                                    f"回放 ep{episode_id} [{arms}] 完成"
                                    if ok
                                    else f"回放 ep{episode_id} 失败"
                                )
                            ),
                        )

                def do_step_replay(episode_id: int | None = None) -> None:
                    """J：首次进入单步会话并到第 0 帧；之后每次前进 1 帧。"""
                    nonlocal zero_force_active, step_replay, replay_mode, replay_buf
                    nonlocal estop_requested
                    replay_mode = None
                    replay_buf = ""
                    estop_requested = False
                    arms = _format_replay_arms(replay_arm_sel, n_robots)

                    if step_replay is None:
                        n_saved = env.replay_buffer.n_episodes
                        if n_saved <= 0:
                            print("No saved episode to step-replay.")
                            return
                        if episode_id is None:
                            episode_id = n_saved - 1
                        episode_id = int(episode_id)
                        loaded = load_episode_actions(env, episode_id)
                        if loaded is None:
                            return
                        actions, rel = loaded
                        _ensure_end_control_for_replay()
                        begin_replay_live_series(
                            web_state,
                            episode_id=episode_id,
                            message=(
                                f"单步 ep{episode_id}[{arms}] → 起点 "
                                f"(0/{len(actions) - 1})"
                            ),
                        )
                        ok = replay_step_to_index(
                            env,
                            actions,
                            0,
                            frequency=frequency,
                            step_duration=float(replay_approach),
                            robot_indices=replay_arm_sel,
                            poll_abort=poll_abort,
                            web_state=web_state,
                            t_elapsed=float(rel[0]) if len(rel) else 0.0,
                        )
                        if not ok:
                            if web_state is not None:
                                web_state.update(
                                    mode="READY" if no_zero_force else "IDLE",
                                    message=(
                                        f"单步 ep{episode_id} "
                                        + ("急停" if estop_requested else "起点失败")
                                    ),
                                )
                            estop_requested = False
                            return
                        step_replay = {
                            "episode_id": episode_id,
                            "actions": actions,
                            "rel": rel,
                            "idx": 0,
                        }
                        print(
                            f"Step replay ep{episode_id} [{arms}]: "
                            f"at 0/{len(actions) - 1}. "
                            "J=next, Esc=leave, X=E-Stop."
                        )
                        if web_state is not None:
                            # 保持 REPLAY：网页继续 Follow LIVE 刷轨迹
                            web_state.update(
                                mode="REPLAY",
                                message=(
                                    f"单步 ep{episode_id}[{arms}]: "
                                    f"0/{len(actions) - 1} · J下一步"
                                ),
                            )
                        return

                    actions = step_replay["actions"]
                    rel = step_replay.get("rel")
                    eid = int(step_replay["episode_id"])
                    nxt = int(step_replay["idx"]) + 1
                    if nxt >= len(actions):
                        print(
                            f"Step replay ep{eid} finished "
                            f"({len(actions)} frames). Press J to restart last ep."
                        )
                        step_replay = None
                        if web_state is not None:
                            web_state.update(
                                mode="READY" if no_zero_force else "IDLE",
                                message=f"单步回放 ep{eid} 结束",
                            )
                        return

                    _ensure_end_control_for_replay()
                    if web_state is not None:
                        web_state.update(
                            mode="REPLAY",
                            message=(
                                f"单步 ep{eid}[{arms}]: "
                                f"{nxt}/{len(actions) - 1}…"
                            ),
                        )
                    t_ep = None
                    if rel is not None and nxt < len(rel):
                        t_ep = float(rel[nxt])
                    ok = replay_step_to_index(
                        env,
                        actions,
                        nxt,
                        frequency=frequency,
                        step_duration=0.12,
                        robot_indices=replay_arm_sel,
                        poll_abort=poll_abort,
                        web_state=web_state,
                        t_elapsed=t_ep,
                    )
                    if not ok:
                        if estop_requested:
                            step_replay = None
                            estop_requested = False
                        return
                    step_replay["idx"] = nxt
                    print(
                        f"Step replay ep{eid} [{arms}]: {nxt}/{len(actions) - 1}"
                    )
                    if web_state is not None:
                        web_state.update(
                            mode="REPLAY",
                            message=(
                                f"单步 ep{eid}[{arms}]: "
                                f"{nxt}/{len(actions) - 1} · J下一步"
                            ),
                        )

                def handle_idle_web_cmds() -> tuple[bool, bool]:
                    """Returns (start_episode, should_quit)."""
                    nonlocal zero_force_active, delete_mode, delete_buf, delete_target
                    nonlocal pose_print_enabled, last_pose_print_t
                    nonlocal replay_mode, replay_buf, step_replay, replay_arm_sel
                    if web_cmds is None:
                        return False, False
                    start = False
                    for cmd in web_cmds.drain():
                        action = cmd.get("action")
                        if action == "quit":
                            return False, True
                        if action == "estop":
                            do_estop()
                            continue
                        if action == "set_replay_arm":
                            which = str(cmd.get("arm", "both")).lower()
                            if which in ("0", "l", "left", "robot0"):
                                _set_replay_arm([0])
                            elif which in ("1", "r", "right", "robot1"):
                                _set_replay_arm([1])
                            else:
                                _set_replay_arm(None)
                            continue
                        if action == "delete_cancel":
                            print("Delete cancelled.")
                            delete_mode = None
                            delete_buf = ""
                            delete_target = None
                            continue
                        if action == "delete_confirm":
                            if is_delete_busy():
                                print("Delete already in progress — ignore.")
                                if web_state is not None:
                                    web_state.update(
                                        message="正在删除中，请稍候…"
                                    )
                                continue
                            tid = cmd.get("episode_id", delete_target)
                            if tid is not None:
                                # 先退出拖拽并钉住，再后台删，避免主循环卡住掉臂
                                if zero_force_active:
                                    leave_drag_stay_pose(env)
                                    zero_force_active = False
                                    if grip_detector is not None:
                                        grip_detector.reset()
                                elif hasattr(env, "set_end_control"):
                                    try:
                                        env.set_end_control()
                                    except Exception as exc:  # noqa: BLE001
                                        print(
                                            f"set_end_control before delete: {exc}"
                                        )
                                schedule_delete_episode(
                                    env,
                                    int(tid),
                                    web_state=web_state,
                                    hold_robot=False,  # 主线程已钉住
                                    done_mode=(
                                        "READY" if no_zero_force else "IDLE"
                                    ),
                                )
                            delete_mode = None
                            delete_buf = ""
                            delete_target = None
                            continue
                        if action == "delete_request":
                            if is_delete_busy():
                                if web_state is not None:
                                    web_state.update(
                                        message="正在删除中，请稍候…"
                                    )
                                continue
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
                        if action in ("trim_request", "trim_confirm"):
                            def _prep_hold_for_trim() -> None:
                                nonlocal zero_force_active
                                if zero_force_active:
                                    leave_drag_stay_pose(env)
                                    zero_force_active = False
                                    if grip_detector is not None:
                                        grip_detector.reset()
                                elif hasattr(env, "set_end_control"):
                                    try:
                                        env.set_end_control()
                                    except Exception as exc:  # noqa: BLE001
                                        print(
                                            f"set_end_control before trim: {exc}"
                                        )

                            handle_trim_web_cmd(
                                web_state,
                                cmd,
                                done_mode=(
                                    "READY" if no_zero_force else "IDLE"
                                ),
                                hold_robot=False,
                                prep_hold_fn=(
                                    _prep_hold_for_trim
                                    if action == "trim_confirm"
                                    else None
                                ),
                            )
                            continue
                        if action in (
                            "dagger_split_request",
                            "dagger_split_confirm",
                        ):
                            handle_dagger_split_web_cmd(
                                web_state,
                                cmd,
                                done_mode=(
                                    "READY" if no_zero_force else "IDLE"
                                ),
                            )
                            continue
                        if action in (
                            "convert_lerobot_request",
                            "convert_lerobot_confirm",
                        ):
                            handle_convert_lerobot_web_cmd(
                                web_state,
                                cmd,
                                done_mode=(
                                    "READY" if no_zero_force else "IDLE"
                                ),
                            )
                            continue
                        if is_delete_busy() and action not in (
                            "quit",
                            "estop",
                            "delete_cancel",
                        ):
                            if web_state is not None:
                                web_state.update(
                                    message="正在处理数据，请稍候…"
                                )
                            continue
                        if delete_mode == "confirm" and action not in (
                            "delete_confirm",
                            "delete_cancel",
                        ):
                            continue
                        if action == "drag_toggle":
                            if not zero_force_active:
                                enter_drag_mode(env)
                                zero_force_active = True
                                if grip_detector is not None:
                                    grip_detector.reset()
                                    print(
                                        f"Double open-close robot{grip_robot_idx} "
                                        "gripper to START recording "
                                        "(or press C)."
                                    )
                            else:
                                leave_zero_force_to_init(
                                    env,
                                    init_targets,
                                    frequency=frequency,
                                    duration=init_duration,
                                )
                                zero_force_active = False
                                if grip_detector is not None:
                                    grip_detector.reset()
                        elif action == "start":
                            if zero_force_active or no_zero_force:
                                start = True
                                step_replay = None
                                if grip_detector is not None:
                                    grip_detector.reset()
                            else:
                                print("Press P / Drag first to enter zero-force mode.")
                        elif action == "save":
                            print("Not recording — nothing to save.")
                        elif action == "drop":
                            print("No active episode to drop.")
                        elif action == "replay":
                            eid = cmd.get("episode_id")
                            do_replay(None if eid is None else int(eid))
                        elif action == "replay_step":
                            eid = cmd.get("episode_id")
                            # 已有单步会话时忽略 episode_id，继续前进
                            if step_replay is not None:
                                do_step_replay(None)
                            else:
                                do_step_replay(None if eid is None else int(eid))
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
                                    if is_delete_busy():
                                        print("Delete already in progress.")
                                    else:
                                        if zero_force_active:
                                            leave_drag_stay_pose(env)
                                            zero_force_active = False
                                            if grip_detector is not None:
                                                grip_detector.reset()
                                        elif hasattr(env, "set_end_control"):
                                            try:
                                                env.set_end_control()
                                            except Exception as exc:  # noqa: BLE001
                                                print(
                                                    f"set_end_control before delete: {exc}"
                                                )
                                        schedule_delete_episode(
                                            env,
                                            int(delete_target),
                                            web_state=web_state,
                                            hold_robot=False,  # 主线程已钉住
                                            done_mode=(
                                                "READY"
                                                if no_zero_force
                                                else "IDLE"
                                            ),
                                        )
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

                            if replay_mode == "input":
                                ch = key_char(key_stroke)
                                if key_stroke == Key.esc:
                                    print("Replay cancelled.")
                                    replay_mode = None
                                    replay_buf = ""
                                elif key_stroke == Key.backspace:
                                    replay_buf = replay_buf[:-1]
                                elif ch is not None and ch.isdigit():
                                    replay_buf += ch
                                elif key_stroke == Key.enter:
                                    n_saved = env.replay_buffer.n_episodes
                                    if n_saved <= 0:
                                        print("No saved episode to replay.")
                                        replay_mode = None
                                        replay_buf = ""
                                    elif not replay_buf.strip():
                                        do_replay(n_saved - 1)
                                    else:
                                        episode_id = parse_episode_id(replay_buf)
                                        if episode_id is None:
                                            print("Type an episode ID (digits), or Enter for last.")
                                        elif episode_id < 0 or episode_id >= n_saved:
                                            print(
                                                f"Episode {episode_id} does not exist "
                                                f"(valid: 0..{n_saved - 1})."
                                            )
                                        else:
                                            do_replay(episode_id)
                                continue

                            if key_stroke == KeyCode(char="p"):
                                step_replay = None
                                if not zero_force_active:
                                    enter_drag_mode(env)
                                    zero_force_active = True
                                    if grip_detector is not None:
                                        grip_detector.reset()
                                    print(
                                        "Press P again to exit and return to init, "
                                        "or C to record."
                                    )
                                    if grip_detector is not None:
                                        print(
                                            f"Or double open-close robot{grip_robot_idx} "
                                            "gripper to START recording."
                                        )
                                else:
                                    leave_zero_force_to_init(
                                        env,
                                        init_targets,
                                        frequency=frequency,
                                        duration=init_duration,
                                    )
                                    zero_force_active = False
                                    if grip_detector is not None:
                                        grip_detector.reset()
                            if key_stroke == KeyCode(char="c"):
                                step_replay = None
                                if zero_force_active or no_zero_force:
                                    start_episode = True
                                    if grip_detector is not None:
                                        grip_detector.reset()
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
                                if is_delete_busy():
                                    print("Delete already in progress — wait.")
                                else:
                                    n_saved = env.replay_buffer.n_episodes
                                    if n_saved <= 0:
                                        print("No saved episode to delete.")
                                    else:
                                        delete_mode = "input"
                                        delete_buf = ""
                                        delete_target = None
                                        replay_mode = None
                                        replay_buf = ""
                                        print(
                                            f"Delete saved episode: type ID "
                                            f"(0..{n_saved - 1}), "
                                            "Enter to confirm, Esc to cancel."
                                        )
                            if key_stroke == KeyCode(char="r"):
                                n_saved = env.replay_buffer.n_episodes
                                if n_saved <= 0:
                                    print("No saved episode to replay.")
                                else:
                                    step_replay = None
                                    replay_mode = "input"
                                    replay_buf = ""
                                    delete_mode = None
                                    delete_buf = ""
                                    print(
                                        f"Full replay: type ID (0..{n_saved - 1}), "
                                        "Enter alone = last, Esc to cancel."
                                    )
                            if key_stroke == KeyCode(char="j"):
                                do_step_replay(None)
                            if key_stroke == KeyCode(char="x"):
                                do_estop()
                            if key_stroke == KeyCode(char="0"):
                                _set_replay_arm([0])
                            if key_stroke == KeyCode(char="1"):
                                _set_replay_arm([1])
                            if key_stroke == KeyCode(char="2"):
                                _set_replay_arm(None)
                            if key_stroke == Key.esc and step_replay is not None:
                                print(
                                    f"Step replay ep{step_replay['episode_id']} cancelled."
                                )
                                step_replay = None
                                if web_state is not None:
                                    web_state.update(
                                        mode="READY" if no_zero_force else "IDLE",
                                        message="已退出单步回放",
                                    )
                        if start_episode:
                            break

                        try:
                            deleting_now = is_delete_busy() or (
                                web_state is not None
                                and bool(getattr(web_state, "deleting", False))
                            )
                            # 删除在后台线程：主循环继续 get_obs，保持通信，避免掉臂
                            obs = env.get_obs()
                            cam_lat = compute_camera_latency(env, obs)
                            if (
                                not deleting_now
                                and grip_detector is not None
                                and zero_force_active
                                and delete_mode is None
                                and replay_mode is None
                            ):
                                gw = gripper_width_from_obs(obs, grip_robot_idx)
                                if gw is not None:
                                    obs_t = float(np.asarray(obs["timestamp"][-1]))
                                    if grip_detector.update(gw, obs_t):
                                        print(
                                            f"Gripper double open-close on "
                                            f"robot{grip_robot_idx} → START recording."
                                        )
                                        start_episode = True
                                        grip_detector.reset()
                            if start_episode:
                                break
                            if pose_print_enabled:
                                now = time.monotonic()
                                if now - last_pose_print_t >= pose_print_interval:
                                    print_current_pose(env, tag="live")
                                    last_pose_print_t = now
                            if deleting_now:
                                n_saved = (
                                    int(getattr(web_state, "n_episodes", 0))
                                    if web_state is not None
                                    else 0
                                )
                            else:
                                n_saved = env.replay_buffer.n_episodes
                            delete_prompt = None
                            prompt_status = "DEL"
                            if deleting_now:
                                prompt_status = "DEL"
                                delete_prompt = "Deleting… wait (do not quit)"
                            elif delete_mode == "input":
                                delete_prompt = (
                                    f"ID:{delete_buf or '_'}  (0..{max(n_saved - 1, 0)}) "
                                    "Enter=ok Esc=cancel"
                                )
                            elif delete_mode == "confirm":
                                delete_prompt = (
                                    f"Delete Ep {delete_target}?  Y=yes N=no"
                                )
                            elif replay_mode == "input":
                                prompt_status = "REP"
                                delete_prompt = (
                                    f"Replay:{replay_buf or '(last)'}  "
                                    f"(0..{max(n_saved - 1, 0)}) Enter=go Esc=cancel"
                                )
                            elif step_replay is not None:
                                prompt_status = "STEP"
                                _sr = step_replay
                                _na = len(_sr["actions"])
                                _arms = _format_replay_arms(replay_arm_sel, n_robots)
                                delete_prompt = (
                                    f"Step ep{_sr['episode_id']}[{_arms}]: "
                                    f"{_sr['idx']}/{_na - 1}  J=next Esc=exit X=stop"
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
                                prompt_status=prompt_status,
                                n_robots=n_robots,
                                camera_latency=cam_lat,
                            )
                            if not no_cv:
                                cv2.imshow("drag_collect_bimanual", vis[..., ::-1])
                                cv2.pollKey()
                            publish_web_frame(
                                web_state,
                                env=env,
                                vis_rgb=vis,
                                clean_rgb=camera_strip_from_obs(
                                    obs, latency=cam_lat
                                ),
                                recording=False,
                                zero_force_active=zero_force_active,
                                no_zero_force=no_zero_force,
                                delete_mode=delete_mode,
                                delete_target=delete_target,
                                n_steps=0,
                                t_elapsed=0.0,
                                episode_id=None,
                                frequency=frequency,
                                camera_latency=cam_lat,
                            )
                        except Exception:
                            pass
                        time.sleep(0.05)

                    # ---------- recording（零力可选；开录后不再重复 set_gravity） ----------
                    episode_id = env.replay_buffer.n_episodes
                    start_delay = 0.5
                    eval_t_start = time.time() + start_delay
                    print(f"\n>>> Episode {episode_id} START")
                    if grip_detector is not None:
                        grip_detector.reset()
                        print(
                            f"Double open-close robot{grip_robot_idx} gripper "
                            "to SAVE (gesture trimmed from zarr), or press S."
                        )
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
                    save_via_grip_gesture = False
                    last_action_rotvecs: list[np.ndarray | None] = [
                        None for _ in range(n_robots)
                    ]

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
                                    elif action == "estop":
                                        print("E-STOP during recording — abort episode.")
                                        abort_episode(env)
                                        do_estop()
                                        stop_drop = True
                                    elif action == "replay":
                                        print("Still recording — save/drop first, then replay.")
                                    elif action == "replay_step":
                                        print("Still recording — save/drop first, then step-replay.")
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
                                elif key_stroke == KeyCode(char="r"):
                                    print("Still recording — save/drop first, then press R.")
                                elif key_stroke == KeyCode(char="j"):
                                    print("Still recording — save/drop first, then press J.")
                                elif key_stroke == KeyCode(char="x"):
                                    print("E-STOP during recording — abort episode.")
                                    abort_episode(env)
                                    do_estop()
                                    stop_drop = True
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
                            cam_lat = compute_camera_latency(env, obs)
                            if pose_print_enabled:
                                if now - last_pose_print_t >= pose_print_interval:
                                    print_current_pose(env, tag="rec")
                                    last_pose_print_t = now

                            grip_save_triggered = False
                            if grip_detector is not None and not (
                                stop_save or stop_drop
                            ):
                                gw = gripper_width_from_obs(obs, grip_robot_idx)
                                if gw is not None:
                                    obs_t = float(np.asarray(obs["timestamp"][-1]))
                                    if grip_detector.update(gw, obs_t):
                                        grip_save_triggered = True
                                        save_via_grip_gesture = True
                                        # 只裁掉末尾手势窗，绝不超过 window（防误删大段）
                                        cut_t = grip_detector.last_gesture_start_t
                                        if cut_t is None:
                                            cut_t = obs_t
                                        max_trim_s = float(
                                            grip_detector.window_s
                                        ) + 0.3
                                        cut_floor = float(obs_t) - max_trim_s
                                        cut_t = max(float(cut_t), cut_floor) - 0.05
                                        before = (
                                            len(env.action_accumulator)
                                            if env.action_accumulator is not None
                                            else 0
                                        )
                                        kept = truncate_episode_before(env, cut_t)
                                        n_steps = kept
                                        trimmed = max(0, before - kept)
                                        print(
                                            f"Gripper double open-close on "
                                            f"robot{grip_robot_idx} → SAVE "
                                            f"(trim {trimmed} steps / "
                                            f"≤{max_trim_s:.1f}s; keep {kept})."
                                        )
                                        stop_save = True
                                        if web_state is not None:
                                            web_state.update(
                                                mode="SAVING",
                                                recording=True,
                                                message=(
                                                    f"夹爪手势保存 episode "
                                                    f"{episode_id}…请稍候"
                                                ),
                                            )

                            if not grip_save_triggered and not (
                                stop_save or stop_drop
                            ):
                                record_pose_action(
                                    env, obs, last_rotvecs=last_action_rotvecs
                                )
                                n_steps += 1
                                append_live_from_obs(
                                    web_state,
                                    obs,
                                    step=n_steps,
                                    t_elapsed=t_elapsed,
                                    n_robots=n_robots,
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
                                camera_latency=cam_lat,
                            )
                            if not no_cv:
                                cv2.imshow("drag_collect_bimanual", vis[..., ::-1])
                                cv2.pollKey()
                            publish_web_frame(
                                web_state,
                                env=env,
                                vis_rgb=vis,
                                clean_rgb=camera_strip_from_obs(
                                    obs, latency=cam_lat
                                ),
                                recording=True,
                                zero_force_active=zero_force_active,
                                no_zero_force=no_zero_force,
                                delete_mode=None,
                                delete_target=None,
                                n_steps=n_steps,
                                t_elapsed=t_elapsed,
                                episode_id=episode_id,
                                frequency=frequency,
                                camera_latency=cam_lat,
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
                                if grip_detector is not None:
                                    grip_detector.reset()
                                if web_state is not None:
                                    web_state.clear_live_series()
                                    web_state.update(
                                        mode="READY" if no_zero_force else (
                                            "DRAG" if zero_force_active else "IDLE"
                                        ),
                                        recording=False,
                                        message="已丢弃，可继续采集",
                                    )
                                # 丢弃后仍保持零力，可直接再按 C / 双开合 录下一条
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
                                    via = (
                                        " [grip gesture, trimmed]"
                                        if save_via_grip_gesture
                                        else ""
                                    )
                                    print(
                                        f"Episode {episode_id} saved{via} "
                                        f"({n_steps} steps @ {frequency} Hz). "
                                        "Press R to replay, E to delete."
                                    )
                                if grip_detector is not None:
                                    grip_detector.reset()

                                # 保存后（S / 夹爪手势 / 超时）: 回 init，再重新进拖拽
                                if web_state is not None:
                                    web_state.update(
                                        mode="SAVING",
                                        recording=False,
                                        message=(
                                            f"已保存 episode {episode_id}，"
                                            "正在回 init…"
                                            if ok
                                            else (
                                                f"保存 episode {episode_id} 失败，"
                                                "正在回 init…"
                                            )
                                        ),
                                    )
                                print(
                                    "Save done — return to init, then re-enter drag..."
                                )
                                leave_zero_force_to_init(
                                    env,
                                    init_targets,
                                    frequency=frequency,
                                    duration=init_duration,
                                )
                                # 与 --no_zero_force 无关：保存回 init 后一律重新进拖拽
                                # （该 flag 只表示「开录不必先按 P」，不是「保存后不要拖拽」）
                                zero_force_active = False
                                time.sleep(0.15)
                                enter_drag_mode(env)
                                zero_force_active = True
                                if grip_detector is not None:
                                    grip_detector.reset()
                                print(
                                    "Returned to init and re-entered drag. "
                                    f"Double open-close robot{grip_robot_idx} "
                                    "gripper to START (or press C)."
                                )

                                if web_state is not None:
                                    web_state.clear_live_series()
                                    web_state.update(
                                        mode=(
                                            "DRAG"
                                            if zero_force_active
                                            else (
                                                "READY"
                                                if no_zero_force
                                                else "IDLE"
                                            )
                                        ),
                                        recording=False,
                                        zero_force_active=bool(zero_force_active),
                                        n_episodes=int(env.replay_buffer.n_episodes),
                                        n_steps=0,
                                        t_elapsed=0.0,
                                        message=(
                                            (
                                                f"已保存 episode {episode_id}，"
                                                "已回 init并进入拖拽"
                                            )
                                            if ok
                                            else (
                                                f"保存 episode {episode_id} "
                                                "失败（无有效帧），已回 init并进入拖拽"
                                            )
                                        ),
                                    )
                                break

                            precise_wait(t_cycle_end, time_func=time.monotonic)
                            last_tick = t_cycle_end

                    except KeyboardInterrupt:
                        print("Interrupted — abort episode.")
                        abort_episode(env)
                        leave_drag_mode(env)
                        zero_force_active = False

                    if grip_detector is not None:
                        grip_detector.reset()
                    if zero_force_active:
                        print(
                            "Ready for next episode. Press C or double open-close "
                            f"robot{grip_robot_idx} gripper to start, "
                            "R to replay, E to delete, or Q to quit.\n"
                        )
                    else:
                        print(
                            "Ready. Press P for zero-force, "
                            "R to replay, E to delete, or C to start.\n"
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
    default="data/collect_bimanual",
    show_default=True,
    help="输出目录（replay_buffer.zarr + videos）",
)
@click.option(
    "--robot_ip",
    default="127.0.0.1",
    show_default=True,
    help="ZMQ 机械臂服务 IP（左右臂共用）",
)
@click.option(
    "--robot_ports",
    default="8765,8766",
    show_default=True,
    help="左右臂 ZMQ 端口，逗号分隔（robot0=左, robot1=右）",
)
@click.option(
    "--frequency",
    "-f",
    default=20.0,
    type=float,
    help="采数频率 Hz（默认 20；对齐用同一 dt）",
)
@click.option(
    "--camera_reorder",
    "-cr",
    default="0,1",
    show_default=True,
    help="腕部 GoPro 在「过滤 RealSense 后」V4L 列表中的索引，映射为 camera0/camera2",
)
@click.option("--obs_width", default=224, type=int)
@click.option("--obs_height", default=224, type=int)
@click.option(
    "--rs_serial",
    default=None,
    type=str,
    help="D435 序列号（默认自动选第一台）",
)
@click.option("--rs_width", default=640, type=int, show_default=True)
@click.option("--rs_height", default=480, type=int, show_default=True)
@click.option("--rs_fps", default=30, type=int, show_default=True)
@click.option(
    "--save_depth/--no_save_depth",
    default=True,
    help="是否采集并写入 camera1_depth（uint16 mm）",
)
@click.option(
    "--rs_color_temperature",
    default=0.0,
    type=float,
    show_default=True,
    help="D435 色温（默认 0=关闭；训练/推理勿单边改）",
)
@click.option(
    "--rs_color_saturation",
    default=1.0,
    type=float,
    show_default=True,
    help="D435 饱和度倍率（默认 1=关闭）",
)
@click.option(
    "--rs_color_contrast",
    default=1.0,
    type=float,
    show_default=True,
    help="D435 对比度倍率（默认 1=关闭）",
)
@click.option(
    "--rs_color_hue_shift",
    default=12.0,
    type=float,
    show_default=True,
    help="D435 蓝→品红色相（默认 12；推理须同参；0=原图）",
)
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
    "--drag_profile",
    type=click.Choice(["light", "stable"], case_sensitive=False),
    default="light",
    show_default=True,
    help=(
        "软阻尼拖拽手感：light=更轻（采数）；stable=更稳防坠（偏沉，"
        "与 DAgger 介入同档）。真零力 server 时此参数无效。"
    ),
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
@click.option(
    "--grip_gesture_robot",
    default=1,
    type=int,
    show_default=True,
    help="双开合手势用的臂索引（默认 1=右臂/can1）",
)
@click.option(
    "--grip_gesture_window",
    default=1.5,
    type=float,
    show_default=True,
    help="两次开合需落在该时间窗内（秒）才算双开合",
)
@click.option(
    "--no_grip_gesture",
    is_flag=True,
    default=False,
    help="关闭夹爪双开合开始/保存手势",
)
@click.option(
    "--replay_approach",
    default=2.5,
    type=float,
    show_default=True,
    help="回放前缓动到 episode 起点的时长（秒）",
)
def main(
    output,
    robot_ip,
    robot_ports,
    frequency,
    camera_reorder,
    obs_width,
    obs_height,
    rs_serial,
    rs_width,
    rs_height,
    rs_fps,
    save_depth,
    rs_color_temperature,
    rs_color_saturation,
    rs_color_contrast,
    rs_color_hue_shift,
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
    drag_profile,
    no_zero_force,
    quit_home_duration,
    web,
    web_host,
    web_port,
    no_cv,
    viewer_only,
    collect_direct,
    grip_gesture_robot,
    grip_gesture_window,
    no_grip_gesture,
    replay_approach,
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
        robot_ports=robot_ports,
        frequency=frequency,
        camera_reorder=camera_reorder,
        obs_width=obs_width,
        obs_height=obs_height,
        rs_serial=rs_serial,
        rs_width=rs_width,
        rs_height=rs_height,
        rs_fps=rs_fps,
        save_depth=save_depth,
        rs_color_temperature=rs_color_temperature,
        rs_color_saturation=rs_color_saturation,
        rs_color_contrast=rs_color_contrast,
        rs_color_hue_shift=rs_color_hue_shift,
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
        drag_profile=drag_profile,
        no_zero_force=no_zero_force,
        quit_home_duration=quit_home_duration,
        web=web,
        web_host=web_host,
        web_port=web_port,
        no_cv=no_cv,
        grip_gesture_robot=grip_gesture_robot,
        grip_gesture_window=grip_gesture_window,
        no_grip_gesture=no_grip_gesture,
        replay_approach=replay_approach,
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
