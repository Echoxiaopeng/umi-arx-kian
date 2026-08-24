"""
ARX5 拖拽采数脚本（重力补偿 / 零力拖拽）。
数据131可能有问题
注：已经验证
不需要起SDK服务，采集到的是x5.urdf（/data1/projects/umi-arx-hzl/modules/arx_x5_python/bimanual/script/x5.urdf）
上面的 link6 的位姿。(X5.urdf / X5_umi.urdf才有eef_link)

和起SDK的ZMQ服务对比：
python python/communication/zmq_server_ac_one.py X5 can3 --gripper-open-readout -3.51 --gripper-width 0.085
原来采集到的数据只要加位置偏移，不需要加姿态旋转了。因为ZMQ服务中同样也做了ee2tcp旋转，是他和umi的轴对齐。

并且更改了link6的姿态，不改变位置。原来SDK中的link6 的 EE 前伸是 X轴；UMI/TCP 前伸是 Z轴。这段代码就是把那套轴对齐过去。


不依赖策略服务。打开重力补偿后手动拖动机械臂，按固定频率记录
相机 + 机器人状态；action 取当前末端位姿 + 夹爪开度。

同步策略:
  1. 每步 env.get_obs()：多相机最近邻对齐，机器人状态插值到相机时间
  2. 用 obs 相机时间戳把 pose-as-action 写入 action_accumulator
  3. env.end_episode()：以 action 时间栅格为轴，插值对齐 obs / 图像写入
     replay_buffer.zarr，并保存 videos/{episode_id}/*.mp4

用法:
    python scripts/collect_drag_umi.py -o data/collect_demo \\
        --init_pose 0.4,0.0,0.25,0,1.57,0 --init_gripper 0.08

    # SSH 无 X11 时用终端按键:
    python scripts/collect_drag_umi.py -o data/collect_demo \\
        --init_pose 0.4,0.0,0.25,0,1.57,0 --terminal_keys

    # 如果上面报错，则执行下面：
    cd /data1/projects/umi-arx-hzl/modules/arx_x5_python
    source setup.sh
    cd /data1/projects/umi-arx-hzl
    python scripts/collect_drag_umi.py --init_pose ...


    python scripts/collect_drag_umi.py -o data/cup_demo2 --init_pose -0.006958,-0.002573,0.120019,2.464006,-2.497232,1.524022 --init_gripper 0.000604 --no_zero_force

键盘:
  P = 进入零力模式（重力补偿，可拖拽）
  C = 开始采集（默认须先按 P；--no_zero_force 时可直接 C）
  S = 停止并保存
  D = 丢弃当前正在录的 episode（未保存）
  E = 删除已保存的指定 episode（空闲时；输入编号后 Y 确认）
  I = 打印当前 TCP 位姿（可直接复制为 --init_pose）
  Q = 回 home 后退出

启动流程: reset_to_home →（可选）缓动到 --init_pose → 等 P 进零力（或 --no_zero_force 直接 C）。
空闲/拖拽/录制时终端会周期性打印当前位姿，便于挑选初始位姿。
不做 episode 级 init 移动；按 C 时臂在哪就从哪开录。

关节置0：
    cansend can3 7FF#00010003
    cansend can3 7FF#00020003
    cansend can3 7FF#00040003
    cansend can3 005#FFFFFFFFFFFFFFFE
    cansend can3 006#FFFFFFFFFFFFFFFE
    cansend can3 007#FFFFFFFFFFFFFFFE
    cansend can3 008#FFFFFFFFFFFFFFFE

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

from modules.arx5_env_with_images import Arx5EnvWithImages  # noqa: E402
from peripherals.keystroke_counter import Key, KeyCode  # noqa: E402
from utils.other_util import precise_wait  # noqa: E402


def _install_arx_log_filter(
    drop_substrings: tuple[str, ...] = ("方舟无限", "ARX方舟无限"),
) -> None:
    """
    过滤 ARX C++ SDK 刷屏（libarx_x5_src.so 打印 ARX方舟无限）。

    必须拦真实 fd：SDK 用 printf 直写 fd；控制器子进程 fork 后会继承
    已重定向的 fd，因此在启动 env 前安装即可，无需改其它源码。
    """
    import threading

    if getattr(_install_arx_log_filter, "_installed", False):
        return
    _install_arx_log_filter._installed = True  # type: ignore[attr-defined]

    drop = tuple(s.encode("utf-8") for s in drop_substrings if s)

    def _pump(read_fd: int, write_fd: int) -> None:
        with os.fdopen(read_fd, "rb", buffering=0) as src:
            buf = b""
            while True:
                try:
                    chunk = src.read(512)
                except Exception:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if any(s in line for s in drop):
                        continue
                    try:
                        os.write(write_fd, line + b"\n")
                    except Exception:
                        return
            if buf and not any(s in buf for s in drop):
                try:
                    os.write(write_fd, buf)
                except Exception:
                    pass

    for stream in (sys.stdout, sys.stderr):
        try:
            raw_fd = stream.fileno()
        except Exception:
            continue
        try:
            saved_fd = os.dup(raw_fd)
            r_fd, w_fd = os.pipe()
            os.dup2(w_fd, raw_fd)
            os.close(w_fd)
            threading.Thread(
                target=_pump,
                args=(r_fd, saved_fd),
                name=f"arx-log-filter-{raw_fd}",
                daemon=True,
            ).start()
        except Exception:
            continue


# 在创建控制器子进程之前安装，fork 后子进程继承过滤后的 fd
_install_arx_log_filter()


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
            "p=零力, c=开始, s=保存, d=丢弃当前, e=删已保存, i=打印当前位姿, q=回home退出"
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
def _default_robots_config():
    return [
        {
            "robot_type": "arx5",
            "robot_ip": "127.0.0.1",
            "robot_port": 8765,
            "can_port": "can3",
            "arm_type": 0,
            "reset_to_home_on_start": True,
            "robot_obs_latency": 0.1867,
            "robot_action_latency": 0.01,
            "tcp_offset": 0.18,
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
    而 SingleArmClient.tcp_pose 读数是长角等价形式；复制 init_pose 必须用这里。
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
    """
    from scipy.spatial.transform import Rotation, Slerp

    if ensure_end_control and hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
            time.sleep(0.05)
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
    start_rots = []
    target_rots = []
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
        start_rots.append(Rotation.from_rotvec(curr6[3:6]))
        target_rots.append(Rotation.from_rotvec(tgt6[3:6]))

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

    cmd_hz = min(float(frequency), 20.0)
    dt = 1.0 / cmd_hz
    steps = max(2, int(duration * cmd_hz))
    batch_size = 30
    t0 = time.time() + 0.3

    for batch_start in range(0, steps, batch_size):
        batch_end = min(batch_start + batch_size, steps)
        actions = []
        timestamps = []
        for k in range(batch_start, batch_end):
            a = ease_in_out_quad(k / (steps - 1))
            key_times = [0.0, 1.0]
            cmd = np.zeros(7 * n_robots, dtype=np.float64)
            for ridx in range(n_robots):
                pos_now = start_pose_all[ridx, :3] + a * (
                    target_pose_all[ridx, :3] - start_pose_all[ridx, :3]
                )
                slerp = Slerp(
                    key_times, Rotation.concatenate([start_rots[ridx], target_rots[ridx]])
                )
                rot_now = slerp([a])[0].as_rotvec()
                grip_now = start_grip_all[ridx] + a * (
                    target_grip_all[ridx] - start_grip_all[ridx]
                )
                cmd[7 * ridx : 7 * ridx + 3] = pos_now
                cmd[7 * ridx + 3 : 7 * ridx + 6] = rot_now
                cmd[7 * ridx + 6] = grip_now
            actions.append(cmd)
            timestamps.append(t0 + k * dt)

        env.exec_actions(
            actions=np.asarray(actions),
            timestamps=np.asarray(timestamps),
            compensate_latency=False,
        )
        if batch_end < steps:
            time.sleep(max(0.0, (batch_end - batch_start) * dt * 0.85))

    remain = (t0 + (steps - 1) * dt + 0.25) - time.time()
    if remain > 0:
        time.sleep(remain)


def move_to_init_pose(
    env: Arx5EnvWithImages,
    target_pose6: np.ndarray,
    *,
    frequency: float,
    duration: float = 2.5,
    init_gripper: float | None = None,
) -> None:
    """启动时缓动到固定 init 位姿（此时尚未进零力）。"""
    n_robots = len(env.robots)
    targets = [(target_pose6, init_gripper) for _ in range(n_robots)]
    move_to_tcp_pose(
        env,
        targets,
        frequency=frequency,
        duration=duration,
        label="init pose",
        ensure_end_control=True,
    )
    print("Init pose reached. Press P to enter zero-force, then C to record.")
    print_current_pose(env, tag="after-init")


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


def check_episode_alignment(env: Arx5EnvWithImages) -> None:
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


def delete_saved_episode(env: Arx5EnvWithImages, episode_id: int) -> bool:
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


def go_home_before_quit(
    env: Arx5EnvWithImages,
    home_targets: list[tuple[np.ndarray, float]],
    *,
    frequency: float,
    duration: float = 3.0,
    zero_force_active: bool = False,
) -> None:
    """
    退出前先离开零力，再缓动回启动时记录的 home TCP 位姿。

    避免零力模式下直接 robot.reset_to_home() / go_home() 造成急跳。
    """
    print("Returning to home before quit...")
    if hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
            # 零力切回末端控制后多等一会，让当前位姿稳住
            time.sleep(0.4 if zero_force_active else 0.15)
        except Exception as exc:  # noqa: BLE001
            print(f"set_end_control warning: {exc}")

    if home_targets:
        move_to_tcp_pose(
            env,
            [(pose, grip) for pose, grip in home_targets],
            frequency=frequency,
            duration=duration,
            label="home",
        )
    else:
        print("No home pose cached; falling back to reset_to_home().")
        for robot_idx, robot in enumerate(env.robots):
            try:
                print(f"  reset_to_home robot{robot_idx}...")
                robot.reset_to_home()
            except Exception as exc:  # noqa: BLE001
                print(f"  reset_to_home robot{robot_idx} failed: {exc}")
    print("Home done. Exiting.")


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
    vis = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    if recording:
        status = "REC"
        help_text = "S save | D drop | I pose | Q quit"
    elif delete_prompt:
        status = "DEL"
        help_text = delete_prompt
    elif zero_force_active:
        status = "DRAG"
        help_text = "C start | I pose | E del ep | Q quit"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--output",
    "-o",
    default="data/collect_demo",
    show_default=True,
    help="输出目录（replay_buffer.zarr + videos）",
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
    help="用终端 stdin 读键（SSH / 无 X11）",
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
    help="启动后缓动到的固定 TCP 位姿: x,y,z,rx,ry,rz（米 & 轴角弧度）；省略则不移动",
)
@click.option(
    "--init_gripper",
    default=None,
    type=float,
    help="启动时目标夹爪开度（米）；默认保持当前开度",
)
@click.option(
    "--init_duration",
    default=2.5,
    type=float,
    show_default=True,
    help="启动缓动到 init_pose 的时长（秒）",
)
@click.option(
    "--no_zero_force",
    is_flag=True,
    default=False,
    help="不要求先进零力模式，可直接 C 开录 / S 保存（P 仍可选）",
)
@click.option(
    "--quit_home_duration",
    default=3.0,
    type=float,
    show_default=True,
    help="按 Q 退出时缓动回 home 的时长（秒）",
)
def main(
    output,
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
    init_gripper,
    init_duration,
    no_zero_force,
    quit_home_duration,
):
    from peripherals.keystroke_counter import KeystrokeCounter

    os.makedirs(output, exist_ok=True)
    dt = 1.0 / frequency
    robots_config = _default_robots_config()
    n_robots = len(robots_config)
    camera_reorder_list = [int(x) for x in camera_reorder.split(",") if x.strip() != ""]
    init_pose6 = parse_pose6(init_pose) if init_pose else None

    print("=" * 60)
    print("ARX5 Drag Data Collection")
    print(f"  output     : {output}")
    print(f"  frequency  : {frequency} Hz")
    print(f"  n_robots   : {n_robots}")
    print(f"  save_images: {save_images}")
    if init_pose6 is None:
        print("  init_pose  : (skip) drag & press I to inspect pose")
    else:
        print(
            f"  init_pose  : {np.array2string(init_pose6, precision=4, separator=',')}"
        )
    print(
        f"  init_gripper: "
        f"{'keep current' if init_gripper is None else f'{init_gripper:.4f} m'}"
    )
    print(f"  init_duration: {init_duration:.1f}s")
    print(f"  no_zero_force: {no_zero_force}")
    if no_zero_force:
        print("  startup: reset_to_home → optional init_pose → C to record (P optional)")
    else:
        print("  startup: reset_to_home → optional init_pose → P then C")
    print("  P=zero-force  C=start  S=save  D=drop  E=del  I=print-pose  Q=home+quit")
    print("=" * 60)

    key_ctx = (
        contextlib.nullcontext(None) if terminal_keys else KeystrokeCounter()
    )
    term_ctx = TerminalKeyReader() if terminal_keys else contextlib.nullcontext(None)

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
            key_source = term_keys if terminal_keys else key_counter

            cv2.setNumThreads(2)
            print("Waiting for camera / env...")
            time.sleep(1.0)
            while not env.is_ready:
                time.sleep(0.1)
            check_episode_alignment(env)
            print("Env ready.")

            home_targets = capture_robot_pose_targets(env)
            print("Home pose captured (used when pressing Q to quit).")

            if init_pose6 is not None:
                move_to_init_pose(
                    env,
                    init_pose6,
                    frequency=frequency,
                    duration=init_duration,
                    init_gripper=init_gripper,
                )
            else:
                if no_zero_force:
                    print(
                        "No --init_pose. Direct record mode: press C to start, "
                        "I to print --init_pose string."
                    )
                else:
                    print(
                        "No --init_pose. Press P to zero-force, drag to a good pose, "
                        "press I to print --init_pose string."
                    )

            zero_force_active = False
            delete_mode = None  # None | "input" | "confirm"
            delete_buf = ""
            delete_target = None
            last_pose_print_t = 0.0
            pose_print_interval = 0.5

            if no_zero_force:
                print("Direct record mode: press C to start (P optional for drag).")

            while True:
                # ---------- idle: optional P -> zero-force, then C -> start ----------
                start_episode = False
                while True:
                    for key_stroke in drain_keys(key_source):
                        if key_stroke == KeyCode(char="q"):
                            print("Quit.")
                            go_home_before_quit(
                                env,
                                home_targets,
                                frequency=frequency,
                                duration=quit_home_duration,
                                zero_force_active=zero_force_active,
                            )
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
                                print("Zero-force ON. Drag freely, then press C to record.")
                            else:
                                print("Already in zero-force mode.")
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
                            try:
                                print_current_pose(env, tag="init-candidate")
                            except Exception as exc:  # noqa: BLE001
                                print(f"Failed to read pose: {exc}")
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
                        cv2.imshow("drag_collect", vis[..., ::-1])
                        cv2.pollKey()
                    except Exception:
                        pass
                    time.sleep(0.05)

                # ---------- recording（零力可选；开录后不再重复 set_gravity） ----------
                episode_id = env.replay_buffer.n_episodes
                start_delay = 0.5
                eval_t_start = time.time() + start_delay
                print(f"\n>>> Episode {episode_id} START")
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

                        for key_stroke in drain_keys(key_source):
                            if key_stroke == KeyCode(char="s"):
                                stop_save = True
                            elif is_drop_key(key_stroke):
                                stop_drop = True
                            elif key_stroke == KeyCode(char="e"):
                                print("Still recording — save/drop first, then press E.")
                            elif key_stroke == KeyCode(char="i"):
                                print_current_pose(env, tag="rec")
                            elif key_stroke == KeyCode(char="q"):
                                print("Quit requested — abort current episode, then go home.")
                                abort_episode(env)
                                go_home_before_quit(
                                    env,
                                    home_targets,
                                    frequency=frequency,
                                    duration=quit_home_duration,
                                    zero_force_active=zero_force_active,
                                )
                                return

                        obs = env.get_obs()
                        if now - last_pose_print_t >= pose_print_interval:
                            print_current_pose(env, tag="rec")
                            last_pose_print_t = now
                        record_pose_action(env, obs)
                        n_steps += 1

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
                        cv2.imshow("drag_collect", vis[..., ::-1])
                        cv2.pollKey()

                        if t_elapsed > max_duration:
                            print(f"Max duration {max_duration}s reached — auto save.")
                            stop_save = True

                        if stop_drop:
                            abort_episode(env)
                            # 丢弃后仍保持零力，可直接再按 C 录下一条
                            break

                        if stop_save:
                            if save_episode(env, episode_id):
                                print(
                                    f"Episode {episode_id} saved "
                                    f"({n_steps} loop steps @ {frequency} Hz). "
                                    "Press E to delete a saved episode."
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


if __name__ == "__main__":
    main()
