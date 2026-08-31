"""
ARX5 三杯挪杯采数脚本（臂固定拍图；人按脚本出题挪杯；同 episode 零力夹取）。

脚本先随机生成任务（方块初始杯子 + 3 次交换），人照做；出题参数即标签。
观察段机械臂固定拍图；满 61 帧后立刻进零力，继续录夹取轨迹，按 S 整段保存。

协议（每条 episode）:
  1. C 出题 → 方块随机放在 initial_cup；第 1 次交换必须包含该杯
  2. F → 录第 0 帧（静止）
  3. 显示 Move k → F → 人交换两杯，同时录 frames_per_move 帧（×3）
  4. F → 进入零力，人拖臂夹取 final_cup 上的杯子（持续录）
  5. S → 保存整段 zarr/videos + labels.jsonl；回 init_pose

同步策略（与拖拽采数相同）:
  1. 每步 env.get_obs()：多相机最近邻对齐，机器人状态插值到相机时间
  2. 用 obs 相机时间戳把 pose-as-action 写入 action_accumulator
  3. env.end_episode()：以 action 时间栅格对齐 obs / 图像写入
     replay_buffer.zarr，并保存 videos/{episode_id}/*.mp4

用法:
    python scripts/collect_drag_umi_cup.py -o data/cup_demo --terminal_keys

    python scripts/collect_drag_umi_cup.py -o data/cup_demo \\
        --frames_per_move 30 --frequency 10 --seed 0
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority

conda activate umi-arx
cd ~/kian/umi-arx-kian
source /home/lenovo/kian/umi-arx-kian/scripts/setup.sh

python scripts/collect_drag_umi_cup.py -o data/cup_demo


conda activate umi-arx-kian
source $CONDA_PREFIX/setup.bash          # 设置 AMENT_PREFIX_PATH 等
source scripts/setup.sh                  # arx_x5_python / LD_LIBRARY_PATH
# 确认 can 已 up，且与脚本 can_port 一致（默认 can3）
python scripts/collect_drag_umi_cup.py -o data/cup_demo

键盘:
  C = 出下一题并进入摆放阶段
  F = 进入下一段观察录制 / 观察结束后进零力夹取
  S = 夹取段结束并保存整条 episode（仅 record_grasp）
  D = 丢弃当前正在录的 episode（未保存）
  E = 删除已保存的指定 episode（空闲时；同步维护 labels.jsonl）
  P = 可选零力开关（idle；再按退出并缓动回 --init_pose）
  I = 开关位姿打印
  Q = 回 home 后退出

杯子编号: 从左到右 0=left, 1=middle, 2=right（预览窗口叠 0/1/2 圆标）

Author: Kian
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import select
import shutil
import sys
import termios
import time
import tty
from dataclasses import dataclass
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

# ---------------------------------------------------------------------------
# Cup task (script generates labels; human executes)
# ---------------------------------------------------------------------------
CUP_NAMES = ("left", "middle", "right")
N_CUPS = 3
N_MOVES = 3
DEFAULT_FRAMES_PER_MOVE = 80
FRAMES_PER_MOVE = DEFAULT_FRAMES_PER_MOVE
FRAMES_TOTAL = 1 + N_MOVES * FRAMES_PER_MOVE  # 1 + 3 * frames_per_move
LABELS_FILENAME = "labels.jsonl"


def set_observe_frame_counts(frames_per_move: int) -> None:
    """Update global observe-segment lengths used by FSM / labels / UI."""
    global FRAMES_PER_MOVE, FRAMES_TOTAL
    fpm = int(frames_per_move)
    if fpm < 1:
        raise click.BadParameter("--frames_per_move must be >= 1")
    FRAMES_PER_MOVE = fpm
    FRAMES_TOTAL = 1 + N_MOVES * FRAMES_PER_MOVE


@dataclass(frozen=True)
class CupTask:
    initial_cup: int
    moves: tuple[tuple[int, int], ...]
    final_cup: int

    def to_label_dict(self, episode_id: int, n_frames: int | None = None) -> dict:
        obs_n = int(FRAMES_TOTAL)
        if n_frames is None:
            n_frames = obs_n
        return {
            "episode_id": int(episode_id),
            "initial_cup": int(self.initial_cup),
            "moves": [list(m) for m in self.moves],
            "final_cup": int(self.final_cup),
            "frames_per_move": int(FRAMES_PER_MOVE),
            "n_observe_frames": obs_n,
            "grasp_start_frame": obs_n,
            "n_frames": int(n_frames),
        }


def apply_swaps(initial_cup: int, moves: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> int:
    """Track which slot the block ends in after successive pairwise cup swaps."""
    pos = int(initial_cup)
    for a, b in moves:
        a_i, b_i = int(a), int(b)
        if pos == a_i:
            pos = b_i
        elif pos == b_i:
            pos = a_i
    return pos


def sample_cup_task(rng: random.Random) -> CupTask:
    """
    方块初始杯位在 0/1/2 中均匀随机；第 1 次交换必须包含该杯。
    后两次交换在三个槽位中均匀随机选一对。
    """
    initial = int(rng.randrange(N_CUPS))
    moves: list[tuple[int, int]] = []
    # First swap must include the cup that currently holds the block.
    other = int(rng.choice([c for c in range(N_CUPS) if c != initial]))
    a, b = sorted((initial, other))
    moves.append((int(a), int(b)))
    for _ in range(N_MOVES - 1):
        a, b = rng.sample(range(N_CUPS), 2)
        # keep smaller index first for stable display
        if a > b:
            a, b = b, a
        moves.append((int(a), int(b)))
    final = apply_swaps(initial, moves)
    return CupTask(initial_cup=initial, moves=tuple(moves), final_cup=final)


def cup_name(idx: int) -> str:
    if 0 <= idx < len(CUP_NAMES):
        return CUP_NAMES[idx]
    return str(idx)


def format_cup_task(task: CupTask, episode_id: int | None = None) -> str:
    lines = []
    if episode_id is not None:
        lines.append(f"Episode {episode_id}")
    lines.append(
        f"Initial: put block under cup {task.initial_cup} ({cup_name(task.initial_cup)})"
    )
    for i, (a, b) in enumerate(task.moves, start=1):
        lines.append(
            f"Move {i}/{N_MOVES}: swap cups {a} ({cup_name(a)}) <-> {b} ({cup_name(b)})"
        )
    lines.append(f"Final (label): cup {task.final_cup} ({cup_name(task.final_cup)})")
    return "\n".join(lines)


def labels_path(output_dir: str | Path) -> Path:
    return Path(output_dir).joinpath(LABELS_FILENAME)


def load_labels(output_dir: str | Path) -> list[dict]:
    path = labels_path(output_dir)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_labels(output_dir: str | Path, rows: list[dict]) -> None:
    path = labels_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_episode_label(
    output_dir: str | Path,
    task: CupTask,
    episode_id: int,
    n_frames: int | None = None,
) -> None:
    rows = load_labels(output_dir)
    rows = [r for r in rows if int(r.get("episode_id", -1)) != int(episode_id)]
    rows.append(
        task.to_label_dict(
            episode_id,
            n_frames=FRAMES_TOTAL if n_frames is None else int(n_frames),
        )
    )
    rows.sort(key=lambda r: int(r["episode_id"]))
    save_labels(output_dir, rows)


def delete_episode_label(output_dir: str | Path, episode_id: int) -> None:
    """Remove one label row and renumber subsequent episode_id values down by 1."""
    rows = load_labels(output_dir)
    if not rows:
        return
    new_rows: list[dict] = []
    for row in rows:
        eid = int(row.get("episode_id", -1))
        if eid == episode_id:
            continue
        if eid > episode_id:
            row = dict(row)
            row["episode_id"] = eid - 1
        new_rows.append(row)
    new_rows.sort(key=lambda r: int(r["episode_id"]))
    save_labels(output_dir, new_rows)


DEFAULT_CUP_OVERLAY_NORM: tuple[tuple[float, float], ...] = (
    (0.25, 0.42),  # 0 left  — tuned on cup_demo @ 224x224
    (0.50, 0.38),  # 1 middle
    (0.75, 0.42),  # 2 right
)
CUP_OVERLAY_REF_SIZE = (224, 224)


def parse_cup_overlay_xy(text: str) -> list[tuple[float, float]]:
    """Parse 'x0,y0;x1,y1;x2,y2' (pixels at 224x224 reference)."""
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if len(parts) != N_CUPS:
        raise click.BadParameter(
            f"--cup_overlay_xy 需要 {N_CUPS} 个点 x,y;x,y;x,y，收到 {len(parts)} 个"
        )
    out: list[tuple[float, float]] = []
    for part in parts:
        vals = [float(v) for v in part.split(",")]
        if len(vals) != 2:
            raise click.BadParameter(
                f"--cup_overlay_xy 每点需 x,y，收到: {part!r}"
            )
        out.append((vals[0], vals[1]))
    return out


def resolve_cup_overlay_positions(
    width: int,
    height: int,
    cup_overlay_xy: str | None,
) -> list[tuple[int, int]]:
    ref_w, ref_h = CUP_OVERLAY_REF_SIZE
    if cup_overlay_xy:
        ref_points = parse_cup_overlay_xy(cup_overlay_xy)
        return [
            (
                int(round(x * width / ref_w)),
                int(round(y * height / ref_h)),
            )
            for x, y in ref_points
        ]
    return [
        (int(round(nx * width)), int(round(ny * height)))
        for nx, ny in DEFAULT_CUP_OVERLAY_NORM
    ]


def highlight_cups_for_phase(
    task: CupTask | None, phase: str, move_idx: int
) -> set[int]:
    if task is None:
        return set()
    if phase in ("wait_setup", "record_frame0"):
        return {task.initial_cup}
    if phase.startswith("wait_move") or phase.startswith("record_move"):
        a, b = task.moves[move_idx]
        return {int(a), int(b)}
    if phase in ("wait_grasp", "record_grasp"):
        return {task.final_cup}
    return set()


def draw_cup_slot_markers(
    vis: np.ndarray,
    positions: list[tuple[int, int]],
    *,
    highlight_cups: set[int] | None = None,
    cam_offset_x: int = 0,
    recording: bool = False,
) -> None:
    """Draw 0/1/2 slot markers on the preview (camera-image coordinates).

    Highlighted cups use cyan/orange while waiting; switch to red while
    recording so the operator can tell the segment is live.
    """
    highlight_cups = highlight_cups or set()
    radius = max(12, min(vis.shape[0], vis.shape[1]) // 12)
    for cup_id, (x, y) in enumerate(positions):
        cx = int(cam_offset_x + x)
        cy = int(y)
        active = cup_id in highlight_cups
        # Colors are RGB (cam buffer is RGB; imshow does [...::-1] → BGR).
        if active and recording:
            ring = (255, 0, 0)  # red while recording
            fill = (180, 0, 0)
        elif active:
            ring = (0, 255, 255)
            fill = (0, 180, 255)
        else:
            ring = (255, 255, 255)
            fill = (40, 40, 40)
        cv2.circle(vis, (cx, cy), radius, ring, 2, lineType=cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), max(radius - 3, 1), fill, -1, lineType=cv2.LINE_AA)
        label = str(cup_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.55, radius / 22.0)
        thickness = max(1, int(round(scale * 2)))
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(
            vis,
            label,
            (tx, ty),
            font,
            scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        name = cup_name(cup_id)[0].upper()
        cv2.putText(
            vis,
            name,
            (cx - 4, cy + radius + 14),
            font,
            0.35,
            ring,
            1,
            lineType=cv2.LINE_AA,
        )


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
            "c=出题, f=下一段/进零力, s=夹取后保存, d=丢弃, "
            "e=删已保存, p=可选零力, i=位姿, q=回home退出"
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
        if ch in ("f", "F"):
            return KeyCode(char="f")
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
            "can_port": "can0",
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
    print("Init pose reached. Arm stays fixed for cup filming. Press C to get a task.")
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

    delete_episode_label(env.output_dir, episode_id)

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

    set_end_control 已在控制器侧用当前位姿重建 pose_interp，不会弹回进零力前位姿。
    未提供 init_pose 时只退出零力、保持当前位置。
    """
    print("Zero-force OFF. Leaving drag mode...")
    if hasattr(env, "set_end_control"):
        try:
            env.set_end_control()
            time.sleep(0.4)
        except Exception as exc:  # noqa: BLE001
            print(f"set_end_control warning: {exc}")

    if init_pose6 is None:
        print("No --init_pose given; staying at current pose.")
        return

    move_to_init_pose(
        env,
        init_pose6,
        frequency=frequency,
        duration=duration,
        init_gripper=init_gripper,
    )


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
    cup_lines: list[str] | None = None,
    phase_text: str | None = None,
    cup_overlay_positions: list[tuple[int, int]] | None = None,
    highlight_cups: set[int] | None = None,
    cup_cam_offset_x: int = 0,
    vis_scale: float = 3.0,
    panel_height: int = 220,
) -> np.ndarray:
    """
    上方放大相机画面（仅叠 0/1/2），下方空白面板写指令/状态。
    """
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
    cam = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    cam = np.ascontiguousarray(cam)

    scale = max(float(vis_scale), 1.0)
    if abs(scale - 1.0) > 1e-6:
        cam = cv2.resize(
            cam,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

    if cup_overlay_positions is not None:
        scaled_positions = [
            (int(round(x * scale)), int(round(y * scale)))
            for x, y in cup_overlay_positions
        ]
        draw_cup_slot_markers(
            cam,
            scaled_positions,
            highlight_cups=highlight_cups,
            cam_offset_x=int(round(cup_cam_offset_x * scale)),
            recording=recording,
        )

    if recording:
        if phase_text and phase_text.startswith("ZERO-FORCE"):
            status = "GRASP"
        else:
            status = "REC"
        help_text = phase_text or "F next | D drop | I pose | Q quit"
    elif delete_prompt:
        status = "DEL"
        help_text = delete_prompt
    elif cup_lines:
        status = "TASK"
        help_text = phase_text or "F=next segment | D drop | Q quit"
    elif zero_force_active:
        status = "DRAG"
        help_text = "C task | P->init | I pose | Q quit"
    elif no_zero_force:
        status = "READY"
        help_text = "C task | F after setup | I pose | Q quit"
    else:
        status = "IDLE"
        help_text = "C task | P zero-force | E del | Q quit"

    h, w = cam.shape[:2]
    panel_h = max(int(panel_height), 160)
    panel = np.full((panel_h, w, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Panel is RGB (same as cam); imshow flips to BGR.
    status_color = (255, 0, 0) if recording else (0, 200, 255)
    if n_steps <= FRAMES_TOTAL:
        frames_text = f"frames={n_steps}/{FRAMES_TOTAL}"
    else:
        frames_text = f"frames={n_steps} (obs {FRAMES_TOTAL}+grasp)"
    header = (
        f"{status}  Ep:{episode_id}  t={t_elapsed:.1f}s  {frames_text}"
    )
    cv2.putText(panel, header, (12, 32), font, 0.75, status_color, 2, cv2.LINE_AA)
    cv2.putText(panel, help_text, (12, 64), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    y = 96
    if cup_lines:
        for line in cup_lines:
            cv2.putText(panel, line, (12, y), font, 0.58, (80, 255, 120), 1, cv2.LINE_AA)
            y += 26
    else:
        cv2.putText(
            panel,
            "Cups: 0=left  1=middle  2=right   (markers on image)",
            (12, y),
            font,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        y += 28

    try:
        for robot_idx, (pose, grip) in enumerate(
            tcp_pose_grip_from_obs(obs, n_robots)
        ):
            if y + 20 >= panel_h - 8:
                break
            cv2.putText(
                panel,
                (
                    f"r{robot_idx} xyz "
                    f"{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}  "
                    f"grip {grip:.3f}"
                ),
                (12, y),
                font,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 22
    except Exception:
        pass

    return np.concatenate([cam, panel], axis=0)


def drain_keys(key_source) -> list:
    if key_source is None:
        return []
    return list(key_source.get_press_events())


def is_drop_key(key_stroke) -> bool:
    return key_stroke in (Key.backspace, KeyCode(char="d"))


def is_advance_key(key_stroke) -> bool:
    """F advances observe segments / enters zero-force grasp."""
    return key_stroke == KeyCode(char="f")


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


def phase_overlay(
    task: CupTask, phase: str, move_idx: int, n_steps: int
) -> tuple[list[str], str]:
    """Return (cup_lines, phase_text) for on-screen guidance."""
    lines = [
        f"Init cup {task.initial_cup} ({cup_name(task.initial_cup)})  "
        f"-> final {task.final_cup} ({cup_name(task.final_cup)})",
    ]
    for i, (a, b) in enumerate(task.moves):
        mark = ">" if i == move_idx and phase.startswith("wait_move") else " "
        if phase.startswith("record_move") and i == move_idx:
            mark = "*"
        if phase in ("wait_grasp", "record_grasp"):
            mark = " "
        lines.append(f"{mark} M{i + 1}: swap {a}<->{b}")
    if phase in ("wait_grasp", "record_grasp"):
        mark = ">" if phase == "wait_grasp" else "*"
        lines.append(
            f"{mark} GRASP cup {task.final_cup} ({cup_name(task.final_cup)})  "
            f"frames={n_steps}"
        )

    if phase == "wait_setup":
        help_t = (
            f"Place block under cup {task.initial_cup} ({cup_name(task.initial_cup)}), "
            "then F"
        )
    elif phase == "record_frame0":
        help_t = f"Recording frame 0  ({n_steps}/{FRAMES_TOTAL})"
    elif phase.startswith("wait_move"):
        a, b = task.moves[move_idx]
        help_t = (
            f"Get ready: swap {a} ({cup_name(a)}) <-> {b} ({cup_name(b)}), then F"
        )
    elif phase.startswith("record_move"):
        a, b = task.moves[move_idx]
        seg = n_steps - 1 - move_idx * FRAMES_PER_MOVE
        help_t = (
            f"REC move {move_idx + 1}: swap {a}<->{b}  "
            f"seg {seg}/{FRAMES_PER_MOVE}  total {n_steps}/{FRAMES_TOTAL}"
        )
    elif phase == "wait_grasp":
        help_t = (
            f"Observe done. Press F to enter ZERO-FORCE and grasp cup "
            f"{task.final_cup} ({cup_name(task.final_cup)})"
        )
    elif phase == "record_grasp":
        help_t = (
            f"ZERO-FORCE: grasp cup {task.final_cup} "
            f"({cup_name(task.final_cup)}), then S save | D drop"
        )
    else:
        help_t = f"frames {n_steps}/{FRAMES_TOTAL}"
    return lines, help_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--output",
    "-o",
    default="data/cup_demo",
    show_default=True,
    help="输出目录（replay_buffer.zarr + videos + labels.jsonl）",
)
@click.option(
    "--frequency",
    "-f",
    default=20.0,
    type=float,
    help="采数频率 Hz（默认 10）",
)
@click.option(
    "--frames_per_move",
    default=DEFAULT_FRAMES_PER_MOVE,
    show_default=True,
    type=int,
    help="每次杯子交换录制的帧数；观察段总长 = 1 + 3×该值",
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
    default="-0.006958,-0.002573,0.120019,2.464006,-2.497232,1.524022",
    show_default=True,
    type=str,
    help="启动后缓动到的固定 TCP 位姿: x,y,z,rx,ry,rz（米 & 轴角弧度）",
)
@click.option(
    "--no_init_pose",
    is_flag=True,
    default=False,
    help="启动时不缓动到 --init_pose（保持 home）",
)
@click.option(
    "--init_gripper",
    default=0.1062,
    show_default=True,
    type=float,
    help="启动时目标夹爪开度（米）",
)
@click.option(
    "--keep_gripper",
    is_flag=True,
    default=False,
    help="启动/回 init 时不改夹爪，保持当前开度",
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
    default=True,
    help="不要求先进零力模式，可直接 C 出题（P 仍可选）",
)
@click.option(
    "--quit_home_duration",
    default=3.0,
    type=float,
    show_default=True,
    help="按 Q 退出时缓动回 home 的时长（秒）",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="任务随机种子（可复现出题顺序）",
)
@click.option(
    "--cup_overlay_xy",
    default=None,
    type=str,
    help="预览上 0/1/2 标记位置（224x224 像素）: x0,y0;x1,y1;x2,y2",
)
@click.option(
    "--vis_scale",
    default=3.0,
    type=float,
    show_default=True,
    help="预览窗口相机画面放大倍数（仅显示，不影响落盘分辨率）",
)
def main(
    output,
    frequency,
    frames_per_move,
    camera_reorder,
    obs_width,
    obs_height,
    camera_obs_horizon,
    robot_obs_horizon,
    camera_obs_latency,
    no_mirror,
    mirror_swap,
    terminal_keys,
    save_images,
    init_pose,
    no_init_pose,
    init_gripper,
    keep_gripper,
    init_duration,
    no_zero_force,
    quit_home_duration,
    seed,
    cup_overlay_xy,
    vis_scale,
):
    from peripherals.keystroke_counter import KeystrokeCounter

    set_observe_frame_counts(frames_per_move)
    os.makedirs(output, exist_ok=True)
    dt = 1.0 / frequency
    robots_config = _default_robots_config()
    n_robots = len(robots_config)
    camera_reorder_list = [int(x) for x in camera_reorder.split(",") if x.strip() != ""]
    init_pose6 = None if no_init_pose else parse_pose6(init_pose)
    gripper_target = None if keep_gripper else float(init_gripper)
    rng = random.Random(seed)
    cup_overlay_positions = resolve_cup_overlay_positions(
        obs_width, obs_height, cup_overlay_xy
    )

    print("=" * 60)
    print("ARX5 Cup Shell Data Collection (scripted labels)")
    print(f"  output     : {output}")
    print(f"  frequency  : {frequency} Hz")
    print(
        f"  frames     : observe {FRAMES_TOTAL} "
        f"(= 1 + {N_MOVES}x{FRAMES_PER_MOVE}) + grasp until S"
    )
    print(f"  frames_per_move: {FRAMES_PER_MOVE}")
    print(f"  n_robots   : {n_robots}")
    print(f"  save_images: {save_images}")
    print(f"  labels     : {labels_path(output)}")
    print(f"  seed       : {seed}")
    print(f"  vis_scale  : {vis_scale}x (preview only)")
    if init_pose6 is None:
        print("  init_pose  : (skip)")
    else:
        print(
            f"  init_pose  : {np.array2string(init_pose6, precision=4, separator=',')}"
        )
    print(
        f"  init_gripper: "
        f"{'keep current' if gripper_target is None else f'{gripper_target:.6f} m'}"
    )
    print(
        "  C=task  F=next segment / zero-force  S=save after grasp  "
        "D=drop  E=del  I=pose  Q=home+quit"
    )
    print("  Cups: 0=left  1=middle  2=right  (overlay on preview)")
    if cup_overlay_xy:
        print(f"  cup_overlay_xy: {cup_overlay_xy}")
    else:
        print(
            "  cup_overlay: default normalized positions "
            f"{DEFAULT_CUP_OVERLAY_NORM}"
        )
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
            # record_*/wait_* 分段：只保留写过 action 的步，等待段图像不进 zarr
            drop_uninitialized_actions=True,
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
                    init_gripper=gripper_target,
                )
            else:
                print("No --init_pose. Press C to get a task (arm stays where it is).")

            zero_force_active = False
            pose_print_enabled = False
            last_pose_print_t = 0.0
            pose_print_interval = 0.5
            delete_mode = None  # None | "input" | "confirm"
            delete_buf = ""
            delete_target = None

            print("Ready: press C to sample a cup task.")

            while True:
                # ---------- idle: C -> new task ----------
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
                                print(
                                    "Zero-force ON (optional). Press P again to exit "
                                    "and return to init_pose, or C for a task."
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
                                print("Press P first, or use --no_zero_force.")
                        if key_stroke == KeyCode(char="s"):
                            print("Not recording — nothing to save.")
                        if is_drop_key(key_stroke):
                            print(
                                "No active episode to drop. "
                                "Press E to delete a saved one."
                            )
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
                            cup_overlay_positions=cup_overlay_positions,
                            vis_scale=vis_scale,
                        )
                        cv2.imshow("cup_collect", vis[..., ::-1])
                        cv2.pollKey()
                    except Exception:
                        pass
                    time.sleep(0.05)

                # ---------- cup episode FSM ----------
                episode_id = env.replay_buffer.n_episodes
                task = sample_cup_task(rng)
                print("\n" + "=" * 50)
                print(format_cup_task(task, episode_id=episode_id))
                print("=" * 50)
                print(
                    f"Place block under cup {task.initial_cup} "
                    f"({cup_name(task.initial_cup)}), then press F."
                )

                phase = "wait_setup"
                move_idx = 0
                n_steps = 0
                episode_started = False
                stop_drop = False
                stop_save = False
                loop_t0 = time.monotonic()
                last_tick = loop_t0
                frames_needed = 0

                try:
                    while True:
                        now = time.monotonic()
                        t_elapsed = now - loop_t0
                        t_cycle_end = last_tick + dt

                        for key_stroke in drain_keys(key_source):
                            if is_drop_key(key_stroke):
                                stop_drop = True
                            elif key_stroke == KeyCode(char="s"):
                                if phase == "record_grasp":
                                    stop_save = True
                                else:
                                    print(
                                        "S only during grasp. "
                                        "Finish observe, then F to enter "
                                        "zero-force, then S to save."
                                    )
                            elif key_stroke == KeyCode(char="e"):
                                print("Still in episode — drop first, then press E.")
                            elif key_stroke == KeyCode(char="i"):
                                pose_print_enabled = not pose_print_enabled
                                if pose_print_enabled:
                                    last_pose_print_t = 0.0
                                    print("Pose print ON (press I again to stop).")
                                    print_current_pose(env, tag="rec")
                                else:
                                    print("Pose print OFF.")
                            elif key_stroke == KeyCode(char="q"):
                                print(
                                    "Quit requested — abort current episode, "
                                    "then go home."
                                )
                                if episode_started:
                                    abort_episode(env)
                                go_home_before_quit(
                                    env,
                                    home_targets,
                                    frequency=frequency,
                                    duration=quit_home_duration,
                                    zero_force_active=zero_force_active,
                                )
                                return
                            elif is_advance_key(key_stroke):
                                if phase == "wait_setup":
                                    start_delay = 0.3
                                    eval_t_start = time.time() + start_delay
                                    print(
                                        f"\n>>> Episode {episode_id} START (frame 0)"
                                    )
                                    env.start_episode(eval_t_start)
                                    if not save_images:
                                        env.image_accumulator = None
                                    precise_wait(eval_t_start, time_func=time.time)
                                    episode_started = True
                                    phase = "record_frame0"
                                    frames_needed = 1
                                    last_tick = time.monotonic()
                                elif phase.startswith("wait_move"):
                                    a, b = task.moves[move_idx]
                                    print(
                                        f">>> Recording move {move_idx + 1}/{N_MOVES}: "
                                        f"swap {a}<->{b} ({FRAMES_PER_MOVE} frames)"
                                    )
                                    phase = f"record_move_{move_idx}"
                                    frames_needed = FRAMES_PER_MOVE
                                    last_tick = time.monotonic()
                                elif phase == "wait_grasp":
                                    print(
                                        f">>> Zero-force ON. Grasp cup "
                                        f"{task.final_cup} "
                                        f"({cup_name(task.final_cup)}), "
                                        "then press S to save."
                                    )
                                    try:
                                        env.set_gravity_compensation()
                                        zero_force_active = True
                                    except Exception as exc:  # noqa: BLE001
                                        print(
                                            "set_gravity_compensation "
                                            f"failed: {exc}"
                                        )
                                    phase = "record_grasp"
                                    last_tick = time.monotonic()
                                elif phase == "record_grasp":
                                    print(
                                        "Already in grasp. Drag to cup "
                                        f"{task.final_cup}, then press S to save."
                                    )

                        if stop_drop:
                            if episode_started:
                                abort_episode(env)
                            else:
                                print(
                                    f"Episode {episode_id} cancelled before recording."
                                )
                            if zero_force_active:
                                leave_zero_force_to_init(
                                    env,
                                    init_pose6,
                                    frequency=frequency,
                                    duration=init_duration,
                                    init_gripper=gripper_target,
                                )
                                zero_force_active = False
                            break

                        if stop_save and phase == "record_grasp" and episode_started:
                            if n_steps < FRAMES_TOTAL:
                                print(
                                    f"Warning: only {n_steps} frames before save "
                                    f"(observe expects {FRAMES_TOTAL})."
                                )
                            if save_episode(env, episode_id):
                                append_episode_label(
                                    output, task, episode_id, n_frames=n_steps
                                )
                                print(
                                    f"Episode {episode_id} saved "
                                    f"({n_steps} frames @ {frequency} Hz; "
                                    f"obs={FRAMES_TOTAL} + grasp="
                                    f"{max(0, n_steps - FRAMES_TOTAL)}). "
                                    f"initial={task.initial_cup} "
                                    f"final={task.final_cup} "
                                    f"-> {labels_path(output)}"
                                )
                            leave_zero_force_to_init(
                                env,
                                init_pose6,
                                frequency=frequency,
                                duration=init_duration,
                                init_gripper=gripper_target,
                            )
                            zero_force_active = False
                            break

                        obs = env.get_obs()
                        if pose_print_enabled:
                            if now - last_pose_print_t >= pose_print_interval:
                                print_current_pose(env, tag="rec")
                                last_pose_print_t = now

                        recording_now = phase.startswith("record_")
                        if recording_now and episode_started:
                            record_pose_action(env, obs)
                            n_steps += 1

                            if phase == "record_grasp":
                                # continuous recording until S
                                pass
                            else:
                                frames_needed -= 1
                                if frames_needed <= 0:
                                    if phase == "record_frame0":
                                        move_idx = 0
                                        phase = "wait_move_0"
                                        a, b = task.moves[0]
                                        print(
                                            f"Frame 0 done. Ready for move 1: "
                                            f"swap {a} ({cup_name(a)}) <-> "
                                            f"{b} ({cup_name(b)}). "
                                            "Press F, then swap."
                                        )
                                    elif phase.startswith("record_move"):
                                        if move_idx + 1 >= N_MOVES:
                                            if n_steps != FRAMES_TOTAL:
                                                print(
                                                    f"Warning: recorded {n_steps} "
                                                    f"observe frames, expected "
                                                    f"{FRAMES_TOTAL}."
                                                )
                                            phase = "wait_grasp"
                                            print(
                                                f"Observe done ({n_steps} frames). "
                                                "Press F to enter zero-force and "
                                                f"grasp cup {task.final_cup} "
                                                f"({cup_name(task.final_cup)})."
                                            )
                                        else:
                                            move_idx += 1
                                            phase = f"wait_move_{move_idx}"
                                            a, b = task.moves[move_idx]
                                            print(
                                                f"Move {move_idx} done. Ready for "
                                                f"move {move_idx + 1}: swap "
                                                f"{a} ({cup_name(a)}) <-> "
                                                f"{b} ({cup_name(b)}). "
                                                "Press F, then swap."
                                            )

                        cup_lines, phase_text = phase_overlay(
                            task, phase, move_idx, n_steps
                        )
                        vis = make_vis_image(
                            obs,
                            episode_id,
                            t_elapsed,
                            n_steps,
                            zero_force_active=zero_force_active,
                            no_zero_force=no_zero_force,
                            recording=recording_now,
                            n_robots=n_robots,
                            cup_lines=cup_lines,
                            phase_text=phase_text,
                            cup_overlay_positions=cup_overlay_positions,
                            highlight_cups=highlight_cups_for_phase(
                                task, phase, move_idx
                            ),
                            vis_scale=vis_scale,
                        )
                        cv2.imshow("cup_collect", vis[..., ::-1])
                        cv2.pollKey()

                        if recording_now:
                            precise_wait(t_cycle_end, time_func=time.monotonic)
                            last_tick = t_cycle_end
                        else:
                            time.sleep(0.05)
                            last_tick = time.monotonic()

                except KeyboardInterrupt:
                    print("Interrupted — abort episode.")
                    if episode_started:
                        abort_episode(env)
                    if zero_force_active:
                        leave_zero_force_to_init(
                            env,
                            init_pose6,
                            frequency=frequency,
                            duration=init_duration,
                            init_gripper=gripper_target,
                        )
                    else:
                        leave_drag_mode(env)
                    zero_force_active = False

                print(
                    "Ready for next episode. Press C for a new task, "
                    "E to delete a saved episode, or Q to quit.\n"
                )

if __name__ == "__main__":
    main()
