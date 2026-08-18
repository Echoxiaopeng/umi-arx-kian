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
    python scripts/collect_drag_umi.py -o data/collect_demo

    # SSH 无 X11 时用终端按键:
    python scripts/collect_drag_umi.py -o data/collect_demo --terminal_keys

    # 如果上面报错，则执行下面：
    cd /data1/projects/umi-arx-hzl/modules/arx_x5_python
    source setup.sh
    cd /data1/projects/umi-arx-hzl
    python scripts/collect_drag_umi.py

键盘:
  P = 进入零力模式（重力补偿，可拖拽）
  C = 开始采集（须先按 P）
  S = 停止并保存
  D = 丢弃当前正在录的 episode（未保存）
  E = 删除已保存的指定 episode（空闲时；输入编号后 Y 确认）
  Q = 退出

不做 reset_to_home / init_delta 作为 episode 起始；按 C 时臂在哪就从哪开录。
进程启动时仍会 reset_to_home 一次（回零）。

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
            "p=零力, c=开始, s=保存, d=丢弃当前, e=删已保存, q=退出"
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


def make_vis_image(
    obs: dict,
    episode_id: int,
    t_elapsed: float,
    n_steps: int,
    *,
    drag_ready: bool = False,
    recording: bool = False,
    delete_prompt: str | None = None,
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
        help_text = "S save | D drop | Q quit"
    elif delete_prompt:
        status = "DEL"
        help_text = delete_prompt
    elif drag_ready:
        status = "DRAG"
        help_text = "C start | E del ep | Q quit"
    else:
        status = "IDLE"
        help_text = "P zero-force | E del ep | Q quit"
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
):
    from peripherals.keystroke_counter import KeystrokeCounter

    os.makedirs(output, exist_ok=True)
    dt = 1.0 / frequency
    robots_config = _default_robots_config()
    n_robots = len(robots_config)
    camera_reorder_list = [int(x) for x in camera_reorder.split(",") if x.strip() != ""]

    print("=" * 60)
    print("ARX5 Drag Data Collection")
    print(f"  output     : {output}")
    print(f"  frequency  : {frequency} Hz")
    print(f"  n_robots   : {n_robots}")
    print(f"  save_images: {save_images}")
    print("  startup: reset_to_home once; episode: no init move")
    print("  P=zero-force  C=start  S=save  D=drop-current  E=del-saved  Q=quit")
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
            print("Env ready. Press P for zero-force, then C to start recording.")

            leave_drag_mode(env)
            drag_ready = False
            delete_mode = None  # None | "input" | "confirm"
            delete_buf = ""
            delete_target = None

            while True:
                # ---------- idle: P -> zero-force, then C -> start ----------
                start_episode = False
                while True:
                    for key_stroke in drain_keys(key_source):
                        if key_stroke == KeyCode(char="q"):
                            print("Quit.")
                            leave_drag_mode(env)
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
                            if not drag_ready:
                                env.set_gravity_compensation()
                                drag_ready = True
                                print("Zero-force ON. Drag freely, then press C to record.")
                            else:
                                print("Already in zero-force mode.")
                        if key_stroke == KeyCode(char="c"):
                            if drag_ready:
                                start_episode = True
                            else:
                                print("Press P first to enter zero-force mode.")
                        if key_stroke == KeyCode(char="s"):
                            print("Not recording — nothing to save.")
                        if is_drop_key(key_stroke):
                            print("No active episode to drop. Press E to delete a saved one.")
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
                            drag_ready=drag_ready,
                            recording=False,
                            delete_prompt=delete_prompt,
                        )
                        cv2.imshow("drag_collect", vis[..., ::-1])
                        cv2.pollKey()
                    except Exception:
                        pass
                    time.sleep(0.05)

                # ---------- recording（已在零力模式，不再重复 set_gravity） ----------
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
                            elif key_stroke == KeyCode(char="q"):
                                print("Quit requested — abort current episode.")
                                abort_episode(env)
                                leave_drag_mode(env)
                                return

                        obs = env.get_obs()
                        record_pose_action(env, obs)
                        n_steps += 1

                        vis = make_vis_image(
                            obs,
                            episode_id,
                            t_elapsed,
                            n_steps,
                            drag_ready=True,
                            recording=True,
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
                    drag_ready = False

                if drag_ready:
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
