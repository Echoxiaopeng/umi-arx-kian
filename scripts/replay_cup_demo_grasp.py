#!/usr/bin/env python3
"""Replay the grasp segment of a cup_demo episode on the real arm.

Purpose
-------
Verify whether the control-layer Direct↔ZMQ conversion is wrong by replaying
the *recorded* grasp TCP trajectory (not the policy).

Two backends
------------
  direct  : send recorded poses as-is via Arx5ControllerDirect (same as collect)
  zmq     : convert Direct link6 → ZMQ eef (origin + eef_offset), then send via ZMQ

Examples
--------
  # Inspect only (no motion)
  python scripts/replay_cup_demo_grasp.py --dry-run

  # Replay with Direct (same stack as collect_drag_umi_cup)
  python scripts/replay_cup_demo_grasp.py --backend direct --can-port can0

  # Replay with ZMQ after Direct→ZMQ conversion
  python scripts/replay_cup_demo_grasp.py --backend zmq \
    --direct-zmq-origin 0.094905,-0.002323,0.158023 --eef-offset 0.145

  # Calibrate origin from current ZMQ pose vs demo grasp-start xyz
  python scripts/replay_cup_demo_grasp.py --backend zmq --calibrate-direct-origin \\
    --eef-offset 0.145
"""

from __future__ import annotations

import json
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import click
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

import os

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from utils.shellgame_openpi import direct_link6_pose_to_zmq_eef  # noqa: E402
from utils.shellgame_openpi import eef_pose_to_link6  # noqa: E402
from utils.shellgame_openpi import parse_csv_vector  # noqa: E402

DEFAULT_DEMO = ROOT_DIR / "data" / "cup_demo"
DEFAULT_EEF_OFFSET = 0.145
DEFAULT_ZMQ_GRIPPER_MIN = 0.0
DEFAULT_ZMQ_GRIPPER_MAX = 0.085
# Direct SingleArmClient: real_pos/5*0.11 → width at fully open real_pos=5
DEFAULT_DIRECT_GRIPPER_FULL_OPEN = 0.11


def invert_gripper_width(
    grip: np.ndarray | float,
    *,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray:
    """Direct/train polarity ↔ ZMQ physical opening: g' = min + max - g."""
    return gripper_min + gripper_max - np.asarray(grip, dtype=np.float64)


def print_gripper_motion_summary(label: str, grippers: np.ndarray) -> None:
    """Explain whether the recorded/commanded gripper actually closes."""
    g = np.asarray(grippers, dtype=np.float64).reshape(-1)
    if len(g) == 0:
        return
    i_min = int(np.argmin(g))
    i_max = int(np.argmax(g))
    delta = float(g[-1] - g[0])
    span = float(g.max() - g.min())
    print(
        f"[{label}] gripper: start={g[0]:.4f} end={g[-1]:.4f} "
        f"min={g.min():.4f}@i{i_min} max={g.max():.4f}@i{i_max} "
        f"Δend-start={delta:+.4f} span={span:.4f} "
        f"(larger=more open for Direct/ZMQ physical width)"
    )
    if span < 0.005:
        print(f"[{label}] WARNING: gripper span < 5mm — almost no motion in recording")
    elif delta > 0.003 and i_max > i_min:
        print(f"[{label}] NOTE: trajectory ends MORE open than start (not a close-at-end grasp)")
    elif delta < -0.003:
        print(f"[{label}] NOTE: trajectory ends more closed than start")


def convert_grippers_for_backend(
    grippers: np.ndarray,
    *,
    backend: str,
    invert_gripper: bool,
    gripper_min: float,
    gripper_max: float,
    gripper_map: str,
    direct_gripper_full_open: float,
) -> np.ndarray:
    """Map recorded Direct gripper widths into the command space of the backend."""
    raw = np.asarray(grippers, dtype=np.float64).copy()
    if backend == "direct":
        return raw

    if backend != "zmq":
        raise ValueError(backend)

    out = raw
    if invert_gripper:
        out = invert_gripper_width(out, gripper_min=gripper_min, gripper_max=gripper_max)

    gripper_map = gripper_map.lower()
    if gripper_map == "scale":
        if direct_gripper_full_open <= 0:
            raise ValueError("direct_gripper_full_open must be positive")
        factor = (gripper_max - gripper_min) / float(direct_gripper_full_open)
        out = gripper_min + out * factor
        print(
            f"[GRIP] scale mode: Direct width × {factor:.4f} "
            f"(full-open {direct_gripper_full_open:.3f}m → ZMQ {gripper_max:.3f}m); "
            f"start {float(raw[0]):.4f} → {float(out[0]):.4f}"
        )
    elif gripper_map == "clip":
        n_outside = int(np.sum((out < gripper_min) | (out > gripper_max)))
        if n_outside:
            print(
                f"[GRIP] clip mode: saturating {n_outside}/{len(out)} values to "
                f"[{gripper_min:.4f}, {gripper_max:.4f}] "
                f"(raw span [{float(raw.min()):.4f}, {float(raw.max()):.4f}])"
            )
        out = np.clip(out, gripper_min, gripper_max)
    elif gripper_map == "linear":
        src_min = float(out.min())
        src_max = float(out.max())
        if src_max - src_min < 1e-6:
            out = np.full_like(out, gripper_max)
            print("[GRIP] linear map: constant gripper in recording; commanding max open")
        else:
            out = gripper_min + (out - src_min) / (src_max - src_min) * (gripper_max - gripper_min)
            print(
                f"[GRIP] linear map: episode min→closed, max→open: "
                f"Direct [{src_min:.4f}, {src_max:.4f}] → ZMQ [{gripper_min:.4f}, {gripper_max:.4f}] "
                f"(start {float(raw[0]):.4f} → {float(out[0]):.4f}; "
                "use only if you want full-range stretch, not absolute opening)"
            )
    elif gripper_map == "none":
        out = out.copy()
    else:
        raise ValueError(f"unknown gripper_map: {gripper_map}")

    return np.clip(out, gripper_min, gripper_max)


DEFAULT_INIT_POSE = "-0.006958,-0.002573,0.120019,2.464006,-2.497232,1.524022"
DEFAULT_INIT_GRIPPER = 0.085


def canonical_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Stable rotvec for controller commands (geometry unchanged)."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    return np.asarray(Rotation.from_matrix(Rotation.from_rotvec(rotvec).as_matrix()).as_rotvec())


def wait_for_enter(prompt: str) -> None:
    """Block until Enter; works when stdin is a TTY or piped."""
    print(prompt, flush=True)
    if sys.stdin.isatty():
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                line = sys.stdin.readline()
                if line == "" or line.endswith("\n"):
                    return
    else:
        sys.stdin.readline()


@contextmanager
def terminal_cbreak():
    """Single-key stdin reads without Enter (TTY only)."""
    if not sys.stdin.isatty():
        yield False
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)


def poll_hold_input(*, timeout_sec: float, cbreak: bool) -> str | None:
    """Return 'replay', 'damping', or None if no input within timeout."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if not ready:
        return None
    if cbreak:
        ch = sys.stdin.read(1)
        if ch in ("\x03",):
            raise KeyboardInterrupt
        if ch in ("\n", "\r"):
            return "damping"
        if ch in ("r", "R"):
            return "replay"
        return None
    line = sys.stdin.readline()
    if line == "":
        return "damping"
    if line.strip().lower() == "r":
        return "replay"
    return "damping"


def merge_action_and_eef_poses(
    pos: np.ndarray,
    rot: np.ndarray,
    grip: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefer recorded action commands; fall back to obs eef when action rows are zero."""
    n = len(pos)
    poses = np.zeros((n, 6), dtype=np.float64)
    grippers = np.zeros(n, dtype=np.float64)
    n_action = 0
    n_eef = 0
    for i in range(n):
        act = np.asarray(action[i], dtype=np.float64).reshape(7)
        use_action = np.linalg.norm(act[:3]) > 1e-6 or act[6] > 1e-6
        if use_action:
            poses[i, :3] = act[:3]
            poses[i, 3:6] = canonical_rotvec(act[3:6])
            grippers[i] = act[6]
            n_action += 1
        else:
            poses[i, :3] = pos[i]
            poses[i, 3:6] = canonical_rotvec(rot[i])
            grippers[i] = grip[i]
            n_eef += 1
    print(f"[traj] merged source: action rows={n_action}, eef fallback rows={n_eef}")
    return poses, grippers


def ease_in_out_quad(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(value < 0.5, 2.0 * value * value, 1.0 - ((-2.0 * value + 2.0) ** 2) / 2.0)


def load_labels(demo_dir: Path) -> list[dict]:
    path = demo_dir / "labels.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_episode_arrays(demo_dir: Path, episode_id: int) -> dict[str, np.ndarray]:
    import zarr

    zarr_path = demo_dir / "replay_buffer.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"missing {zarr_path}")
    root = zarr.open(str(zarr_path), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if not 0 <= episode_id < len(ends):
        raise IndexError(f"episode {episode_id} out of range [0, {len(ends)})")
    start = 0 if episode_id == 0 else int(ends[episode_id - 1])
    end = int(ends[episode_id])
    data = root["data"]
    return {
        "action": np.asarray(data["action"][start:end], dtype=np.float64),
        "eef_pos": np.asarray(data["robot0_eef_pos"][start:end], dtype=np.float64),
        "eef_rot": np.asarray(data["robot0_eef_rot_axis_angle"][start:end], dtype=np.float64),
        "gripper": np.asarray(data["robot0_gripper_width"][start:end], dtype=np.float64).reshape(-1),
        "timestamp": np.asarray(data["timestamp"][start:end], dtype=np.float64),
    }


def build_grasp_trajectory(
    arrays: dict[str, np.ndarray],
    *,
    grasp_start_frame: int,
    source: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (poses Nx6, grippers N, dt_median). Prefer eef_* over action zeros."""
    n = len(arrays["eef_pos"])
    if not 0 <= grasp_start_frame < n:
        raise ValueError(f"grasp_start_frame={grasp_start_frame} out of range for n={n}")

    pos = arrays["eef_pos"][grasp_start_frame:]
    rot = arrays["eef_rot"][grasp_start_frame:]
    grip = arrays["gripper"][grasp_start_frame:]
    action = arrays["action"][grasp_start_frame:]
    ts = arrays["timestamp"][grasp_start_frame:]

    if source == "action":
        poses = action[:, :6].copy()
        grippers = action[:, 6].copy()
        bad = np.linalg.norm(poses[:, :3], axis=1) < 1e-6
        poses[bad, :3] = pos[bad]
        poses[bad, 3:6] = np.stack([canonical_rotvec(r) for r in rot[bad]], axis=0)
        grippers[bad] = grip[bad]
        for i in range(len(poses)):
            poses[i, 3:6] = canonical_rotvec(poses[i, 3:6])
    elif source == "auto":
        poses, grippers = merge_action_and_eef_poses(pos, rot, grip, action)
    elif source == "eef":
        poses = np.concatenate([pos, np.stack([canonical_rotvec(r) for r in rot], axis=0)], axis=-1)
        grippers = grip.copy()
    else:
        raise ValueError(f"unknown source: {source}")

    if len(ts) >= 2:
        dt = float(np.median(np.diff(ts)))
    else:
        dt = 0.1
    if not np.isfinite(dt) or dt <= 1e-4:
        dt = 0.1
    return poses, grippers.astype(np.float64), dt


def convert_poses_for_backend(
    poses: np.ndarray,
    *,
    backend: str,
    direct_origin3: np.ndarray | None,
    eef_offset: float,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if backend == "direct":
        return poses.copy()
    if backend != "zmq":
        raise ValueError(backend)
    if direct_origin3 is None:
        raise ValueError("zmq backend requires --direct-zmq-origin or --calibrate-direct-origin")
    out = np.zeros_like(poses)
    for i, pose in enumerate(poses):
        out[i] = direct_link6_pose_to_zmq_eef(pose, direct_origin3, eef_offset)
    return out


def print_traj_summary(
    *,
    label: str,
    poses: np.ndarray,
    grippers: np.ndarray,
    dt: float,
) -> None:
    xyz = poses[:, :3]
    path = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))) if len(xyz) > 1 else 0.0
    tcp_z0 = Rotation.from_rotvec(poses[0, 3:6]).as_matrix()[:, 2]
    tcp_z1 = Rotation.from_rotvec(poses[-1, 3:6]).as_matrix()[:, 2]
    print(f"[{label}] n={len(poses)} dt≈{dt:.4f}s duration≈{len(poses) * dt:.2f}s")
    print(f"[{label}] start xyz={np.array2string(xyz[0], precision=4)} grip={grippers[0]:.4f}")
    print(f"[{label}] end   xyz={np.array2string(xyz[-1], precision=4)} grip={grippers[-1]:.4f}")
    print(
        f"[{label}] Δxyz={np.array2string(xyz[-1] - xyz[0], precision=4)} "
        f"path_len={path:.4f}m z=[{xyz[:, 2].min():.4f},{xyz[:, 2].max():.4f}]"
    )
    print(f"[{label}] tcp_z_world start={np.array2string(tcp_z0, precision=4)}")
    print(f"[{label}] tcp_z_world end  ={np.array2string(tcp_z1, precision=4)}")


def move_smooth(
    robot,
    target_pose6: np.ndarray,
    target_gripper: float,
    *,
    duration: float,
    frequency: float,
) -> None:
    state = robot.get_state()
    start_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
    start_grip = float(state["gripper_position"])
    target = np.asarray(target_pose6, dtype=np.float64).reshape(6)
    steps = max(2, int(round(duration * frequency)))
    blend = ease_in_out_quad(np.linspace(0.0, 1.0, steps))
    positions = start_pose[:3] + blend[:, None] * (target[:3] - start_pose[:3])
    rots = Slerp(
        [0.0, 1.0],
        Rotation.from_rotvec(np.stack([start_pose[3:], target[3:]], axis=0)),
    )(blend).as_rotvec()
    grips = start_grip + blend * (float(target_gripper) - start_grip)
    t0 = time.time() + 0.25
    for i in range(steps):
        pose = np.concatenate([positions[i], rots[i]])
        robot.schedule_waypoint(
            pose=pose,
            gripper_pos=float(grips[i]),
            target_time=t0 + i / frequency,
        )
    time.sleep(max(0.0, t0 + (steps - 1) / frequency - time.time()) + 0.3)


def replay_waypoints(
    robot,
    poses: np.ndarray,
    grippers: np.ndarray,
    *,
    dt: float,
    speed: float,
    lead_time: float,
    sample_hz: float,
) -> dict[str, float]:
    """Schedule absolute TCP waypoints; sample tracking error while playing."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    play_dt = float(dt) / float(speed)
    t0 = time.time() + float(lead_time)
    timestamps = t0 + np.arange(len(poses), dtype=np.float64) * play_dt
    for pose, grip, ts in zip(poses, grippers, timestamps, strict=True):
        robot.schedule_waypoint(pose=pose, gripper_pos=float(grip), target_time=float(ts))

    err_pos: list[float] = []
    err_rot: list[float] = []
    err_grip: list[float] = []
    sample_dt = 1.0 / max(1.0, sample_hz)
    deadline = float(timestamps[-1]) + 0.35
    idx = 0
    while time.time() < deadline:
        now = time.time()
        while idx + 1 < len(timestamps) and timestamps[idx + 1] <= now:
            idx += 1
        state = robot.get_state()
        cur = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
        cur_grip = float(state["gripper_position"])
        tgt = poses[idx]
        err_pos.append(float(np.linalg.norm(cur[:3] - tgt[:3])))
        err_rot.append(
            float((Rotation.from_rotvec(cur[3:]) * Rotation.from_rotvec(tgt[3:]).inv()).magnitude())
        )
        err_grip.append(abs(cur_grip - float(grippers[idx])))
        time.sleep(sample_dt)

    # Hold final pose briefly.
    robot.servoL(pose=poses[-1], gripper_pos=float(grippers[-1]), duration=max(0.1, play_dt))
    time.sleep(0.2)
    final_state = robot.get_state()
    final = np.asarray(final_state["ActualTCPPose"], dtype=np.float64).reshape(6)
    final_grip = float(final_state["gripper_position"])
    final_err = float(np.linalg.norm(final[:3] - poses[-1, :3]))
    return {
        "mean_pos_err": float(np.mean(err_pos)) if err_pos else float("nan"),
        "max_pos_err": float(np.max(err_pos)) if err_pos else float("nan"),
        "mean_rot_err": float(np.mean(err_rot)) if err_rot else float("nan"),
        "max_rot_err": float(np.max(err_rot)) if err_rot else float("nan"),
        "mean_grip_err": float(np.mean(err_grip)) if err_grip else float("nan"),
        "max_grip_err": float(np.max(err_grip)) if err_grip else float("nan"),
        "final_pos_err": final_err,
        "final_grip": final_grip,
        "final_grip_cmd": float(grippers[-1]),
        "final_grip_err": abs(final_grip - float(grippers[-1])),
    }


def print_replay_stats(stats: dict[str, float]) -> None:
    print("[replay] tracking stats:")
    print(
        f"  mean_pos_err={stats['mean_pos_err']*1000:.1f} mm  "
        f"max_pos_err={stats['max_pos_err']*1000:.1f} mm"
    )
    print(
        f"  mean_rot_err={np.rad2deg(stats['mean_rot_err']):.2f} deg  "
        f"max_rot_err={np.rad2deg(stats['max_rot_err']):.2f} deg"
    )
    print(
        f"  mean_grip_err={stats['mean_grip_err']*1000:.1f} mm  "
        f"max_grip_err={stats['max_grip_err']*1000:.1f} mm  "
        f"final grip live={stats['final_grip']:.4f} cmd={stats['final_grip_cmd']:.4f}"
    )
    print(f"  final_pos_err={stats['final_pos_err']*1000:.1f} mm")
    if stats["final_pos_err"] < 0.02 and stats["max_pos_err"] < 0.05:
        print("[verdict] pose tracking OK")
    else:
        print(
            "[verdict] pose tracking poor — check conversion / controller / "
            "whether arm started at the demo grasp pose"
        )
    if stats["max_grip_err"] > 0.015:
        print(
            "[verdict] gripper tracking poor — try --invert-gripper if polarity is wrong, "
            "or check ZMQ gripper range"
        )
    elif stats["max_grip_err"] <= 0.015:
        print("[verdict] gripper tracking OK")


def approach_grasp_start(
    robot,
    cmd_poses: np.ndarray,
    grippers: np.ndarray,
    *,
    approach_duration: float,
) -> None:
    print(
        f"[approach] moving to grasp-start ({approach_duration:.1f}s), "
        f"grip_cmd={float(grippers[0]):.4f} ..."
    )
    move_smooth(
        robot,
        cmd_poses[0],
        float(grippers[0]),
        duration=approach_duration,
        frequency=20.0,
    )
    after_state = robot.get_state()
    after = np.asarray(after_state["ActualTCPPose"], dtype=np.float64).reshape(6)
    after_grip = float(after_state["gripper_position"])
    approach_err = float(np.linalg.norm(after[:3] - cmd_poses[0, :3]))
    print(
        f"[approach] done; |Δxyz|={approach_err * 1000:.1f} mm  "
        f"grip live={after_grip:.4f} cmd={float(grippers[0]):.4f} "
        f"|Δgrip|={abs(after_grip - float(grippers[0])) * 1000:.1f} mm"
    )
    if approach_err > 0.03:
        raise RuntimeError(
            f"Approach missed grasp-start by {approach_err:.3f}m; aborting replay for safety"
        )


def replay_grasp_segment(
    robot,
    cmd_poses: np.ndarray,
    grippers: np.ndarray,
    *,
    dt: float,
    speed: float,
    lead_time: float,
    sample_hz: float,
) -> dict[str, float]:
    print(f"[replay] scheduling {len(cmd_poses)} waypoints at speed={speed:.2f} ...")
    stats = replay_waypoints(
        robot,
        cmd_poses,
        grippers,
        dt=dt,
        speed=speed,
        lead_time=lead_time,
        sample_hz=sample_hz,
    )
    print_replay_stats(stats)
    return stats


def make_robot(backend: str, shm_manager: SharedMemoryManager, **kwargs):
    if backend == "direct":
        from modules.arx5_controller_direct import Arx5ControllerDirect

        return Arx5ControllerDirect(
            shm_manager=shm_manager,
            can_port=kwargs["can_port"],
            arm_type=int(kwargs["arm_type"]),
            frequency=200,
            verbose=True,
            reset_to_home_on_start=False,
        )
    if backend == "zmq":
        from modules.arx5_controller import Arx5Controller

        return Arx5Controller(
            shm_manager=shm_manager,
            robot_ip=kwargs["robot_ip"],
            robot_port=int(kwargs["robot_port"]),
            frequency=200,
            verbose=True,
        )
    raise ValueError(backend)


@click.command()
@click.option("--demo-dir", type=click.Path(path_type=Path), default=DEFAULT_DEMO, show_default=True)
@click.option("--episode", default=0, show_default=True, type=int)
@click.option(
    "--backend",
    type=click.Choice(["direct", "zmq"], case_sensitive=False),
    required=False,
    default=None,
    help="direct=raw recorded poses; zmq=Direct→ZMQ convert then send.",
)
@click.option(
    "--source",
    type=click.Choice(["auto", "eef", "action"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="auto=action when valid else eef (best for Direct); eef=obs only; action=command only.",
)
@click.option("--direct-zmq-origin", default=None, type=str, help="x,y,z meters for Direct→ZMQ.")
@click.option(
    "--calibrate-direct-origin",
    is_flag=True,
    help="With zmq: origin = live_zmq_link6_xyz - demo_grasp_start_xyz (arm must match grasp start).",
)
@click.option("--eef-offset", default=DEFAULT_EEF_OFFSET, show_default=True, type=float)
@click.option(
    "--invert-gripper/--no-invert-gripper",
    default=False,
    show_default=True,
    help=(
        "ZMQ only: g' = min+max-g. Use if Direct recording is larger=close while "
        "ZMQ is larger=open (same as eval_arx5_light_wam)."
    ),
)
@click.option(
    "--gripper-min-width",
    default=DEFAULT_ZMQ_GRIPPER_MIN,
    show_default=True,
    type=float,
    help="ZMQ gripper clip/invert lower bound (meters, larger=more open).",
)
@click.option(
    "--gripper-max-width",
    default=DEFAULT_ZMQ_GRIPPER_MAX,
    show_default=True,
    type=float,
    help="ZMQ gripper clip/invert upper bound (meters, larger=more open).",
)
@click.option(
    "--gripper-map",
    type=click.Choice(["scale", "clip", "linear", "none"], case_sensitive=False),
    default="scale",
    show_default=True,
    help=(
        "ZMQ only: scale=Direct×(zmq_max/0.11) keeps absolute opening (recommended); "
        "clip=saturate at max; linear=stretches episode min→closed (start may look closed); "
        "none=pass through."
    ),
)
@click.option(
    "--direct-gripper-full-open",
    default=DEFAULT_DIRECT_GRIPPER_FULL_OPEN,
    show_default=True,
    type=float,
    help="Direct collector full-open width (m) for --gripper-map=scale.",
)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot-port", default=8765, show_default=True, type=int)
@click.option("--can-port", default="can0", show_default=True)
@click.option("--arm-type", default=0, show_default=True, type=int)
@click.option("--approach-duration", default=3.0, show_default=True, type=float)
@click.option(
    "--move-to-init/--no-move-to-init",
    default=False,
    show_default=True,
    help="Direct:缓动到采集默认 init_pose 后再去 grasp-start（臂在 home 时建议开启）。",
)
@click.option(
    "--init-pose",
    default=DEFAULT_INIT_POSE,
    show_default=True,
    type=str,
    help="--move-to-init 使用的 Direct TCP 位姿 x,y,z,rx,ry,rz。",
)
@click.option("--init-gripper", default=DEFAULT_INIT_GRIPPER, show_default=True, type=float)
@click.option("--init-duration", default=3.0, show_default=True, type=float)
@click.option("--speed", default=1.0, show_default=True, type=float, help=">1 faster, <1 slower.")
@click.option("--lead-time", default=0.35, show_default=True, type=float)
@click.option("--sample-hz", default=20.0, show_default=True, type=float)
@click.option("--dry-run", is_flag=True, help="Print trajectory / conversion only; no robot motion.")
@click.option("--yes", "-y", is_flag=True, help="Skip interactive confirm.")
def main(
    demo_dir: Path,
    episode: int,
    backend: str | None,
    source: str,
    direct_zmq_origin: str | None,
    calibrate_direct_origin: bool,  # noqa: FBT001
    eef_offset: float,
    invert_gripper: bool,  # noqa: FBT001
    gripper_min_width: float,
    gripper_max_width: float,
    gripper_map: str,
    direct_gripper_full_open: float,
    robot_ip: str,
    robot_port: int,
    can_port: str,
    arm_type: int,
    approach_duration: float,
    move_to_init: bool,  # noqa: FBT001
    init_pose: str,
    init_gripper: float,
    init_duration: float,
    speed: float,
    lead_time: float,
    sample_hz: float,
    dry_run: bool,  # noqa: FBT001
    yes: bool,  # noqa: FBT001
) -> None:
    demo_dir = demo_dir.expanduser().resolve()
    labels = load_labels(demo_dir)
    label = next((row for row in labels if int(row["episode_id"]) == int(episode)), None)
    if label is None:
        raise click.ClickException(f"episode {episode} not found in {demo_dir / 'labels.jsonl'}")
    if not gripper_min_width < gripper_max_width:
        raise click.BadParameter("gripper-min-width must be < gripper-max-width")

    arrays = load_episode_arrays(demo_dir, episode)
    grasp_start = int(label.get("grasp_start_frame", 0))
    raw_poses, raw_grippers, dt = build_grasp_trajectory(
        arrays, grasp_start_frame=grasp_start, source=source.lower()
    )

    print("=" * 64)
    print(f"demo={demo_dir}")
    print(f"episode={episode} label={label}")
    print(f"zarr_frames={len(arrays['eef_pos'])} grasp_start={grasp_start} source={source}")
    print_traj_summary(label="recorded_direct", poses=raw_poses, grippers=raw_grippers, dt=dt)
    print_gripper_motion_summary("recorded_direct", raw_grippers)

    if backend is None:
        if dry_run:
            backend = "direct"
            print("[dry-run] backend not set; showing recorded Direct poses only")
        else:
            raise click.ClickException("Please pass --backend direct|zmq (or --dry-run)")

    backend = backend.lower()
    direct_origin3: np.ndarray | None = None
    if direct_zmq_origin is not None:
        direct_origin3 = parse_csv_vector(direct_zmq_origin, size=3, name="--direct-zmq-origin")

    if backend == "zmq" and dry_run and direct_origin3 is None and not calibrate_direct_origin:
        raise click.ClickException(
            "zmq dry-run needs --direct-zmq-origin (or run without dry-run + --calibrate-direct-origin)"
        )

    grippers = convert_grippers_for_backend(
        raw_grippers,
        backend=backend,
        invert_gripper=invert_gripper and backend == "zmq",
        gripper_min=gripper_min_width,
        gripper_max=gripper_max_width,
        gripper_map=gripper_map if backend == "zmq" else "none",
        direct_gripper_full_open=direct_gripper_full_open,
    )
    if backend == "zmq":
        print_gripper_motion_summary("command_zmq", grippers)
        print(
            f"[GRIP] zmq: invert={invert_gripper} map={gripper_map} "
            f"clip=[{gripper_min_width:.4f},{gripper_max_width:.4f}] "
            f"cmd span=[{float(grippers.min()):.4f},{float(grippers.max()):.4f}]"
        )

    # For dry-run zmq with known origin, convert and print now.
    if dry_run:
        if backend == "zmq":
            assert direct_origin3 is not None
            cmd_poses = convert_poses_for_backend(
                raw_poses,
                backend="zmq",
                direct_origin3=direct_origin3,
                eef_offset=eef_offset,
            )
            print(
                f"[ALIGN] direct-zmq-origin={np.array2string(direct_origin3, precision=6)} "
                f"eef_offset={eef_offset:.3f}"
            )
            print_traj_summary(label="command_zmq", poses=cmd_poses, grippers=grippers, dt=dt)
        print("[dry-run] done; no robot motion.")
        return

    if not yes:
        click.confirm(
            f"About to move the arm with backend={backend}. Cups/workspace clear?",
            abort=True,
        )

    with SharedMemoryManager() as shm_manager:
        robot = make_robot(
            backend,
            shm_manager,
            can_port=can_port,
            arm_type=arm_type,
            robot_ip=robot_ip,
            robot_port=robot_port,
        )
        robot.start(wait=True)
        damped = False
        try:
            while not robot.is_ready:
                time.sleep(0.05)

            if backend == "direct" and hasattr(robot, "set_end_control"):
                robot.set_end_control()
                time.sleep(0.1)

            if backend == "direct" and move_to_init:
                init_pose6 = parse_csv_vector(init_pose, size=6, name="--init-pose")
                print(
                    f"[init] moving to collect init_pose "
                    f"{np.array2string(init_pose6, precision=4)} ({init_duration:.1f}s)..."
                )
                move_smooth(
                    robot,
                    init_pose6,
                    float(init_gripper),
                    duration=init_duration,
                    frequency=20.0,
                )

            live = np.asarray(robot.get_state()["ActualTCPPose"], dtype=np.float64).reshape(6)
            live_grip = float(robot.get_state()["gripper_position"])
            print(f"[live] pose={np.array2string(live, precision=4)} grip={live_grip:.4f}")

            if backend == "zmq" and calibrate_direct_origin:
                live_link6 = eef_pose_to_link6(live, eef_offset)
                direct_origin3 = live_link6[:3] - raw_poses[0, :3]
                print(
                    "[ALIGN] calibrated direct-zmq-origin from live ZMQ vs demo grasp-start: "
                    f"{np.array2string(direct_origin3, precision=6, separator=',')}"
                )

            cmd_poses = convert_poses_for_backend(
                raw_poses,
                backend=backend,
                direct_origin3=direct_origin3,
                eef_offset=eef_offset,
            )
            if backend == "zmq":
                print(
                    f"[ALIGN] direct-zmq-origin={np.array2string(direct_origin3, precision=6)} "
                    f"eef_offset={eef_offset:.3f}"
                )
                print_traj_summary(label="command_zmq", poses=cmd_poses, grippers=grippers, dt=dt)

            approach_grasp_start(
                robot,
                cmd_poses,
                grippers,
                approach_duration=approach_duration,
            )
            replay_grasp_segment(
                robot,
                cmd_poses,
                grippers,
                dt=dt,
                speed=speed,
                lead_time=lead_time,
                sample_hz=sample_hz,
            )

            hold_pose = cmd_poses[-1].copy()
            hold_grip = float(grippers[-1])
            hold_hint = (
                "Press r to replay grasp, Enter to enter damping"
                if sys.stdin.isatty()
                else "Type r + Enter to replay grasp, Enter alone for damping"
            )
            print(
                f"[hold] keeping final cmd pose + grip={hold_grip:.4f}. {hold_hint}..."
            )
            enter_damping = False
            with terminal_cbreak() as cbreak:
                while not enter_damping:
                    robot.servoL(pose=hold_pose, gripper_pos=hold_grip, duration=0.2)
                    action = poll_hold_input(timeout_sec=0.15, cbreak=cbreak)
                    if action == "replay":
                        print("[hold] r pressed — replaying grasp segment...")
                        approach_grasp_start(
                            robot,
                            cmd_poses,
                            grippers,
                            approach_duration=approach_duration,
                        )
                        replay_grasp_segment(
                            robot,
                            cmd_poses,
                            grippers,
                            dt=dt,
                            speed=speed,
                            lead_time=lead_time,
                            sample_hz=sample_hz,
                        )
                        hold_pose = cmd_poses[-1].copy()
                        hold_grip = float(grippers[-1])
                        print(f"[hold] replay done. {hold_hint}...")
                    elif action == "damping":
                        enter_damping = True

            print("[hold] entering damping...")
            robot.set_to_damping()
            time.sleep(0.5)
            wait_for_enter("[damping] arm in damping. Press Enter again to release controller and exit...")
            damped = True
        finally:
            if not damped:
                try:
                    robot.set_to_damping()
                    time.sleep(0.2)
                except Exception:  # noqa: BLE001
                    pass
            robot.stop(wait=True)
            print("Robot stopped.")


if __name__ == "__main__":
    main()
