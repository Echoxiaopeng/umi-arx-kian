#!/usr/bin/env python3
"""Smoke-test schedule_joint_waypoint / command_joint_actions on real ARX5.

Needs ZMQ server(s) started with --joint (not default cartesian).
        
    conda activate arx-py310 && cd /home/lenovo/kian/arx5-sdk
    python python/communication/zmq_server.py X5_umi can0 --port 8765 --joint --no-idle-reset-home

    # 单臂测试
    conda activate umi-arx
    python scripts/test_command_joint_actions.py --robot-ports 8765 --go
    
    # 双臂测试，再step步内运动到delta-rad的偏差
    conda activate umi-arx
    cd /home/lenovo/kian/umi-arx-kian
    python scripts/test_command_joint_actions.py --robot-ports 8765,8766 --go --steps 10 --delta-rad 0.05
"""

from __future__ import annotations

from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import sys
import time

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

from modules.arx5_controller import Arx5Controller  # noqa: E402


def _load_kai0_eval():
    path = ROOT / "scripts" / "eval_arx5_kai0_bimanual.py"
    spec = importlib.util.spec_from_file_location("eval_arx5_kai0_bimanual", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_kai0 = _load_kai0_eval()
command_joint_actions_bimanual = _kai0.command_joint_actions


class _RobotOnlyEnv:
    def __init__(self, robots: list[Arx5Controller]) -> None:
        self.robots = robots

    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]


def parse_ports(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) not in (1, 2):
        raise click.BadParameter(
            "--robot-ports needs one port (single arm) or two (bimanual), "
            "e.g. 8765 or 8765,8766"
        )
    return values


def measured_q7(env: _RobotOnlyEnv) -> np.ndarray:
    """Stacked [q6, grip] per arm → shape (7,) or (14,)."""
    parts: list[float] = []
    for state in env.get_robot_state():
        joints = np.asarray(state["ActualQ"], dtype=np.float64).reshape(6)
        grip = float(state["gripper_position"])
        parts.extend(joints.tolist())
        parts.append(grip)
    value = np.asarray(parts, dtype=np.float64)
    expected = 7 * len(env.robots)
    if value.shape != (expected,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"Invalid state: shape={value.shape}, value={value}")
    return value


def schedule_actions(
    env: _RobotOnlyEnv,
    actions: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """1 arm: schedule_joint_waypoint; 2 arms: kai0 command_joint_actions."""
    n_robots = len(env.robots)
    actions = np.asarray(actions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if n_robots == 1:
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(f"Expected [T,7] for one arm, got {actions.shape}")
        receive_time = time.time()
        is_new = timestamps > receive_time
        if not np.any(is_new):
            return
        for action, target_time in zip(actions[is_new], timestamps[is_new]):
            env.robots[0].schedule_joint_waypoint(
                np.asarray(action, dtype=np.float64),
                target_time=float(target_time),
            )
        return
    command_joint_actions_bimanual(env, actions, timestamps)


def hold_current(env: _RobotOnlyEnv) -> None:
    q = measured_q7(env)
    schedule_actions(
        env,
        q.reshape(1, -1),
        np.asarray([time.time() + 0.05], dtype=np.float64),
    )


@click.command()
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option(
    "--robot-ports",
    default="8765",
    show_default=True,
    help="One port (single arm) or two comma-separated (bimanual).",
)
@click.option("--frequency", default=20.0, type=float, show_default=True)
@click.option("--steps", default=2, type=click.IntRange(2, 50), show_default=True)
@click.option(
    "--delta-rad",
    default=0.02,
    type=float,
    show_default=True,
    help="Peak offset on joint_index (radians).",
)
@click.option("--joint-index", default=0, type=click.IntRange(0, 5), show_default=True)
@click.option(
    "--go",
    is_flag=True,
    help="Actually schedule motion. Without this flag, only print the plan.",
)
def main(
    robot_ip: str,
    robot_ports: str,
    frequency: float,
    steps: int,
    delta_rad: float,
    joint_index: int,
    go: bool,  # noqa: FBT001
) -> None:
    if frequency <= 0 or delta_rad <= 0:
        raise click.BadParameter("frequency and delta-rad must be positive")

    ports = parse_ports(robot_ports)
    n_robots = len(ports)
    action_dim = 7 * n_robots
    dt = 1.0 / frequency
    half = steps // 2

    with SharedMemoryManager() as shm_manager:
        robots = [
            Arx5Controller(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                robot_port=port,
                frequency=100,
                verbose=True,
                reset_to_home_on_start=False,
            )
            for port in ports
        ]
        for robot in robots:
            robot.start()
        try:
            print(f"Waiting for {n_robots} controller(s) on ports {ports}...")
            deadline = time.time() + 15.0
            while time.time() < deadline:
                if all(robot.is_ready for robot in robots):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("Controllers did not become ready in time")

            env = _RobotOnlyEnv(robots)
            before = measured_q7(env)
            print(f"measured_state before: {np.round(before, 4)}")

            actions = np.tile(before, (steps, 1))
            for i in range(steps):
                if i <= half:
                    scale = i / max(half, 1)
                else:
                    scale = (steps - 1 - i) / max(steps - 1 - half, 1)
                for arm in range(n_robots):
                    col = 7 * arm + joint_index
                    actions[i, col] = before[col] + scale * delta_rad

            command_start = time.time() + 0.15
            timestamps = np.arange(steps, dtype=np.float64) * dt + command_start
            print(
                f"plan arms={n_robots} dim={action_dim} steps={steps} "
                f"dt={dt:.3f}s duration≈{steps * dt:.2f}s "
                f"joint{joint_index} ±{delta_rad} rad"
            )
            print(f"first={np.round(actions[0], 4)}")
            print(f"mid  ={np.round(actions[half], 4)}")
            print(f"last ={np.round(actions[-1], 4)}")
            print(f"timestamps[0]={timestamps[0]:.3f} now={time.time():.3f}")

            if not go:
                print("Dry-run only. Re-run with --go to move the arm(s).")
                return

            print("Scheduling joint waypoints...")
            schedule_actions(env, actions, timestamps)
            deadline = float(timestamps[-1]) + 0.2
            while time.time() < deadline:
                time.sleep(0.02)

            after = measured_q7(env)
            print(f"measured_state after:  {np.round(after, 4)}")
            err = np.abs(after - before)
            print(f"|after-before| max={err.max():.4f}  per-dim={np.round(err, 4)}")

            hold_current(env)
            time.sleep(0.2)

            mid_expected = float(before[joint_index] + delta_rad)
            if err.max() > 0.15:
                print(
                    "WARNING: final pose far from start — inspect arms / "
                    "lower --delta-rad / check ZMQ joint mode."
                )
            else:
                print(
                    "OK: returned near start. Visually confirm arm(s) briefly "
                    f"wagged joint{joint_index} "
                    f"(expected peak ≈ {mid_expected:.4f})."
                )
        finally:
            for robot in robots:
                try:
                    robot.stop()
                except Exception as exc:  # noqa: BLE001
                    print(f"stop warning: {exc}")


if __name__ == "__main__":
    main()
