#!/usr/bin/env python3
"""只读查看 ARX5 ZMQ 末端位姿定义。

用法:
  # 先启动 ZMQ controller，再:
  python scripts/test_zmq_ee_pose.py
  python scripts/test_zmq_ee_pose.py --ip 127.0.0.1 --port 8765 --hz 10

按键 (终端焦点在脚本上):
  r  把当前位姿记为参考点，之后打印相对参考的 delta
  c  清除参考点
  h  调用 reset_to_home()（会动臂！）
  q  退出

默认只读，不会发运动指令。
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.arx5_zmq_client import Arx5Client, ee2tcp  # noqa: E402


def fmt3(v: np.ndarray, prec: int = 4) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ",".join(f"{x:.{prec}f}" for x in v) + "]"


def fmt_deg(rad: np.ndarray, prec: int = 1) -> str:
    return fmt3(np.rad2deg(np.asarray(rad, dtype=np.float64)), prec=prec)


def rotvec_angle_deg(rotvec: np.ndarray) -> float:
    return float(np.rad2deg(np.linalg.norm(rotvec)))


def print_pose_block(tag: str, ee: np.ndarray, tcp: np.ndarray, grip: float, joints: np.ndarray) -> None:
    ee = np.asarray(ee, dtype=np.float64).reshape(6)
    tcp = np.asarray(tcp, dtype=np.float64).reshape(6)
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)

    ee_pos, ee_rpy = ee[:3], ee[3:6]
    tcp_pos, tcp_rv = tcp[:3], tcp[3:6]
    tcp_rpy = R.from_rotvec(tcp_rv).as_euler("xyz")
    ee_rv = R.from_euler("xyz", ee_rpy).as_rotvec()

    print(f"\n===== {tag} =====")
    print(f"gripper_pos = {grip:.4f}")
    print(f"joint_pos   = {fmt3(joints, prec=3)} rad")
    print(f"             {fmt_deg(joints)} deg")
    print("--- EE  (ZMQ raw: xyz + rpy) ---")
    print(f"  pos       = {fmt3(ee_pos)} m")
    print(f"  rpy       = {fmt3(ee_rpy)} rad  |  {fmt_deg(ee_rpy)} deg")
    print(f"  as_rotvec = {fmt3(ee_rv)}  |angle|={rotvec_angle_deg(ee_rv):.1f} deg")
    print("--- TCP (ee2tcp: xyz + rotvec) ---")
    print(f"  pos       = {fmt3(tcp_pos)} m   (与 EE xyz 相同)")
    print(f"  rotvec    = {fmt3(tcp_rv)}  |angle|={rotvec_angle_deg(tcp_rv):.1f} deg")
    print(f"  as_rpy    = {fmt3(tcp_rpy)} rad  |  {fmt_deg(tcp_rpy)} deg")
    print(f"  |pos|     = {np.linalg.norm(tcp_pos):.4f} m")


def print_delta(ref_tcp: np.ndarray, cur_tcp: np.ndarray) -> None:
    ref = np.asarray(ref_tcp, dtype=np.float64).reshape(6)
    cur = np.asarray(cur_tcp, dtype=np.float64).reshape(6)
    dpos = cur[:3] - ref[:3]
    R_ref = R.from_rotvec(ref[3:6])
    R_cur = R.from_rotvec(cur[3:6])
    drot = (R_ref.inv() * R_cur).as_rotvec()
    print("--- vs REF (TCP) ---")
    print(f"  dpos      = {fmt3(dpos)} m  |dpos|={np.linalg.norm(dpos)*1000:.1f} mm")
    print(f"  drotvec   = {fmt3(drot)}  |dangle|={rotvec_angle_deg(drot):.2f} deg")


class RawKeyReader:
    """非阻塞读单键；非 tty 时退化为无输入。"""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._is_tty = sys.stdin.isatty()
        self._old: list | None = None
        if self._is_tty:
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def read(self) -> str | None:
        if not self._is_tty:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)

    def close(self) -> None:
        if self._is_tty and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX5 ZMQ EE/TCP pose inspector (read-only)")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hz", type=float, default=5.0, help="print rate")
    parser.add_argument(
        "--once",
        action="store_true",
        help="print once and exit",
    )
    args = parser.parse_args()

    period = 1.0 / max(args.hz, 0.1)
    client = Arx5Client(zmq_ip=args.ip, zmq_port=args.port)

    state = client.get_state()
    print("\nstate keys:", sorted(state.keys()))
    print(
        "hint: home 时若 pos≈[0,0,0]，说明笛卡尔原点在 home 末端；"
        "姿态若非零，则姿态不是相对 home 归零。"
    )
    print("keys: [r]ef  [c]lear  [h]ome(moves!)  [q]uit\n")

    ref_tcp: np.ndarray | None = None
    keys = RawKeyReader()
    try:
        while True:
            state = client.get_state()
            ee = np.asarray(state["ee_pose"], dtype=np.float64).reshape(6)
            tcp = np.asarray(ee2tcp(ee), dtype=np.float64).reshape(6)
            # 也核对 property
            tcp_prop = np.asarray(client.tcp_pose, dtype=np.float64).reshape(6)
            grip = float(state["gripper_pos"])
            joints = np.asarray(state["joint_pos"], dtype=np.float64)

            ts = time.strftime("%H:%M:%S")
            print_pose_block(f"{ts}  tcp://{args.ip}:{args.port}", ee, tcp, grip, joints)
            if not np.allclose(tcp, tcp_prop, atol=1e-9):
                print(f"WARN: ee2tcp != client.tcp_pose  delta={fmt3(tcp - tcp_prop)}")
            if ref_tcp is not None:
                print_delta(ref_tcp, tcp)

            if args.once:
                break

            t_end = time.time() + period
            while time.time() < t_end:
                ch = keys.read()
                if ch is None:
                    time.sleep(0.01)
                    continue
                ch = ch.lower()
                if ch == "q":
                    print("quit.")
                    return
                if ch == "r":
                    ref_tcp = tcp.copy()
                    print(f"\n[ref set] tcp pos={fmt3(ref_tcp[:3])} rotvec={fmt3(ref_tcp[3:])}")
                elif ch == "c":
                    ref_tcp = None
                    print("\n[ref cleared]")
                elif ch == "h":
                    print("\n[home] reset_to_home() ...")
                    client.reset_to_home()
                    print("[home] done")
                # drain remaining sleep after key
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        keys.close()
        print("done.")


if __name__ == "__main__":
    main()
