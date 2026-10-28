import multiprocessing as mp
import enum
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, cast
import numpy as np
from utils.other_util import precise_wait
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

from modules.pose_trajectory_interpolator import (
    JointTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
)
import time
from modules.arx5_zmq_client import Arx5Client
import numpy.typing as npt

from multiprocessing import Value
import ctypes


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET_TO_HOME = 3
    ADD_WAYPOINT = 4
    UPDATE_TRAJECTORY = 5
    SET_JOINT_TARGET = 6
    SET_JOINT_TRAJECTORY = 7
    SET_GRAVITY_COMPENSATION = 8
    SET_TO_DAMPING = 9
    SET_END_CONTROL = 10
    RESTORE_GAINS_AFTER_DRAG = 11
    SCHEDULE_JOINT_WAYPOINT = 12


# 拖拽夹爪：轻回弹 + 跟手；旧硬墙默认关（曾偏猛）。
# 死区/同号门限防振荡；回弹只朝张开。
# 捏死后静止不再顶开（否则握紧会被力弹开）；闭合跟手略加强。
DRAG_GRIPPER_WALL_ENABLE = False
DRAG_GRIPPER_WALL_SOFT_LO = 0.028
DRAG_GRIPPER_WALL_HARD_LO = 0.016
DRAG_GRIPPER_WALL_SOFT_TAU = 0.45
DRAG_GRIPPER_WALL_HARD_TAU = 0.95
DRAG_GRIPPER_WALL_OPEN_FLOOR = 0.35
DRAG_GRIPPER_WALL_HI = 0.082
DRAG_GRIPPER_WALL_HI_TAU = 0.25
# 闭侧回弹：仅在「偏紧且仍在动/张开」时轻顶，静止握紧不出力
DRAG_GRIPPER_REBOUND_ENABLE = True
DRAG_GRIPPER_REBOUND_LO = 0.024  # m，更深才回弹（原 0.032 太早顶）
DRAG_GRIPPER_REBOUND_HARD_LO = 0.012
DRAG_GRIPPER_REBOUND_TAU = 0.10  # Nm，软段（原 0.22）
DRAG_GRIPPER_REBOUND_HARD_TAU = 0.18  # Nm，夹很紧（原 0.38）
DRAG_GRIPPER_REBOUND_FLOOR = 0.04  # 软段底力（原 0.12）
# 跟手：闭合略加强，张开略减弱，减轻「闭合偏硬」
DRAG_GRIPPER_VEL_EPS = 0.032
DRAG_GRIPPER_FRICTION_OPEN = 0.12
DRAG_GRIPPER_FRICTION_CLOSE = 0.10
DRAG_GRIPPER_TORQUE_TO_ACC = 0.05
DRAG_GRIPPER_ACC_THRESHOLD = 0.02
DRAG_GRIPPER_TORQUE_MAX = 1.2
DRAG_GRIPPER_VEL_SIGN_MIN = 4
DRAG_GRIPPER_STUCK_VEL = 0.02
DRAG_GRIPPER_STUCK_TAU = 0.55
# 开口低于此且近似静止 → 视为故意握紧，禁止回弹顶开
DRAG_GRIPPER_HOLD_CLOSED_LO = 0.020
# 阻尼拖拽（无真零力时）：纯 kp=0 + 极小 kd 会立刻垂臂。
# 用硬增益比例托住重力；仅在「明显是人手在拖」时才让 hold 跟随，
# 否则重力慢坠也会被当成跟手，hold 跟着掉（日志里 z 从 0.38→-0.01）。
# kd 相对 set_to_damping 后再乘此系数（越大越不易下坠，拖拽略沉）。
DRAG_ARM_KD_SCALE = 0.20
# 相对进入拖拽前硬 kp 的比例；0.06 托不住双臂+相机，约 0.15–0.20 才稳。
DRAG_ARM_KP_SCALE = 0.18
# hold 跟随：要够大的速度+误差才跟手；门槛过低会把重力下坠当成拖动。
DRAG_HOLD_POS_DEADBAND_M = 0.025
DRAG_HOLD_VEL_THRESH = 0.40  # rad/s, max |qd|
DRAG_HOLD_JOINT_ERR = 0.10  # rad, L2 of first 6 joints


def _looks_like_joint_server_error(exc: BaseException) -> bool:
    """True only when EE cmds are clearly unavailable (--joint server).

    Do NOT match ``EVAN_SYNC_AFTER_DRAG``: an unknown-sync error on a
    cartesian collect server is not joint-mode, and falling back to
    ``SET_JOINT_POSITION`` hot-swaps cartesian→joint and bricks spdlog.
    """
    msg = str(exc)
    return (
        "--joint" in msg
        or "SET_EE_POSE" in msg
        or "cartesian server" in msg
    )


def _measured_joint7(robot_client: Arx5Client) -> np.ndarray:
    robot_client.get_state()
    return np.concatenate(
        [
            np.asarray(robot_client.joint_pos, dtype=np.float64).reshape(6),
            [float(robot_client.gripper_pos)],
        ]
    )


def _sync_measured_tcp_gripper(robot_client: Arx5Client) -> None:
    """把 EE/TCP 与夹爪指令钉在实测值，避免进拖拽后仍追旧 setpoint。"""
    robot_client.get_state()
    robot_client.set_tcp_pose(robot_client.tcp_pose, robot_client.gripper_pos)


def _pin_measured_setpoint(robot_client: Arx5Client) -> str:
    """Pin controller setpoint to measured pose.

    Prefer cartesian ``sync_after_drag`` / TCP pin (UMI collect path). On
    ``--joint`` Kai0 servers ``EVAN_SYNC_AFTER_DRAG`` still returns OK (joint
    pin), so we must probe ``SET_EE_POSE`` before claiming cartesian hold.
    Returns which path was used: ``cartesian_sync`` | ``tcp`` | ``joint``.

    Always try TCP before committing to joint: joint cmds on a cartesian
    server can hot-swap and brick spdlog.
    """
    robot_client.get_state()
    sync_ok = False
    sync_exc: BaseException | None = None
    try:
        robot_client.sync_after_drag()
        sync_ok = True
    except Exception as exc:  # noqa: BLE001
        sync_exc = exc

    # Probe EE: joint servers reject this even after a successful sync.
    try:
        _sync_measured_tcp_gripper(robot_client)
        return "cartesian_sync" if sync_ok else "tcp"
    except Exception as tcp_exc:  # noqa: BLE001
        jointish = _looks_like_joint_server_error(tcp_exc) or (
            sync_exc is not None and _looks_like_joint_server_error(sync_exc)
        )
        if not jointish:
            raise tcp_exc
        q7 = _measured_joint7(robot_client)
        robot_client.set_joint_position(q7)
        return "joint"


def _closed_wall_torque(g: float) -> float:
    """闭限位外软内硬：正力矩 = 张开方向。进墙即有 OPEN_FLOOR 底力。"""
    if g >= DRAG_GRIPPER_WALL_SOFT_LO:
        return 0.0
    floor = float(DRAG_GRIPPER_WALL_OPEN_FLOOR)
    soft = float(DRAG_GRIPPER_WALL_SOFT_TAU)
    hard = float(DRAG_GRIPPER_WALL_HARD_TAU)
    if g >= DRAG_GRIPPER_WALL_HARD_LO:
        span = max(DRAG_GRIPPER_WALL_SOFT_LO - DRAG_GRIPPER_WALL_HARD_LO, 1e-6)
        alpha = (DRAG_GRIPPER_WALL_SOFT_LO - g) / span
        alpha = min(1.0, max(0.0, alpha))
        return floor + (soft - floor) * alpha
    span = max(DRAG_GRIPPER_WALL_HARD_LO, 1e-6)
    beta = (DRAG_GRIPPER_WALL_HARD_LO - g) / span
    beta = min(1.0, max(0.0, beta))
    return soft + (hard - soft) * (beta * beta)


def _rebound_open_torque(g: float) -> float:
    """偏紧区间张开回弹（比硬墙温和）；力矩已整体调弱。"""
    if g >= DRAG_GRIPPER_REBOUND_LO:
        return 0.0
    soft = float(DRAG_GRIPPER_REBOUND_TAU)
    hard = float(DRAG_GRIPPER_REBOUND_HARD_TAU)
    floor = float(DRAG_GRIPPER_REBOUND_FLOOR)
    if g >= DRAG_GRIPPER_REBOUND_HARD_LO:
        span = max(
            DRAG_GRIPPER_REBOUND_LO - DRAG_GRIPPER_REBOUND_HARD_LO, 1e-6
        )
        alpha = (DRAG_GRIPPER_REBOUND_LO - g) / span
        alpha = min(1.0, max(0.0, alpha))
        return floor + (soft - floor) * alpha
    span = max(DRAG_GRIPPER_REBOUND_HARD_LO, 1e-6)
    beta = (DRAG_GRIPPER_REBOUND_HARD_LO - g) / span
    beta = min(1.0, max(0.0, beta))
    return soft + (hard - soft) * (beta * beta)


def _vel_same_sign_count(gripper_vel_hist: list[float], sign: float) -> int:
    n = 0
    for vv in gripper_vel_hist:
        if abs(float(vv)) >= DRAG_GRIPPER_VEL_EPS and float(vv) * sign > 0:
            n += 1
    return n


def _gripper_assist_torque(
    gripper_vel_hist: list[float],
    dt: float,
    gripper_pos: float | None = None,
) -> float:
    """张开回弹（全程偏开）+ 同号跟手；可选硬墙。"""
    torque = 0.0
    g = float(gripper_pos) if gripper_pos is not None else None
    v_now = float(gripper_vel_hist[-1]) if gripper_vel_hist else 0.0
    clearly_closing = (
        v_now <= -DRAG_GRIPPER_VEL_EPS
        and _vel_same_sign_count(gripper_vel_hist, -1.0)
        >= DRAG_GRIPPER_VEL_SIGN_MIN
    )
    clearly_opening = (
        v_now >= DRAG_GRIPPER_VEL_EPS
        and _vel_same_sign_count(gripper_vel_hist, 1.0)
        >= DRAG_GRIPPER_VEL_SIGN_MIN
    )

    # 1) 跟手：同向持续才出力（防噪声振荡）；闭合全程跟手（含偏紧）
    if len(gripper_vel_hist) >= 5 and dt > 0 and (
        clearly_opening or clearly_closing
    ):
        v = gripper_vel_hist
        gripper_acc = 0.5 * (v[-1] - v[0]) / (dt * (len(v) - 1)) + 0.5 * (
            v[-2] - v[2]
        ) / (dt * (len(v) - 3))
        if gripper_acc > DRAG_GRIPPER_ACC_THRESHOLD:
            gripper_acc -= DRAG_GRIPPER_ACC_THRESHOLD
        elif gripper_acc < -DRAG_GRIPPER_ACC_THRESHOLD:
            gripper_acc += DRAG_GRIPPER_ACC_THRESHOLD
        else:
            gripper_acc = 0.0
        torque += DRAG_GRIPPER_TORQUE_TO_ACC * float(gripper_acc)
        if clearly_opening:
            torque += DRAG_GRIPPER_FRICTION_OPEN
        else:
            torque -= DRAG_GRIPPER_FRICTION_CLOSE

    # 2) 回弹：偏紧时轻顶开，但故意握紧（近闭且静止）或明确闭合时不顶
    holding_closed = (
        g is not None
        and g <= DRAG_GRIPPER_HOLD_CLOSED_LO
        and abs(v_now) < DRAG_GRIPPER_STUCK_VEL
    )
    if (
        DRAG_GRIPPER_REBOUND_ENABLE
        and g is not None
        and not clearly_closing
        and not holding_closed
    ):
        rb = _rebound_open_torque(g)
        if rb > 0:
            if torque < 0:
                torque = 0.0
            torque = max(torque, rb)
            if clearly_opening:
                torque = max(torque, rb + 0.5 * DRAG_GRIPPER_FRICTION_OPEN)

    # 3) 旧硬墙（默认关）
    if DRAG_GRIPPER_WALL_ENABLE and g is not None:
        wall = _closed_wall_torque(g)
        if wall > 0:
            if torque < 0:
                torque = 0.0
            torque = max(torque + wall, float(DRAG_GRIPPER_WALL_OPEN_FLOOR))
            if abs(v_now) < DRAG_GRIPPER_STUCK_VEL:
                torque = max(torque, float(DRAG_GRIPPER_STUCK_TAU))
        if g > DRAG_GRIPPER_WALL_HI:
            if torque > 0:
                torque = 0.0
            depth = (g - DRAG_GRIPPER_WALL_HI) / 0.01
            torque -= DRAG_GRIPPER_WALL_HI_TAU * min(1.0, max(0.0, depth))

    if torque > DRAG_GRIPPER_TORQUE_MAX:
        torque = DRAG_GRIPPER_TORQUE_MAX
    elif torque < -DRAG_GRIPPER_TORQUE_MAX:
        torque = -DRAG_GRIPPER_TORQUE_MAX
    return float(torque)


class Arx5Controller(mp.Process):
    pass

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip: str,
        robot_port: int,
        # robot_joint_port: int,
        launch_timeout: float = 10,
        frequency: float = 100,
        get_max_k: Optional[int] = None,
        verbose: bool = False,
        receive_latency: float = 0.0,
        reset_to_home_on_start: bool = True,
    ):
        super().__init__(name="Arx5Controller")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        # self.robot_joint_port = robot_joint_port

        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "gripper_pos": 0.0,
            "joint_angles": np.zeros((7,), dtype=np.float64),
            # "joint_traj": np.zeros((10, 7), dtype=np.float64),  # optional
            # "timestamps": np.zeros((10,), dtype=np.float64),  # optional
            "duration": 0.0,
            "target_time": 0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )
        self.verbose = verbose

        # build ring buffer
        receive_keys = [
            ("ActualTCPPose", "tcp_pose"),
            ("ActualQ", "joint_pos"),
            ("ActualQd", "joint_vel"),
            ("gripper_position", "gripper_pos"),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if "joint" in func_name:
                example[key] = np.zeros(6)
            elif "tcp_pose" in func_name:
                example[key] = np.zeros(6)
        example["gripper_position"] = 0.0

        example["robot_receive_timestamp"] = time.time()
        example["robot_timestamp"] = time.time()

        if get_max_k is None:
            get_max_k = int(frequency * 5)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )
        self.launch_timeout = launch_timeout
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        self.frequency = frequency
        self.receive_latency = receive_latency
        self.reset_to_home_on_start = bool(reset_to_home_on_start)

        # Will be initialized in the subprocess
        self.robot_client: Arx5Client
        # self.robot_joint_port: Arx5Client
        self.reset_success = Value(ctypes.c_bool, False)
        # 父进程在 set_end_control / restore_gains / enter-drag 后必须等待，否则会抢跑
        # （日志：Moving to home 插在 gain_alpha_0.10 与 0.15 之间，from 读到 drag 前位姿）
        self.end_control_done = Value(ctypes.c_bool, True)
        self.restore_gains_done = Value(ctypes.c_bool, True)
        self.gravity_compensation_done = Value(ctypes.c_bool, True)

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Arx5Controller] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
        if self.verbose:
            print(f"[Arx5Controller] Controller process terminated at {self.pid}")

    def start_wait(self):
        print(
            f"[Arx5Controller] Waiting for controller process to be ready "
            f"({self.robot_ip}:{self.robot_port})",
            flush=True,
        )
        print(f"{self.launch_timeout=}", flush=True)
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive(), (
            f"[Arx5Controller] process died during launch "
            f"({self.robot_ip}:{self.robot_port})"
        )
        if not self.is_ready:
            print(
                f"[Arx5Controller] WARNING: launch timeout "
                f"({self.launch_timeout}s) but not ready yet "
                f"({self.robot_ip}:{self.robot_port}). "
                "常见原因: 该端口 ZMQ GET_STATE 卡住 / server 无响应。"
                "（进程仍活着，外层会继续等）",
                flush=True,
            )

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= command methods ============
    def servoL(self, pose: npt.NDArray[np.float64], gripper_pos: float, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "duration": duration,
        }
        self.input_queue.put(message)

    def schedule_waypoint(
        self, pose: npt.NDArray[np.float64], gripper_pos: float, target_time: float
    ):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def set_joint_target(
        self, joint_angles: npt.NDArray[np.float64], target_time: float
    ):
        """
        设置关节角目标 (joint-space control)。
        joint_angles: 6 关节 + 夹爪，shape=(7,)
        target_time:  Unix 墙钟时间戳；控制器内插值到该时刻。
        """
        assert self.is_alive()
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        assert joint_angles.shape == (7,), f"Expected 7 joint angles, got {joint_angles.shape}"

        message = {
            "cmd": Command.SET_JOINT_TARGET.value,
            "joint_angles": joint_angles,
            "target_time": float(target_time),
        }
        self.input_queue.put(message)

    def schedule_joint_waypoint(
        self, joint_angles: npt.NDArray[np.float64], target_time: float
    ):
        """
        关节空间等价于 schedule_waypoint：按 target_time 插入路点并插值执行。
        joint_angles: 6 关节 + 夹爪，shape=(7,)
        target_time:  Unix 墙钟时间戳
        """
        assert self.is_alive()
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        assert joint_angles.shape == (7,), f"Expected 7 joint angles, got {joint_angles.shape}"

        message = {
            "cmd": Command.SCHEDULE_JOINT_WAYPOINT.value,
            "joint_angles": joint_angles,
            "target_time": float(target_time),
        }
        self.input_queue.put(message)

    def add_waypoint(
        self, pose: npt.NDArray[np.float64], gripper_pos: float, target_time: float
    ):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.ADD_WAYPOINT.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def update_trajectory(self):
        message = {
            "cmd": Command.UPDATE_TRAJECTORY.value,
        }
        self.input_queue.put(message)

    def reset_to_home(self):
        self.reset_success.value = False
        message = {"cmd": Command.RESET_TO_HOME.value}
        self.input_queue.put(message)
        while not self.reset_success.value:
            print("waiting for reset")
            time.sleep(0.1)

    
    def set_gravity_compensation(self, wait: bool = True, timeout: float = 5.0):
        """进入拖拽/零力；默认等待控制器真正切完，避免回 init 后立刻调用却未生效。"""
        self.gravity_compensation_done.value = False
        message = {"cmd": Command.SET_GRAVITY_COMPENSATION.value}
        self.input_queue.put(message)
        if wait:
            t0 = time.time()
            while not self.gravity_compensation_done.value:
                if time.time() - t0 > timeout:
                    print("[Arx5Controller] set_gravity_compensation wait timeout")
                    break
                time.sleep(0.02)

    def set_end_control(self, wait: bool = True, timeout: float = 5.0):
        """退出零力：同步当前位置，升到中等轨迹增益，重建插值器。"""
        self.end_control_done.value = False
        message = {"cmd": Command.SET_END_CONTROL.value}
        self.input_queue.put(message)
        if wait:
            t0 = time.time()
            while not self.end_control_done.value:
                if time.time() - t0 > timeout:
                    print("[Arx5Controller] set_end_control wait timeout")
                    break
                time.sleep(0.02)

    def restore_gains_after_drag(self, wait: bool = True, timeout: float = 3.0):
        """到达目标位姿后再恢复满增益（避免在拖拽点硬锁）。"""
        self.restore_gains_done.value = False
        message = {"cmd": Command.RESTORE_GAINS_AFTER_DRAG.value}
        self.input_queue.put(message)
        if wait:
            t0 = time.time()
            while not self.restore_gains_done.value:
                if time.time() - t0 > timeout:
                    print("[Arx5Controller] restore_gains_after_drag wait timeout")
                    break
                time.sleep(0.02)

    def set_to_damping(self):
        message = {"cmd": Command.SET_TO_DAMPING.value}
        self.input_queue.put(message)

    # ========= main loop in process ============
    def run(self):
        self.robot_client = Arx5Client(self.robot_ip, self.robot_port)
        if self.reset_to_home_on_start:
            self.robot_client.reset_to_home()
        else:
            self.robot_client.get_state()
        # self.robot_joint_client = Arx5Client(self.robot_ip, self.robot_joint_port)
        # self.robot_joint_client.reset_to_home()
        # self.robot_client.set_gain()
        time.sleep(1)
        # self.robot_client.set_to_damping()
        # gain = self.robot_client.get_gain()
        # gain['kd'] = gain['kd'] * 0.2
        gain = self.robot_client.get_gain()
        gain["kp"] = np.array([300, 300, 400, 80, 50, 30])
        self.robot_client.set_gain(gain)
        # self.robot_joint_client.set_gain(gain)
        np.set_printoptions(precision=3, suppress=True)

        self.waypoint_buffer = []

        try:

            dt = 1 / self.frequency
            self.robot_client.get_state()
            curr_pose = self.robot_client.tcp_pose
            curr_gripper_pos = self.robot_client.gripper_pos
            curr_joint = np.concatenate(
                [
                    np.asarray(self.robot_client.joint_pos, dtype=np.float64).reshape(6),
                    [float(self.robot_client.gripper_pos)],
                ]
            )
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            last_joint_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]), poses=np.array([curr_pose])
            )
            gripper_pos_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
            )
            joint_interp = JointTrajectoryInterpolator(
                times=np.array([curr_t]), joints=np.array([curr_joint])
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            is_gravity_compensation = False
            joint_control_active = False
            # ZMQ rejects set_tcp_pose until RESET_TO_HOME. If we skipped home on
            # start, stay passive (get_state only) until home or joint control.
            allow_ee_pose = bool(self.reset_to_home_on_start)
            saved_gain_before_drag = None
            # Damping-drag soft hold (None on true GC). Pose/joint target while idle.
            drag_hold_pose: np.ndarray | None = None
            drag_hold_gripper: float | None = None
            drag_hold_joint: np.ndarray | None = None
            drag_hold_mode: str | None = None  # "tcp" | "joint"
            drag_gripper_vel_hist: list[float] = [0.0] * 5
            _ee_pose_fail_t = 0.0
            _joint_pose_fail_t = 0.0
            # --joint servers historically lack SET_GRIPPER_TORQUE; skip after 1 fail.
            drag_grip_tau_disabled = False

            def _seed_joint_interp_from_state():
                nonlocal joint_interp, last_joint_waypoint_time
                self.robot_client.get_state()
                q = np.concatenate(
                    [
                        np.asarray(self.robot_client.joint_pos, dtype=np.float64).reshape(
                            6
                        ),
                        [float(self.robot_client.gripper_pos)],
                    ]
                )
                t_seed = time.monotonic()
                joint_interp = JointTrajectoryInterpolator(
                    times=np.array([t_seed]), joints=np.array([q])
                )
                last_joint_waypoint_time = t_seed

            def _schedule_joint_waypoint_cmd(joint_angles, target_time_wall: float):
                nonlocal joint_interp, last_joint_waypoint_time, joint_control_active
                target_time = time.monotonic() - time.time() + float(target_time_wall)
                if not joint_control_active:
                    _seed_joint_interp_from_state()
                joint_interp = joint_interp.schedule_waypoint(
                    joint=joint_angles,
                    time=target_time,
                    curr_time=t_now,
                    last_waypoint_time=last_joint_waypoint_time,
                )
                last_joint_waypoint_time = target_time
                joint_control_active = True

            def print_drag_diag(tag: str, gain_override=None) -> None:
                """只读诊断：不发送位姿、模式或增益命令。"""
                try:
                    self.robot_client.get_state()
                    tcp = np.asarray(self.robot_client.tcp_pose, dtype=np.float64)
                    ee = np.asarray(self.robot_client.ee_pose, dtype=np.float64)
                    qd = np.asarray(self.robot_client.joint_vel, dtype=np.float64)
                    tau = np.asarray(self.robot_client.joint_torque, dtype=np.float64)
                    gain_now = (
                        gain_override
                        if gain_override is not None
                        else self.robot_client.get_gain()
                    )
                    kp = np.asarray(gain_now["kp"], dtype=np.float64)
                    kd = np.asarray(gain_now["kd"], dtype=np.float64)
                    print(
                        f"[DRAG_DIAG] t={time.time():.6f} tag={tag}\n"
                        f"  tcp={np.array2string(tcp, precision=5, separator=',')}\n"
                        f"  ee ={np.array2string(ee, precision=5, separator=',')}\n"
                        f"  qd ={np.array2string(qd, precision=5, separator=',')} "
                        f"max_abs={float(np.max(np.abs(qd))) if qd.size else 0.0:.6f}\n"
                        f"  tau={np.array2string(tau, precision=4, separator=',')}\n"
                        f"  grip={float(self.robot_client.gripper_pos):.6f} "
                        f"kp={np.array2string(kp, precision=2, separator=',')} "
                        f"kd={np.array2string(kd, precision=2, separator=',')}"
                    )
                except Exception as diag_exc:
                    print(f"[DRAG_DIAG] t={time.time():.6f} tag={tag} ERROR={diag_exc}")

            while keep_running:
                t_now = time.monotonic()
                # 零力/阻尼拖拽：不下发插值器 TCP（否则把臂拉回目标）。
                # 夹爪：kp=0 + 力矩前馈，尽量让手捏跟动（非键盘）。
                if joint_control_active and not is_gravity_compensation:
                    try:
                        joint_cmd = np.asarray(joint_interp(t_now), dtype=np.float64)
                        self.robot_client.set_joint_position(joint_cmd)
                    except Exception as e:
                        if time.monotonic() - _joint_pose_fail_t > 1.0:
                            _joint_pose_fail_t = time.monotonic()
                            print(f"[Arx5Controller] set_joint_position failed: {e}")
                        try:
                            self.robot_client.get_state()
                        except Exception:
                            pass
                elif (
                    allow_ee_pose
                    and not is_gravity_compensation
                    and not joint_control_active
                ):
                    pose_cmd = pose_interp(t_now)
                    gripper_cmd = float(gripper_pos_interp(t_now)[0])
                    try:
                        self.robot_client.set_tcp_pose(pose_cmd, gripper_cmd)
                    except Exception as e:
                        # ZMQ: consecutive EE pose L2 must be < 0.1；
                        # 尚未 RESET_TO_HOME；或服务器是 --joint（无 SET_EE_POSE）。
                        # 单次失败不应打死控制器进程。
                        msg = str(e)
                        if (
                            "RESET_TO_HOME" in msg
                            or "--joint" in msg
                            or "SET_EE_POSE" in msg
                            or "cartesian server" in msg
                        ):
                            allow_ee_pose = False
                        if time.monotonic() - _ee_pose_fail_t > 1.0:
                            _ee_pose_fail_t = time.monotonic()
                            print(f"[Arx5Controller] set_tcp_pose failed: {e}")
                        try:
                            self.robot_client.get_state()
                        except Exception:
                            pass
                elif not is_gravity_compensation and not joint_control_active:
                    # Passive hold: read state only (no EE / joint commands yet).
                    try:
                        self.robot_client.get_state()
                    except Exception:
                        pass
                elif is_gravity_compensation:
                    # 真零力：只读状态。阻尼拖拽：软 hold（静止钉住、拖动跟随）+ 夹爪力矩。
                    self.robot_client.get_state()
                    if drag_hold_mode is not None:
                        try:
                            qd = np.asarray(
                                self.robot_client.joint_vel, dtype=np.float64
                            ).reshape(-1)
                            moving = (
                                qd.size > 0
                                and float(np.max(np.abs(qd)))
                                >= float(DRAG_HOLD_VEL_THRESH)
                            )
                            if drag_hold_mode == "tcp" and drag_hold_pose is not None:
                                meas = np.asarray(
                                    self.robot_client.tcp_pose, dtype=np.float64
                                ).reshape(6)
                                g_meas = float(self.robot_client.gripper_pos)
                                pos_err = float(
                                    np.linalg.norm(meas[:3] - drag_hold_pose[:3])
                                )
                                if (
                                    moving
                                    and pos_err >= float(DRAG_HOLD_POS_DEADBAND_M)
                                ):
                                    drag_hold_pose = meas.copy()
                                    drag_hold_gripper = g_meas
                                grip_cmd = (
                                    float(drag_hold_gripper)
                                    if drag_hold_gripper is not None
                                    else g_meas
                                )
                                self.robot_client.set_tcp_pose(
                                    drag_hold_pose, grip_cmd
                                )
                            elif (
                                drag_hold_mode == "joint"
                                and drag_hold_joint is not None
                            ):
                                meas_q = _measured_joint7(self.robot_client)
                                q_err = float(
                                    np.linalg.norm(
                                        meas_q[:6] - drag_hold_joint[:6]
                                    )
                                )
                                # 慢速下坠不更新 hold，否则 soft kp 永远追不上重力。
                                if moving and q_err >= float(DRAG_HOLD_JOINT_ERR):
                                    drag_hold_joint = meas_q.copy()
                                self.robot_client.set_joint_position(
                                    drag_hold_joint
                                )
                        except Exception as hold_exc:
                            # --joint servers: sync may look cartesian but EE fails.
                            if (
                                drag_hold_mode == "tcp"
                                and _looks_like_joint_server_error(hold_exc)
                            ):
                                try:
                                    drag_hold_joint = _measured_joint7(
                                        self.robot_client
                                    )
                                    drag_hold_mode = "joint"
                                    drag_hold_pose = None
                                    drag_hold_gripper = None
                                    self.robot_client.set_joint_position(
                                        drag_hold_joint
                                    )
                                    print(
                                        "[Arx5Controller] soft-hold fallback "
                                        "tcp -> joint (--joint server)"
                                    )
                                except Exception as joint_hold_exc:
                                    if time.monotonic() - _ee_pose_fail_t > 1.0:
                                        _ee_pose_fail_t = time.monotonic()
                                        print(
                                            "[Arx5Controller] drag soft-hold "
                                            f"joint fallback failed: {joint_hold_exc}"
                                        )
                            elif time.monotonic() - _ee_pose_fail_t > 1.0:
                                _ee_pose_fail_t = time.monotonic()
                                print(
                                    "[Arx5Controller] drag soft-hold failed: "
                                    f"{hold_exc}"
                                )
                    try:
                        gvel = float(self.robot_client.gripper_vel)
                    except Exception:
                        gvel = 0.0
                    try:
                        gpos = float(self.robot_client.gripper_pos)
                    except Exception:
                        gpos = None
                    drag_gripper_vel_hist.append(gvel)
                    drag_gripper_vel_hist.pop(0)
                    assist_tau = _gripper_assist_torque(
                        drag_gripper_vel_hist, dt, gripper_pos=gpos
                    )
                    # 每拍下发（含 0），避免上次辅助力矩残留继续推夹爪
                    if not drag_grip_tau_disabled:
                        try:
                            self.robot_client.set_gripper_torque(assist_tau)
                        except Exception as grip_tau_exc:
                            if _looks_like_joint_server_error(grip_tau_exc):
                                drag_grip_tau_disabled = True
                                print(
                                    "[Arx5Controller] set_gripper_torque unsupported "
                                    "on this ZMQ server (--joint); "
                                    "drag will use hand squeeze without torque assist"
                                )
                            else:
                                if not hasattr(self, "_grip_tau_fail_t"):
                                    self._grip_tau_fail_t = 0.0
                                if (
                                    time.monotonic() - float(self._grip_tau_fail_t)
                                    > 1.0
                                ):
                                    self._grip_tau_fail_t = time.monotonic()
                                    print(
                                        "[Arx5Controller] set_gripper_torque failed: "
                                        f"{grip_tau_exc} "
                                        f"(gpos={gpos}, tau={assist_tau:.3f})"
                                    )

                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(self.robot_client, func_name)
                t_recv = time.time()
                state["robot_receive_timestamp"] = t_recv
                state["robot_timestamp"] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # if self.verbose:
                #     print(f"Current: {state['ActualTCPPose']} target: {pose_cmd}, gripper: {state['gripper_position']:.3f}/{gripper_cmd:.3f}")

                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands["cmd"])
                except Empty:
                    commands = {}
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        joint_control_active = False
                        allow_ee_pose = True
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        gripper_pos_interp = gripper_pos_interp.drive_to_waypoint(
                            pose=[command["gripper_pos"], 0, 0, 0, 0, 0],
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                f"[Arx5Controller] New pose target: {target_pose} duration {duration:.3f}s"
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        joint_control_active = False
                        allow_ee_pose = True
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        target_time = time.monotonic() - time.time() + target_time
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        gripper_pos_interp = gripper_pos_interp.schedule_waypoint(
                            pose=[command["gripper_pos"], 0, 0, 0, 0, 0],
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time

                    elif cmd == Command.RESET_TO_HOME.value:
                        self.robot_client.reset_to_home()
                        self.robot_client.get_state()
                        is_gravity_compensation = False
                        joint_control_active = False
                        allow_ee_pose = True

                        self.ring_buffer.clear()
                        state = dict()
                        for key, func_name in self.receive_keys:
                            state[key] = getattr(self.robot_client, func_name)
                        t_recv = time.time()
                        state["robot_receive_timestamp"] = t_recv
                        state["robot_timestamp"] = t_recv - self.receive_latency
                        self.ring_buffer.put(state)
                        if self.verbose:
                            print(f"[Arx5Controller] Reset to home")

                        curr_pose = self.robot_client.tcp_pose
                        curr_gripper_pos = self.robot_client.gripper_pos
                        curr_t = time.monotonic()
                        last_waypoint_time = curr_t
                        pose_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]), poses=np.array([curr_pose])
                        )
                        gripper_pos_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]),
                            poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
                        )
                        _seed_joint_interp_from_state()
                        self.reset_success.value = True
                        gain = self.robot_client.get_gain()
                        gain["kp"] = np.array([300, 300, 400, 80, 50, 30])
                        self.robot_client.set_gain(gain)
                        # self.robot_joint_client.set_gain(gain)

                    elif cmd == Command.ADD_WAYPOINT.value:
                        if len(self.waypoint_buffer) > 0:
                            last_waypoint_time = self.waypoint_buffer[-1]["target_time"]
                            if command["target_time"] <= last_waypoint_time:
                                print(
                                    f"[Arx5Controller] Waypoint time {command['target_time']:.3f} is not in the future, skipping"
                                )
                                continue
                        self.waypoint_buffer.append(command)
                    elif cmd == Command.UPDATE_TRAJECTORY.value:
                        if len(self.waypoint_buffer) == 0:
                            print(
                                f"[Arx5Controller] No new waypoints to update trajectory"
                            )
                            continue
                        start_time = time.monotonic()
                        # Dynamic latency matching
                        matching_dt = 0.01
                        pose_samples = np.zeros((3, 6))
                        pose_samples[0, :] = pose_interp(t_now - matching_dt)
                        pose_samples[1, :] = pose_interp(t_now)
                        pose_samples[2, :] = pose_interp(t_now + matching_dt)

                        input_poses = np.array(
                            [cmd["target_pose"] for cmd in self.waypoint_buffer]
                        )  # (N, 6)
                        input_times = np.array(
                            [cmd["target_time"] for cmd in self.waypoint_buffer]
                        )
                        input_gripper_pos = np.array(
                            [cmd["gripper_pos"] for cmd in self.waypoint_buffer]
                        )
                        input_pose_interp = PoseTrajectoryInterpolator(
                            times=input_times - input_times[0], poses=input_poses
                        )

                        latency_precision = 0.02
                        max_latency = 1.2
                        smoothing_time = 0.5
                        errors = []
                        error_weights = np.array(
                            [1, 1, 1, 0.1, 0.1, 0.1]
                        )  # x, y, z, rx, ry, rz
                        for latency in np.arange(
                            matching_dt, max_latency, latency_precision
                        ):
                            input_pose_samples = np.zeros((3, 6))
                            input_pose_samples[0, :] = input_pose_interp(
                                latency - matching_dt
                            )
                            input_pose_samples[1, :] = input_pose_interp(latency)
                            input_pose_samples[2, :] = input_pose_interp(
                                latency + matching_dt
                            )
                            error = np.sum(
                                np.abs(input_pose_samples - pose_samples)
                                * error_weights
                            )
                            errors.append(error)
                        errors = np.array(errors)
                        best_latency = np.arange(
                            matching_dt, max_latency, latency_precision
                        )[np.argmin(errors)]
                        # best_latency = 0.0

                        smoothened_input_poses = input_poses
                        new_times = input_times - input_times[0] + t_now - best_latency
                        for i in range((smoothened_input_poses.shape[0])):
                            if new_times[i] < t_now:
                                smoothened_input_poses[i] = pose_interp(new_times[i])
                            elif t_now <= new_times[i] < t_now + smoothing_time:
                                alpha = (new_times[i] - t_now) / smoothing_time
                                smoothened_input_poses[i] = (1 - alpha) * pose_interp(
                                    new_times[i]
                                ) + alpha * input_poses[i]
                            else:
                                smoothened_input_poses[i] = input_poses[i]

                        pose_interp = PoseTrajectoryInterpolator(
                            times=new_times, poses=smoothened_input_poses
                        )
                        extended_gripper_pos = np.zeros_like(input_poses)
                        extended_gripper_pos[:, 0] = input_gripper_pos
                        gripper_pos_interp = PoseTrajectoryInterpolator(
                            times=new_times, poses=extended_gripper_pos
                        )
                        if self.verbose:
                            print(
                                f"[Arx5Controller] latency: {best_latency:.3f}s, error: {errors.min():.3f}, time: {time.monotonic() - start_time:.3f}s"
                            )
                        # clear buffer
                        self.waypoint_buffer = []
                    elif cmd == Command.SET_JOINT_TARGET.value:
                        joint_angles = command.get("joint_angles", None)
                        target_time = float(command.get("target_time", time.time()))
                        if joint_angles is None:
                            print(
                                "[Arx5Controller] Warning: missing joint_angles "
                                "field in SET_JOINT_TARGET"
                            )
                            continue
                        try:
                            _schedule_joint_waypoint_cmd(joint_angles, target_time)
                            if self.verbose:
                                print(
                                    f"[Arx5Controller] set_joint_target -> {joint_angles} "
                                    f"@ {target_time:.3f}"
                                )
                        except Exception as e:
                            print(f"[Arx5Controller] Exception in set_joint_target: {e}")
                    elif cmd == Command.SCHEDULE_JOINT_WAYPOINT.value:
                        joint_angles = command.get("joint_angles", None)
                        target_time = float(command.get("target_time", time.time()))
                        if joint_angles is None:
                            print(
                                "[Arx5Controller] Warning: missing joint_angles "
                                "field in SCHEDULE_JOINT_WAYPOINT"
                            )
                            continue
                        try:
                            _schedule_joint_waypoint_cmd(joint_angles, target_time)
                            if self.verbose:
                                print(
                                    "[Arx5Controller] schedule_joint_waypoint -> "
                                    f"{joint_angles} @ {target_time:.3f}"
                                )
                        except Exception as e:
                            print(
                                "[Arx5Controller] Exception in "
                                f"schedule_joint_waypoint: {e}"
                            )
                    elif cmd == Command.SET_JOINT_TRAJECTORY.value:
                        joint_traj = command.get("joint_traj", None)
                        timestamps = command.get("timestamps", None)
                        gripper_traj = command.get("gripper_traj", None)

                        if joint_traj is None or timestamps is None:
                            print("[Arx5Controller] Missing trajectory data.")
                            continue

                        try:
                            self.robot_joint_client.set_joint_position_traj(
                                joint_traj, timestamps, gripper_traj
                            )
                            if self.verbose:
                                print(f"[Arx5Controller] Executing joint trajectory with {len(timestamps)} points.")
                        except Exception as e:
                            print(f"[Arx5Controller] Exception in SET_JOINT_TRAJECTORY: {e}")

                    elif cmd == Command.SET_GRAVITY_COMPENSATION.value:
                        # 优先真零力；ac_one / 普通 zmq_server 不支持时回退到
                        # 软 hold 阻尼拖拽（少量 kp 抗坠 + 拖动跟手）。
                        try:
                            print_drag_diag("enter_drag.before")
                            entered = False
                            drag_gripper_vel_hist = [0.0] * 5
                            drag_hold_pose = None
                            drag_hold_gripper = None
                            drag_hold_joint = None
                            drag_hold_mode = None
                            try:
                                self.robot_client.set_gravity_compensation(True)
                                # 真零力路径也要把夹爪位置环清零，否则墙力矩会被 kp 顶住
                                try:
                                    g_gc = self.robot_client.get_gain()
                                    g_gc = {
                                        "kp": np.asarray(g_gc["kp"], dtype=np.float64),
                                        "kd": np.asarray(g_gc["kd"], dtype=np.float64),
                                        "gripper_kp": 0.0,
                                        "gripper_kd": 0.0,
                                    }
                                    self.robot_client.set_gain(g_gc)
                                except Exception as grip_gain_exc:
                                    print(
                                        "[Arx5Controller] zero gripper gain after GC "
                                        f"warning: {grip_gain_exc}"
                                    )
                                is_gravity_compensation = True
                                entered = True
                                print("[Arx5Controller] set_gravity_compensation -> True")
                            except Exception as e:
                                print(
                                    "[Arx5Controller] SET_GRAVITY_COMPENSATION unsupported "
                                    f"({e}); falling back to soft-hold damping drag."
                                )
                                try:
                                    saved_gain_before_drag = self.robot_client.get_gain()
                                    hard_kp = np.asarray(
                                        saved_gain_before_drag["kp"], dtype=np.float64
                                    )
                                    hard_kd = np.asarray(
                                        saved_gain_before_drag["kd"], dtype=np.float64
                                    )
                                    try:
                                        pin = _pin_measured_setpoint(self.robot_client)
                                        print(
                                            "[Arx5Controller] pre-damping pin via "
                                            f"{pin}"
                                        )
                                    except Exception as sync_exc:
                                        pin = "failed"
                                        print(
                                            "[Arx5Controller] pre-damping pin "
                                            f"warning: {sync_exc}"
                                        )
                                    self.robot_client.get_state()
                                    hold_pose = np.asarray(
                                        self.robot_client.tcp_pose, dtype=np.float64
                                    ).reshape(6).copy()
                                    hold_grip = float(self.robot_client.gripper_pos)
                                    hold_joint = _measured_joint7(self.robot_client)

                                    self.robot_client.set_to_damping()
                                    damp = self.robot_client.get_gain()
                                    damp_kd = np.asarray(
                                        damp["kd"], dtype=np.float64
                                    )
                                    target_soft = {
                                        "kp": hard_kp * float(DRAG_ARM_KP_SCALE),
                                        "kd": damp_kd * float(DRAG_ARM_KD_SCALE),
                                        "gripper_kp": 0.0,
                                        "gripper_kd": 0.0,
                                    }
                                    # 分级降增益，避免 kp→0 悬崖；每级都钉回切入位姿
                                    soft = target_soft
                                    for alpha in (0.35, 0.65, 1.0):
                                        soft = {
                                            "kp": (1.0 - alpha) * hard_kp
                                            + alpha * target_soft["kp"],
                                            "kd": (1.0 - alpha) * hard_kd
                                            + alpha * target_soft["kd"],
                                            "gripper_kp": 0.0,
                                            "gripper_kd": 0.0,
                                        }
                                        self.robot_client.set_gain(soft)
                                        try:
                                            if pin == "joint":
                                                self.robot_client.set_joint_position(
                                                    hold_joint
                                                )
                                            else:
                                                self.robot_client.set_tcp_pose(
                                                    hold_pose, hold_grip
                                                )
                                        except Exception as hold_pin_exc:
                                            if (
                                                pin != "joint"
                                                and _looks_like_joint_server_error(
                                                    hold_pin_exc
                                                )
                                            ):
                                                pin = "joint"
                                                try:
                                                    self.robot_client.set_joint_position(
                                                        hold_joint
                                                    )
                                                    print(
                                                        "[Arx5Controller] ramp hold "
                                                        "switched tcp -> joint"
                                                    )
                                                except Exception as joint_pin_exc:
                                                    print(
                                                        "[Arx5Controller] ramp hold "
                                                        f"joint pin warning: {joint_pin_exc}"
                                                    )
                                            else:
                                                print(
                                                    "[Arx5Controller] ramp hold pin "
                                                    f"warning: {hold_pin_exc}"
                                                )
                                        time.sleep(0.05)

                                    if pin == "joint":
                                        drag_hold_mode = "joint"
                                        drag_hold_joint = hold_joint.copy()
                                        drag_hold_pose = None
                                        drag_hold_gripper = None
                                    else:
                                        drag_hold_mode = "tcp"
                                        drag_hold_pose = hold_pose.copy()
                                        drag_hold_gripper = hold_grip
                                        drag_hold_joint = None

                                    is_gravity_compensation = True
                                    entered = True
                                    print_drag_diag(
                                        "enter_drag.after_damping_soft",
                                        gain_override=soft,
                                    )
                                    print(
                                        "[Arx5Controller] soft-hold damping drag ON "
                                        f"(arm kp*={DRAG_ARM_KP_SCALE}, "
                                        f"kd*={DRAG_ARM_KD_SCALE}; "
                                        f"hold={drag_hold_mode}; "
                                        "gripper kp=0 + torque assist). "
                                        "Drag freely; idle pose is held."
                                    )
                                except Exception as e2:
                                    is_gravity_compensation = False
                                    saved_gain_before_drag = None
                                    drag_hold_pose = None
                                    drag_hold_gripper = None
                                    drag_hold_joint = None
                                    drag_hold_mode = None
                                    print(
                                        f"[Arx5Controller] damping drag FAILED: {e2}"
                                    )
                            if entered:
                                # Stop stale joint traj; drag loop must not resume it.
                                joint_control_active = False
                                # True-GC: still sync once. Soft-hold: do NOT re-pin
                                # to a already-sagging pose (would bake in the drop).
                                if drag_hold_mode is None:
                                    try:
                                        pin = _pin_measured_setpoint(self.robot_client)
                                        print(
                                            f"[Arx5Controller] enter-drag pin via {pin}"
                                        )
                                    except Exception as sync_exc:
                                        print(
                                            "[Arx5Controller] enter-drag pin warning: "
                                            f"{sync_exc}"
                                        )
                                try:
                                    _seed_joint_interp_from_state()
                                except Exception:
                                    pass
                                try:
                                    g_now = self.robot_client.get_gain()
                                    parts = ["follow"]
                                    if DRAG_GRIPPER_REBOUND_ENABLE:
                                        parts.append("rebound")
                                    if DRAG_GRIPPER_WALL_ENABLE:
                                        parts.append("wall")
                                    if drag_hold_mode is not None:
                                        parts.append(f"soft_hold:{drag_hold_mode}")
                                    print(
                                        "[Arx5Controller] drag gain check: "
                                        f"gripper_kp={float(g_now['gripper_kp']):.4f} "
                                        f"gripper_kd={float(g_now['gripper_kd']):.4f} "
                                        f"arm_kp={np.array2string(np.asarray(g_now['kp']), precision=1)} "
                                        f"assist={'+'.join(parts)} "
                                        f"open={DRAG_GRIPPER_FRICTION_OPEN} "
                                        f"close={DRAG_GRIPPER_FRICTION_CLOSE} "
                                        f"rebound_lo={DRAG_GRIPPER_REBOUND_LO}/"
                                        f"{DRAG_GRIPPER_REBOUND_TAU}"
                                    )
                                except Exception:
                                    pass
                            if not entered:
                                print(
                                    "  Hint: start zmq_server_gravity_compensation.py, "
                                    "or collect with --no_zero_force (skip P)."
                                )
                            self.waypoint_buffer = []
                        finally:
                            self.gravity_compensation_done.value = True

                    elif cmd == Command.SET_END_CONTROL.value:
                        # 日志结论：sync 后先抬 kp 时，臂被猛拉向 home（qd 0.06→3.7）。
                        # 疑似 set_gain 会重置伺服目标。因此必须：
                        #   set_gain → 立刻 sync_after_drag 重新钉住实测位姿。
                        # 且不要在拖拽点拉满 kp；只用中等轨迹增益，满增益延后。
                        try:
                            print_drag_diag("exit_drag.request_received")
                            # Stop soft-hold tracking before gain ramp / settle.
                            drag_hold_pose = None
                            drag_hold_gripper = None
                            drag_hold_joint = None
                            drag_hold_mode = None
                            # 若此前是真零力，先关 GC，避免回程/再进拖拽时 server 状态脏
                            try:
                                self.robot_client.set_gravity_compensation(False)
                            except Exception:
                                pass
                            settle_deadline = time.monotonic() + 0.5
                            settled_samples = 0
                            while time.monotonic() < settle_deadline:
                                self.robot_client.get_state()
                                qd = np.asarray(
                                    self.robot_client.joint_vel, dtype=np.float64
                                )
                                if qd.size > 0 and float(np.max(np.abs(qd))) < 0.08:
                                    settled_samples += 1
                                    if settled_samples >= 3:
                                        break
                                else:
                                    settled_samples = 0
                                time.sleep(0.05)
                            print(
                                "[Arx5Controller] drag settle "
                                f"samples={settled_samples}/3"
                            )
                            print_drag_diag("exit_drag.after_settle")

                            self.waypoint_buffer = []
                            # clear 后 counter=0，get() 会读到 buffer 末槽的陈旧数据；
                            # 必须立刻写入当前实测位姿，避免父进程读到 drag 前的 home。
                            self.ring_buffer.clear()
                            self.robot_client.get_state()
                            state = dict()
                            for key, func_name in self.receive_keys:
                                state[key] = getattr(self.robot_client, func_name)
                            t_recv = time.time()
                            state["robot_receive_timestamp"] = t_recv
                            state["robot_timestamp"] = t_recv - self.receive_latency
                            self.ring_buffer.put(state)

                            track_alpha = 0.25
                            try:
                                soft = self.robot_client.get_gain()
                                hard = (
                                    saved_gain_before_drag
                                    if saved_gain_before_drag is not None
                                    else soft
                                )
                                soft_kp = np.asarray(soft["kp"], dtype=np.float64)
                                soft_kd = np.asarray(soft["kd"], dtype=np.float64)
                                hard_kp = np.asarray(hard["kp"], dtype=np.float64)
                                hard_kd = np.asarray(hard["kd"], dtype=np.float64)
                                soft_gkp = float(soft["gripper_kp"])
                                soft_gkd = float(soft["gripper_kd"])
                                hard_gkp = float(hard["gripper_kp"])
                                hard_gkd = float(hard["gripper_kd"])

                                # 更细前几级；每级都是 set_gain 后立刻 pin（顺序很关键）
                                exit_pin_mode = None
                                for alpha in (0.03, 0.06, 0.10, 0.15, 0.20, track_alpha):
                                    blended = {
                                        "kp": (1.0 - alpha) * soft_kp + alpha * hard_kp,
                                        "kd": (1.0 - alpha) * soft_kd + alpha * hard_kd,
                                        "gripper_kp": (1.0 - alpha) * soft_gkp
                                        + alpha * hard_gkp,
                                        "gripper_kd": (1.0 - alpha) * soft_gkd
                                        + alpha * hard_gkd,
                                    }
                                    self.robot_client.set_gain(blended)
                                    try:
                                        exit_pin_mode = _pin_measured_setpoint(
                                            self.robot_client
                                        )
                                    except Exception as e_sync:
                                        print(
                                            "[Arx5Controller] pin after set_gain "
                                            f"failed: {e_sync}"
                                        )
                                    print_drag_diag(
                                        f"exit_drag.gain_alpha_{alpha:.2f}",
                                        gain_override=blended,
                                    )
                                    time.sleep(0.10)
                                print(
                                    "[Arx5Controller] exit drag: track gain "
                                    f"~{track_alpha:.0%} (full gain deferred)"
                                    + (
                                        f", pin={exit_pin_mode}"
                                        if exit_pin_mode is not None
                                        else ""
                                    )
                                )
                            except Exception as e_gain:
                                print(f"[Arx5Controller] restore gain warning: {e_gain}")
                                try:
                                    _pin_measured_setpoint(self.robot_client)
                                except Exception:
                                    pass

                            try:
                                _pin_measured_setpoint(self.robot_client)
                            except Exception:
                                pass
                            self.robot_client.get_state()
                            curr_pose = self.robot_client.tcp_pose
                            curr_gripper_pos = self.robot_client.gripper_pos
                            curr_t = time.monotonic()
                            last_waypoint_time = curr_t
                            pose_interp = PoseTrajectoryInterpolator(
                                times=np.array([curr_t]), poses=np.array([curr_pose])
                            )
                            gripper_pos_interp = PoseTrajectoryInterpolator(
                                times=np.array([curr_t]),
                                poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
                            )
                            # Seed joint interp but keep inactive while gains ramp in
                            # RESTORE_GAINS: otherwise pin + active hold fight → shake.
                            try:
                                _seed_joint_interp_from_state()
                            except Exception as seed_exc:
                                print(
                                    "[Arx5Controller] exit-drag joint seed warning: "
                                    f"{seed_exc}"
                                )
                            joint_control_active = False
                            state = dict()
                            for key, func_name in self.receive_keys:
                                state[key] = getattr(self.robot_client, func_name)
                            t_recv = time.time()
                            state["robot_receive_timestamp"] = t_recv
                            state["robot_timestamp"] = t_recv - self.receive_latency
                            self.ring_buffer.put(state)
                            print_drag_diag("exit_drag.before_tcp_loop")

                            is_gravity_compensation = False
                        finally:
                            self.end_control_done.value = True

                    elif cmd == Command.RESTORE_GAINS_AFTER_DRAG.value:
                        try:
                            print_drag_diag("restore_full_gain.before")
                            # Avoid dual command streams (interp hold + pin) at rising kp.
                            joint_control_active = False
                            if saved_gain_before_drag is not None:
                                try:
                                    # 到达目标后再拉满；先 pin 再 set_gain，随后立刻再 pin
                                    try:
                                        _pin_measured_setpoint(self.robot_client)
                                    except Exception:
                                        pass
                                    soft = self.robot_client.get_gain()
                                    hard = saved_gain_before_drag
                                    soft_kp = np.asarray(soft["kp"], dtype=np.float64)
                                    soft_kd = np.asarray(soft["kd"], dtype=np.float64)
                                    hard_kp = np.asarray(hard["kp"], dtype=np.float64)
                                    hard_kd = np.asarray(hard["kd"], dtype=np.float64)
                                    soft_gkp = float(soft["gripper_kp"])
                                    soft_gkd = float(soft["gripper_kd"])
                                    hard_gkp = float(hard["gripper_kp"])
                                    hard_gkd = float(hard["gripper_kd"])
                                    for alpha in (0.4, 0.7, 1.0):
                                        blended = {
                                            "kp": (1.0 - alpha) * soft_kp
                                            + alpha * hard_kp,
                                            "kd": (1.0 - alpha) * soft_kd
                                            + alpha * hard_kd,
                                            "gripper_kp": (1.0 - alpha) * soft_gkp
                                            + alpha * hard_gkp,
                                            "gripper_kd": (1.0 - alpha) * soft_gkd
                                            + alpha * hard_gkd,
                                        }
                                        self.robot_client.set_gain(blended)
                                        try:
                                            pin = _pin_measured_setpoint(
                                                self.robot_client
                                            )
                                            if alpha >= 0.99:
                                                print(
                                                    "[Arx5Controller] full-gain pin "
                                                    f"via {pin}"
                                                )
                                        except Exception as pin_exc:
                                            print(
                                                "[Arx5Controller] full-gain pin "
                                                f"warning: {pin_exc}"
                                            )
                                        print_drag_diag(
                                            f"restore_full_gain.alpha_{alpha:.1f}",
                                            gain_override=blended,
                                        )
                                        time.sleep(0.10)
                                    # Post-restore hold:
                                    # - cartesian collect: EE pose stream OK
                                    # - Kai0 --joint: keep passive after pin;
                                    #   never enable joint_control tick-hold
                                    #   here (fought hold_current/policy and
                                    #   shook both arms).
                                    try:
                                        self.robot_client.get_state()
                                        curr_pose = self.robot_client.tcp_pose
                                        curr_gripper_pos = (
                                            self.robot_client.gripper_pos
                                        )
                                        curr_t = time.monotonic()
                                        last_waypoint_time = curr_t
                                        pose_interp = PoseTrajectoryInterpolator(
                                            times=np.array([curr_t]),
                                            poses=np.array([curr_pose]),
                                        )
                                        gripper_pos_interp = (
                                            PoseTrajectoryInterpolator(
                                                times=np.array([curr_t]),
                                                poses=np.array(
                                                    [
                                                        [
                                                            curr_gripper_pos,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                        ]
                                                    ]
                                                ),
                                            )
                                        )
                                        _seed_joint_interp_from_state()
                                        joint_control_active = False
                                        try:
                                            self.robot_client.set_tcp_pose(
                                                curr_pose, curr_gripper_pos
                                            )
                                            allow_ee_pose = True
                                            print(
                                                "[Arx5Controller] post-restore "
                                                "EE hold"
                                            )
                                        except Exception as probe_exc:
                                            allow_ee_pose = False
                                            if _looks_like_joint_server_error(
                                                probe_exc
                                            ):
                                                print(
                                                    "[Arx5Controller] post-restore "
                                                    "passive pin (--joint)"
                                                )
                                            else:
                                                print(
                                                    "[Arx5Controller] post-restore "
                                                    f"EE probe warning: {probe_exc}"
                                                )
                                    except Exception as seed_exc:
                                        print(
                                            "[Arx5Controller] post-restore seed "
                                            f"warning: {seed_exc}"
                                        )
                                        allow_ee_pose = False
                                        joint_control_active = False
                                    print(
                                        "[Arx5Controller] full gain restored after arrive"
                                    )
                                except Exception as e_gain:
                                    print(
                                        "[Arx5Controller] full gain restore warning: "
                                        f"{e_gain}"
                                    )
                                saved_gain_before_drag = None
                            else:
                                print("[Arx5Controller] no saved gain to restore")
                                try:
                                    self.robot_client.get_state()
                                    curr_pose = self.robot_client.tcp_pose
                                    curr_gripper_pos = self.robot_client.gripper_pos
                                    curr_t = time.monotonic()
                                    last_waypoint_time = curr_t
                                    pose_interp = PoseTrajectoryInterpolator(
                                        times=np.array([curr_t]),
                                        poses=np.array([curr_pose]),
                                    )
                                    gripper_pos_interp = PoseTrajectoryInterpolator(
                                        times=np.array([curr_t]),
                                        poses=np.array(
                                            [[curr_gripper_pos, 0, 0, 0, 0, 0]]
                                        ),
                                    )
                                    _seed_joint_interp_from_state()
                                except Exception:
                                    pass
                                allow_ee_pose = False
                                joint_control_active = False
                            print_drag_diag("restore_full_gain.after")
                        finally:
                            self.restore_gains_done.value = True

                    elif cmd == Command.SET_TO_DAMPING.value:
                        self.robot_client.set_to_damping()
                        print(f"[Arx5Controller] set_to_damping")
                        is_gravity_compensation = False
                        self.waypoint_buffer = []

                    else:
                        keep_running = False
                        print(f"[Arx5Controller] Unknown command {cmd}")
                        break
                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                    if self.verbose:
                        print(
                            "[Arx5Controller] Controller process is ready "
                            f"({self.robot_ip}:{self.robot_port})",
                            flush=True,
                        )
                iter_idx += 1
                # if self.verbose:
                #     print(f"[Arx5Controller] Actual frequency {1/(time.monotonic() - t_now)} Hz")

        finally:
            print("[Arx5Controller] Setting robot to damping")
            self.robot_client.set_to_damping()
            # self.robot_joint_client.set_to_damping()
            del self.robot_client
            # del self.robot_joint_client
            self.ready_event.set()
            if self.verbose:
                print("[Arx5Controller] Controller process terminated")
