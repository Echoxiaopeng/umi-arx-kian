import multiprocessing as mp
import enum
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, cast
import numpy as np
from utils.other_util import precise_wait
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

from modules.pose_trajectory_interpolator import PoseTrajectoryInterpolator
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
    ):
        super().__init__(name="Arx5Controller")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        # self.robot_joint_port = robot_joint_port

        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "gripper_pos": 0.0,
            # "joint_angles": np.zeros((7,), dtype=np.float64),
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

        # Will be initialized in the subprocess
        self.robot_client: Arx5Client
        # self.robot_joint_port: Arx5Client
        self.reset_success = Value(ctypes.c_bool, False)
        # 父进程在 set_end_control / restore_gains 后必须等待，否则会抢跑回程
        # （日志：Moving to home 插在 gain_alpha_0.10 与 0.15 之间，from 读到 drag 前位姿）
        self.end_control_done = Value(ctypes.c_bool, True)
        self.restore_gains_done = Value(ctypes.c_bool, True)

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
        print(f"[Arx5Controller] Waiting for controller process to be ready")
        print(f"{self.launch_timeout=}")
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

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
        设置关节角目标 (joint-space control)
        joint_angles: 7个关节角（弧度）
        target_time:  目标执行时间（Unix时间戳）
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

    
    def set_gravity_compensation(self):
        message = {"cmd": Command.SET_GRAVITY_COMPENSATION.value}
        self.input_queue.put(message)

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
        self.robot_client.reset_to_home()
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
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]), poses=np.array([curr_pose])
            )
            gripper_pos_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            is_gravity_compensation = False
            saved_gain_before_drag = None

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
                # 零力模式下停止下发 TCP，否则会把臂拉回插值器目标
                if not is_gravity_compensation:
                    pose_cmd = pose_interp(t_now)
                    gripper_cmd = float(gripper_pos_interp(t_now)[0])
                    try:
                        self.robot_client.set_tcp_pose(pose_cmd, gripper_cmd)
                    except Exception as e:
                        # ZMQ ac_one: consecutive EE pose L2 must be < 0.1；
                        # 单次失败不应打死控制器进程。
                        print(f"[Arx5Controller] set_tcp_pose failed: {e}")
                        try:
                            self.robot_client.get_state()
                        except Exception:
                            pass
                else:
                    self.robot_client.get_state()

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
                        # 新增：处理关节空间命令
                        joint_angles = command.get("joint_angles", None)
                        target_time = float(command.get("target_time", time.time()))

                        if joint_angles is None:
                            print("[Arx5Controller] Warning: missing joint_angles field in SET_JOINT_TARGET")
                            continue

                        try:
                            # 调用底层客户端接口
                            self.robot_joint_client.set_joint_position(joint_angles)
                            if self.verbose:
                                print(f"[Arx5Controller] set_joint_target -> {joint_angles}")
                        except AttributeError:
                            print("[Arx5Controller] Error: robot_joint_client has no set_joint_position() method!")
                        except Exception as e:
                            print(f"[Arx5Controller] Exception in set_joint_target: {e}")
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
                        # 优先真零力；ac_one 不支持时回退到阻尼拖拽（teach_replay 同款）
                        print_drag_diag("enter_drag.before")
                        entered = False
                        try:
                            self.robot_client.set_gravity_compensation(True)
                            is_gravity_compensation = True
                            entered = True
                            print("[Arx5Controller] set_gravity_compensation -> True")
                        except Exception as e:
                            print(
                                "[Arx5Controller] SET_GRAVITY_COMPENSATION unsupported "
                                f"({e}); falling back to SET_TO_DAMPING drag."
                            )
                            try:
                                saved_gain_before_drag = self.robot_client.get_gain()
                                self.robot_client.set_to_damping()
                                # 阻尼拖拽期间把位置刚度清零。仅降低 kd 不够：
                                # EVAN_SYNC_AFTER_DRAG 一切回位置控制就会立刻使用 kp，
                                # 若 kp 仍是约 300，会在 Q/P 时产生明显冲击。
                                gain = self.robot_client.get_gain()
                                soft = {
                                    "kp": np.zeros_like(
                                        np.asarray(gain["kp"], dtype=np.float64)
                                    ),
                                    "kd": np.asarray(gain["kd"], dtype=np.float64) * 0.1,
                                    "gripper_kp": float(gain["gripper_kp"]),
                                    "gripper_kd": float(gain["gripper_kd"]),
                                }
                                self.robot_client.set_gain(soft)
                                is_gravity_compensation = True  # 停发 TCP，允许拖动
                                entered = True
                                print_drag_diag(
                                    "enter_drag.after_damping_soft", gain_override=soft
                                )
                                print(
                                    "[Arx5Controller] damping drag ON "
                                    "(kd*=0.1, TCP commands paused)"
                                )
                            except Exception as e2:
                                is_gravity_compensation = False
                                saved_gain_before_drag = None
                                print(f"[Arx5Controller] damping drag FAILED: {e2}")
                        if not entered:
                            print(
                                "  Hint: start zmq_server_gravity_compensation.py, "
                                "or collect with --no_zero_force (skip P)."
                            )
                        self.waypoint_buffer = []

                    elif cmd == Command.SET_END_CONTROL.value:
                        # 日志结论：sync 后先抬 kp 时，臂被猛拉向 home（qd 0.06→3.7）。
                        # 疑似 set_gain 会重置伺服目标。因此必须：
                        #   set_gain → 立刻 sync_after_drag 重新钉住实测位姿。
                        # 且不要在拖拽点拉满 kp；只用中等轨迹增益，满增益延后。
                        try:
                            print_drag_diag("exit_drag.request_received")
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

                                # 更细前几级；每级都是 set_gain 后立刻 sync（顺序很关键）
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
                                        self.robot_client.sync_after_drag()
                                    except Exception as e_sync:
                                        print(
                                            "[Arx5Controller] sync_after_drag after "
                                            f"set_gain failed: {e_sync}"
                                        )
                                    print_drag_diag(
                                        f"exit_drag.gain_alpha_{alpha:.2f}",
                                        gain_override=blended,
                                    )
                                    time.sleep(0.10)
                                print(
                                    "[Arx5Controller] exit drag: track gain "
                                    f"~{track_alpha:.0%} (full gain deferred)"
                                )
                            except Exception as e_gain:
                                print(f"[Arx5Controller] restore gain warning: {e_gain}")
                                try:
                                    self.robot_client.sync_after_drag()
                                except Exception:
                                    pass

                            try:
                                self.robot_client.sync_after_drag()
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
                            if saved_gain_before_drag is not None:
                                try:
                                    # 到达目标后再拉满；先 sync 再 set_gain，随后立刻再 sync
                                    try:
                                        self.robot_client.sync_after_drag()
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
                                            self.robot_client.sync_after_drag()
                                        except Exception:
                                            # 已在目标附近时，用当前 TCP 钉住即可
                                            self.robot_client.get_state()
                                            self.robot_client.set_tcp_pose(
                                                self.robot_client.tcp_pose,
                                                self.robot_client.gripper_pos,
                                            )
                                        print_drag_diag(
                                            f"restore_full_gain.alpha_{alpha:.1f}",
                                            gain_override=blended,
                                        )
                                        time.sleep(0.10)
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
                        print("[Arx5Controller] Controller process is ready")
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
