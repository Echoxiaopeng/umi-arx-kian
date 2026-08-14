"""
Adapter class to use SingleArm interface as a drop-in replacement for Arx5Client.
This allows Arx5Controller to use the arx_x5_python SDK directly instead of ZMQ.
"""

from typing import Optional, Union, Dict, Tuple, cast
import numpy as np
import numpy.typing as npt
import time

from modules.arx_x5_python.bimanual.script.single_arm import SingleArm

def quaternion_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    参数：
        quat: np.ndarray, 长度为 4 的数组 [w, x, y, z]
    返回：
        roll, pitch, yaw: 以弧度为单位的欧拉角
    """
    w, x, y, z = quat

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将欧拉角（roll, pitch, yaw）转换为四元数。
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def rotvec_to_rpy(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """将旋转向量转换为 roll-pitch-yaw"""
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(rotvec).as_euler('xyz')


def rpy_to_rotvec(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """将 roll-pitch-yaw 转换为旋转向量"""
    from scipy.spatial.transform import Rotation as R
    return R.from_euler('xyz', rpy).as_rotvec()


def rpy2rotm(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert roll-pitch-yaw angles to rotation matrix"""
    roll, pitch, yaw = rpy
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    return np.dot(R_z, np.dot(R_y, R_x))


def rotm2rotvec(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert rotation matrix to rotation vector"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        k = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        k = k / (2 * np.sin(theta))
        return theta * k


def ee_rpy_to_tcp_rotvec(ee_rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将 EE 的 RPY 转换为 TCP 的 rotvec
    (与 arx5_zmq_client.py 中的 ee2tcp 类似的坐标变换)
    """
    ee_rot_mat = rpy2rotm(ee_rpy)
    ee2tcp_rot_mat = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ])
    tcp_rot_mat = ee_rot_mat @ ee2tcp_rot_mat
    tcp_rotvec = rotm2rotvec(tcp_rot_mat)
    
    # Normalize to alternate representation
    angle_rad = np.linalg.norm(tcp_rotvec)
    if angle_rad > 1e-6:
        vec = tcp_rotvec / angle_rad
        alternate_angle_rad = 2 * np.pi - angle_rad
        tcp_rotvec = -vec * alternate_angle_rad
    
    return tcp_rotvec


def tcp_rotvec_to_ee_rpy(tcp_rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将 TCP 的 rotvec 转换为 EE 的 RPY
    """
    from scipy.spatial.transform import Rotation as R
    
    tcp_rot_mat = R.from_rotvec(tcp_rotvec).as_matrix()
    tcp2ee_rot_mat = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ])
    ee_rot_mat = tcp_rot_mat @ tcp2ee_rot_mat
    ee_rpy = R.from_matrix(ee_rot_mat).as_euler('xyz')
    return ee_rpy


def real_pos2gripper_width(real_pos: float) -> float:
    """将实际位置转换为夹爪宽度"""
    return real_pos / 5.0 * 0.11

def gripper_width2real_pos(gripper_width: float) -> float:
    """将夹爪宽度转换为实际位置"""

    return gripper_width / 0.11 * 5.0



class SingleArmClient:
    """
    Adapter class that wraps SingleArm to provide the same interface as Arx5Client.
    This allows Arx5Controller to use it as a drop-in replacement.
    """
    
    def __init__(
        self,
        can_port: str = "can0",
        arm_type: int = 0,
        urdf_path: Optional[str] = None,
    ):
        """
        初始化 SingleArm 适配器
        
        Args:
            can_port: CAN 总线端口名称
            arm_type: 机械臂类型 (0: x5, 1: x5_master, 2: x5_2025)
            urdf_path: URDF 文件路径（可选，SingleArm 会自动选择，保留参数以兼容旧接口）
        """
        # urdf_path 参数保留以兼容旧接口，但 SingleArm 会自动选择 URDF 文件
        _ = urdf_path
        
        # 构建 SingleArm 需要的 config
        # 注意: SingleArm 使用 "type" 而不是 "arm_type"
        arm_config: Dict[str, Union[str, int]] = {
            "can_port": can_port,
            "type": arm_type,
        }
        
        # SingleArm 的 __init__ 会自动调用 arx_x(500, 2000, 10)
        self.arm = SingleArm(arm_config)
        
        self.can_port = can_port
        self.arm_type = arm_type
        
        # 内部状态缓存
        self.latest_state: Dict[str, Union[npt.NDArray[np.float64], float]] = {}
        self._gravity_compensation_enabled = False
        
        # 记录启动时间，用于计算相对时间戳（与 Arx5Client 格式一致）
        self._start_time = time.time()
        
        print(f"SingleArmClient initialized on {can_port}, type={arm_type}")
        self.get_state()
        print("Initial state fetched")
    
    def get_state(self) -> Dict[str, Union[npt.NDArray[np.float64], float]]:
        """获取机器人当前状态，格式与 Arx5Client.get_state() 完全一致"""
        # 获取末端位姿 (xyzwxyz 格式: x, y, z, w, x, y, z)
        ee_pose_raw = self.arm.get_ee_pose()
        
        # 提取位置和四元数
        pos = np.array([ee_pose_raw[0], ee_pose_raw[1], ee_pose_raw[2]])
        quat = np.array([ee_pose_raw[3], ee_pose_raw[4], ee_pose_raw[5], ee_pose_raw[6]])
        
        # 转换为 RPY
        roll, pitch, yaw = quaternion_to_euler(quat)
        ee_rpy = np.array([roll, pitch, yaw])
        
        # EE pose (position + rpy) - shape (6,)
        ee_pose = np.concatenate([pos, ee_rpy])
        
        # 获取关节状态 (可能是 6 或 7 个，第 7 个是夹爪)
        joint_pos_raw = np.array(self.arm.get_joint_positions())
        joint_vel_raw = np.array(self.arm.get_joint_velocities())
        joint_torque_raw = np.array(self.arm.get_joint_currents())
        
        # 提取夹爪数据 (第 7 个关节，如果存在)
        if len(joint_pos_raw) > 6:
            gripper_pos = float(joint_pos_raw[6])
            gripper_vel = float(joint_vel_raw[6]) if len(joint_vel_raw) > 6 else 0.0
            gripper_torque = float(joint_torque_raw[6]) if len(joint_torque_raw) > 6 else 0.0
            # 只保留前 6 个关节
            joint_pos = joint_pos_raw[:6]
            joint_vel = joint_vel_raw[:6]
            joint_torque = joint_torque_raw[:6]
        else:
            gripper_pos = 0.0
            gripper_vel = 0.0
            gripper_torque = 0.0
            joint_pos = joint_pos_raw
            joint_vel = joint_vel_raw
            joint_torque = joint_torque_raw
        
        # 使用相对时间戳（从启动开始计算），与 Arx5Client 格式一致
        relative_timestamp = time.time() - self._start_time
        
        self.latest_state = {
            "timestamp": relative_timestamp,
            "ee_pose": ee_pose,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
            "gripper_pos": real_pos2gripper_width(gripper_pos),
            "gripper_vel": gripper_vel,
            "gripper_torque": gripper_torque,
        }
        
        return self.latest_state


    def set_ee_pose(
        self,
        pose_6d: npt.NDArray[np.float64],
        gripper_pos: Union[float, None] = None
    ) -> Dict[str, Union[npt.NDArray[np.float64], float]]:
        """
        设置末端位姿 (EE pose: x, y, z, roll, pitch, yaw)
        """
        pos = pose_6d[:3]
        rpy = pose_6d[3:6]
        
        # 转换为四元数
        quat = euler_to_quaternion(rpy[0], rpy[1], rpy[2])
        
        # 使用 SingleArm 的 set_ee_pose(pos, quat) 接口
        self.arm.set_ee_pose(pos=pos, quat=quat)
        
        # 设置夹爪
        if gripper_pos is not None:
            self.arm.set_catch_pos(gripper_width2real_pos(gripper_pos))
        
        return self.get_state()

    def set_mode(self, mode: int):
        self.arm.set_mode(mode)
        return self.get_state()
    
    def set_tcp_pose(
        self,
        tcp_pose_6d: npt.NDArray[np.float64],
        gripper_pos: Union[float, None] = None,
    ) -> Dict[str, Union[npt.NDArray[np.float64], float]]:
        """
        设置 TCP 位姿 (TCP pose: x, y, z, rx, ry, rz - rotvec)
        需要转换为 EE pose
        """
        pos = tcp_pose_6d[:3]
        tcp_rotvec = tcp_pose_6d[3:6]
        
        # 转换 TCP rotvec 到 EE rpy
        ee_rpy = tcp_rotvec_to_ee_rpy(tcp_rotvec)
        
        ee_pose = np.concatenate([pos, ee_rpy])
        return self.set_ee_pose(ee_pose, gripper_pos)
    
    def reset_to_home(self):
        """回到初始位置"""
        self.arm.go_home()
        time.sleep(2.0)  # 等待回到初始位置
        self.get_state()
    
    def set_to_damping(self):
        """设置阻尼模式（保护模式）"""
        self.arm.protect_mode()
        self.get_state()
    
    def set_to_normal(self):
        """设置正常模式（通过设置关节位置触发）"""
        # SingleArm 没有直接的 normal_mode，通过设置当前位置来触发
        current_joint_pos = np.array(self.arm.get_joint_positions())
        self.arm.set_joint_positions(current_joint_pos[:6])
        self.get_state()

    def set_gravity_compensation(self):
        """设置重力补偿模式"""
        self.arm.set_mode(3)
        self.get_state()

    def set_end_control(self):
        """设置结束控制模式"""
        self.arm.set_mode(4)
        self.get_state()
    
    def get_gravity_compensation(self) -> bool:
        """获取重力补偿状态"""
        return self._gravity_compensation_enabled
    
    def get_gain(self) -> Dict[str, Union[npt.NDArray[np.float64], float]]:
        """获取增益参数（SingleArm 可能不支持，返回默认值）"""
        return {
            "kp": np.array([300, 300, 400, 80, 50, 30]),
            "kd": np.array([10, 10, 10, 5, 5, 5]),
        }
    
    def set_gain(self, gain: Dict[str, Union[npt.NDArray[np.float64], float]]):
        """设置增益参数（SingleArm SDK 不支持，此方法为空实现以保持兼容性）"""
        _ = gain  # 标记参数已使用，避免 linter 警告
    
    def set_joint_position(
        self,
        joint_pos: npt.NDArray[np.float64],
    ) -> Dict[str, Union[npt.NDArray[np.float64], float]]:
        """设置关节位置"""
        # SingleArm 的 set_joint_positions 内部会调用 set_arm_status(5)
        self.arm.set_joint_positions(joint_pos)
        return self.get_state()
    
    # ========= Properties (与 Arx5Client 兼容) =========
    
    @property
    def timestamp(self) -> float:
        return cast(float, self.latest_state.get("timestamp", time.time()))
    
    @property
    def ee_pose(self) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self.latest_state.get("ee_pose", np.zeros(6)))
    
    @property
    def joint_pos(self) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self.latest_state.get("joint_pos", np.zeros(6)))
    
    @property
    def joint_vel(self) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self.latest_state.get("joint_vel", np.zeros(6)))
    
    @property
    def joint_torque(self) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self.latest_state.get("joint_torque", np.zeros(6)))
    
    @property
    def gripper_pos(self) -> float:
        return cast(float, self.latest_state.get("gripper_pos", 0.0))
    
    @property
    def gripper_vel(self) -> float:
        return cast(float, self.latest_state.get("gripper_vel", 0.0))
    
    @property
    def gripper_torque(self) -> float:
        return cast(float, self.latest_state.get("gripper_torque", 0.0))
    
    @property
    def tcp_pose(self) -> npt.NDArray[np.float64]:
        """获取 TCP 位姿 (x, y, z, rx, ry, rz - rotvec)"""
        ee_pose = self.ee_pose
        pos = ee_pose[:3]
        ee_rpy = ee_pose[3:6]
        
        # 转换 EE rpy 到 TCP rotvec
        tcp_rotvec = ee_rpy_to_tcp_rotvec(ee_rpy)
        
        return np.concatenate([pos, tcp_rotvec])
    
    def __del__(self):
        print("SingleArmClient closed")
