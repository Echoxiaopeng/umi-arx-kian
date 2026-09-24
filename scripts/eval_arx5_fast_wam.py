"""

真实机器人 UMI 策略评估脚本（Fast-WAM 适配版）。

对齐数据集 / 训练标签:
  - data/umi_pick_elec_lerobot_real 及 FastWAM replay_buffer_lerobot_rel*
  - FastWAM dataloader 经 StepRelativeEEF10ToCurrentRelative 后，
    action horizon 为 current-relative:
      A_i = T_cur^{-1} @ T_{t0+i+1}
    decode: T_i = T_cur @ A_i（每步相对同一当前 EE，不链式积分）

通过 WebSocket 连接 Fast-WAM 推理服务（serve_real_robot_policy.py），
不需要在本脚本内加载模型权重。

用法:
# 1. 先启动 Fast-WAM 推理服务（另一个终端）:


# 2. 再运行本脚本:
python /data1/projects/umi-arx-kian/scripts/eval_arx5_fast_wam.py   -o data_local/umi_eval_output   --policy_host 112.91.138.78   --policy_port 5566 --terminal_keys -eh 4 --no_binarize_gripper


================ Human in control ==============
Robot movement:
    - SpaceMouse 移动 → 控制机器人末端 EEF（默认锁定在 xy 平面）
    - SpaceMouse 右键 → 解锁 z 轴
    - SpaceMouse 左键 → 启用旋转轴

Recording control:
    - 按 "C" → 开始评估（控制权交给 policy）
    - 按 "Q" → 退出程序

================ Policy in control ==============
Recording control:
    - 按 "S" → 停止评估，控制权交还给人

与 umi_pick_elec_lerobot_real / FastWAM current-relative 对齐:
  1. 图像 float[0,1] → uint8
  2. state (20): flatten(eef_pos[2,3], eef_rot[2,6], gripper[2,1])
     相对当前 EE（UMI get_real_umi_obs_dict）:
       [pos_hist(3), pos_curr(3≈0), rot_hist(6), rot_curr(6≈I), grip_hist, grip_curr]
     夹爪 state[18:20] 使用训练集极性（越大越夹紧），由 _build_umi_state 从 ZMQ 读数转换
  3. action (10): current-relative waypoint [pos(3), rot6d(6), gripper]
     A_i = T_cur^{-1} @ T_{target,i}
     SE(3) decode（相对同一当前 EE，不链式）:
       p_i = p_cur + R_cur @ A_i_pos
       R_i = R_cur @ R(A_i_rot6)
     模型输出的 action[...,9] 同样是训练集极性；下发前 invert 回 ZMQ 物理开口
  4. rot6 约定: 旋转矩阵前两行展平 [r00,r01,r02, r10,r11,r12]
  5. 夹爪极性（采集 zarr / 训练集 vs 真机 ZMQ）:
       - 训练/采集 robot0_gripper_width: 电机行程映射值，越大越夹紧（张开≈0.03，夹紧≈0.07）
       - 真机 ZMQ gripper_position: 物理开口宽度（米），越大越张开（min≈闭合，max≈全开）
       - 线性互转: g' = gripper_min + gripper_max - g（对称，可逆）
       - state 上传: ZMQ → 训练；action 下发: 训练 → ZMQ
       - --no_invert_gripper 同时关闭 state/action 两侧转换
       - 人工 SpaceMouse / init_gripper_delta 始终操作 ZMQ 物理开口，不 invert
  6. 训练/采集使用 link6，ZMQ 读写 eef_link：
     eef_link → 沿 TCP Z 退回 0.145m 到 link6 → 在 link6 上构造 state / decode action
     → 沿新姿态 TCP Z 前移 0.145m 到 eef_link → 下发绝对 TCP

Policy 控制循环（串行，不开流水线）:
  每轮: get_obs 拍 1 张腕部图 + 读 20d state
        → client.infer（阻塞，图+state 一起发给 Fast-WAM）
        → 只执行 chunk 前 exec_horizon 步（其余丢掉）
        → env.exec_actions(绝对 TCP, timestamps=time.time() 墙钟)
        → precise_wait(最后一个 waypoint 的 timestamp + 约 1 帧)
        → 再进入下一轮拍照

  时间戳:
    下发时现算: command_start_time = time.time() + 0.05
    第 k 步: command_start_time + k * dt（dt = 1/frequency）
    exec_actions / 控制器 / precise_wait 都按同一套 time.time() 墙钟对齐
    （控制器内部会把墙钟目标时刻转成 monotonic）

  exec_actions 不阻塞: 只把带时间戳的 waypoint 丢进控制器立刻返回。
  本脚本用 precise_wait(action_timestamps[-1]) 等本批排期结束，再拍下一张，
  避免推理慢时下一轮拍照叠在上一轮还在跑的动作上。

  推理慢（例如 1s）且场景/机械臂在推理期间基本静止时:
    观测与执行几何上仍大致对应拍照时的位姿，但执行会整体晚一个 infer_s。
    闭环纠偏也会变慢。这不是 wait 能消掉的。

  --rebase_decode（默认开）:
    推理结束后再读实时 TCP 做相对→绝对积分基准。
    旧用途: 推理期间上一轮还在动，防止 step0 拽回旧起点。
    等完上一轮再拍之后，拍照位姿与推理结束位姿应接近，通常可关
    --no_rebase_decode。日志 rebase_shift_mm 接近 0 即可确认。

# ssh 远程运行方式：
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority

python /data1/projects/umi-arx-kian/scripts/eval_arx5_fast_wam.py \
    -o data_local/umi_eval_output \
    --policy_port 5566 --terminal_keys\
    -eh 16 \
    -f 10


    --policy_host 112.91.138.78 \

"""

import sys
sys.path.insert(0, '/data1/projects/umi-arx-hzl')

import asyncio
import atexit
import contextlib
import os
import select
import sys
import termios
import tty
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import msgpack
import numpy as np
import scipy.spatial.transform as st
import websockets

# ===========================================================================
# 你的原项目模块（utils/, peripherals/, modules/）
# 这些需要在你原项目的 PYTHONPATH 中
# ===========================================================================
from utils.other_util import precise_wait
from peripherals.keystroke_counter import KeystrokeCounter, Key, KeyCode
from peripherals.spacemouse_shared_memory import Spacemouse
from modules.arx5_env import Arx5Env

# ===========================================================================
# msgpack numpy 序列化支持
# ===========================================================================

def _msgpack_default(obj):
    """将 numpy array / generic 序列化为 msgpack 兼容的 dict。"""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    raise TypeError(f"Unsupported type: {type(obj)}")


def _msgpack_decode(obj):
    """递归解码 msgpack 中的 __ndarray__ 为 numpy array。"""
    if isinstance(obj, dict):
        if b"__ndarray__" in obj:
            return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return {k: _msgpack_decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_msgpack_decode(v) for v in obj]
    return obj


# ===========================================================================
# 与 FastWAM current-relative 训练标签对齐
# Action (10): [pos(3), rot6d(6), gripper(abs)]
#   A_i = T_cur^{-1} @ T_{target,i}（horizon 内每步相对同一当前 EE）
#   decode: T_i = T_cur @ A_i
#           p_i = p_cur + R_cur @ A_i_pos
#           R_i = R_cur @ R_from_rot6(A_i_rot6)
# State  (20): [pos_hist(3), pos_curr(3), rot_hist(6), rot_curr(6), grip_hist, grip_curr]
#   - 位姿: 相对当前 EE（curr≈0 / I；hist = T_cur^{-1} @ T_hist）
#   - 旋转: 6D = 旋转矩阵前两行 [r00,r01,r02, r10,r11,r12]
#   - 夹爪: 训练集绝对值，越大越夹紧；真机 ZMQ 为物理开口（越大越张开）
# ===========================================================================

ACTION_DIM = 10
STATE_DIM = 20
GRIPPER_ACTION_INDEX = 9
IDENTITY_ROT6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
# X5.urdf: eef_link = link6 + 0.145m along tool x = TCP Z after ee2tcp.
X5_EEF_OFFSET_TCP_Z = 0.145
# umi_pick_elec_lerobot_real dataset_stats.json → state.default.global_{min,max}[18:20]
# 统计量在训练集极性下：min≈张开，max≈夹紧
TRAIN_STATE_GRIP_MIN = 0.01154826
TRAIN_STATE_GRIP_MAX = 0.0768357366
# ZMQ 物理开口参考上界（用于 invert 与 exec_actions clip，非训练统计）
DEFAULT_GRIPPER_MAX_WIDTH = 0.085


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """服务端期望 HWC uint8。Arx5Env(obs_float32=True) 常返回 [0,1] float。"""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected HWC RGB image, got shape {tuple(array.shape)}")
    array = array[..., :3]
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = array.astype(np.float32)
    if float(np.nanmax(array)) <= 1.5:
        array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).round().astype(np.uint8))


def _invert_gripper_width(
    grip: np.ndarray,
    *,
    gripper_min_width: float,
    gripper_max_width: float,
) -> np.ndarray:
    """训练集极性（越大越夹紧）↔ ZMQ 物理开口（越大越张开）。

    线性对翻: g' = gripper_min + gripper_max - g，在 [min, max] 内自逆。
    """
    return (
        gripper_min_width
        + gripper_max_width
        - np.asarray(grip, dtype=np.float64)
    )


def _shift_tcp_along_z(
    pos: np.ndarray,
    rotvec: np.ndarray,
    dz: float,
) -> np.ndarray:
    """p' = p + R_tcp @ [0, 0, dz]，姿态保持不变。"""
    position = np.asarray(pos, dtype=np.float64).reshape(3)
    if abs(float(dz)) < 1e-12:
        return position.copy()
    rotation = st.Rotation.from_rotvec(
        np.asarray(rotvec, dtype=np.float64).reshape(3)
    )
    return position + rotation.apply(
        np.array([0.0, 0.0, float(dz)], dtype=np.float64)
    )


def _x5_eef_to_link6(
    pos: np.ndarray,
    rotvec: np.ndarray,
    dz: float,
) -> np.ndarray:
    """ZMQ X5 eef_link 位置 → 训练/采集使用的 link6 位置。"""
    return _shift_tcp_along_z(pos, rotvec, -dz)


def _link6_to_x5_eef(
    pos: np.ndarray,
    rotvec: np.ndarray,
    dz: float,
) -> np.ndarray:
    """训练/采集使用的 link6 位置 → ZMQ X5 eef_link 位置。"""
    return _shift_tcp_along_z(pos, rotvec, dz)


def _abs_exec_link6_to_x5_eef(abs_exec: np.ndarray, dz: float) -> np.ndarray:
    """将 (N,7) link6 绝对 waypoint 转成 eef_link waypoint。"""
    out = np.asarray(abs_exec, dtype=np.float64).copy()
    if abs(float(dz)) < 1e-12:
        return out
    for i in range(out.shape[0]):
        out[i, :3] = _link6_to_x5_eef(out[i, :3], out[i, 3:6], dz)
    return out


def _read_tcp_pose(obs: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """读机器人当前绝对 TCP：pos(3), rotvec(3), gripper（ZMQ 物理开口，未 invert）。"""
    pos = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
    rotvec = np.asarray(obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64).reshape(3)
    grip = float(np.asarray(obs["robot0_gripper_width"][-1]).squeeze())
    return pos, rotvec, grip


def _read_live_tcp_pose(env) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    直接从机器人 ring buffer 读最新绝对 TCP（不走相机对齐，开销远低于 get_obs）。

    返回 (pos(3), rotvec(3), gripper, robot_timestamp)；
    gripper 为 ZMQ 物理开口（越大越张开，未 invert）。
    """
    states = env.get_robot_state()
    state = states[0] if isinstance(states, (list, tuple)) else states
    pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(-1)[:6]
    grip = float(np.asarray(state["gripper_position"]).squeeze())
    ts = float(np.asarray(state["robot_receive_timestamp"]).squeeze())
    return pose[:3].copy(), pose[3:6].copy(), grip, ts


def _read_tcp_pose_at(obs: dict, index: int) -> tuple[np.ndarray, np.ndarray, float]:
    """读 obs horizon 中指定 index 的绝对 TCP 与 ZMQ 夹爪（物理开口，未 invert）。"""
    pos = np.asarray(obs["robot0_eef_pos"][index], dtype=np.float64).reshape(3)
    rotvec = np.asarray(obs["robot0_eef_rot_axis_angle"][index], dtype=np.float64).reshape(3)
    grip = float(np.asarray(obs["robot0_gripper_width"][index]).squeeze())
    return pos, rotvec, grip


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 6D（前两行展平）。与训练侧 rotation.py 约定一致。"""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return matrix[:2, :].reshape(6).copy()


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """6D → 旋转矩阵（Gram-Schmidt；前两行约定）。"""
    d6 = np.asarray(d6, dtype=np.float64).reshape(6)
    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def _rotation_6d_to_rotvec(d6: np.ndarray) -> np.ndarray:
    return st.Rotation.from_matrix(_rotation_6d_to_matrix(d6)).as_rotvec()


def _rotvec_to_rotation_6d(rotvec: np.ndarray) -> np.ndarray:
    return _matrix_to_rotation_6d(
        st.Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()
    )


def _clip_rotation_6d(d6: np.ndarray, max_angle_rad: float) -> np.ndarray:
    """按相对转角幅值裁剪 6D 旋转；超限则按比例缩小。"""
    rotvec = _rotation_6d_to_rotvec(d6)
    angle = float(np.linalg.norm(rotvec))
    if angle <= max_angle_rad or angle < 1e-12:
        return np.asarray(d6, dtype=np.float64).reshape(6)
    return _rotvec_to_rotation_6d(rotvec * (max_angle_rad / angle))


def _pose_relative_to(
    pos: np.ndarray,
    rotvec: np.ndarray,
    *,
    ref_pos: np.ndarray,
    ref_rotvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 T_rel = T_ref^{-1} @ T，返回 (pos_rel, rot6_rel)。
      pos_rel = R_ref^T @ (pos - ref_pos)
      R_rel   = R_ref^T @ R
    """
    r_ref = st.Rotation.from_rotvec(np.asarray(ref_rotvec, dtype=np.float64).reshape(3))
    r = st.Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3))
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    ref_pos = np.asarray(ref_pos, dtype=np.float64).reshape(3)
    pos_rel = r_ref.inv().apply(pos - ref_pos)
    rot6_rel = _matrix_to_rotation_6d((r_ref.inv() * r).as_matrix())
    return pos_rel, rot6_rel


def _build_umi_state(
    obs: dict,
    *,
    eef_offset: float = 0.0,
    invert_gripper: bool = True,
    gripper_min_width: float = 0.011,
    gripper_max_width: float = DEFAULT_GRIPPER_MAX_WIDTH,
) -> np.ndarray:
    """
    构造 20 维 UMI state（与训练分布一致）:
        [pos_hist(3), pos_curr(3), rot_hist(6), rot_curr(6), grip_hist, grip_curr]

    需要 robot_obs_horizon >= 2。

    位姿（link6 坐标系，相对当前 EE）:
        eef_offset>0 时先把 ZMQ eef_link 沿 TCP Z 退回 link6
        curr → pos=0, rot=I
        hist → T_cur^{-1} @ T_hist

    夹爪:
        训练集为绝对值、越大越夹紧
        默认 invert_gripper=True：把 ZMQ 物理开口转为训练极性后再写入 state[18:20]
    """
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    if eef_pos.ndim != 2 or eef_pos.shape[0] < 2:
        raise ValueError(
            f"Need robot_obs_horizon>=2 for 20-d UMI state, got eef_pos shape {eef_pos.shape}"
        )
    pos_hist, rotvec_hist, grip_hist = _read_tcp_pose_at(obs, -2)
    pos_curr, rotvec_curr, grip_curr = _read_tcp_pose_at(obs, -1)
    pos_hist = _x5_eef_to_link6(pos_hist, rotvec_hist, eef_offset)
    pos_curr = _x5_eef_to_link6(pos_curr, rotvec_curr, eef_offset)

    pos_hist_rel, rot_hist_rel = _pose_relative_to(
        pos_hist, rotvec_hist, ref_pos=pos_curr, ref_rotvec=rotvec_curr
    )
    pos_curr_rel = np.zeros(3, dtype=np.float64)
    rot_curr_rel = IDENTITY_ROT6.copy()

    grip_pair = np.array([grip_hist, grip_curr], dtype=np.float64)
    if invert_gripper:
        grip_pair = _invert_gripper_width(
            grip_pair,
            gripper_min_width=gripper_min_width,
            gripper_max_width=gripper_max_width,
        )

    return np.concatenate(
        [
            pos_hist_rel,
            pos_curr_rel,
            rot_hist_rel,
            rot_curr_rel,
            grip_pair,
        ]
    ).astype(np.float32)


def _clip_umi_state_for_model(
    state: np.ndarray,
    *,
    grip_min: float = TRAIN_STATE_GRIP_MIN,
    grip_max: float = TRAIN_STATE_GRIP_MAX,
    clip_grip: bool = True,
) -> tuple[np.ndarray, bool]:
    """上传前将 state[18:20] clip 到训练集 gripper 统计范围（训练极性，越大越夹紧）。"""
    state = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    if not clip_grip:
        return state, False
    before = state[18:20].copy()
    state[18:20] = np.clip(state[18:20], float(grip_min), float(grip_max))
    clipped = not np.allclose(before, state[18:20], rtol=0.0, atol=1e-9)
    return state, clipped


def _delta_chunk_to_absolute_pose(
    action_chunk: np.ndarray,
    *,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
    gripper_min_width: float,
    gripper_max_width: float,
    binarize_gripper: bool = True,
    gripper_close_threshold: float = 0.055,
    invert_gripper: bool = True,
) -> np.ndarray:
    """
    模型输出 → 机器人可执行的绝对 TCP 轨迹。

    ---------------------------------------------------------------------------
    模型给的是什么？（current-relative，不是逐步相对）
    ---------------------------------------------------------------------------
    action_chunk[i] = A_i = [pos(3), rot6d(6), grip(1)]

    约定（与 FastWAM StepRelativeEEF10ToCurrentRelative 训练标签一致）:
        A_i  描述的是「从【当前】末端到【第 i 个目标】」的相对位姿
        A_i = T_cur^{-1} @ T_i
        即：horizon 里每一步都锚定在同一张拍照时的 cur，而不是上一步。

    例子（只看位置）:
        当前末端在世界系 p_cur
        A_0.pos = [0.01, 0, 0]  → 第 0 个目标在末端前方 1cm
        A_3.pos = [0.04, 0, 0]  → 第 3 个目标在末端前方 4cm（仍相对 cur，不是相对 A_2）

    ---------------------------------------------------------------------------
    和「旧版链式积分」的区别（容易混）
    ---------------------------------------------------------------------------
    旧 step-relative（已废弃）:
        a_i = T_i^{-1} @ T_{i+1}          # 每步相对「上一步自己」
        执行时必须链式:  T <- T @ a_i     # 走完第 i 步，基准才变成新位姿

    现在 current-relative:
        A_i = T_cur^{-1} @ T_i            # 每步相对「同一当前」
        执行时各自独立:  T_i = T_cur @ A_i
        循环里 cur / R_cur 全程不变，不要写成 T <- T @ A_i。

    公式展开（SE(3) 左乘）:
        T_i = T_cur @ A_i
        ⇒  p_i = p_cur + R_cur @ A_i.pos
        ⇒  R_i = R_cur @ R_from_rot6(A_i.rot6)

    ---------------------------------------------------------------------------
    夹爪
    ---------------------------------------------------------------------------
    action[..., 9] 是训练集极性（越大越夹紧）。
    invert 后变成 ZMQ 物理开口（越大越张开），再可选二值化 / clip。

    返回 (N, 7): [x,y,z, rx,ry,rz, grip_zmq]
    """
    deltas = np.asarray(action_chunk, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action chunk (N, {ACTION_DIM}), got {deltas.shape}")

    out = np.zeros((deltas.shape[0], 7), dtype=np.float64)

    # 解码基准 = 推理时的「当前」末端（通常是拍照/rebase 后的 link6 位姿）。
    # 下面整个 for 循环都用这一对 (pos_base, r_base)，不会随 i 更新。
    pos_base = np.asarray(cur_pos, dtype=np.float64).reshape(3).copy()
    r_base = st.Rotation.from_rotvec(np.asarray(cur_rotvec, dtype=np.float64).reshape(3))

    # ----- 夹爪：训练极性 → ZMQ 物理开口 -----
    abs_grip = deltas[:, GRIPPER_ACTION_INDEX].copy()
    if invert_gripper:
        abs_grip = _invert_gripper_width(
            abs_grip,
            gripper_min_width=gripper_min_width,
            gripper_max_width=gripper_max_width,
        )
    if binarize_gripper:
        # 阈值在「物理开口」空间：开口小于阈值 → 视为闭合
        abs_grip = np.where(abs_grip < gripper_close_threshold, gripper_min_width, gripper_max_width)
    abs_grip = np.clip(abs_grip, gripper_min_width, gripper_max_width)

    # ----- 位姿：每个 A_i 独立左乘到同一 T_cur -----
    for i in range(deltas.shape[0]):
        # p_i = p_cur + R_cur @ A_i.pos
        #   A_i.pos 是「在当前 EE 坐标系里」的位移；
        #   r_base.apply(...) 把它转到世界系，再加到当前世界系位置上。
        out[i, :3] = pos_base + r_base.apply(deltas[i, :3])

        # R_i = R_cur @ R(A_i.rot6)
        #   A_i.rot6 → 相对当前 EE 的旋转；左乘 R_cur 得到世界系姿态。
        #   注意: 这里始终是 r_base * Rd，不是「上一步的 R」再乘 Rd。
        r_i = r_base * st.Rotation.from_matrix(_rotation_6d_to_matrix(deltas[i, 3:9]))
        out[i, 3:6] = r_i.as_rotvec()

        out[i, 6] = abs_grip[i]
    return out


def _clip_action_chunk(
    action_chunk: np.ndarray,
    *,
    max_delta_pos: float,
    max_delta_rot: float,
) -> tuple[np.ndarray, int, int]:
    """安全限位（current-relative waypoints）。

    不能直接对每个 A_i 做 |pos| clip（后期 waypoint 离当前 EE 可以很远）。
    做法: A → 逐步相对 a_i = A_{i-1}^{-1}@A_i → clip a_i → 再复合回 A'。
    返回 (clipped, n_pos_clip, n_rot_clip)。
    """
    waypoints = np.asarray(action_chunk, dtype=np.float64).copy()
    if waypoints.ndim != 2 or waypoints.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action chunk (N, {ACTION_DIM}), got {waypoints.shape}")

    n = waypoints.shape[0]
    n_pos_clip = 0
    n_rot_clip = 0

    prev_pos = np.zeros(3, dtype=np.float64)
    prev_rot = np.eye(3, dtype=np.float64)
    cum_pos = np.zeros(3, dtype=np.float64)
    cum_rot = np.eye(3, dtype=np.float64)

    out = waypoints.copy()
    for i in range(n):
        cur_pos = waypoints[i, :3]
        cur_rot = _rotation_6d_to_matrix(waypoints[i, 3:9])
        rel_rot = prev_rot.T @ cur_rot
        rel_pos = prev_rot.T @ (cur_pos - prev_pos)

        before_pos = rel_pos.copy()
        rel_pos = np.clip(rel_pos, -max_delta_pos, max_delta_pos)
        n_pos_clip += int(np.sum(np.abs(before_pos) > max_delta_pos))

        before_rot6 = _matrix_to_rotation_6d(rel_rot)
        after_rot6 = _clip_rotation_6d(before_rot6, max_delta_rot)
        if not np.allclose(before_rot6, after_rot6, atol=1e-8):
            n_rot_clip += 1
        rel_rot = _rotation_6d_to_matrix(after_rot6)

        cum_pos = cum_pos + cum_rot @ rel_pos
        cum_rot = cum_rot @ rel_rot
        out[i, :3] = cum_pos
        out[i, 3:9] = _matrix_to_rotation_6d(cum_rot)
        out[i, 9] = waypoints[i, 9]

        prev_pos = cum_pos.copy()
        prev_rot = cum_rot.copy()

    return out, n_pos_clip, n_rot_clip


def _fmt_vec3(vec: np.ndarray, *, unit: str = "") -> str:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    return f"[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]{unit}"


def _unit_vec3(vec: np.ndarray) -> np.ndarray | None:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _dir_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    ua = _unit_vec3(a)
    ub = _unit_vec3(b)
    if ua is None or ub is None:
        return None
    return float(np.dot(ua, ub))


def _fmt_dir_line(
    label: str,
    vec: np.ndarray,
    *,
    frame: str = "",
    note: str = "",
) -> str:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    mag_mm = float(np.linalg.norm(v) * 1000.0)
    u = _unit_vec3(v)
    frame_tag = f" ({frame})" if frame else ""
    note_tag = f" {note}" if note else ""
    if u is None:
        return f"  {label}{frame_tag}: |Δ|={mag_mm:.2f}mm (零位移){note_tag}"
    return (
        f"  {label}{frame_tag}: û={_fmt_vec3(u)} |Δ|={mag_mm:.2f}mm{note_tag}"
    )


def _print_direction_triplet_compare(
    *,
    title: str,
    model_step_ee: np.ndarray,
    decode_step_ee: np.ndarray,
    model_step_world: np.ndarray,
    decode_step_world: np.ndarray,
    actual_exec_world: np.ndarray | None,
) -> dict[str, float | None]:
    """打印「模型首步 / 解码世界 / 实际执行」三方向对比。"""
    cos_model_decode = _dir_cosine(model_step_world, decode_step_world)
    cos_decode_actual = None
    cos_model_actual = None
    if actual_exec_world is not None:
        cos_decode_actual = _dir_cosine(decode_step_world, actual_exec_world)
        cos_model_actual = _dir_cosine(model_step_world, actual_exec_world)

    clipped = not np.allclose(model_step_ee, decode_step_ee, atol=1e-8)
    print(f"[DIAG][DIR] {title}")
    print(_fmt_dir_line("模型首步", model_step_ee, frame="EE"))
    print(_fmt_dir_line("模型首步", model_step_world, frame="世界"))
    decode_note = "[clip]" if clipped else ""
    print(_fmt_dir_line("解码后", decode_step_ee, frame="EE", note=decode_note))
    print(_fmt_dir_line("解码后", decode_step_world, frame="世界"))
    if actual_exec_world is not None:
        print(_fmt_dir_line("实际执行", actual_exec_world, frame="世界"))
    else:
        print("  实际执行 (世界): (待下周期观测)")

    cos_parts: list[str] = []
    if cos_model_decode is not None:
        cos_parts.append(f"cos(模型→解码)={cos_model_decode:+.3f}")
    if cos_decode_actual is not None:
        cos_parts.append(f"cos(解码→实际)={cos_decode_actual:+.3f}")
    if cos_model_actual is not None:
        cos_parts.append(f"cos(模型→实际)={cos_model_actual:+.3f}")
    if cos_parts:
        print(f"  {' | '.join(cos_parts)}")

    return {
        "cos_model_decode": cos_model_decode,
        "cos_decode_actual": cos_decode_actual,
        "cos_model_actual": cos_model_actual,
    }


def _rot_angle_deg(r_a: st.Rotation, r_b: st.Rotation) -> float:
    return float(np.degrees((r_a.inv() * r_b).magnitude()))


def _absolute_pose_to_delta_step(
    abs_pose: np.ndarray,
    *,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
) -> np.ndarray:
    """单步 decode 的逆：从绝对位姿反推 current-relative action（10 维，用于自检）。

    abs_pose[6] 为 ZMQ 物理开口；返回 delta[9] 仍为 ZMQ 值（未转回训练极性）。
    """
    abs_pose = np.asarray(abs_pose, dtype=np.float64).reshape(7)
    cur_pos = np.asarray(cur_pos, dtype=np.float64).reshape(3)
    r_cur = st.Rotation.from_rotvec(np.asarray(cur_rotvec, dtype=np.float64).reshape(3))
    r_next = st.Rotation.from_rotvec(abs_pose[3:6])
    # T_rel = T_cur^{-1} @ T_next
    dpos = r_cur.inv().apply(abs_pose[:3] - cur_pos)
    drot6 = _matrix_to_rotation_6d((r_cur.inv() * r_next).as_matrix())
    return np.concatenate([dpos, drot6, [abs_pose[6]]])


def _summarize_action_steps(raw_exec: np.ndarray) -> str:
    """一行摘要：每步位移/转角/夹爪（raw_exec[...,9] 为训练极性；pos 相对当前 EE）。"""
    raw_exec = np.asarray(raw_exec, dtype=np.float64)
    pos_mm = np.linalg.norm(raw_exec[:, :3], axis=1) * 1000.0
    rot_deg = np.array(
        [float(np.degrees(np.linalg.norm(_rotation_6d_to_rotvec(r)))) for r in raw_exec[:, 3:9]],
        dtype=np.float64,
    )
    grips = raw_exec[:, GRIPPER_ACTION_INDEX]
    return (
        f"pos_mm/from_cur={np.array2string(pos_mm, precision=2, suppress_small=True)} "
        f"rot_deg/from_cur={np.array2string(rot_deg, precision=2, suppress_small=True)} "
        f"grip_abs={np.array2string(grips, precision=4, suppress_small=True)}"
    )


def _diagnose_scheduling(
    action_timestamps: np.ndarray,
    *,
    n_exec: int,
) -> dict:
    """检查有多少 waypoint 因 timestamp 已过期而被 exec_actions 丢弃。"""
    receive_time = time.time()
    ts = np.asarray(action_timestamps[:n_exec], dtype=np.float64)
    is_new = ts > receive_time
    n_keep = int(np.sum(is_new))
    return {
        "receive_time": receive_time,
        "n_exec": n_exec,
        "n_keep": n_keep,
        "n_drop": n_exec - n_keep,
        "t0_minus_now_ms": float((ts[0] - receive_time) * 1000.0) if n_exec else 0.0,
        "dt_ms": float((ts[1] - ts[0]) * 1000.0) if n_exec > 1 else 0.0,
    }


def _collect_diag_hints(
    *,
    track_pos_err_mm: float | None,
    track_rot_err_deg: float | None,
    infer_s: float,
    exec_budget_s: float,
    n_pos_clip: int,
    n_rot_clip: int,
    n_drop: int,
    decode_roundtrip_pos_mm: float,
    decode_roundtrip_rot_deg: float,
    moved_pos_mm: float | None,
    raw_exec: np.ndarray,
    move_dir_cos: float | None = None,
    cos_decode_actual: float | None = None,
    state_rel: np.ndarray | None = None,
) -> list[str]:
    """根据阈值给出「最可能出问题在哪一段」的提示。"""
    hints: list[str] = []

    if state_rel is not None:
        state_rel = np.asarray(state_rel, dtype=np.float64).reshape(-1)
        if state_rel.shape[0] >= STATE_DIM:
            pos_curr = state_rel[3:6]
            rot_curr = state_rel[12:18]
            pos_hist = state_rel[0:3]
            rot_hist_ang = float(np.linalg.norm(_rotation_6d_to_rotvec(state_rel[6:12])))
            if np.linalg.norm(pos_curr) > 1e-3 or not np.allclose(
                rot_curr, IDENTITY_ROT6, atol=1e-2
            ):
                hints.append(
                    "[STATE] pos_curr/rot_curr 应近 0/I；检查 EE-relative 转换。"
                )
            if np.linalg.norm(pos_hist) > 0.05 or rot_hist_ang > 0.35:
                hints.append(
                    "[STATE] hist 相对当前偏大（|pos_hist|>5cm 或 |rot_hist|>0.35rad）；"
                    "确认 robot_obs_horizon=2 与控制频率是否匹配数据集。"
                )
        else:
            pos_rel = state_rel[:3]
            if np.linalg.norm(pos_rel) > 0.20:
                hints.append("[STATE] relative 位姿偏大；检查 state 构造。")

    if infer_s > exec_budget_s:
        hints.append(
            f"[TIMING] 推理 {infer_s:.3f}s > 本周期执行预算 {exec_budget_s:.3f}s "
            f"(exec_horizon*dt)；开环超前，跟踪会变差。"
        )
    if n_drop > 0:
        hints.append(
            f"[SCHED] {n_drop}/{len(raw_exec)} 个 waypoint 因 timestamp 已过期被丢弃；"
            "检查推理耗时或 command_start_time。"
        )
    if n_pos_clip > 0 or n_rot_clip > 0:
        hints.append(
            f"[MODEL/CLIP] 有 {n_pos_clip} 个位置分量、{n_rot_clip} 个旋转步被安全限位截断；"
            "模型输出可能过大或尺度不对。"
        )
    if decode_roundtrip_pos_mm > 0.05 or decode_roundtrip_rot_deg > 0.1:
        hints.append(
            f"[DECODE] roundtrip 误差 pos={decode_roundtrip_pos_mm:.3f}mm "
            f"rot={decode_roundtrip_rot_deg:.3f}deg；检查 current-relative→abs 转换。"
        )
    if track_pos_err_mm is not None and track_pos_err_mm > 10.0:
        hints.append(
            f"[EXEC] 上周期末命令 vs 本周期初状态 位置误差 {track_pos_err_mm:.1f}mm；"
            "机器人没跟上或时序不对。"
        )
    if track_rot_err_deg is not None and track_rot_err_deg > 8.0:
        hints.append(
            f"[EXEC] 上周期末命令 vs 本周期初状态 旋转误差 {track_rot_err_deg:.1f}deg；"
            "机器人没跟上或旋转约定不一致。"
        )

    if np.all(np.linalg.norm(raw_exec[:, :3], axis=1) < 0.0001):
        hints.append("[MODEL] 位置 delta 几乎全零；确认图像/state 是否正确送入服务。")

    if track_pos_err_mm is not None and moved_pos_mm is not None:
        pred_move = float(np.linalg.norm(raw_exec[0, :3]) * 1000.0)
        if pred_move > 0.3 and moved_pos_mm < 0.2:
            hints.append(
                "[EXEC/MODEL] 模型要求移动但机器人几乎没动；优先查执行层或夹爪/碰撞。"
            )

    dir_cos = cos_decode_actual if cos_decode_actual is not None else move_dir_cos
    if dir_cos is not None:
        if dir_cos < 0.0:
            hints.append(
                "[MODEL/DECODE] 解码世界方向 vs 实际执行相反 (cos<0)；"
                "检查坐标系、镜像或旋转约定。"
            )
        elif dir_cos < 0.3:
            hints.append(
                "[MODEL/DECODE] 解码世界方向 vs 实际执行偏差大 (cos<0.3)；"
                "检查相机镜像或 state/action frame。"
            )

    if not hints:
        hints.append(
            "[OK] 本周期未发现明显异常；若任务仍失败，更可能是模型策略/观测分布问题。"
        )
    return hints


def _print_policy_cycle_diag(
    *,
    cycle_idx: int,
    episode_id: int,
    iter_idx: int,
    obs_latency: float,
    exec_budget_s: float,
    wrist_img: np.ndarray,
    model_cur_pos: np.ndarray,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
    cur_grip: float,
    state_rel: np.ndarray,
    last_cmd_abs: np.ndarray | None,
    last_model_state_pos: np.ndarray | None,
    last_state_pos: np.ndarray | None,
    last_state_rotvec: np.ndarray | None,
    last_state_grip: float | None,
    prev_model_step_ee: np.ndarray | None,
    prev_decode_step_ee: np.ndarray | None,
    prev_model_step_world: np.ndarray | None,
    prev_decode_step_world: np.ndarray | None,
    raw_action_chunk: np.ndarray,
    action_chunk: np.ndarray,
    abs_model_exec: np.ndarray,
    abs_exec: np.ndarray,
    infer_s: float,
    n_exec: int,
    max_delta_pos: float,
    max_delta_rot: float,
    sched: dict,
    debug_fixed_delta: str | None,
    decode_base_pos: np.ndarray | None = None,
    decode_base_rotvec: np.ndarray | None = None,
    rebase_shift_mm: float | None = None,
    rebase_rot_deg: float | None = None,
    rebase_age_s: float | None = None,
    invert_gripper: bool = True,
    gripper_min_width: float = 0.011,
    gripper_max_width: float = DEFAULT_GRIPPER_MAX_WIDTH,
) -> None:
    """打印一个 control cycle 的结构化诊断。"""
    raw_exec = np.asarray(raw_action_chunk[:n_exec], dtype=np.float64)
    clipped_exec = np.asarray(action_chunk[:n_exec], dtype=np.float64)

    # clip 统计：对 current-relative 看相邻 waypoint 的逐步相对幅值
    n_pos_clip = 0
    n_rot_clip = 0
    prev_pos = np.zeros(3, dtype=np.float64)
    prev_rot = np.eye(3, dtype=np.float64)
    for i in range(raw_exec.shape[0]):
        cur_p = raw_exec[i, :3]
        cur_r = _rotation_6d_to_matrix(raw_exec[i, 3:9])
        rel_pos = prev_rot.T @ (cur_p - prev_pos)
        rel_rot = prev_rot.T @ cur_r
        n_pos_clip += int(np.sum(np.abs(rel_pos) > max_delta_pos))
        angle = float(np.linalg.norm(_rotation_6d_to_rotvec(_matrix_to_rotation_6d(rel_rot))))
        if angle > max_delta_rot + 1e-8:
            n_rot_clip += 1
        prev_pos = cur_p
        prev_rot = cur_r

    track_pos_err_mm = None
    track_rot_err_deg = None
    moved_pos_mm = None
    moved_rot_deg = None
    dgrip_mm = None
    move_dir_cos = None
    cos_decode_actual = None

    base_model_pos = (
        np.asarray(model_cur_pos, dtype=np.float64)
        if decode_base_pos is None
        else np.asarray(decode_base_pos, dtype=np.float64)
    )
    base_rotvec = (
        np.asarray(cur_rotvec, dtype=np.float64)
        if decode_base_rotvec is None
        else np.asarray(decode_base_rotvec, dtype=np.float64)
    )

    r_cur_diag = st.Rotation.from_rotvec(base_rotvec)
    cur_model_step_ee = raw_exec[0, :3]
    cur_decode_step_ee = clipped_exec[0, :3]
    cur_model_step_world = r_cur_diag.apply(cur_model_step_ee)
    cur_decode_step_world = r_cur_diag.apply(cur_decode_step_ee)
    actual_exec_world = None
    if last_model_state_pos is not None:
        actual_exec_world = model_cur_pos - last_model_state_pos

    print("\n" + "=" * 72)
    print(
        f"[DIAG] cycle={cycle_idx} episode={episode_id} iter={iter_idx} "
        f"budget={exec_budget_s:.3f}s obs_latency={obs_latency:.3f}s"
    )
    if debug_fixed_delta is not None:
        print(f"[DIAG][MODE] debug_fixed_delta={debug_fixed_delta} (跳过模型推理)")

    print(
        f"[DIAG][OBS] img={wrist_img.shape}/{wrist_img.dtype} mean={wrist_img.mean():.1f} "
        f"eef_pos={_fmt_vec3(cur_pos, unit='m')} "
        f"link6_pos={_fmt_vec3(model_cur_pos, unit='m')} "
        f"abs_rotvec={_fmt_vec3(cur_rotvec, unit='rad')} "
        f"zmq_grip={cur_grip:.4f}m"
    )
    state_arr = np.asarray(state_rel, dtype=np.float64).reshape(-1)
    print(
        f"[DIAG][OBS][state20→model] {np.array2string(state_arr, precision=4)}"
    )
    if state_arr.shape[0] >= STATE_DIM:
        print(
            f"[DIAG][OBS][state20] pos_hist={_fmt_vec3(state_arr[0:3], unit='m')} "
            f"pos_curr={_fmt_vec3(state_arr[3:6], unit='m')} "
            f"grip_hist={state_arr[18]:.4f} grip_curr={state_arr[19]:.4f} "
            f"(training polarity, larger=close)"
        )
    print(
        f"[DIAG][OBS][tcp_abs] "
        f"{np.array2string(np.concatenate([cur_pos, cur_rotvec, [cur_grip]]), precision=4)}"
    )

    if last_cmd_abs is not None and last_state_pos is not None:
        track_pos_err_mm = float(np.linalg.norm(cur_pos - last_cmd_abs[:3]) * 1000.0)
        r_cmd = st.Rotation.from_rotvec(last_cmd_abs[3:6])
        r_cur = st.Rotation.from_rotvec(cur_rotvec)
        track_rot_err_deg = _rot_angle_deg(r_cmd, r_cur)
        track_grip_err_mm = abs(cur_grip - float(last_cmd_abs[6])) * 1000.0

        moved_pos_mm = float(np.linalg.norm(cur_pos - last_state_pos) * 1000.0)
        moved_rot_deg = _rot_angle_deg(
            st.Rotation.from_rotvec(last_state_rotvec),
            st.Rotation.from_rotvec(cur_rotvec),
        )
        dgrip_mm = (cur_grip - last_state_grip) * 1000.0

        print(
            f"[DIAG][TRACK] 上周期命令 vs 本周期观测 | "
            f"pos_err={track_pos_err_mm:.1f}mm "
            f"rot_err={track_rot_err_deg:.2f}deg "
            f"grip_err={track_grip_err_mm:.1f}mm"
        )
        print(
            f"[DIAG][TRACK] 机器人实际运动 | "
            f"move_pos={moved_pos_mm:.1f}mm "
            f"move_rot={moved_rot_deg:.2f}deg "
            f"dgrip={dgrip_mm:+.1f}mm"
        )
        print(
            f"[DIAG][TRACK] last_cmd pos={_fmt_vec3(last_cmd_abs[:3], unit='m')} "
            f"grip={last_cmd_abs[6]:.4f}m"
        )
    else:
        print("[DIAG][TRACK] (首周期，无上周期命令可对比)")

    if (
        prev_model_step_ee is not None
        and prev_decode_step_ee is not None
        and prev_model_step_world is not None
        and prev_decode_step_world is not None
        and actual_exec_world is not None
    ):
        dir_stats = _print_direction_triplet_compare(
            title="上周期预测 vs 本周期实际执行",
            model_step_ee=prev_model_step_ee,
            decode_step_ee=prev_decode_step_ee,
            model_step_world=prev_model_step_world,
            decode_step_world=prev_decode_step_world,
            actual_exec_world=actual_exec_world,
        )
        cos_decode_actual = dir_stats["cos_decode_actual"]
        move_dir_cos = cos_decode_actual

    _print_direction_triplet_compare(
        title="本周期预测（下周期将与「实际执行」对比）",
        model_step_ee=cur_model_step_ee,
        decode_step_ee=cur_decode_step_ee,
        model_step_world=cur_model_step_world,
        decode_step_world=cur_decode_step_world,
        actual_exec_world=None,
    )

    timing_flag = "OVER_BUDGET" if infer_s > exec_budget_s else "ok"
    print(
        f"[DIAG][MODEL] infer={infer_s:.3f}s {timing_flag} "
        f"chunk={raw_action_chunk.shape} exec={n_exec}"
    )
    print(f"[DIAG][MODEL] {_summarize_action_steps(raw_exec)}")
    if n_pos_clip or n_rot_clip:
        print(
            f"[DIAG][MODEL][clip] consecutive_step pos_elems={n_pos_clip} rot_steps={n_rot_clip} "
            f"(limits pos±{max_delta_pos}m rot±{max_delta_rot}rad)"
        )
    print(f"[DIAG][MODEL][raw_exec]\n{raw_exec}")

    if rebase_shift_mm is not None:
        age_txt = "n/a" if rebase_age_s is None else f"{rebase_age_s * 1000:.0f}ms"
        rot_txt = "n/a" if rebase_rot_deg is None else f"{rebase_rot_deg:.2f}deg"
        print(
            f"[DIAG][REBASE] 解码基准=推理后最新位姿 | "
            f"vs 观测 pos={rebase_shift_mm:.1f}mm rot={rot_txt} state_age={age_txt}"
        )

    # decode 自检：每步都相对同一 decode base（current-relative）
    implied_delta = _absolute_pose_to_delta_step(
        abs_model_exec[0], cur_pos=base_model_pos, cur_rotvec=base_rotvec
    )
    decode_pos_err_mm = float(np.linalg.norm(implied_delta[:3] - clipped_exec[0, :3]) * 1000.0)
    decode_rot_err_deg = _rot_angle_deg(
        st.Rotation.from_matrix(_rotation_6d_to_matrix(clipped_exec[0, 3:9])),
        st.Rotation.from_matrix(_rotation_6d_to_matrix(implied_delta[3:9])),
    )
    print(
        f"[DIAG][DECODE] step0 abs pos={_fmt_vec3(abs_exec[0, :3], unit='m')} "
        f"rotvec={_fmt_vec3(abs_exec[0, 3:6], unit='rad')} "
        f"zmq_grip={abs_exec[0, 6]:.4f}m"
    )
    print(
        f"[DIAG][DECODE] step0 A_from_cur pos={_fmt_vec3(clipped_exec[0, :3], unit='m')} "
        f"drot6={np.array2string(clipped_exec[0, 3:9], precision=4)} "
        f"train_grip={clipped_exec[0, GRIPPER_ACTION_INDEX]:.4f}m"
    )
    implied_zmq_grip = float(implied_delta[GRIPPER_ACTION_INDEX])
    model_train_grip = float(clipped_exec[0, GRIPPER_ACTION_INDEX])
    if invert_gripper:
        implied_train_grip = float(
            _invert_gripper_width(
                np.asarray([implied_zmq_grip]),
                gripper_min_width=gripper_min_width,
                gripper_max_width=gripper_max_width,
            )[0]
        )
    else:
        implied_train_grip = implied_zmq_grip
    print(
        f"[DIAG][DECODE][roundtrip] pos_err={decode_pos_err_mm:.4f}mm "
        f"rot_err={decode_rot_err_deg:.4f}deg "
        f"grip_err={abs(implied_train_grip - model_train_grip)*1000:.3f}mm "
        f"(train {model_train_grip:.4f} vs zmq {implied_zmq_grip:.4f})"
    )
    if n_exec > 1:
        step1_implied = _absolute_pose_to_delta_step(
            abs_model_exec[1],
            cur_pos=base_model_pos,
            cur_rotvec=base_rotvec,
        )
        print(
            f"[DIAG][DECODE] step1 A_from_cur pos={_fmt_vec3(clipped_exec[1, :3], unit='m')} "
            f"（相对同一当前 EE，非相对 step0）"
        )
        print(
            f"[DIAG][DECODE] step1 roundtrip pos_err="
            f"{np.linalg.norm(step1_implied[:3]-clipped_exec[1,:3])*1000:.4f}mm"
        )
    print(f"[DIAG][DECODE][abs_link6]\n{abs_model_exec}")
    print(f"[DIAG][DECODE][abs_eef_exec]\n{abs_exec}")

    print(
        f"[DIAG][SCHED] submit={sched['n_exec']} keep={sched['n_keep']} drop={sched['n_drop']} "
        f"t0-now={sched['t0_minus_now_ms']:+.1f}ms dt={sched['dt_ms']:.1f}ms"
    )

    hints = _collect_diag_hints(
        track_pos_err_mm=track_pos_err_mm,
        track_rot_err_deg=track_rot_err_deg,
        infer_s=infer_s,
        exec_budget_s=exec_budget_s,
        n_pos_clip=n_pos_clip,
        n_rot_clip=n_rot_clip,
        n_drop=sched["n_drop"],
        decode_roundtrip_pos_mm=decode_pos_err_mm,
        decode_roundtrip_rot_deg=decode_rot_err_deg,
        moved_pos_mm=moved_pos_mm,
        raw_exec=raw_exec,
        move_dir_cos=move_dir_cos,
        cos_decode_actual=cos_decode_actual,
        state_rel=state_rel,
    )
    print("[DIAG][HINT]")
    for hint in hints:
        print(f"  → {hint}")


# ===========================================================================
# Fast-WAM 远程推理客户端
# ===========================================================================

class LightWAMClient:
    """通过 WebSocket 连接 Fast-WAM serve_real_robot_policy.py 推理服务。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5566,
        connect_timeout: float = 5.0,
        request_timeout: float = 5.0,
    ):
        self._url = f"ws://{host}:{port}"
        self._connect_timeout = float(connect_timeout)
        self._request_timeout = float(request_timeout)
        self._loop = asyncio.new_event_loop()
        self._ws = None
        self._metadata = None
        self._closed = False
        atexit.register(self.close)

    def infer(self, wrist_image: np.ndarray, state: np.ndarray, prompt: str) -> np.ndarray:
        """
        发送单相机 observation 到 Fast-WAM 服务，返回 action_chunk。

        参数:
            wrist_image: HWC uint8 ndarray, shape (H, W, 3)，腕部相机图片
            state:       float32 ndarray, shape (20,)，EE-relative UMI state
            prompt:      str，任务描述

        返回:
            action_chunk: float32 ndarray, shape (action_horizon, 10)
              current-relative SE(3): A_i = T_cur^{-1} @ T_{target,i}
              gripper 维为训练集极性（越大越夹紧）
        """
        wrist_image = _to_uint8_hwc(wrist_image)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != STATE_DIM:
            raise ValueError(f"Expected state shape ({STATE_DIM},), got {state.shape}")
        obs = {
            "prompt": prompt,
            "images": {"wrist_image": wrist_image},
            "state": state,
            "apply_gripper_postprocess": False,
            "apply_real_gripper_snap": False,
        }
        if self._closed:
            raise RuntimeError("LightWAMClient is closed")
        return self._loop.run_until_complete(self._infer_async(obs))

    async def _close_ws(self) -> None:
        if self._ws is None:
            return
        ws, self._ws = self._ws, None
        try:
            await ws.close()
        except Exception:
            pass

    async def _ensure_connected(self) -> None:
        if self._ws is not None:
            return
        self._ws = await asyncio.wait_for(
            websockets.connect(
                self._url,
                max_size=None,
                open_timeout=self._connect_timeout,
                close_timeout=1.0,
            ),
            timeout=self._connect_timeout + 1.0,
        )
        self._metadata = await asyncio.wait_for(
            self._ws.recv(),
            timeout=self._request_timeout,
        )

    async def _infer_async(self, obs: dict) -> np.ndarray:
        for attempt in range(2):
            try:
                await self._ensure_connected()
                await asyncio.wait_for(
                    self._ws.send(msgpack.packb(obs, default=_msgpack_default)),
                    timeout=self._request_timeout,
                )
                response = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self._request_timeout,
                )
                if isinstance(response, str):
                    raise RuntimeError(f"Policy server returned an error:\n{response}")
                result = msgpack.unpackb(response, object_hook=_msgpack_decode)
                if not isinstance(result, dict) or "actions" not in result:
                    raise RuntimeError(
                        f"Invalid policy response: expected dict with 'actions', got {type(result)}"
                    )
                return np.asarray(result["actions"], dtype=np.float32)
            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                await self._close_ws()
                if attempt == 0:
                    continue
                raise RuntimeError(
                    f"Policy request failed after reconnecting to {self._url}"
                ) from exc
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._loop.is_closed():
            self._loop.run_until_complete(self._close_ws())
            self._loop.close()


# ===========================================================================
# 辅助函数
# ===========================================================================

DEFAULT_TASK_PROMPT = "pick up the electronic component and place it into the designated box"


def easeInOutQuad(t: float) -> float:
    """二次缓入缓出插值，t ∈ [0, 1]。"""
    t *= 2.0
    if t < 1.0:
        return 0.5 * t * t
    t -= 1.0
    return -0.5 * (t * (t - 2.0) - 1.0)


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
            "c=启动策略, s=停止, q=退出, b=删除 episode"
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
        if ch in ("c", "C"):
            return KeyCode(char="c")
        if ch in ("q", "Q"):
            return KeyCode(char="q")
        if ch in ("s", "S"):
            return KeyCode(char="s")
        if ch == "1":
            return KeyCode(char="1")
        if ch in ("b", "B", "\x7f", "\b"):
            return Key.backspace
        return None


def _collect_key_events(key_counter, terminal_reader):
    events = []
    if key_counter is not None:
        events.extend(key_counter.get_press_events())
    if terminal_reader is not None:
        events.extend(terminal_reader.get_press_events())
    return events


def _clear_key_events(key_counter, terminal_reader):
    if key_counter is not None:
        key_counter.clear()
    if terminal_reader is not None:
        terminal_reader.clear()


# ===========================================================================
# CLI
# ===========================================================================

@click.command()
@click.option("--output", "-o", default="./fast_wam_output", help="输出目录")
@click.option("--policy_host", default="127.0.0.1", help="Fast-WAM 推理服务地址")
@click.option("--policy_port", default=5566, type=int, help="Fast-WAM 推理服务端口")
@click.option("--policy_connect_timeout", default=5.0, type=float, show_default=True,
              help="连接推理服务的超时（秒）")
@click.option("--policy_request_timeout", default=5.0, type=float, show_default=True,
              help="单次推理请求的超时（秒）")
@click.option("--task_prompt", default=DEFAULT_TASK_PROMPT, help="任务描述")
@click.option("--steps_per_inference", "-si", default=32, type=int,
              help="每次推理请求服务端返回的动作步数；实际只执行前 exec_horizon 步")
@click.option("--exec_horizon", "-eh", default=32, type=int, show_default=True,
              help="每轮实际下发并等待执行完的帧数（≤ steps_per_inference）；"
                   "例: -eh 4 @ 20Hz 表示每轮执行约 0.2s")
@click.option("--max_duration", "-md", default=2000000, help="每 episode 最大时长（秒）")
@click.option("--frequency", "-f", default=20, type=float,
              help="控制频率 (Hz)；默认与训练数据集 20 fps 对齐")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="SpaceMouse 命令延迟（秒）")
@click.option("--camera_reorder", "-cr", default="0")
@click.option("--no_mirror", "-nm", is_flag=True, default=False)
@click.option("--mirror_swap", is_flag=True, default=False)
@click.option("--init_joints", "-j", is_flag=True, default=False, help="是否初始化关节配置")
@click.option("--init_delta", type=str, default="0,0,0,0.1,-0.1,-1.0",
              show_default=True,
              help="位姿微调: dx,dy,dz,drx,dry,drz（米/弧度）")
@click.option("--init_gripper_delta", type=float, default=0.11,
              help="夹爪开度微调（米，ZMQ 物理开口；越大越张开）")
@click.option("--gripper_min_width", type=float, default=0.011,
              help="ZMQ 物理开口下界（米，≈闭合；也用于 invert 映射）")
@click.option("--gripper_max_width", type=float, default=DEFAULT_GRIPPER_MAX_WIDTH,
              help="ZMQ 物理开口上界（米，≈全开；也用于 invert 映射）")
@click.option("--max_delta_pos", type=float, default=0.05,
              help="相邻 waypoint 逐步相对位置安全限位（米）")
@click.option("--max_delta_rot", type=float, default=0.3,
              help="相邻 waypoint 逐步相对旋转安全限位（弧度）")
@click.option("--no_binarize_gripper", is_flag=True, default=False,
              help="不对 invert 后的 ZMQ 物理开口做开/关二值化")
@click.option(
    "--no_invert_gripper",
    is_flag=True,
    default=False,
    help="关闭 state/action 夹爪极性互转（默认: ZMQ↔训练，g'=min+max-g）",
)
@click.option(
    "--clip_state_grip/--no_clip_state_grip",
    default=False,
    show_default=True,
    help="上传前将 state[18:20] clip 到训练集 gripper 统计范围（训练极性）",
)
@click.option(
    "--train_state_grip_min",
    type=float,
    default=TRAIN_STATE_GRIP_MIN,
    show_default=True,
    help="训练集 state gripper 下界（训练极性，≈全开）",
)
@click.option(
    "--train_state_grip_max",
    type=float,
    default=TRAIN_STATE_GRIP_MAX,
    show_default=True,
    help="训练集 state gripper 上界（训练极性，≈全夹）",
)
@click.option("--terminal_keys", is_flag=True, default=False,
              help="从 SSH 终端 stdin 读 c/q/s/b（不依赖 pynput 全局键盘）")
@click.option(
    "--debug_fixed_delta",
    default=None,
    help="调试用：跳过模型，使用固定 10 维 current-relative action；grip 维为训练极性。"
         "注意: 会 tile 到整个 horizon，各步指向同一相对目标。",
)
@click.option(
    "--rebase_decode/--no_rebase_decode",
    default=True,
    show_default=True,
    help="推理结束后用最新实时 TCP 做相对→绝对解码基准。"
         "上一轮动作会等完再拍时通常可关（--no_rebase_decode）；"
         "仍开则作为限速未到位时的保险",
)
@click.option(
    "--eef_offset",
    type=float,
    default=X5_EEF_OFFSET_TCP_Z,
    show_default=True,
    help="训练数据是 link6、ZMQ 控制点是 eef_link；沿 TCP Z 的位置偏移。设为 0 可关闭。",
)
def main(
    output,
    policy_host,
    policy_port,
    policy_connect_timeout,
    policy_request_timeout,
    task_prompt,
    steps_per_inference,
    exec_horizon,
    max_duration,
    frequency,
    command_latency,
    camera_reorder,
    no_mirror,
    mirror_swap,
    init_joints,
    init_delta,
    init_gripper_delta,
    gripper_min_width,
    gripper_max_width,
    max_delta_pos,
    max_delta_rot,
    no_binarize_gripper,
    no_invert_gripper,
    clip_state_grip,
    train_state_grip_min,
    train_state_grip_max,
    terminal_keys,
    debug_fixed_delta,
    rebase_decode,
    eef_offset,
):
    # -------------------------------------------------------------------------
    # 1. 基础设置
    # -------------------------------------------------------------------------
    pid = os.getpid()
    os.sched_setaffinity(pid, [7])

    gripper_min_width = max(0.0, float(gripper_min_width))
    max_gripper_width = float(gripper_max_width)
    if max_gripper_width <= gripper_min_width:
        raise click.BadParameter("--gripper_max_width must be > --gripper_min_width")
    train_state_grip_min = float(train_state_grip_min)
    train_state_grip_max = float(train_state_grip_max)
    if train_state_grip_max <= train_state_grip_min:
        raise click.BadParameter("--train_state_grip_max must be > --train_state_grip_min")
    if steps_per_inference <= 0:
        raise click.BadParameter("--steps_per_inference must be > 0")
    if exec_horizon <= 0:
        raise click.BadParameter("--exec_horizon must be > 0")
    if exec_horizon > steps_per_inference:
        raise click.BadParameter(
            "--exec_horizon must be <= --steps_per_inference"
        )
    if frequency <= 0:
        raise click.BadParameter("--frequency must be > 0")
    if policy_connect_timeout <= 0 or policy_request_timeout <= 0:
        raise click.BadParameter("--policy timeouts must be > 0")
    eef_offset = float(eef_offset)
    invert_gripper = not no_invert_gripper

    fixed_delta_vec = None
    if debug_fixed_delta is not None:
        vals = [float(v) for v in debug_fixed_delta.split(",")]
        if len(vals) != ACTION_DIM:
            raise click.BadParameter(
                f"--debug_fixed_delta must have {ACTION_DIM} comma-separated values "
                "(pos×3,rot6×6,grip)"
            )
        fixed_delta_vec = np.asarray(vals, dtype=np.float32)
        print(f"[DIAG] debug_fixed_delta enabled: {fixed_delta_vec}")

    gripper_speed = 0.02
    cartesian_speed = 0.4
    orientation_speed = 0.8

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output, "action"), exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. 连接 Fast-WAM 推理服务
    # -------------------------------------------------------------------------
    client = LightWAMClient(
        host=policy_host,
        port=policy_port,
        connect_timeout=policy_connect_timeout,
        request_timeout=policy_request_timeout,
    )
    print(f"Fast-WAM policy service: ws://{policy_host}:{policy_port}")
    print(f"Task prompt: {task_prompt}")
    if abs(eef_offset) > 1e-12:
        print(
            "[ALIGN] link6↔eef_link enabled "
            f"(offset={eef_offset:.3f}m along TCP Z; rotation unchanged)"
        )
    else:
        print("[ALIGN] eef_offset=0: link6↔eef_link conversion disabled")
    print(
        "[ALIGN] gripper polarity: "
        + (
            "state: ZMQ→training; action: training→ZMQ "
            "(larger=close ↔ larger=open)"
            if invert_gripper
            else "no inversion (state/action use ZMQ opening as-is)"
        )
    )
    print(
        "[CHANGE] UMI pipeline: "
        f"state={STATE_DIM}d (EE-relative pose + training-polarity grip); "
        f"action={ACTION_DIM}d (current-relative SE(3) waypoints + training-polarity grip); "
        f"decode: T_i=T_cur@A_i (NOT chain); grip invert on upload/exec; "
        f"clip_state_grip={clip_state_grip} "
        f"train_grip=[{train_state_grip_min:.4f},{train_state_grip_max:.4f}]; "
        "images uint8"
    )
    print(
        "[ALIGN] decode base: "
        + (
            "推理后实时位姿 (rebase; 抑制 chunk 切换回跳)"
            if rebase_decode
            else "推理前观测位姿 (no rebase)"
        )
    )
    print(
        "[DIAG] 日志分段: [OBS]=输入  [TRACK]=执行跟踪  [DIR]=方向对比  "
        "[MODEL]=推理输出  [REBASE]=解码基准  [DECODE]=current-rel→abs  "
        "[SCHED]=下发时序  [HINT]=问题定位"
    )

    # -------------------------------------------------------------------------
    # 3. 机器人配置（单 ARX5）
    # -------------------------------------------------------------------------
    obs_res = (224, 224)  # UMI wrist_image 分辨率

    robots_config = [
        {
            "robot_type": "arx5",
            "robot_ip": "127.0.0.1",
            "robot_port": 8765,
            "robot_obs_latency": 0.20,
            "robot_action_latency": 0.01,
            "tcp_offset": 0,
            "height_threshold": 0.08,
            "sphere_radius": 0.1,
            "sphere_center": [0, -0.06, -0.185],
        }
    ]

    # -------------------------------------------------------------------------
    # 4. 初始化硬件
    # -------------------------------------------------------------------------
    dt = 1.0 / frequency

    key_counter_ctx = (
        contextlib.nullcontext(None)
        if terminal_keys
        else KeystrokeCounter()
    )
    terminal_reader_ctx = (
        TerminalKeyReader()
        if terminal_keys
        else contextlib.nullcontext(None)
    )

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=0.1) as sm, \
             key_counter_ctx as key_counter, \
             terminal_reader_ctx as terminal_reader, \
             Arx5Env(
                 output_dir=output,
                 robots_config=robots_config,
                 frequency=frequency,
                 obs_image_resolution=obs_res,
                 obs_float32=True,
                 camera_reorder=[int(x) for x in camera_reorder],
                 init_joints=init_joints,
                 enable_multi_cam_vis=False,
                 camera_obs_latency=0.17,
                 camera_obs_horizon=2,
                 robot_obs_horizon=2,
                 no_mirror=no_mirror,
                 mirror_swap=mirror_swap,
                 max_pos_speed=4.0,
                 max_rot_speed=10.0,
                 shm_manager=shm_manager,
             ) as env:

            cv2.setNumThreads(2)
            print("Waiting for camera...")
            time.sleep(1.0)

            while not env.is_ready:
                time.sleep(0.1)
            print("Env is ready")

            # -----------------------------------------------------------------
            # 5. 可选位姿微调
            # -----------------------------------------------------------------
            if init_delta is not None or abs(init_gripper_delta) > 0.0:
                delta6 = np.zeros(6, dtype=np.float64)
                if init_delta is not None:
                    vals = [float(v) for v in init_delta.split(",")]
                    assert len(vals) == 6
                    delta6[:] = vals

                robot_states = env.get_robot_state()
                start_pose = robot_states[0]["ActualTCPPose"].astype(np.float64).copy()
                start_grip = float(robot_states[0]["gripper_position"])

                tgt_pose = start_pose + delta6
                tgt_grip = np.clip(start_grip + init_gripper_delta,
                                   gripper_min_width, max_gripper_width)

                warmup_sec = 2.5
                steps = max(2, int(warmup_sec * frequency))
                actions, timestamps = [], []
                t0 = time.time() + 0.3

                for k in range(steps):
                    a = easeInOutQuad(k / (steps - 1))
                    cmd = np.zeros(7, dtype=np.float64)
                    cmd[:6] = start_pose + a * (tgt_pose - start_pose)
                    cmd[6] = np.clip(start_grip + a * (tgt_grip - start_grip),
                                     gripper_min_width, max_gripper_width)
                    actions.append(cmd)
                    timestamps.append(t0 + k * dt)

                env.exec_actions(
                    actions=np.asarray(actions),
                    timestamps=np.asarray(timestamps),
                    compensate_latency=False,
                )
                time.sleep(warmup_sec + 0.2)
                print("微调完成")

            # -----------------------------------------------------------------
            # 6. Warmup: 试跑一次推理
            # -----------------------------------------------------------------
            print("Warming up policy inference...")
            obs = env.get_obs()
            wrist_img = _to_uint8_hwc(obs["camera0_rgb"][-1])
            state_raw = _build_umi_state(
                obs,
                eef_offset=eef_offset,
                invert_gripper=invert_gripper,
                gripper_min_width=gripper_min_width,
                gripper_max_width=max_gripper_width,
            )
            state, state_grip_clipped = _clip_umi_state_for_model(
                state_raw,
                grip_min=train_state_grip_min,
                grip_max=train_state_grip_max,
                clip_grip=clip_state_grip,
            )
            zmq_grip = float(np.asarray(obs["robot0_gripper_width"][-1]).squeeze())
            print(
                f"warmup state20: pos_hist={state[:3]}, pos_curr={state[3:6]} (~0), "
                f"rot_curr={state[12:18]} (~I), "
                f"grip_hist={state[18]:.4f}, grip_curr={state[19]:.4f} "
                f"(zmq_grip={zmq_grip:.4f})"
            )
            if state_grip_clipped:
                print(
                    f"  [STATE] clipped grip for model upload: "
                    f"{state_raw[18:20]} -> {state[18:20]}"
                )

            if fixed_delta_vec is None:
                raw_action = client.infer(wrist_img, state, task_prompt)
                print(f"  Warmup inference done. action_chunk shape: {raw_action.shape}")
            else:
                print("  Fixed-delta debug mode: policy connection and warmup inference skipped.")
            print("Ready!")

            # =================================================================
            # 7. 主循环: Human ↔ Policy 切换
            # =================================================================
            while True:
                # =============================================================
                # 7.1 人工控制
                # =============================================================
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs["ActualTCPPose"] for rs in robot_states])
                gripper_target_pos = np.asarray([rs["gripper_position"] for rs in robot_states])
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0

                episode_id = env.replay_buffer.n_episodes
                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    vis_img = obs["camera0_rgb"][-1]
                    obs = env.get_obs()

                    text = f"Episode: {episode_id}"
                    vis_show = np.asarray(vis_img)
                    if vis_show.dtype != np.uint8:
                        vis_show = _to_uint8_hwc(vis_show)
                    else:
                        vis_show = vis_show.copy()
                    cv2.putText(
                        vis_show,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0, 0, 0),
                    )
                    cv2.putText(
                        vis_show,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )
                    cv2.imshow("default", vis_show[..., ::-1])
                    _ = cv2.pollKey()

                    press_events = _collect_key_events(key_counter, terminal_reader)
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char="c"):
                            start_policy = True
                        elif key_stroke == Key.backspace:
                            if terminal_keys or click.confirm("Drop episode?"):
                                env.drop_episode()
                                _clear_key_events(key_counter, terminal_reader)
                        elif key_stroke == KeyCode(char="1"):
                            control_robot_idx_list = [0]

                    if start_policy:
                        break

                    precise_wait(t_sample)

                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (0.5 / frequency) * cartesian_speed
                    drot_xyz = sm_state[3:] * (1.5 / frequency) * orientation_speed
                    drot = st.Rotation.from_euler("xyz", drot_xyz)

                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (
                            drot * st.Rotation.from_rotvec(target_pose[robot_idx, 3:])
                        ).as_rotvec()

                    dpos = 0.0
                    if sm.is_button_pressed(0) and sm.is_button_pressed(1):
                        print("Reset to home...")
                        for robot_idx in control_robot_idx_list:
                            env.robots[robot_idx].reset_to_home()
                            st2 = env.get_robot_state()
                            target_pose[robot_idx] = np.stack(
                                [rs["ActualTCPPose"] for rs in st2]
                            )[robot_idx]
                            gripper_target_pos[robot_idx] = np.asarray(
                                [rs["gripper_position"] for rs in st2]
                            )[robot_idx]
                    elif sm.is_button_pressed(0):
                        dpos = -gripper_speed / frequency
                    elif sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency

                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(
                            gripper_target_pos[robot_idx] + dpos,
                            gripper_min_width, max_gripper_width,
                        )

                    action = np.zeros(7)
                    action[:6] = target_pose[0]
                    action[6] = np.clip(gripper_target_pos[0],
                                        gripper_min_width, max_gripper_width)

                    env.exec_actions(
                        actions=[action],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        compensate_latency=False,
                    )
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # =============================================================
                # 7.2 Policy 控制
                # =============================================================
                episode_active = False
                try:
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    episode_active = True

                    obs = env.get_obs()

                    frame_latency = 1.0 / 60.0
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Policy in control!")

                    iter_idx = 0
                    cycle_idx = 0
                    last_cmd_abs = None
                    last_model_state_pos = None
                    last_state_pos = None
                    last_state_rotvec = None
                    last_state_grip = None
                    exec_budget_s = exec_horizon * dt
                    rebase_warned = False
                    prev_model_step_ee = None
                    prev_decode_step_ee = None
                    prev_model_step_world = None
                    prev_decode_step_world = None

                    while True:
                        # -----------------------------------------------------
                        # 1. 拍照 + 读机器人状态
                        # -----------------------------------------------------
                        obs = env.get_obs()
                        obs_timestamps = obs["timestamp"]
                        obs_latency = time.time() - obs_timestamps[-1]

                        s = time.time()
                        wrist_img = _to_uint8_hwc(obs["camera0_rgb"][-1])
                        state_raw = _build_umi_state(
                            obs,
                            eef_offset=eef_offset,
                            invert_gripper=invert_gripper,
                            gripper_min_width=gripper_min_width,
                            gripper_max_width=max_gripper_width,
                        )
                        state, state_grip_clipped = _clip_umi_state_for_model(
                            state_raw,
                            grip_min=train_state_grip_min,
                            grip_max=train_state_grip_max,
                            clip_grip=clip_state_grip,
                        )
                        if state_grip_clipped:
                            print(
                                f"[STATE] clipped grip for model upload: "
                                f"{state_raw[18:20]} -> {state[18:20]}"
                            )
                        cur_pos_robot, cur_rotvec, cur_grip = _read_tcp_pose(obs)
                        cur_pos_link6 = _x5_eef_to_link6(
                            cur_pos_robot,
                            cur_rotvec,
                            eef_offset,
                        )

                        episode_id = env.replay_buffer.n_episodes

                        # -----------------------------------------------------
                        # 2. 阻塞推理
                        # -----------------------------------------------------
                        if fixed_delta_vec is not None:
                            raw_action_chunk = np.tile(
                                fixed_delta_vec, (steps_per_inference, 1)
                            )
                        else:
                            raw_action_chunk = client.infer(wrist_img, state, task_prompt)
                        infer_s = time.time() - s

                        if raw_action_chunk.ndim != 2 or raw_action_chunk.shape[1] != ACTION_DIM:
                            raise ValueError(
                                f"Fast-WAM action chunk must have shape (N, {ACTION_DIM}), "
                                f"got {raw_action_chunk.shape}"
                            )
                        if len(raw_action_chunk) < steps_per_inference:
                            raise ValueError(
                                "Fast-WAM action chunk is shorter than --steps_per_inference: "
                                f"{len(raw_action_chunk)} < {steps_per_inference}"
                            )
                        raw_action_chunk = raw_action_chunk[:steps_per_inference]

                        action_chunk, _, _ = _clip_action_chunk(
                            raw_action_chunk,
                            max_delta_pos=max_delta_pos,
                            max_delta_rot=max_delta_rot,
                        )

                        # -----------------------------------------------------
                        # 3. 解码基准位姿（current-relative → 绝对 TCP）
                        # -----------------------------------------------------
                        base_pos_robot = cur_pos_robot
                        base_rotvec = cur_rotvec
                        base_pos_link6 = cur_pos_link6
                        rebase_shift_mm = None
                        rebase_rot_deg = None
                        rebase_age_s = None
                        if rebase_decode:
                            try:
                                (
                                    live_pos,
                                    live_rotvec,
                                    _live_grip,
                                    live_ts,
                                ) = _read_live_tcp_pose(env)
                            except Exception as rebase_err:  # noqa: BLE001
                                if not rebase_warned:
                                    print(
                                        f"[REBASE] 读取实时位姿失败，退回观测位姿: {rebase_err}"
                                    )
                                    rebase_warned = True
                            else:
                                rebase_shift_mm = float(
                                    np.linalg.norm(live_pos - cur_pos_robot) * 1000.0
                                )
                                rebase_rot_deg = _rot_angle_deg(
                                    st.Rotation.from_rotvec(cur_rotvec),
                                    st.Rotation.from_rotvec(live_rotvec),
                                )
                                rebase_age_s = time.time() - live_ts
                                base_pos_robot = live_pos
                                base_rotvec = live_rotvec
                                base_pos_link6 = _x5_eef_to_link6(
                                    live_pos,
                                    live_rotvec,
                                    eef_offset,
                                )

                        abs_link6 = _delta_chunk_to_absolute_pose(
                            action_chunk[:exec_horizon],
                            cur_pos=base_pos_link6,
                            cur_rotvec=base_rotvec,
                            gripper_min_width=gripper_min_width,
                            gripper_max_width=max_gripper_width,
                            binarize_gripper=not no_binarize_gripper,
                            invert_gripper=invert_gripper,
                        )
                        abs_exec = _abs_exec_link6_to_x5_eef(abs_link6, eef_offset)

                        # -----------------------------------------------------
                        # 4. 排期时间戳
                        # -----------------------------------------------------
                        command_start_time = time.time() + 0.05
                        action_timestamps = (
                            np.arange(exec_horizon, dtype=np.float64) * dt
                            + command_start_time
                        )
                        sched = _diagnose_scheduling(action_timestamps, n_exec=exec_horizon)

                        _print_policy_cycle_diag(
                            cycle_idx=cycle_idx,
                            episode_id=episode_id,
                            iter_idx=iter_idx,
                            obs_latency=obs_latency,
                            exec_budget_s=exec_budget_s,
                            wrist_img=wrist_img,
                            model_cur_pos=cur_pos_link6,
                            cur_pos=cur_pos_robot,
                            cur_rotvec=cur_rotvec,
                            cur_grip=cur_grip,
                            state_rel=state,
                            last_cmd_abs=last_cmd_abs,
                            last_model_state_pos=last_model_state_pos,
                            last_state_pos=last_state_pos,
                            last_state_rotvec=last_state_rotvec,
                            last_state_grip=last_state_grip,
                            prev_model_step_ee=prev_model_step_ee,
                            prev_decode_step_ee=prev_decode_step_ee,
                            prev_model_step_world=prev_model_step_world,
                            prev_decode_step_world=prev_decode_step_world,
                            raw_action_chunk=raw_action_chunk,
                            action_chunk=action_chunk,
                            abs_model_exec=abs_link6,
                            abs_exec=abs_exec,
                            infer_s=infer_s,
                            n_exec=exec_horizon,
                            max_delta_pos=max_delta_pos,
                            max_delta_rot=max_delta_rot,
                            sched=sched,
                            debug_fixed_delta=debug_fixed_delta,
                            decode_base_pos=base_pos_link6,
                            decode_base_rotvec=base_rotvec,
                            rebase_shift_mm=rebase_shift_mm,
                            rebase_rot_deg=rebase_rot_deg,
                            rebase_age_s=rebase_age_s,
                            invert_gripper=invert_gripper,
                            gripper_min_width=gripper_min_width,
                            gripper_max_width=max_gripper_width,
                        )

                        env.exec_actions(
                            actions=abs_exec,
                            timestamps=action_timestamps,
                            dynamic_latency=False,
                        )

                        if sched["n_keep"] > 0:
                            last_cmd_abs = abs_exec[-1].copy()
                        else:
                            last_cmd_abs = np.concatenate(
                                [base_pos_robot, base_rotvec, [cur_grip]]
                            ).astype(np.float64)
                        last_model_state_pos = cur_pos_link6.copy()
                        last_state_pos = cur_pos_robot.copy()
                        last_state_rotvec = cur_rotvec.copy()
                        last_state_grip = cur_grip

                        r_store = st.Rotation.from_rotvec(base_rotvec)
                        prev_model_step_ee = raw_action_chunk[0, :3].astype(np.float64).copy()
                        prev_decode_step_ee = action_chunk[0, :3].astype(np.float64).copy()
                        prev_model_step_world = r_store.apply(prev_model_step_ee)
                        prev_decode_step_world = r_store.apply(prev_decode_step_ee)

                        vis_img = _to_uint8_hwc(obs["camera0_rgb"][-1]).copy()
                        text = f"Episode: {episode_id}, Time: {time.monotonic() - t_start:.1f}s"
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255),
                        )
                        cv2.imshow("default", vis_img[..., ::-1])
                        _ = cv2.pollKey()

                        press_events = _collect_key_events(key_counter, terminal_reader)
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char="s"):
                                print("Stopped by user.")
                                stop_episode = True

                        if time.time() - eval_t_start > max_duration:
                            print("Max duration reached.")
                            stop_episode = True
                        if stop_episode:
                            break

                        precise_wait(
                            float(action_timestamps[-1]) + frame_latency,
                            time_func=time.time,
                        )
                        iter_idx += exec_horizon
                        cycle_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                finally:
                    if episode_active:
                        env.end_episode()

                print("Stopped. Returning to human control.")


if __name__ == "__main__":
    main()
