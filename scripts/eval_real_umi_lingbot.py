"""
真实机器人 UMI 策略评估脚本（LingBot-VA 协议 / WAM state-action 语义）。

通过 WebSocket 连接 LingBot-VA 推理服务（wan_va.wan_va_server, CONFIG_NAME=umi），
不需要在本脚本内加载模型权重。

注：
由于真机数据采集的是link6的位姿，所以状态和动作都需要转换，转换成eef_link的位姿。
0.145偏移对应的是X5.urdf中的eef_link和link6的偏移。
ZMQ 读到的 eef_link
  → 减 R @ [0,0,0.145] 回到 link6
  → 在 link6 上做 p += R @ dpos, R = R @ R_delta
  → 再加 R_next @ [0,0,0.145] 变成 eef_link
  → exec_actions

URDF 里那 0.145 m 是在 link6 自己的 X 上（法兰前伸），rpy=0 所以姿态不变。
ee2tcp 已经把轴转成 UMI/TCP 约定：
TCP Z  =  link6 X   ← 前伸
TCP X  = -link6 Y
TCP Y  = -link6 Z


p_eef = p_link6 + R_tcp @ [0, 0, 0.145]
不是又把点往 TCP 的 X 挪 0.145。若在 TCP 位姿上再加 [0.145, 0, 0]，方向就错了。

与 Light-WAM 的 umi_pick_elec_lerobot_real state/action 对齐:
  - state (20,): hist/current EE-relative pose + 两帧绝对夹爪
  - action (10,): EE-frame dpos + 前两行相对 rot6d + gripper(absolute)
  - 执行: p_next = p_cur + R_cur @ dpos; R_next = R_cur @ R_delta
  - 夹爪: 训练数据越大越夹紧；ZMQ 越大越张开。策略输出先取反；
    判定夹取时下发更紧的 --gripper_grasp_width，其他时候用原开度。
    （--no_invert_gripper 可关掉取反；--no_binarize_gripper 可关掉夹取加紧）。
    SpaceMouse / init_gripper_delta 不取反。
  - 采集/训练是 link6 TCP；本脚本走 ZMQ X5，下发前沿 TCP Z 加 0.145 m
    变成 eef_link（旋转不变）。--eef_offset 0 可关掉。
  - 协议 (async, 默认, 论文 Algorithm 2):
      reset → cold infer(a0)
      loop: 执行 a_t ‖ (fdm_ground(a_t) → infer a_{t+1}) → commit_real(真实关键帧, a_t)
  - 协议 (sync, --sync_infer):
      reset → infer(obs 图像) → 执行 → compute_kv_cache(state=上一 chunk 原始 action)
  - infer obs 只包含图像；20 维 WAM state 仅用于本地诊断
  - FDM/KV/commit 顶层 state 是本轮实际提交部分对应的原始模型 action 张量

用法:
    # 1. GPU 机启动 server（务必加载微调权重，例如 step400 评测包）:
    #    在 wan_va/configs/va_umi_cfg.py 将
    #    wan22_pretrained_model_name_or_path 指到 checkpoints/lingbot-va-umi-step400
    conda activate Lingbot
    cd /path/to/lingbot-va
    CUDA_VISIBLE_DEVICES=0 NGPU=1 CONFIG_NAME=umi bash script/run_launch_va_server_sync.sh
   
    # 默认 host=0.0.0.0 port=29536

    # 可选远程推理：
python scripts/eval_real_umi_lingbot.py --policy_host 58.251.254.222 --policy_port 5566 --terminal_keys --sync_infer

    # 2. 本机（机械臂 / 相机 / SpaceMouse）:
    conda activate umi-arx-kian
    python scripts/eval_real_umi_lingbot.py --policy_port 5566
    # 部分执行示例: --exec_horizon 8（须为模型 H 的整数倍）
    # 同步回退: 加 --sync_infer

键盘:
  Human: C=交给 policy, Q=退出, B=删除 episode
  Policy: S=停评估交还人

SSH 远程控制（无 X11）:

export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
conda activate umi-arx-kian
python scripts/eval_real_umi_lingbot.py --terminal_keys --policy_port 5566

python scripts/eval_real_umi_lingbot.py --terminal_keys --policy_port 5566 --sync_infer

python scripts/eval_real_umi_lingbot.py --policy_host 58.251.254.222 --policy_port 5566

测试：
    2026-7-30 目前在真机上可以正常运行，有靠近目标，但是抓取失败。使用的checkpoint是500步的，训练数据集大小是89 Episodes.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import select
import sys
import termios
import tty
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import websockets.sync.client
from websockets.exceptions import ConnectionClosed
from openpi_client import msgpack_numpy

# ---------------------------------------------------------------------------
# 路径：UMI 真机工程（lingbot-va 客户端可内嵌，无需本机装完整 wan_va）
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()


def _ensure_umi_project(umi_project_root: str) -> None:
    root = Path(umi_project_root).expanduser().resolve()
    if not root.is_dir():
        raise click.ClickException(f"umi_project_root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _ensure_lingbot_project(lingbot_project_root: str | None) -> None:
    """可选：把 lingbot-va 根目录加入 PYTHONPATH（使用官方 wan_va client 时）。"""
    if not lingbot_project_root:
        return
    root = Path(lingbot_project_root).expanduser().resolve()
    if not root.is_dir():
        raise click.ClickException(f"lingbot_project_root not found: {root}")
    if not (root / "wan_va").is_dir():
        raise click.ClickException(
            f"lingbot_project_root has no wan_va/ package: {root}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _create_websocket_client(
    host: str,
    port: int,
    *,
    connect_timeout: float,
    request_timeout: float,
):
    try:
        from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
            WebsocketClientPolicy,
        )

        print(
            "[lingbot] Using wan_va WebsocketClientPolicy "
            "(timeout/reconnect settings are managed by that client)"
        )
        return WebsocketClientPolicy(host=host, port=port)
    except ModuleNotFoundError:
        print(
            "[lingbot] wan_va not found; using bundled websocket client "
            "(openpi msgpack)"
        )
        return _BundledWebsocketClient(
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )


def _assert_lingbot_server(metadata: dict, *, host: str, port: int) -> None:
    if not isinstance(metadata, dict):
        return
    if "checkpoint_path" in metadata or "action_horizon" in metadata:
        raise click.ClickException(
            f"ws://{host}:{port} 上是 Light-WAM 服务，不是 LingBot-VA。\n"
            f"  - Light-WAM: 用对应 light-wam 评测脚本 / 端口\n"
            "  - LingBot-VA: 先启动 wan_va server（默认 29536），再 "
            "--policy_port 29536"
        )


class _BundledWebsocketClient:
    """LingBot-VA websocket client（无需安装 lingbot-va 仓库）。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29536,
        api_key: str | None = None,
        connect_timeout: float = 5.0,
        request_timeout: float = 100.0,
    ):
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout = float(connect_timeout)
        self._request_timeout = float(request_timeout)
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        deadline = time.monotonic() + self._connect_timeout
        last_error: Exception | None = None
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10,
                    open_timeout=max(0.1, deadline - time.monotonic()),
                )
                metadata = msgpack_numpy.unpackb(
                    conn.recv(timeout=self._request_timeout)
                )
                return conn, metadata
            except (ConnectionRefusedError, OSError, TimeoutError) as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out connecting to LingBot-VA at {self._uri} "
                        f"after {self._connect_timeout:.1f}s"
                    ) from last_error
                print(f"Waiting for LingBot-VA server at {self._uri} ... ({exc})")
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    def _close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def infer(self, obs: dict) -> dict:
        for attempt in range(2):
            try:
                self._ws.send(self._packer.pack(obs))
                response = self._ws.recv(timeout=self._request_timeout)
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")
                return msgpack_numpy.unpackb(response)
            except (ConnectionClosed, OSError, TimeoutError) as exc:
                self._close()
                if attempt == 0:
                    self._ws, self._server_metadata = self._wait_for_server()
                    continue
                raise RuntimeError(
                    f"LingBot-VA request failed after reconnecting to {self._uri}"
                ) from exc
        raise AssertionError("unreachable")


# ===========================================================================
# WAM state/action helpers（与 umi_pick_elec_lerobot_real 一致）
# Action: [dx,dy,dz, drot6_0..5, gripper(absolute)]  → 10 维
# rot6d = 旋转矩阵前两行展平；位置 delta 位于当前 EE 坐标系
# ===========================================================================

CAM_KEY = "observation.images.wrist_image"
# Wan streaming VAE 每 4 张输入图像编码为一个 temporal latent。
OBS_FRAMES_PER_LATENT = 4
ACTION_DIM = 10
STATE_DIM = 20
GRIPPER_ACTION_INDEX = 9
IDENTITY_ROT6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _invert_gripper_width(
    grip: np.ndarray,
    *,
    gripper_min_width: float,
    gripper_max_width: float,
) -> np.ndarray:
    """数据集约定（越大越夹紧）↔ ZMQ/Arx5Env 约定（越大越张开）。

    线性对翻：g' = min + max - g，把 [min, max] 映射到自身。
    例如 min=0.011, max=0.085 时，策略 0.07（夹紧）→ 0.026，
    再按 <0.05 二值化就会下发 min（闭合）。
    """
    return gripper_min_width + gripper_max_width - np.asarray(grip, dtype=np.float64)


# X5.urdf: eef_link = link6 + 0.145 m along tool x = TCP Z after ee2tcp.
X5_EEF_OFFSET_TCP_Z = 0.1  #貌似这个工作更加正常
# X5_EEF_OFFSET_TCP_Z = 0.145

def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 6D（前两行展平）。"""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return matrix[:2, :].reshape(6).copy()


def _rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D → 旋转矩阵（Gram-Schmidt；前两行约定）。"""
    values = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = values[:3], values[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def _rotation_6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    return st.Rotation.from_matrix(_rotation_6d_to_matrix(rot6d)).as_rotvec()


def _rotvec_to_rotation_6d(rotvec: np.ndarray) -> np.ndarray:
    return _matrix_to_rotation_6d(
        st.Rotation.from_rotvec(
            np.asarray(rotvec, dtype=np.float64).reshape(3)
        ).as_matrix()
    )


def _clip_rotation_6d(rot6d: np.ndarray, max_angle_rad: float) -> np.ndarray:
    """按相对转角幅值裁剪 6D 旋转。"""
    rotvec = _rotation_6d_to_rotvec(rot6d)
    angle = float(np.linalg.norm(rotvec))
    if angle <= max_angle_rad or angle < 1e-12:
        return np.asarray(rot6d, dtype=np.float64).reshape(6)
    return _rotvec_to_rotation_6d(rotvec * (max_angle_rad / angle))


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """服务端期望 HWC uint8；Arx5Env(obs_float32=True) 常返回 [0,1] float。"""
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


def _shift_tcp_along_z(
    pos: np.ndarray, rotvec: np.ndarray, dz: float
) -> np.ndarray:
    """p' = p + R_tcp @ [0, 0, dz]。姿态不变。"""
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
    pos: np.ndarray, rotvec: np.ndarray, dz: float
) -> np.ndarray:
    """ZMQ X5 的 eef_link 位置 → 采集用的 link6 位置。"""
    return _shift_tcp_along_z(pos, rotvec, -dz)


def _link6_to_x5_eef(
    pos: np.ndarray, rotvec: np.ndarray, dz: float
) -> np.ndarray:
    """采集用的 link6 位置 → ZMQ X5 的 eef_link 位置。"""
    return _shift_tcp_along_z(pos, rotvec, dz)


def _abs_exec_link6_to_x5_eef(abs_exec: np.ndarray, dz: float) -> np.ndarray:
    """(N, 7) 绝对 TCP：xyz 从 link6 挪到 eef_link，rotvec/gripper 不变。"""
    out = np.asarray(abs_exec, dtype=np.float64).copy()
    if abs(float(dz)) < 1e-12:
        return out
    for i in range(out.shape[0]):
        out[i, :3] = _link6_to_x5_eef(out[i, :3], out[i, 3:6], dz)
    return out


def _read_tcp_pose(obs: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """读机器人当前绝对 TCP：pos(3), rotvec(3), gripper_width。"""
    pos = np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
    rotvec = np.asarray(obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float64).reshape(
        3
    )
    grip = float(np.asarray(obs["robot0_gripper_width"][-1]).squeeze())
    return pos, rotvec, grip


def _read_tcp_pose_at(
    obs: dict, index: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """读 obs horizon 中指定 index 的绝对 TCP。"""
    pos = np.asarray(obs["robot0_eef_pos"][index], dtype=np.float64).reshape(3)
    rotvec = np.asarray(
        obs["robot0_eef_rot_axis_angle"][index], dtype=np.float64
    ).reshape(3)
    grip = float(np.asarray(obs["robot0_gripper_width"][index]).squeeze())
    return pos, rotvec, grip


def _pose_relative_to(
    pos: np.ndarray,
    rotvec: np.ndarray,
    *,
    ref_pos: np.ndarray,
    ref_rotvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 T_rel = T_ref^-1 @ T，返回位置和前两行 rot6d。"""
    r_ref = st.Rotation.from_rotvec(
        np.asarray(ref_rotvec, dtype=np.float64).reshape(3)
    )
    r_pose = st.Rotation.from_rotvec(
        np.asarray(rotvec, dtype=np.float64).reshape(3)
    )
    pos_rel = r_ref.inv().apply(
        np.asarray(pos, dtype=np.float64).reshape(3)
        - np.asarray(ref_pos, dtype=np.float64).reshape(3)
    )
    rot6_rel = _matrix_to_rotation_6d((r_ref.inv() * r_pose).as_matrix())
    return pos_rel, rot6_rel


def _build_umi_state(
    obs: dict,
    *,
    eef_offset: float = 0.0,
) -> np.ndarray:
    """
    构造 WAM 20 维 state：
      [pos_hist(3), pos_curr(3), rot_hist(6), rot_curr(6),
       grip_hist, grip_curr]
    hist/current 位姿均相对当前 EE，因此 current 为 0 / I。
    eef_offset>0 时先把机器人 eef_link 退回 link6，与训练数据一致。
    """
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    if eef_pos.ndim != 2 or eef_pos.shape[0] < 2:
        raise ValueError(
            "Need robot_obs_horizon>=2 for 20-d UMI state, "
            f"got eef_pos shape {eef_pos.shape}"
        )
    pos_hist, rot_hist, grip_hist = _read_tcp_pose_at(obs, -2)
    pos_curr, rot_curr, grip_curr = _read_tcp_pose_at(obs, -1)
    pos_hist = _x5_eef_to_link6(pos_hist, rot_hist, eef_offset)
    pos_curr = _x5_eef_to_link6(pos_curr, rot_curr, eef_offset)
    pos_hist_rel, rot_hist_rel = _pose_relative_to(
        pos_hist,
        rot_hist,
        ref_pos=pos_curr,
        ref_rotvec=rot_curr,
    )
    return np.concatenate(
        [
            pos_hist_rel,
            np.zeros(3, dtype=np.float64),
            rot_hist_rel,
            IDENTITY_ROT6,
            [grip_hist, grip_curr],
        ]
    ).astype(np.float32)


def _make_obs_dict(wrist_image: np.ndarray) -> dict:
    """LingBot server 实际消费的图像观测。"""
    return {CAM_KEY: _to_uint8_hwc(wrist_image)}


def _collect_keyframes_during_exec(
    env,
    *,
    n_exec: int,
    action_timestamps: np.ndarray,
    actions_per_temporal: int,
    observation_delay: float,
) -> list[dict]:
    """按官方协议为每个 latent temporal frame 均匀采集 4 张图。"""
    from utils.other_util import precise_wait

    if n_exec % actions_per_temporal != 0:
        raise ValueError(
            f"n_exec={n_exec} must align to H={actions_per_temporal}"
        )
    if actions_per_temporal % OBS_FRAMES_PER_LATENT != 0:
        raise ValueError(
            f"H={actions_per_temporal} must be divisible by "
            f"OBS_FRAMES_PER_LATENT={OBS_FRAMES_PER_LATENT}"
        )
    sample_interval = actions_per_temporal // OBS_FRAMES_PER_LATENT
    key_frames: list[dict] = []
    for j in range(n_exec):
        if (j + 1) % sample_interval == 0:
            # 等待 waypoint 生效并给相机至少一个成像周期，避免读到动作前一帧。
            precise_wait(
                float(action_timestamps[j]) + observation_delay,
                time_func=time.time,
            )
            obs_j = env.get_obs()
            key_frames.append(_make_obs_dict(obs_j["camera0_rgb"][-1]))
    return key_frames


def _flatten_lingbot_action(
    action: np.ndarray, *, skip_first_temporal: bool
) -> np.ndarray:
    """
    LingBot-VA postprocess_action 输出 (C, F, H)，UMI rot6d 为 C=10。
    展平为 (N, 10)。
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 3:
        raise ValueError(f"Expected lingbot action (C,F,H), got {action.shape}")
    c, f, h = action.shape
    if c < ACTION_DIM:
        raise ValueError(
            f"Expected >= {ACTION_DIM} WAM action channels, got {action.shape}. "
            "确认 server 使用 va_umi_cfg.used_action_channel_ids = range(10)。"
        )
    start = 1 if skip_first_temporal and f > 1 else 0
    steps = [action[:ACTION_DIM, i, j] for i in range(start, f) for j in range(h)]
    if not steps:
        raise ValueError("Flattened action is empty")
    return np.stack(steps, axis=0)


def _postprocess_exec_gripper(
    grip: np.ndarray,
    *,
    gripper_min_width: float,
    gripper_max_width: float,
    invert_gripper: bool = True,
    tighten_on_grasp: bool = True,
    gripper_close_threshold: float = 0.05,
    gripper_grasp_width: float = 0.0,
) -> np.ndarray:
    """策略夹爪 → ZMQ 下发开度。

    先按需极性翻转；夹取（开度 < threshold）时改用更紧的 grasp_width，
    其他时候保留翻转后的原开度。
    """
    abs_grip = np.asarray(grip, dtype=np.float64).copy()
    if invert_gripper:
        abs_grip = _invert_gripper_width(
            abs_grip,
            gripper_min_width=gripper_min_width,
            gripper_max_width=gripper_max_width,
        )
    if tighten_on_grasp:
        abs_grip = np.where(
            abs_grip < gripper_close_threshold, gripper_grasp_width, abs_grip
        )
        exec_min = min(gripper_min_width, gripper_grasp_width)
    else:
        exec_min = gripper_min_width
    exec_min = max(0.0, float(exec_min))
    return np.clip(abs_grip, exec_min, gripper_max_width)


def _delta_chunk_to_absolute_pose(
    action_chunk: np.ndarray,
    *,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
    gripper_min_width: float,
    gripper_max_width: float,
    binarize_gripper: bool = True,
    gripper_close_threshold: float = 0.05,
    invert_gripper: bool = True,
    gripper_grasp_width: float = 0.0,
) -> np.ndarray:
    """
    将 EE-relative SE(3) action 链式复合为绝对 TCP：
      p <- p + R @ dpos
      R <- R @ R_from_row_rot6d
      grip: 可选先从数据集极性翻到 ZMQ 开口；
            夹取时下发 gripper_grasp_width，其他时候用原开度
            （--no_binarize_gripper 时全程用原开度）
    """
    deltas = np.asarray(action_chunk, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action chunk (N, {ACTION_DIM}), got {deltas.shape}")

    out = np.zeros((deltas.shape[0], 7), dtype=np.float64)
    pos = np.asarray(cur_pos, dtype=np.float64).reshape(3).copy()
    r_cur = st.Rotation.from_rotvec(np.asarray(cur_rotvec, dtype=np.float64).reshape(3))
    abs_grip = _postprocess_exec_gripper(
        deltas[:, GRIPPER_ACTION_INDEX],
        gripper_min_width=gripper_min_width,
        gripper_max_width=gripper_max_width,
        invert_gripper=invert_gripper,
        tighten_on_grasp=binarize_gripper,
        gripper_close_threshold=gripper_close_threshold,
        gripper_grasp_width=gripper_grasp_width,
    )

    for i in range(deltas.shape[0]):
        pos = pos + r_cur.apply(deltas[i, :3])
        r_delta = st.Rotation.from_matrix(
            _rotation_6d_to_matrix(deltas[i, 3:9])
        )
        r_cur = r_cur * r_delta
        out[i, :3] = pos
        out[i, 3:6] = r_cur.as_rotvec()
        out[i, 6] = abs_grip[i]
    return out


def _clip_action_chunk(
    action_chunk: np.ndarray,
    *,
    max_delta_pos: float,
    max_delta_rot: float,
) -> tuple[np.ndarray, int, int]:
    """位置逐维 clip；旋转按相对转角缩放。"""
    clipped = np.asarray(action_chunk, dtype=np.float64).copy()
    if clipped.ndim != 2 or clipped.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action chunk (N, {ACTION_DIM}), got {clipped.shape}")

    n_pos_clip = int(np.sum(np.abs(clipped[:, :3]) > max_delta_pos))
    clipped[:, :3] = np.clip(clipped[:, :3], -max_delta_pos, max_delta_pos)

    n_rot_clip = 0
    for i in range(clipped.shape[0]):
        before = clipped[i, 3:9].copy()
        after = _clip_rotation_6d(before, max_delta_rot)
        if not np.allclose(before, after, atol=1e-8):
            n_rot_clip += 1
        clipped[i, 3:9] = after
    return clipped, n_pos_clip, n_rot_clip


def _absolute_pose_to_delta_step(
    abs_pose: np.ndarray,
    *,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
) -> np.ndarray:
    """单步 WAM decode 的逆；位置 delta 位于当前 EE 坐标系。"""
    absolute = np.asarray(abs_pose, dtype=np.float64).reshape(7)
    position = np.asarray(cur_pos, dtype=np.float64).reshape(3)
    r_cur = st.Rotation.from_rotvec(
        np.asarray(cur_rotvec, dtype=np.float64).reshape(3)
    )
    r_next = st.Rotation.from_rotvec(absolute[3:6])
    dpos_ee = r_cur.inv().apply(absolute[:3] - position)
    drot6 = _matrix_to_rotation_6d((r_cur.inv() * r_next).as_matrix())
    return np.concatenate([dpos_ee, drot6, [absolute[6]]])


def _fmt_vec3(vec: np.ndarray, *, unit: str = "") -> str:
    values = np.asarray(vec, dtype=np.float64).reshape(3)
    return (
        f"[{values[0]:+.4f}, {values[1]:+.4f}, {values[2]:+.4f}]{unit}"
    )


def _dir_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    va = np.asarray(a, dtype=np.float64).reshape(3)
    vb = np.asarray(b, dtype=np.float64).reshape(3)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(va / na, vb / nb))


def _summarize_action_steps(action_chunk: np.ndarray) -> str:
    actions = np.asarray(action_chunk, dtype=np.float64)
    pos_mm = np.linalg.norm(actions[:, :3], axis=1) * 1000.0
    rot_deg = np.asarray(
        [
            np.degrees(
                st.Rotation.from_matrix(
                    _rotation_6d_to_matrix(row[3:9])
                ).magnitude()
            )
            for row in actions
        ]
    )
    return (
        f"pos_mm/step={np.array2string(pos_mm, precision=2)} "
        f"rot_deg/step={np.array2string(rot_deg, precision=2)} "
        f"grip_abs={np.array2string(actions[:, 9], precision=4)}"
    )


def _diagnose_scheduling(
    action_timestamps: np.ndarray, *, n_exec: int
) -> dict[str, float | int]:
    """检查 waypoint 在提交时是否已经过期。"""
    receive_time = time.time()
    timestamps = np.asarray(action_timestamps[:n_exec], dtype=np.float64)
    is_new = timestamps > receive_time
    n_keep = int(np.sum(is_new))
    return {
        "receive_time": receive_time,
        "n_exec": n_exec,
        "n_keep": n_keep,
        "n_drop": n_exec - n_keep,
        "t0_minus_now_ms": (
            float((timestamps[0] - receive_time) * 1000.0) if n_exec else 0.0
        ),
        "dt_ms": (
            float((timestamps[1] - timestamps[0]) * 1000.0)
            if n_exec > 1
            else 0.0
        ),
    }


def _print_policy_cycle_diag(
    *,
    cycle_idx: int,
    episode_id: int,
    iter_idx: int,
    obs_latency: float,
    budget_s: float,
    wrist_img: np.ndarray,
    state_dbg: np.ndarray,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
    cur_grip: float,
    last_cmd_abs: np.ndarray | None,
    last_state_pos: np.ndarray | None,
    last_state_rotvec: np.ndarray | None,
    last_state_grip: float | None,
    prev_model_step_world: np.ndarray | None,
    prev_decode_step_world: np.ndarray | None,
    raw_chunk: np.ndarray,
    action_chunk: np.ndarray,
    abs_exec: np.ndarray,
    infer_s: float,
    n_exec: int,
    n_pos_clip: int,
    n_rot_clip: int,
    sched: dict[str, float | int],
    debug_fixed_delta: str | None,
) -> None:
    """打印 WAM state 与 EE-relative action 的结构化控制诊断。"""
    raw_exec = np.asarray(raw_chunk[:n_exec], dtype=np.float64)
    clipped_exec = np.asarray(action_chunk[:n_exec], dtype=np.float64)
    r_cur = st.Rotation.from_rotvec(cur_rotvec)
    current_model_world = r_cur.apply(raw_exec[0, :3])
    current_decode_world = r_cur.apply(clipped_exec[0, :3])
    actual_move_world = (
        None
        if last_state_pos is None
        else np.asarray(cur_pos) - np.asarray(last_state_pos)
    )
    decode_actual_cos = (
        None
        if actual_move_world is None or prev_decode_step_world is None
        else _dir_cosine(prev_decode_step_world, actual_move_world)
    )

    print("\n" + "=" * 72)
    print(
        f"[DIAG] cycle={cycle_idx} episode={episode_id} iter={iter_idx} "
        f"budget={budget_s:.3f}s obs_latency={obs_latency:.3f}s"
    )
    if debug_fixed_delta is not None:
        print(f"[DIAG][MODE] fixed WAM EE-relative delta={debug_fixed_delta}")
    print(
        f"[DIAG][OBS] img={wrist_img.shape}/{wrist_img.dtype} "
        f"mean={wrist_img.mean():.1f} pos={_fmt_vec3(cur_pos, unit='m')} "
        f"rotvec={_fmt_vec3(cur_rotvec, unit='rad')} grip={cur_grip:.4f}m"
    )
    print(
        "[DIAG][OBS][state20->model] "
        f"{np.array2string(np.asarray(state_dbg), precision=4)}"
    )
    print(
        f"[DIAG][OBS][state20] pos_hist={_fmt_vec3(state_dbg[:3], unit='m')} "
        f"pos_curr={_fmt_vec3(state_dbg[3:6], unit='m')} "
        f"grip_hist={state_dbg[18]:.4f} grip_curr={state_dbg[19]:.4f}"
    )

    track_pos_err_mm = track_rot_err_deg = moved_pos_mm = None
    if last_cmd_abs is not None and last_state_pos is not None:
        track_pos_err_mm = float(np.linalg.norm(cur_pos - last_cmd_abs[:3]) * 1000.0)
        track_rot_err_deg = float(
            np.degrees(
                (
                    st.Rotation.from_rotvec(last_cmd_abs[3:6]).inv()
                    * st.Rotation.from_rotvec(cur_rotvec)
                ).magnitude()
            )
        )
        moved_pos_mm = float(np.linalg.norm(actual_move_world) * 1000.0)
        moved_rot_deg = float(
            np.degrees(
                (
                    st.Rotation.from_rotvec(last_state_rotvec).inv()
                    * st.Rotation.from_rotvec(cur_rotvec)
                ).magnitude()
            )
        )
        print(
            f"[DIAG][TRACK] cmd_err pos={track_pos_err_mm:.1f}mm "
            f"rot={track_rot_err_deg:.2f}deg "
            f"grip={abs(cur_grip-float(last_cmd_abs[6]))*1000.0:.1f}mm | "
            f"actual pos={moved_pos_mm:.1f}mm rot={moved_rot_deg:.2f}deg "
            f"dgrip={(cur_grip-float(last_state_grip))*1000.0:+.1f}mm"
        )
    else:
        print("[DIAG][TRACK] first cycle")

    if (
        actual_move_world is not None
        and prev_model_step_world is not None
        and prev_decode_step_world is not None
    ):
        print(
            "[DIAG][DIR] previous model/decode/actual (all world frame): "
            f"model={_fmt_vec3(prev_model_step_world)} "
            f"decode={_fmt_vec3(prev_decode_step_world)} "
            f"actual={_fmt_vec3(actual_move_world)} "
            f"cos(decode,actual)={decode_actual_cos}"
        )
    print(
        "[DIAG][DIR] current: "
        f"model_ee={_fmt_vec3(raw_exec[0, :3])} "
        f"decode_ee={_fmt_vec3(clipped_exec[0, :3])} "
        f"model_world={_fmt_vec3(current_model_world)} "
        f"decode_world={_fmt_vec3(current_decode_world)}"
    )
    print(
        f"[DIAG][MODEL] infer={infer_s:.3f}s "
        f"{'OVER_BUDGET' if infer_s > budget_s else 'ok'} "
        f"chunk={raw_chunk.shape} exec={n_exec}"
    )
    print(f"[DIAG][MODEL] {_summarize_action_steps(raw_exec)}")
    print(
        f"[DIAG][MODEL][clip] pos_elems={n_pos_clip} rot_steps={n_rot_clip}"
    )
    print(f"[DIAG][MODEL][raw_exec]\n{raw_exec}")

    implied = _absolute_pose_to_delta_step(
        abs_exec[0], cur_pos=cur_pos, cur_rotvec=cur_rotvec
    )
    pos_roundtrip_mm = float(
        np.linalg.norm(implied[:3] - clipped_exec[0, :3]) * 1000.0
    )
    rot_roundtrip_deg = float(
        np.degrees(
            (
                st.Rotation.from_matrix(
                    _rotation_6d_to_matrix(implied[3:9])
                ).inv()
                * st.Rotation.from_matrix(
                    _rotation_6d_to_matrix(clipped_exec[0, 3:9])
                )
            ).magnitude()
        )
    )
    print(
        f"[DIAG][DECODE][roundtrip] pos={pos_roundtrip_mm:.4f}mm "
        f"rot={rot_roundtrip_deg:.4f}deg "
        f"grip={abs(implied[9]-clipped_exec[0, 9])*1000.0:.3f}mm"
    )
    print(f"[DIAG][DECODE][abs_exec]\n{abs_exec}")
    print(
        f"[DIAG][SCHED] submit={sched['n_exec']} keep={sched['n_keep']} "
        f"drop={sched['n_drop']} t0-now={sched['t0_minus_now_ms']:+.1f}ms "
        f"dt={sched['dt_ms']:.1f}ms"
    )

    hints: list[str] = []
    state_array = np.asarray(state_dbg, dtype=np.float64).reshape(-1)
    if state_array.shape != (STATE_DIM,):
        hints.append(f"[STATE] 期望 {STATE_DIM} 维，实际 {state_array.shape}。")
    elif np.linalg.norm(state_array[3:6]) > 1e-6 or not np.allclose(
        state_array[12:18], IDENTITY_ROT6, atol=1e-5
    ):
        hints.append("[STATE] current pose 应为 0/I；检查 EE-relative state 构造。")
    if infer_s > budget_s:
        hints.append("[TIMING] 推理超过当前 LingBot 控制周期预算。")
    if int(sched["n_drop"]) > 0:
        hints.append("[SCHED] 存在过期 waypoint；检查推理耗时和启动提前量。")
    if n_pos_clip or n_rot_clip:
        hints.append("[MODEL/CLIP] 模型输出被安全限位，检查动作尺度。")
    if pos_roundtrip_mm > 0.05 or rot_roundtrip_deg > 0.1:
        hints.append("[DECODE] delta→absolute roundtrip 异常。")
    if track_pos_err_mm is not None and track_pos_err_mm > 10.0:
        hints.append("[EXEC] 机器人位置未跟上上周期末命令。")
    if track_rot_err_deg is not None and track_rot_err_deg > 8.0:
        hints.append("[EXEC] 机器人旋转未跟上上周期末命令。")
    if decode_actual_cos is not None and decode_actual_cos < 0.3:
        hints.append("[MODEL/DECODE] 预测世界方向与实际运动方向偏差较大。")
    if moved_pos_mm is not None and np.linalg.norm(raw_exec[0, :3]) > 3e-4 and moved_pos_mm < 0.2:
        hints.append("[EXEC] 模型要求移动但机器人几乎未动。")
    if not hints:
        hints.append("[OK] 本周期未发现明显的输入、解码或执行异常。")
    print("[DIAG][HINT]")
    for hint in hints:
        print(f"  -> {hint}")


# ===========================================================================
# LingBot-VA 远程推理客户端
# ===========================================================================


class LingBotVAClient:
    """包装 WebsocketClientPolicy，对齐 wan_va_server.infer 协议。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29536,
        *,
        connect_timeout: float = 5.0,
        request_timeout: float = 100.0,
    ):
        self._client = _create_websocket_client(
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )
        _assert_lingbot_server(
            self._client.get_server_metadata(), host=host, port=port
        )
        self._first_chunk = True
        self._last_chunk_start: int | None = None

    def reset(self, prompt: str) -> None:
        self._client.infer(dict(reset=True, prompt=prompt))
        self._first_chunk = True
        self._last_chunk_start = None

    def infer_chunk(
        self,
        wrist_image: np.ndarray,
        prompt: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        返回:
            flat_deltas: (N, 10)
            raw_action:  (C, F, H) 原始返回，供 compute_kv_cache 的 state=
        """
        obs_dict = _make_obs_dict(wrist_image)
        # obs 可为单 dict；server 会包成 list
        ret = self._client.infer(dict(obs=obs_dict, prompt=prompt))
        if "action" not in ret:
            raise RuntimeError(f"Server response missing 'action': keys={list(ret)}")
        raw = np.asarray(ret["action"])
        self._last_chunk_start = (
            1 if self._first_chunk and raw.shape[1] > 1 else 0
        )
        flat = _flatten_lingbot_action(raw, skip_first_temporal=self._first_chunk)
        self._first_chunk = False
        return flat, raw

    def action_state_for_exec(
        self, raw_action: np.ndarray, n_exec: int
    ) -> tuple[np.ndarray, int]:
        """切片本轮实际执行对应的原始 action，供 KV / FDM / commit_real。

        必须在下一次 infer_chunk 之前调用：async 会先 predict 下一 chunk，
        否则 _last_chunk_start 会被覆盖。
        """
        if self._last_chunk_start is None:
            raise RuntimeError("action_state_for_exec called before infer_chunk")

        raw = np.asarray(raw_action)
        if raw.ndim != 3:
            raise ValueError(f"Expected raw action (C,F,H), got {raw.shape}")
        _, temporal_count, actions_per_temporal = raw.shape
        if n_exec <= 0 or n_exec % actions_per_temporal != 0:
            raise ValueError(
                f"n_exec={n_exec} must be a positive multiple of "
                f"H={actions_per_temporal}"
            )

        executed_temporals = n_exec // actions_per_temporal
        start = self._last_chunk_start
        end = start + executed_temporals
        if end > temporal_count:
            raise ValueError(
                f"Executed temporal range [{start}:{end}] exceeds F={temporal_count}"
            )

        # 首 chunk 的 F=0 是初始条件；server 同时 prepend init_latent，因此也要保留。
        if start == 1:
            cache_action = raw[:, :end, :]
        else:
            cache_action = raw[:, :executed_temporals, :]
        return cache_action, executed_temporals

    def update_kv_cache(
        self,
        key_frames: list[dict],
        raw_action: np.ndarray,
        n_exec: int,
        prompt: str,
    ) -> tuple[int, int]:
        if not key_frames:
            raise ValueError("key_frames must not be empty for KV cache update")

        cache_action, executed_temporals = self.action_state_for_exec(
            raw_action, n_exec
        )
        expected_key_frames = executed_temporals * OBS_FRAMES_PER_LATENT
        if len(key_frames) != expected_key_frames:
            raise ValueError(
                f"Expected {expected_key_frames} KV key frames for "
                f"{executed_temporals} executed temporal frames, got {len(key_frames)}"
            )

        # state 是实际提交部分对应的原始模型 action，不是机器人 TCP state。
        self._client.infer(
            dict(
                obs=key_frames,
                compute_kv_cache=True,
                imagine=False,
                state=cache_action,
                prompt=prompt,
            )
        )
        return executed_temporals, cache_action.shape[1]

    def fdm_ground(self, cache_action: np.ndarray, prompt: str) -> None:
        """Paper Algorithm 2: 用已知 a_t grounding 想象视觉，同时机器人执行。"""
        self._client.infer(
            dict(
                fdm_ground=True,
                state=np.asarray(cache_action),
                prompt=prompt,
            )
        )

    def commit_real(
        self,
        key_frames: list[dict],
        cache_action: np.ndarray,
        executed_temporals: int,
        prompt: str,
    ) -> None:
        """用真实关键帧替换 FDM 想象 visual；保留已预测的下一 chunk。"""
        if not key_frames:
            raise ValueError("key_frames must not be empty for commit_real")
        expected_key_frames = executed_temporals * OBS_FRAMES_PER_LATENT
        if len(key_frames) != expected_key_frames:
            raise ValueError(
                f"Expected {expected_key_frames} commit key frames for "
                f"{executed_temporals} executed temporal frames, got {len(key_frames)}"
            )
        self._client.infer(
            dict(
                commit_real=True,
                obs=key_frames,
                state=np.asarray(cache_action),
                prompt=prompt,
            )
        )


# ===========================================================================
# 辅助
# ===========================================================================

DEFAULT_GRIPPER_MAX_WIDTH = 0.085 # 改成这个他才正常工作
# DEFAULT_GRIPPER_MAX_WIDTH = 0.05 # 改成这个他在刚开始的时候就会闭合夹爪。
DEFAULT_TASK_PROMPT = (
    "pick up the electronic component and place it into the designated box"
)


def easeInOutQuad(t: float) -> float:
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
        from peripherals.keystroke_counter import Key, KeyCode

        events = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            event = self._char_to_event(ch, Key, KeyCode)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _char_to_event(ch: str, Key, KeyCode):
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
@click.option("--output", "-o", default="./lingbot_umi_output", help="输出目录")
@click.option(
    "--umi_project_root",
    default="/data1/projects/umi-arx-hzl",
    show_default=True,
    help="UMI/ARX5 真机工程根目录（含 utils/, peripherals/, modules/）",
)
@click.option(
    "--lingbot_project_root",
    default="",
    show_default=True,
    help="可选 lingbot-va 根目录（含 wan_va/）；不设则用内嵌 websocket client",
)
@click.option("--policy_host", default="127.0.0.1", help="LingBot-VA server 地址")
@click.option(
    "--policy_port",
    default=29536,
    type=int,
    show_default=True,
    help="LingBot-VA server 端口",
)
@click.option(
    "--policy_connect_timeout",
    default=300.0,
    type=float,
    show_default=True,
    help="连接 LingBot-VA 服务的超时（秒；内嵌 client）",
)
@click.option(
    "--policy_request_timeout",
    default=300.0,
    type=float,
    show_default=True,
    help="单次 LingBot-VA 请求的超时（秒；内嵌 client）",
)
@click.option("--task_prompt", default=DEFAULT_TASK_PROMPT, help="任务描述")
@click.option(
    "--steps_per_inference",
    "-si",
    default=8,
    type=int,
    help="固定动作调试模式每轮生成的步数",
)
@click.option(
    "--exec_horizon",
    "-eh",
    default=8,
    type=int,
    show_default=True,
    help="每次实际执行的 action 数；必须是模型 H 的整数倍",
)
@click.option(
    "--use_full_chunk",
    is_flag=True,
    default=False,
    help="执行完整展平 chunk（忽略 exec_horizon）",
)
@click.option("--max_duration", "-md", default=2000000, help="每 episode 最大时长（秒）")
@click.option(
    "--frequency",
    "-f",
    default=20,
    type=float,
    show_default=True,
    help="动作执行频率；应与训练数据原始动作频率一致 (Hz)",
)
@click.option(
    "--command_latency", "-cl", default=0.01, type=float, help="SpaceMouse 命令延迟（秒）"
)
@click.option("--camera_reorder", "-cr", default="0")
@click.option("--no_mirror", "-nm", is_flag=True, default=False)
@click.option("--mirror_swap", is_flag=True, default=False)
@click.option("--init_joints", "-j", is_flag=True, default=False, help="是否初始化关节配置")
@click.option(
    "--init_delta",
    type=str,
    default="0,0,0,0.1,-0.1,-1.0",
    show_default=True,
    help="位姿微调: dx,dy,dz,drx,dry,drz（米/弧度）",
)
@click.option("--init_gripper_delta", type=float, default=0.11, help="夹爪开度微调（米）")
@click.option("--gripper_min_width", type=float, default=0.01, help="夹爪最小开度（米）")
@click.option(
    "--gripper_max_width",
    type=float,
    default=DEFAULT_GRIPPER_MAX_WIDTH,
    help="夹爪最大开度（米）",
)
@click.option("--max_delta_pos", type=float, default=0.08, help="单步位置 delta 安全限位（米）")
@click.option(
    "--max_delta_rot",
    type=float,
    default=0.3,
    show_default=True,
    help="单步相对旋转 delta 安全限位（弧度）",
)
@click.option(
    "--no_binarize_gripper",
    is_flag=True,
    default=False,
    help="关掉夹取加紧：全程使用策略原开度（仍会按需极性翻转并 clip）",
)
@click.option(
    "--gripper_close_threshold",
    type=float,
    default=0.045,
    show_default=True,
    help="ZMQ 开度（米）低于该值视为夹取，下发 --gripper_grasp_width",
)
@click.option(
    "--gripper_grasp_width",
    type=float,
    default=0.011,
    show_default=True,
    help="夹取时下发的开度（米），比策略输出更紧。其他时候用原开度",
)
@click.option(
    "--no_invert_gripper",
    is_flag=True,
    default=False,
    help="不要把策略夹爪从数据集极性（越大越夹紧）翻到 ZMQ 开口（越大越张开）",
)
@click.option(
    "--cpu_affinity",
    default="7",
    show_default=True,
    help="os.sched_setaffinity CPU 列表，逗号分隔；空字符串表示不绑核",
)
@click.option(
    "--terminal_keys",
    is_flag=True,
    default=False,
    help="从 SSH 终端 stdin 读 c/q/s/b",
)
@click.option(
    "--debug_fixed_delta",
    default=None,
    help="跳过策略，重复固定 10 维 WAM EE-relative action：pos×3,rot6×6,grip",
)
@click.option(
    "--eef_offset",
    type=float,
    default=X5_EEF_OFFSET_TCP_Z,
    show_default=True,
    help="训练数据是 link6，ZMQ X5 控制点是 eef_link。"
    "相对动作在 link6 上展开后，沿 TCP Z 加该偏移再下发；旋转不变。"
    "Direct / 已经是 eef 数据时设为 0。",
)
@click.option(
    "--async_infer/--sync_infer",
    default=True,
    show_default=True,
    help="异步 Algorithm 2（执行 ‖ FDM+下一 chunk 预测；支持 --exec_horizon 部分执行）；"
    "--sync_infer 回退旧闭环",
)
def main(
    output,
    umi_project_root,
    lingbot_project_root,
    policy_host,
    policy_port,
    policy_connect_timeout,
    policy_request_timeout,
    task_prompt,
    steps_per_inference,
    exec_horizon,
    use_full_chunk,
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
    gripper_close_threshold,
    gripper_grasp_width,
    no_invert_gripper,
    cpu_affinity,
    terminal_keys,
    debug_fixed_delta,
    eef_offset,
    async_infer,
):
    _ensure_umi_project(umi_project_root)
    _ensure_lingbot_project(lingbot_project_root or os.environ.get("LINGBOT_VA_ROOT"))

    from utils.other_util import precise_wait
    from peripherals.keystroke_counter import KeystrokeCounter, Key, KeyCode
    from peripherals.spacemouse_shared_memory import Spacemouse
    from modules.arx5_env import Arx5Env

    if cpu_affinity.strip():
        cpus = [int(x) for x in cpu_affinity.split(",") if x.strip() != ""]
        os.sched_setaffinity(os.getpid(), cpus)

    gripper_min_width = max(0.0, float(gripper_min_width))
    max_gripper_width = float(gripper_max_width)
    if max_gripper_width <= gripper_min_width:
        raise click.BadParameter("--gripper_max_width must be > --gripper_min_width")
    gripper_close_threshold = float(gripper_close_threshold)
    gripper_grasp_width = float(gripper_grasp_width)
    if gripper_close_threshold <= 0.0:
        raise click.BadParameter("--gripper_close_threshold must be > 0")
    if gripper_grasp_width < 0.0:
        raise click.BadParameter("--gripper_grasp_width must be >= 0")
    if gripper_grasp_width >= max_gripper_width:
        raise click.BadParameter("--gripper_grasp_width must be < --gripper_max_width")
    if steps_per_inference <= 0:
        raise click.BadParameter("--steps_per_inference must be > 0")
    if exec_horizon <= 0:
        raise click.BadParameter("--exec_horizon must be > 0")
    if frequency <= 0:
        raise click.BadParameter("--frequency must be > 0")
    if policy_connect_timeout <= 0 or policy_request_timeout <= 0:
        raise click.BadParameter("--policy timeouts must be > 0")
    if async_infer and debug_fixed_delta is not None:
        raise click.BadParameter(
            "--async_infer 不能与 --debug_fixed_delta 同时使用；请加 --sync_infer"
        )

    fixed_delta_vec = None
    if debug_fixed_delta is not None:
        try:
            values = [float(value) for value in debug_fixed_delta.split(",")]
        except ValueError as exc:
            raise click.BadParameter(
                "--debug_fixed_delta values must be numbers"
            ) from exc
        if len(values) != ACTION_DIM:
            raise click.BadParameter(
                f"--debug_fixed_delta must have {ACTION_DIM} comma-separated values"
            )
        fixed_delta_vec = np.asarray(values, dtype=np.float64)
        print(f"[DIAG] debug_fixed_delta enabled: {fixed_delta_vec}")

    gripper_speed = 0.02
    cartesian_speed = 0.4
    orientation_speed = 0.8

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output, "action"), exist_ok=True)

    client = None
    if fixed_delta_vec is None:
        print(f"Connecting to LingBot-VA at ws://{policy_host}:{policy_port} ...")
        client = LingBotVAClient(
            host=policy_host,
            port=policy_port,
            connect_timeout=policy_connect_timeout,
            request_timeout=policy_request_timeout,
        )
        print(f"Connected. Task prompt: {task_prompt}")
    else:
        print("LingBot-VA connection skipped in fixed-delta debug mode.")
    eef_offset = float(eef_offset)
    print(
        f"[MODE] {'async_fdm (Algorithm 2)' if async_infer else 'sync'} "
        f"| WAM state20 + EE-frame action10 (row-rot6d); "
        "LingBot FDM/KV/commit state=raw model action"
    )
    if abs(eef_offset) > 1e-12:
        print(
            f"[ALIGN] policy frame=link6 → exec frame=X5 eef_link "
            f"(+{eef_offset:.3f} m along TCP Z; rotation unchanged)"
        )
    else:
        print("[ALIGN] eef_offset=0: no link6↔eef_link conversion")
    invert_gripper = not no_invert_gripper
    tighten_on_grasp = not no_binarize_gripper
    if invert_gripper:
        polarity = "invert dataset (large=close) → ZMQ (large=open)"
    else:
        polarity = "no invert; treat policy output as ZMQ opening"
    if tighten_on_grasp:
        print(
            f"[ALIGN] gripper: {polarity}; "
            f"grasp if < {gripper_close_threshold:.3f}m → "
            f"{gripper_grasp_width:.4f}m, else original width"
        )
    else:
        print(f"[ALIGN] gripper: {polarity}; always original width")

    obs_res = (256, 256)
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
    dt = 1.0 / frequency

    key_counter_ctx = (
        contextlib.nullcontext(None) if terminal_keys else KeystrokeCounter()
    )
    terminal_reader_ctx = (
        TerminalKeyReader() if terminal_keys else contextlib.nullcontext(None)
    )

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=0.1) as sm, key_counter_ctx as key_counter, terminal_reader_ctx as terminal_reader, Arx5Env(
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
                tgt_grip = np.clip(
                    start_grip + init_gripper_delta, gripper_min_width, max_gripper_width
                )

                warmup_sec = 2.5
                steps = max(2, int(warmup_sec * frequency))
                actions, timestamps = [], []
                t0 = time.time() + 0.3
                for k in range(steps):
                    a = easeInOutQuad(k / (steps - 1))
                    cmd = np.zeros(7, dtype=np.float64)
                    cmd[:6] = start_pose + a * (tgt_pose - start_pose)
                    cmd[6] = np.clip(
                        start_grip + a * (tgt_grip - start_grip),
                        gripper_min_width,
                        max_gripper_width,
                    )
                    actions.append(cmd)
                    timestamps.append(t0 + k * dt)

                env.exec_actions(
                    actions=np.asarray(actions),
                    timestamps=np.asarray(timestamps),
                    compensate_latency=False,
                )
                time.sleep(warmup_sec + 0.2)
                print("微调完成")

            if client is not None:
                print("Warming up LingBot-VA inference...")
                client.reset(task_prompt)
                obs = env.get_obs()
                wrist_img = _to_uint8_hwc(obs["camera0_rgb"][-1])
                state_dbg = _build_umi_state(obs, eef_offset=eef_offset)
                print(
                    f"warmup state20: pos_hist={state_dbg[:3]}, "
                    f"pos_curr={state_dbg[3:6]} (~0), "
                    f"rot_curr={state_dbg[12:18]} (~I), "
                    f"grip={state_dbg[18:20]}"
                )
                flat, raw = client.infer_chunk(wrist_img, task_prompt)
                print(f"Warmup done. flat={flat.shape}, raw={raw.shape}")
                client.reset(task_prompt)
            print("Ready!")

            while True:
                # =============================================================
                # Human
                # =============================================================
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs["ActualTCPPose"] for rs in robot_states])
                gripper_target_pos = np.asarray(
                    [rs["gripper_position"] for rs in robot_states]
                )
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                episode_id = env.replay_buffer.n_episodes
                obs = env.get_obs()

                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    vis_img = obs["camera0_rgb"][-1]
                    obs = env.get_obs()

                    text = f"Episode: {episode_id}"
                    vis_show = np.asarray(vis_img)
                    vis_show = (
                        _to_uint8_hwc(vis_show)
                        if vis_show.dtype != np.uint8
                        else vis_show.copy()
                    )
                    for thickness, color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
                        cv2.putText(
                            vis_show,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            lineType=cv2.LINE_AA,
                            thickness=thickness,
                            color=color,
                        )
                    cv2.imshow("default", vis_show[..., ::-1])
                    _ = cv2.pollKey()

                    press_events = _collect_key_events(key_counter, terminal_reader)
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            env.end_episode()
                            raise SystemExit(0)
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

                    dpos_g = 0.0
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
                        dpos_g = -gripper_speed / frequency
                    elif sm.is_button_pressed(1):
                        dpos_g = gripper_speed / frequency

                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(
                            gripper_target_pos[robot_idx] + dpos_g,
                            gripper_min_width,
                            max_gripper_width,
                        )

                    action = np.zeros(7)
                    action[:6] = target_pose[0]
                    action[6] = np.clip(
                        gripper_target_pos[0], gripper_min_width, max_gripper_width
                    )
                    env.exec_actions(
                        actions=[action],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        compensate_latency=False,
                    )
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # =============================================================
                # Policy
                # =============================================================
                episode_active = False
                try:
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    episode_active = True

                    frame_latency = 1.0 / 60.0
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Policy in control!")
                    if client is not None:
                        client.reset(task_prompt)

                    iter_idx = 0
                    cycle_idx = 0
                    last_cmd_abs = None
                    last_state_pos = None
                    last_state_rotvec = None
                    last_state_grip = None
                    prev_model_step_world = None
                    prev_decode_step_world = None

                    # Cold start PREDICT only for async Algorithm 2.
                    pending_flat = pending_raw = None
                    if async_infer:
                        assert client is not None
                        obs0 = env.get_obs()
                        wrist_img0 = _to_uint8_hwc(obs0["camera0_rgb"][-1])
                        t_cold = time.time()
                        pending_flat, pending_raw = client.infer_chunk(
                            wrist_img0, task_prompt
                        )
                        print(
                            f"[DIAG][MODEL] cold_infer={time.time() - t_cold:.3f}s "
                            f"flat={pending_flat.shape} raw={pending_raw.shape} "
                            f"mode=async_fdm"
                        )

                    while True:
                        obs = env.get_obs()
                        obs_timestamps = obs["timestamp"]
                        obs_latency = time.time() - obs_timestamps[-1]

                        s = time.time()
                        wrist_img = _to_uint8_hwc(obs["camera0_rgb"][-1])
                        state_dbg = _build_umi_state(obs, eef_offset=eef_offset)
                        cur_pos_robot, cur_rotvec, cur_grip = _read_tcp_pose(obs)
                        cur_pos = _x5_eef_to_link6(
                            cur_pos_robot, cur_rotvec, eef_offset
                        )

                        episode_id = env.replay_buffer.n_episodes
                        if fixed_delta_vec is not None:
                            flat_chunk = np.tile(
                                fixed_delta_vec, (steps_per_inference, 1)
                            )
                            raw_action = None
                            infer_s = time.time() - s
                        elif async_infer:
                            flat_chunk = pending_flat
                            raw_action = pending_raw
                            infer_s = 0.0  # filled after overlap GPU work
                        else:
                            assert client is not None
                            flat_chunk, raw_action = client.infer_chunk(
                                wrist_img, task_prompt
                            )
                            infer_s = time.time() - s

                        n_exec = (
                            len(flat_chunk)
                            if use_full_chunk
                            else min(exec_horizon, len(flat_chunk))
                        )
                        if n_exec <= 0:
                            raise RuntimeError("No actions to execute from lingbot chunk")
                        if raw_action is not None:
                            actions_per_temporal = int(raw_action.shape[2])
                            if n_exec % actions_per_temporal != 0:
                                raise RuntimeError(
                                    f"exec_horizon produced n_exec={n_exec}, but KV "
                                    f"updates require complete temporal groups of "
                                    f"H={actions_per_temporal} actions"
                                )
                        else:
                            actions_per_temporal = n_exec
                        budget_s = n_exec * dt

                        action_chunk, n_pos_clip, n_rot_clip = _clip_action_chunk(
                            flat_chunk,
                            max_delta_pos=max_delta_pos,
                            max_delta_rot=max_delta_rot,
                        )

                        abs_link6 = _delta_chunk_to_absolute_pose(
                            action_chunk[:n_exec],
                            cur_pos=cur_pos,
                            cur_rotvec=cur_rotvec,
                            gripper_min_width=gripper_min_width,
                            gripper_max_width=max_gripper_width,
                            binarize_gripper=tighten_on_grasp,
                            gripper_close_threshold=gripper_close_threshold,
                            invert_gripper=invert_gripper,
                            gripper_grasp_width=gripper_grasp_width,
                        )
                        abs_exec = _abs_exec_link6_to_x5_eef(abs_link6, eef_offset)

                        command_start_time = time.time() + 0.05
                        action_timestamps = (
                            np.arange(n_exec, dtype=np.float64) * dt + command_start_time
                        )
                        sched = _diagnose_scheduling(
                            action_timestamps, n_exec=n_exec
                        )
                        _print_policy_cycle_diag(
                            cycle_idx=cycle_idx,
                            episode_id=episode_id,
                            iter_idx=iter_idx,
                            obs_latency=obs_latency,
                            budget_s=budget_s,
                            wrist_img=wrist_img,
                            state_dbg=state_dbg,
                            cur_pos=cur_pos_robot,
                            cur_rotvec=cur_rotvec,
                            cur_grip=cur_grip,
                            last_cmd_abs=last_cmd_abs,
                            last_state_pos=last_state_pos,
                            last_state_rotvec=last_state_rotvec,
                            last_state_grip=last_state_grip,
                            prev_model_step_world=prev_model_step_world,
                            prev_decode_step_world=prev_decode_step_world,
                            raw_chunk=flat_chunk,
                            action_chunk=action_chunk,
                            abs_exec=abs_exec,
                            infer_s=infer_s,
                            n_exec=n_exec,
                            n_pos_clip=n_pos_clip,
                            n_rot_clip=n_rot_clip,
                            sched=sched,
                            debug_fixed_delta=debug_fixed_delta,
                        )

                        next_flat = next_raw = None
                        if client is not None and async_infer:
                            # 在下一 chunk infer 前先切片 a_t，避免 _last_chunk_start 被覆盖。
                            cache_action, executed_temporals = (
                                client.action_state_for_exec(raw_action, n_exec)
                            )

                            def _exec_and_collect():
                                env.exec_actions(
                                    actions=abs_exec,
                                    timestamps=action_timestamps,
                                    dynamic_latency=False,
                                )
                                return _collect_keyframes_during_exec(
                                    env,
                                    n_exec=n_exec,
                                    action_timestamps=action_timestamps,
                                    actions_per_temporal=actions_per_temporal,
                                    observation_delay=frame_latency,
                                )

                            t_overlap0 = time.time()
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=1
                            ) as pool:
                                fut = pool.submit(_exec_and_collect)
                                t_gpu0 = time.time()
                                client.fdm_ground(cache_action, task_prompt)
                                next_flat, next_raw = client.infer_chunk(
                                    wrist_img, task_prompt
                                )
                                gpu_s = time.time() - t_gpu0
                                key_frames = fut.result()
                            overlap_s = time.time() - t_overlap0
                            infer_s = gpu_s
                            t_commit0 = time.time()
                            client.commit_real(
                                key_frames,
                                cache_action,
                                executed_temporals,
                                task_prompt,
                            )
                            commit_s = time.time() - t_commit0
                            pending_flat, pending_raw = next_flat, next_raw
                            print(
                                f"[DIAG][MODEL] async gpu(fdm+infer)={gpu_s:.3f}s "
                                f"overlap_wall={overlap_s:.3f}s "
                                f"commit_real={commit_s:.3f}s "
                                f"{'OVER_BUDGET' if (overlap_s + commit_s) > budget_s else 'ok'} "
                                f"next_flat={next_flat.shape} "
                                f"collected={len(key_frames)} "
                                f"executed_F={executed_temporals} "
                                f"cached_F={cache_action.shape[1]}"
                            )
                        elif client is not None:
                            env.exec_actions(
                                actions=abs_exec,
                                timestamps=action_timestamps,
                                dynamic_latency=False,
                            )
                            key_frames = _collect_keyframes_during_exec(
                                env,
                                n_exec=n_exec,
                                action_timestamps=action_timestamps,
                                actions_per_temporal=actions_per_temporal,
                                observation_delay=frame_latency,
                            )
                            (
                                executed_temporals,
                                cached_temporals,
                            ) = client.update_kv_cache(
                                key_frames,
                                raw_action,
                                n_exec,
                                task_prompt,
                            )
                            print(
                                f"[DIAG][CACHE] collected={len(key_frames)} "
                                f"executed_F={executed_temporals} "
                                f"cached_F={cached_temporals}"
                            )
                        else:
                            env.exec_actions(
                                actions=abs_exec,
                                timestamps=action_timestamps,
                                dynamic_latency=False,
                            )
                            # 固定动作调试没有 KV 关键帧采集，因此显式等待本轮
                            # waypoint 执行完成，避免下一轮立即覆盖尚未执行的轨迹。
                            precise_wait(
                                float(action_timestamps[-1]) + frame_latency,
                                time_func=time.time,
                            )

                        n_keep = int(sched["n_keep"])
                        if n_keep > 0:
                            # timestamp 单调递增；若前部过期，保留下来的是尾部 waypoint。
                            last_cmd_abs = abs_exec[-1].copy()
                        else:
                            last_cmd_abs = np.concatenate(
                                [cur_pos_robot, cur_rotvec, [cur_grip]]
                            ).astype(np.float64)
                        last_state_pos = cur_pos_robot.copy()
                        last_state_rotvec = cur_rotvec.copy()
                        last_state_grip = cur_grip
                        r_store = st.Rotation.from_rotvec(cur_rotvec)
                        prev_model_step_world = r_store.apply(
                            flat_chunk[0, :3].astype(np.float64)
                        )
                        prev_decode_step_world = r_store.apply(
                            action_chunk[0, :3].astype(np.float64)
                        )

                        vis_img = wrist_img.copy()
                        text = (
                            f"Episode: {episode_id}, "
                            f"Time: {time.monotonic() - t_start:.1f}s"
                        )
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

                        iter_idx += n_exec
                        cycle_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                finally:
                    if episode_active:
                        env.end_episode()

                print("Stopped. Returning to human control.")


if __name__ == "__main__":
    main()
