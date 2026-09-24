"""UMI pose helpers shared by OpenPI eval and UmiArxEnv.

State: episode-relative SE(3) as pos(3)+rot6d(6)+gripper (10).
Action: current-EEF-relative waypoint; execute with T_abs = T_curr @ T_rel.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = values[:3], values[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return matrix[:2, :].reshape(6).copy()


def pose9_to_mat(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float64).reshape(9)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot6d_to_matrix(pose9[3:9])
    mat[:3, 3] = pose9[:3]
    return mat


def mat_to_pose9(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return np.concatenate([mat[:3, 3], matrix_to_rot6d(mat[:3, :3])], axis=0)


def invert_mat(T: np.ndarray) -> np.ndarray:
    R_mat = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_mat.T
    out[:3, 3] = -R_mat.T @ t
    return out


def pos_rotvec_to_mat(pos: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()
    mat[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return mat


def mat_to_pos_rotvec(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    pos = mat[:3, 3].copy()
    rotvec = R.from_matrix(mat[:3, :3]).as_rotvec()
    return pos, rotvec


def shift_tcp_along_z(pos: np.ndarray, rotvec: np.ndarray, dz: float) -> np.ndarray:
    """p' = p + R_tcp @ [0, 0, dz]; rotation unchanged."""
    if abs(float(dz)) < 1e-12:
        return np.asarray(pos, dtype=np.float64).reshape(3).copy()
    r = R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3))
    return np.asarray(pos, dtype=np.float64).reshape(3) + r.apply(np.array([0.0, 0.0, float(dz)]))


def x5_eef_to_link6(pos: np.ndarray, rotvec: np.ndarray, eef_offset: float) -> np.ndarray:
    """ZMQ eef_link → link6 (policy / training frame)."""
    return shift_tcp_along_z(pos, rotvec, -float(eef_offset))


def link6_to_x5_eef(pos: np.ndarray, rotvec: np.ndarray, eef_offset: float) -> np.ndarray:
    """link6 → ZMQ eef_link for exec_actions."""
    return shift_tcp_along_z(pos, rotvec, float(eef_offset))


def episode_rel_state10(
    pos: np.ndarray,
    rotvec: np.ndarray,
    grip: float,
    *,
    ref_pos: np.ndarray,
    ref_rotvec: np.ndarray,
) -> np.ndarray:
    """state = se3_to_vec(inv(T_0) @ T_t) + gripper → (10,)."""
    T_0 = pos_rotvec_to_mat(ref_pos, ref_rotvec)
    T_t = pos_rotvec_to_mat(pos, rotvec)
    T_rel = invert_mat(T_0) @ T_t
    return np.concatenate([mat_to_pose9(T_rel), [float(grip)]]).astype(np.float32)


def rel_action10_to_abs7(
    action10: np.ndarray,
    *,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
) -> np.ndarray:
    """T_abs = T_curr @ T_rel; return [xyz, rotvec, grip] (7,)."""
    a = np.asarray(action10, dtype=np.float64).reshape(10)
    T_curr = pos_rotvec_to_mat(cur_pos, cur_rotvec)
    T_rel = pose9_to_mat(a[:9])
    T_abs = T_curr @ T_rel
    pos, rotvec = mat_to_pos_rotvec(T_abs)
    out = np.zeros(7, dtype=np.float64)
    out[:3] = pos
    out[3:6] = rotvec
    out[6] = a[9]
    return out


def rel_chunk_to_abs7(
    action_chunk: np.ndarray,
    *,
    cur_pos: np.ndarray,
    cur_rotvec: np.ndarray,
) -> np.ndarray:
    """Convert (N, 10) current-EEF-relative chunk to (N, 7) absolute TCP.

    Each waypoint is relative to the *same* observation-time EEF (not chained
    body-delta). Matches RelCurrentEefActions training: inv(T_t) @ T_{t+1+k}.
    """
    chunk = np.asarray(action_chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 10:
        raise ValueError(f"Expected action chunk (N, 10), got {chunk.shape}")
    out = np.zeros((chunk.shape[0], 7), dtype=np.float64)
    for i in range(chunk.shape[0]):
        out[i] = rel_action10_to_abs7(chunk[i], cur_pos=cur_pos, cur_rotvec=cur_rotvec)
    return out


def invert_gripper_width(
    grip: np.ndarray | float,
    *,
    gripper_min_width: float,
    gripper_max_width: float,
) -> np.ndarray:
    """Dataset (large=close) → ZMQ (large=open)."""
    return gripper_min_width + gripper_max_width - np.asarray(grip, dtype=np.float64)


def postprocess_gripper(
    grip: np.ndarray | float,
    *,
    gripper_min_width: float,
    gripper_max_width: float,
    invert: bool = True,
    binarize: bool = True,
    close_threshold: float = 0.05,
    grasp_width: float | None = None,
) -> np.ndarray:
    """策略夹爪 → ZMQ 下发开度。

    invert: 训练极性（大=夹紧）↔ ZMQ（大=张开）。
    binarize: 夹取加紧——ZMQ 开度 < close_threshold 时改发 grasp_width，
              张开侧保持连续值（不是硬二值开/关）。
    """
    g = np.asarray(grip, dtype=np.float64).copy()
    if invert:
        g = invert_gripper_width(
            g, gripper_min_width=gripper_min_width, gripper_max_width=gripper_max_width
        )
    target_grasp = (
        float(gripper_min_width) if grasp_width is None else float(grasp_width)
    )
    if binarize:
        g = np.where(g < close_threshold, target_grasp, g)
    exec_min = min(float(gripper_min_width), target_grasp)
    exec_min = max(0.0, exec_min)
    return np.clip(g, exec_min, gripper_max_width)


def to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img
