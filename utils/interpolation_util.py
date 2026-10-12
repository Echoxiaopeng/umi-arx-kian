import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, axis=0, bounds_error=False, fill_value=(x[0], x[-1])
    )
    return gripper_interp


def nearest_rotvec_equivalent(rvec: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """在 2π 周期等价 rotvec 中选最接近 reference 的一个（SO(3) 不变）。

    ``Rotation.as_rotvec()`` 把转角落在 [0, π]；当真实姿态角靠近 π 时，
    相邻帧可能在 ``r`` 与 ``-r``（即 ``axis*(θ-2π)``）之间翻转，欧氏距离≈2π，
    但测地距离几乎为 0。用本函数可去掉这种表示突变。
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    reference = np.asarray(reference, dtype=np.float64).reshape(3)
    ang = float(np.linalg.norm(rvec))
    if ang < 1e-12:
        return rvec.copy()
    axis = rvec / ang
    best = rvec
    best_d = float(np.linalg.norm(rvec - reference))
    for k in range(-3, 4):
        cand = axis * (ang + 2.0 * np.pi * k)
        d = float(np.linalg.norm(cand - reference))
        if d < best_d:
            best_d = d
            best = cand
    return best


def continuize_rotvecs(rotvecs: np.ndarray) -> np.ndarray:
    """将 rotvec 序列在 R^3 中连续化，不改变每帧对应的姿态。"""
    out = np.asarray(rotvecs, dtype=np.float64)
    if out.ndim == 1:
        return out.copy()
    out = out.copy()
    if len(out) <= 1:
        return out
    for i in range(1, len(out)):
        out[i] = nearest_rotvec_equivalent(out[i], out[i - 1])
    return out


def continuize_action_rotvecs(actions: np.ndarray, n_robots: int) -> np.ndarray:
    """连续化 action 里每臂的 rotvec 段（每臂 7 维: xyz + rotvec + grip）。"""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0:
        return actions
    out = actions.copy()
    for robot_idx in range(int(n_robots)):
        sl = slice(7 * robot_idx + 3, 7 * robot_idx + 6)
        if out.shape[1] < sl.stop:
            break
        out[:, sl] = continuize_rotvecs(out[:, sl])
    return out


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:, :3]
        rot = st.Rotation.from_rotvec(x[:, 3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)

    @property
    def x(self):
        return self.pos_interp.x

    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = np.asarray(rot.as_rotvec(), dtype=np.float64)
        # Slerp 姿态连续，但 as_rotvec 在 ±π 处会翻转表示
        if rvec.ndim == 1:
            pose = np.concatenate([pos, rvec], axis=-1)
            return pose
        rvec = continuize_rotvecs(rvec)
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose


def get_gripper_calibration_interpolator(aruco_measured_width, aruco_actual_width):
    """
    Assumes the minimum width in aruco_actual_width
    is measured when the gripper is fully closed
    and maximum width is when the gripper is fully opened
    """
    aruco_measured_width = np.array(aruco_measured_width)
    aruco_actual_width = np.array(aruco_actual_width)
    assert len(aruco_measured_width) == len(aruco_actual_width)
    assert len(aruco_actual_width) >= 2
    aruco_min_width = np.min(aruco_actual_width)
    gripper_actual_width = aruco_actual_width - aruco_min_width
    interp = get_interp1d(aruco_measured_width, gripper_actual_width)
    return interp
