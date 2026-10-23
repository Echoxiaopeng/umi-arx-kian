from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:, :3]
            rot = st.Rotation.from_rotvec(poses[:, 3:])

            self.pos_interp = si.interp1d(times, pos, axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:, :3] = self.pos_interp.y
            poses[:, 3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)

    def drive_to_waypoint(
        self, pose, time, curr_time, max_pos_speed=np.inf, max_rot_speed=np.inf
    ) -> "PoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)

        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(
        self,
        pose,
        time,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ) -> "PoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)

        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, int):
            is_single = True
            t = np.array([t])
        if isinstance(t, float):
            is_single = True
            t = np.array([t])

        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose


class JointTrajectoryInterpolator:
    """Linear interpolator for joint (+ gripper) waypoints with schedule_waypoint semantics."""

    def __init__(self, times: np.ndarray, joints: np.ndarray):
        assert len(times) >= 1
        assert len(joints) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim == 1:
            joints = joints.reshape(1, -1)
        assert joints.ndim == 2

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._joints = joints
            self._dim = joints.shape[1]
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self._dim = joints.shape[1]
            self.joint_interp = si.interp1d(
                times, joints, axis=0, assume_sorted=True
            )

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        return self.joint_interp.x

    @property
    def joints(self) -> np.ndarray:
        if self.single_step:
            return self._joints
        return np.asarray(self.joint_interp.y, dtype=np.float64)

    @property
    def dim(self) -> int:
        return self._dim

    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.unique(np.concatenate([[start_t], keep_times, [end_t]]))
        all_joints = self(all_times)
        if all_joints.ndim == 1:
            all_joints = all_joints.reshape(1, -1)
        return JointTrajectoryInterpolator(times=all_times, joints=all_joints)

    def drive_to_waypoint(
        self, joint, time, curr_time, max_joint_speed=np.inf
    ) -> "JointTrajectoryInterpolator":
        assert max_joint_speed > 0
        time = max(time, curr_time)
        curr_joint = np.asarray(self(curr_time), dtype=np.float64).reshape(-1)
        joint = np.asarray(joint, dtype=np.float64).reshape(-1)
        assert joint.shape == curr_joint.shape
        dist = float(np.linalg.norm(joint - curr_joint))
        duration = max(time - curr_time, dist / max_joint_speed)
        assert duration >= 0
        last_waypoint_time = curr_time + duration
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        joints = np.append(trimmed_interp.joints, [joint], axis=0)
        return JointTrajectoryInterpolator(times, joints)

    def schedule_waypoint(
        self,
        joint,
        time,
        max_joint_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ) -> "JointTrajectoryInterpolator":
        assert max_joint_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)
        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        duration = time - end_time
        end_joint = np.asarray(trimmed_interp(end_time), dtype=np.float64).reshape(-1)
        joint = np.asarray(joint, dtype=np.float64).reshape(-1)
        assert joint.shape == end_joint.shape
        dist = float(np.linalg.norm(joint - end_joint))
        duration = max(duration, dist / max_joint_speed)
        assert duration >= 0
        insert_time = end_time + duration
        times = np.append(trimmed_interp.times, [insert_time], axis=0)
        joints = np.append(trimmed_interp.joints, [joint], axis=0)
        return JointTrajectoryInterpolator(times, joints)

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, (int, float, numbers.Real)):
            is_single = True
            t = np.array([float(t)])
        elif not isinstance(t, np.ndarray):
            t = np.asarray(t, dtype=np.float64)

        if self.single_step:
            joints = np.broadcast_to(self._joints[0], (len(t), self._dim)).copy()
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            joints = np.asarray(self.joint_interp(t), dtype=np.float64)
            if joints.ndim == 1:
                joints = joints.reshape(1, -1)

        if is_single:
            return joints[0]
        return joints
