from bimanual import SingleArm
from typing import Dict, Any
import numpy as np
import time

def test_single_arm(single_arm: SingleArm, duration: float = 10.0, dt: float = 0.01):
    #single_arm.go_home()
    # while(1):
    position = np.array([0.0, 0.0, 0.05])  # x, y, z 位置
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数表示方向

    success = single_arm.set_ee_pose(pos=position, quat=quaternion)

    time.sleep(1)

    single_arm.set_mode(3)

    # time.sleep(1)

    # single_arm.set_mode(5)
    # time.sleep(1)

    # single_arm.set_catch_pos(5.0)

    # success = single_arm.set_ee_pose(pos=position, quat=quaternion)
    
    count = 0
    while(count < 1000):
        print(f"testing ..., count: {count}")
        # print(single_arm.get_ee_pose())
        print(single_arm.get_joint_positions())
        time.sleep(1)
        count += 1


    # single_arm.set_catch_pos(0.0)
    # time.sleep(1)
    # print(single_arm.get_joint_positions())

    # single_arm.go_home()
    #print(single_arm.get_ee_pose())
    #print(single_arm.get_joint_positions())
        
    #positions = [0.5, 1.0, -0.5]  # 指定每个关节的位置
    #joint_names = ["joint1", "joint2", "joint3"]  # 对应关节的名称

    #success = single_arm.set_joint_positions(positions=positions, joint_names=joint_names)
        
if __name__ == "__main__":
    arm_config: Dict[str, Any] = {
        "can_port": "can1",
        "type": 0,
        # Add necessary configuration parameters for the left arm
    }
    single_arm = SingleArm(arm_config)
    test_single_arm(single_arm)