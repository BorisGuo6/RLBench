import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointVelocity, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutBooksOnBookshelf, ReachTarget, CloseBox, PutShoesInBox, PickAndLift, PickUpCup, OpenWineBottle
from rlbench.backend.robot import Robot
from pyquaternion import Quaternion

import os
import sys
from typing import Dict, Tuple

import torch
from pytorch3d import transforms
from model import AnyGrasp

# curobo
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.util_file import get_assets_path, join_path
from pyrep.objects import Object

import imageio
from PIL import Image
import open3d as o3d

SAVE_DIR = 'tmp'
os.makedirs(SAVE_DIR, exist_ok=True)
def images_to_video(images, video_path, frame_size=(1920, 1080), fps=30):
    if not images:
        print("No images found in the specified directory!")
        return

    writer = imageio.get_writer(video_path, fps=30)

    for image in images:

        if image.shape[1] > frame_size[0] or image.shape[0] > frame_size[1]:
            print("Warning: frame size is smaller than the one of the images.")
            print("Images will be resized to match frame size.")
            image = np.array(Image.fromarray(image).resize(frame_size))

        writer.append_data(image)

    writer.close()
    print("Video created successfully!")


def pose_in_robot_base_frame(robot: Robot, action: np.ndarray):
    act_pos, act_quat = action[:3], action[3:] # x,y,z,qx,qy,qz,qw
    act_quat = np.roll(act_quat, 1)
    
    act_rotation = transforms.quaternion_to_matrix(torch.tensor(act_quat)).numpy()
    
    # Construct transformation matrices
    T_world_action = np.eye(4)
    T_world_action[:3, :3] = act_rotation
    T_world_action[:3, 3] = act_pos
    
    T_world_robot = robot.arm.get_matrix()
    T_world_robot[:3, :3] = np.eye(3)
    T_robot_world = np.linalg.inv(T_world_robot)
    # get_matrix() see: https://github.com/stepjam/PyRep/blob/8f420be8064b1970aae18a9cfbc978dfb15747ef/pyrep/objects/object.py#L311
    
    # Transform the action pose into the robot base frame
    T_robot_action = T_robot_world @ T_world_action

    # Extract new pose in robot frame
    action_pos_robot = T_robot_action[:3, 3]
    action_rot_robot = T_robot_action[:3, :3]
    action_quat_robot = transforms.matrix_to_quaternion(torch.tensor(action_rot_robot)).numpy()  # qw, qx, qy, qz
    
    # Final robot-frame pose: (position, quaternion)
    pose = np.concatenate([action_pos_robot, action_quat_robot], axis=0)
    
    return pose

obs_config = ObservationConfig()
obs_config.set_all(True)
env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointPosition(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)
env.launch()

task = env.get_task(OpenWineBottle)

ckpt_path = '/home/wbj/wbj/graspnet-baseline/logs/log_rs/checkpoint-rs.tar'
anygrasp = AnyGrasp(ckpt_path=ckpt_path)

steps = 180
obs = None
image_list = []

print('Reset Episode')
descriptions, obs = task.reset()
row_1 = np.concatenate([np.array(obs.front_rgb), np.array(obs.wrist_rgb), np.array(obs.overhead_rgb)], axis=1)
row_2 = np.concatenate([np.array(obs.left_shoulder_rgb), np.array(obs.right_shoulder_rgb), np.array(obs.front_rgb)], axis=1)
image_list.append(np.concatenate([row_1, row_2], axis=0))
print(descriptions)

raw_action = anygrasp.step(obs)

# Option 1: Using curobo IK solver

tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

# convert action to robot coordinates

goal_action = pose_in_robot_base_frame(env._scene.robot, raw_action)

# goal_action = np.concatenate([raw_action[:3] - env._scene.robot.arm.get_position(), np.roll(raw_action[3:],1)], axis=0)


interpolation_dt = 0.01
collision_activation_distance = 0.02
# create motion gen with a cuboid cache to be able to load obstacles later:
motion_gen_cfg = MotionGenConfig.load_from_robot_config(
    "franka.yml",
    "collision_base.yml",
    tensor_args,
    num_ik_seeds=50,
    interpolation_dt=interpolation_dt,
    # collision_activation_distance=collision_activation_distance,
)
motion_gen = MotionGen(motion_gen_cfg)
motion_gen.warmup()

retract_cfg = motion_gen.get_retract_config()
# goal_pose = Pose(torch.tensor(goal_action[:3], dtype=torch.float32).cuda(), torch.tensor(goal_action[3:], dtype=torch.float32).cuda())
goal_pose = Pose(torch.tensor(goal_action[:3], dtype=torch.float32).cuda(), torch.tensor(goal_action[3:], dtype=torch.float32).cuda())
# goal_pose = Pose(torch.tensor([0.25,0.0,0.25], dtype=torch.float32).cuda(), torch.tensor(goal_action[3:], dtype=torch.float32).cuda())
q_start = JointState.from_position(
    tensor_args.to_device([obs.joint_positions]),
    joint_names=[
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
)

result = motion_gen.plan_single(q_start, goal_pose)
    
if result.success.item():
    # this contains a linearly interpolated trajectory with fixed dt
    interpolated_solution = result.get_interpolated_plan()
    print("Plannning Success")
else:
    print("Failed") 


for action in interpolated_solution.position:
    action = action.cpu().numpy()
    action = np.concatenate([action, np.array([1.0])], axis=0)
    print('Action: ', action)
    obs, reward, done = task.step(action)
    row_1 = np.concatenate([np.array(obs.front_rgb), np.array(obs.wrist_rgb), np.array(obs.overhead_rgb)], axis=1)
    row_2 = np.concatenate([np.array(obs.left_shoulder_rgb), np.array(obs.right_shoulder_rgb), np.array(obs.front_rgb)], axis=1)
    image_list.append(np.concatenate([row_1, row_2], axis=0))
    if done:
        break
    
action[-1] = 0.0
print('Action: ', action)
obs, reward, done = task.step(action)
row_1 = np.concatenate([np.array(obs.front_rgb), np.array(obs.wrist_rgb), np.array(obs.overhead_rgb)], axis=1)
row_2 = np.concatenate([np.array(obs.left_shoulder_rgb), np.array(obs.right_shoulder_rgb), np.array(obs.front_rgb)], axis=1)
image_list.append(np.concatenate([row_1, row_2], axis=0))
    
print('Distance between goal and current pose:', np.linalg.norm(obs.gripper_pose[:3] - raw_action[:3]))
print('Current Gripper Pose:', obs.gripper_pose)
images_to_video(image_list, os.path.join(SAVE_DIR, 'graspnet_curobo_test.mp4'), fps=30)
print('Done')
env.shutdown()
