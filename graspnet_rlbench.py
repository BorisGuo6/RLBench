import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import OpenDoor, PutBooksOnBookshelf, ReachTarget, CloseBox, PutShoesInBox, PickAndLift, PickUpCup, OpenWineBottle, PushButton

import os
import sys
from typing import Dict, Tuple

import torch
from pytorch3d import transforms
from model import AnyGrasp
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
        writer.append_data(image)

    writer.close()
    print("Video created successfully!")


obs_config = ObservationConfig()
obs_config.set_all(True)
env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)
env.launch()

task = env.get_task(PushButton)

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

goal_action = anygrasp.step(obs)

# goal_action[:3] = np.array([0.3, 0.0, env._scene.robot.arm.get_position()[2]])
# goal_action[3:] = obs.gripper_pose[3:]

gripper = [1.0]
print('Action:', goal_action)
print('Current Gripper Pose:', obs.gripper_pose)
print('Distance:', np.linalg.norm(obs.gripper_pose - goal_action))
action = np.concatenate([obs.gripper_pose, gripper], axis=-1)

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

try:
    rot_start = R.from_quat(obs.gripper_pose[3:])
    rot_end = R.from_quat(goal_action[3:])
except ValueError as e:
    print(f"Error creating Rotation object: {e}")
    print("Please ensure input quaternions are valid and normalized.")
    exit()

num_intermediate_steps = steps # How many poses to generate BETWEEN start and end
num_total_poses = num_intermediate_steps + 2 # Total poses including start and end

# Setup Slerp for orientation
# Slerp needs key times (0 and 1 for start and end) and corresponding Rotation objects
key_times = [0, 1]
key_rots = R.concatenate([rot_start, rot_end])
slerp_interpolator = Slerp(key_times, key_rots)

# --- Generate Transitional Poses ---
transitional_poses_list = []
# Generate points from t=0 to t=1, inclusive
interpolation_times = np.linspace(0, 1, num_total_poses)

print(f"Start Pose: {obs.gripper_pose}")
print(f"End Pose:   {goal_action}")
print(f"\nGenerating {num_intermediate_steps} transitional poses:")

for i, t in enumerate(interpolation_times):
    interp_pos = (1 - t) * obs.gripper_pose[:3] + t * goal_action[:3]
    interp_rot = slerp_interpolator([t])[0]

    interp_quat_coeffs = interp_rot.as_quat() # x,y,z,w

    current_pose = list(interp_pos) + list(interp_quat_coeffs)
    transitional_poses_list.append(current_pose)
    

transitional_poses_list.append(transitional_poses_list[-1])

for i, pose in enumerate(transitional_poses_list):
    if i < len(transitional_poses_list) - 1:
        gripper = [1.0]
        cur_action = np.concatenate([pose, gripper], axis=-1)
    else:
        gripper = [0.0]
        cur_action = np.concatenate([pose, gripper], axis=-1)
    
        
    print('Action:', cur_action)
    try:
        obs, reward, terminate = task.step(cur_action)
        row_1 = np.concatenate([np.array(obs.front_rgb), np.array(obs.wrist_rgb), np.array(obs.overhead_rgb)], axis=1)
        row_2 = np.concatenate([np.array(obs.left_shoulder_rgb), np.array(obs.right_shoulder_rgb), np.array(obs.front_rgb)], axis=1)
        image_list.append(np.concatenate([row_1, row_2], axis=0))
    except Exception as e:
        images_to_video(image_list, os.path.join(SAVE_DIR, 'graspnet_error_test.mp4'), fps=60)
        print('Error during task step:', e)
        exit()


images_to_video(image_list, os.path.join(SAVE_DIR, 'graspnet_test.mp4'), fps=60)
print('Done')
env.shutdown()