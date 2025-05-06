import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

import torch
# from graspnetAPI import GraspGroup
# from graspnet import GraspNet, pred_decode

import os
import sys
import imageio
from PIL import Image

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

# class AnyGrasp:
#     def __init__(self, ckpt_path):
#         self.ckpt_path = ckpt_path
#         self.net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
#             cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.net.to(self.device)
#         # Load checkpoint
#         checkpoint = torch.load(self.ckpt_path)
#         self.net.load_state_dict(checkpoint['model_state_dict'])
#         self.start_epoch = checkpoint['epoch']
#         print("-> loaded checkpoint %s (epoch: %d)"%(self.ckpt_path, self.start_epoch))
#         # set model to eval mode
#         self.net.eval()
        
#     def process_input(self, obs):
#         pass
    
#     def predict_grasp(self, obs):
#         processed_obs = self.process_input(obs)
       
#         with torch.no_grad():
#             pred = self.net(processed_obs)
#             grasp_group = pred_decode(pred, self.device)

#         return grasp_group
    
class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(env.action_shape)

training_steps = 120
episode_length = 40
obs = None
image_list = []
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action = agent.act(obs)
    print(action)
    obs, reward, terminate = task.step(action)
    image_list.append(np.array(obs.front_rgb))

images_to_video(image_list, os.path.join(SAVE_DIR, 'single_task_rl.mp4'), fps=30)
print('Done')
env.shutdown()
