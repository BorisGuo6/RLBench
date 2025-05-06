import sys
import os
sys.path.append('/data/wbj/miniconda3/envs/rlbench/lib/python3.9/site-packages')

import numpy as np
import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import rlbench

import imageio
from PIL import Image

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

env = gym.make('rlbench/reach_target-vision-v0', render_mode="rgb_array")

SAVE_DIR = 'tmp'
os.makedirs(SAVE_DIR, exist_ok=True)

image_list = []
training_steps = 120
episode_length = 40

for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _, _ = env.step(env.action_space.sample())
    breakpoint()
    image_list.append(np.array(obs['front_rgb']))
    # print(obs.shape, reward)
    env.render()  # Note: rendering increases step time.

print('Done')
images_to_video(image_list, os.path.join(SAVE_DIR, 'video.mp4'), fps=30)
fps = benchmark_step(env, target_duration=10)
print(f"FPS: {fps:.2f}")
env.close()
