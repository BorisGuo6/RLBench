import torch
import os
import time
import numpy as np
import csv
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from scipy.spatial.transform import Rotation as R

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import OpenWineBottle, TakeFrameOffHanger, TakePlateOffColoredDishRack, RemoveCups, CloseBox, OpenDoor, OpenDrawer
from rlbench.tasks import StraightenRope,  InsertOntoSquarePeg, PlaceShapeInShapeSorter, PlugChargerInPowerSupply, PutKnifeInKnifeBlock, PutShoesInBox, SlideCabinetOpenAndPlaceCups
from rlbench.observation_config import ObservationConfig
from rlbench.backend.exceptions import InvalidActionError


# ==== 初始化 OpenVLA 模型 ====
print("Loading OpenVLA model...")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True, local_files_only=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
).to("cuda:0")
print("Model loaded.")

# 定义任务列表
task_list = [
    # OpenWineBottle,
    # TakeFrameOffHanger,
    # TakePlateOffColoredDishRack,
    # RemoveCups,
    # CloseBox,
    # OpenDoor,
    # OpenDrawer,
    # StraightenRope,
    # InsertOntoSquarePeg,
    # PlaceShapeInShapeSorter,
    PlugChargerInPowerSupply,
    # PutKnifeInKnifeBlock,
    # PutShoesInBox,
    # SlideCabinetOpenAndPlaceCups
]

# ==== 初始化 RLBench 环境 ====
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.left_shoulder_camera.rgb = True
obs_config.left_shoulder_camera.depth = False

action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaIK(),
    gripper_action_mode=Discrete()
)

print("Initializing RLBench environment...")
env = Environment(action_mode, obs_config=obs_config, headless=False)
env.launch()

# 创建结果保存目录
results_dir = "./task_results"
os.makedirs(results_dir, exist_ok=True)

# 准备结果CSV文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(results_dir, f"openvla_results_{timestamp}.csv")

# 测试参数
num_episodes = 50  # 每个任务测试的次数
max_steps_per_episode = 60

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Task Name', 'Success Count', 'Total Episodes', 'Success Rate(%)'])
    
    # 遍历测试每个任务
    for task_class in task_list:
        task_name = task_class.__name__
        print(f"\n===== 开始测试任务: {task_name} =====")
        
        try:
            # 获取当前任务
            task = env.get_task(task_class)
            success_count = 0
            
            # 测试当前任务多次
            for ep in range(num_episodes):
                print(f"\n[任务: {task_name}] [Episode {ep+1}/{num_episodes}]")
                descriptions, obs = task.reset()
                print("descriptions:", descriptions)
                
                for step in range(max_steps_per_episode):
                    print(f"执行步骤 {step+1}/{max_steps_per_episode}")
                    
                    # 获取图像
                    img_array = obs.left_shoulder_rgb
                    img = Image.fromarray(img_array)
                    
                    # 动态生成提示，包含任务名称
                    prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"
                    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
                    
                    # 预测动作
                    with torch.no_grad():
                        action = vla.predict_action(
                            **inputs, 
                            unnorm_key="fractal20220817_data",
                            do_sample=False
                        )
                    
                    # 处理动作输出
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    print("Raw OpenVLA action:", action)
                    
                    # 解析动作
                    delta_pos = action[:3]
                    delta_euler = action[3:6]
                    gripper_action = action[6]
                    
                    # 获取当前机器人姿态
                    current_pose = env._robot.arm.get_tip().get_pose()
                    print("current_pose:", current_pose)
                    current_pos = current_pose[:3]
                    current_quat = current_pose[3:]
                    
                    # 计算新位置
                    new_pose = current_pos + delta_pos
                    
                    # 检查是否在工作空间内
                    if not env._scene.check_target_in_workspace(new_pose):
                        print("new_pose is out of workspace")
                        continue
                    
                    # 计算新的旋转
                    delta_rot = R.from_euler('xyz', delta_euler)
                    current_rot = R.from_quat(current_quat)
                    new_rot = delta_rot * current_rot
                    new_quat = new_rot.as_quat()
                    
                    # 归一化四元数
                    new_quat = new_quat / np.linalg.norm(new_quat)
                    print("new_quat:", new_quat)
                    
                    if not np.isclose(np.linalg.norm(new_quat), 1.0):
                        print('Action contained non unit quaternion!')
                        new_quat = new_quat / np.linalg.norm(new_quat)
                    
                    # 组合最终动作
                    action_rlbench = np.concatenate([new_pose, new_quat, np.array([gripper_action])])
                    print(f"[Step {step+1}] 执行动作:", action_rlbench)
                    
                    # 执行动作
                    try:
                        obs, reward, terminate = task.step(action_rlbench)
                        
                        # 检查任务是否完成
                        if terminate:
                            print(f"[Episode {ep+1}] 在第 {step+1} 步成功完成任务!")
                            success_count += 1
                            break
                        elif step == max_steps_per_episode - 1:
                            print(f"[Episode {ep+1}] 失败: 达到最大步数 {max_steps_per_episode}")
                    except InvalidActionError as e:
                        print(f"IK解算失败: {e}")
                        print("跳过当前动作，继续下一步")
            
            # 计算当前任务的成功率
            success_rate = (success_count / num_episodes) * 100
            
            # 输出结果
            print(f"\n=== {task_name} 评估结果 ===")
            print(f"尝试次数: {num_episodes}")
            print(f"成功次数: {success_count}")
            print(f"成功率: {success_rate:.1f}%")
            
            # 将结果写入CSV
            csv_writer.writerow([task_name, success_count, num_episodes, f"{success_rate:.1f}"])
            csvfile.flush()  # 确保立即写入
            
        except Exception as e:
            # 处理任务执行中的错误
            print(f"测试任务 {task_name} 时出错: {e}")
            csv_writer.writerow([task_name, "ERROR", num_episodes, "0.0"])
            csvfile.flush()

print(f"\n所有任务测试完成! 结果已保存至: {csv_filename}")
env.shutdown()