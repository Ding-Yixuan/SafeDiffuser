import os
# 1. 【修复】设置无头模式，防止 X11/Display 报错
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import numpy as np
import d4rl.pointmaze
from gym.envs.registration import register

# 定义地图 (保持列表格式比较好看，后面我们会转换)
CUSTOM_MAZE1 = [
    "#######",
    "#OOOOO#",
    "#O#OOO#",
    "#OOOOO#",
    "#OOO#O#",
    "#OOO#G#",
    "#######"
]

# 临时注册一个测试环境
env_name = 'maze2d-custom-calibration-v0'

# 2. 【修复】将列表转换为 D4RL 需要的字符串格式
# D4RL 要求地图是一个长字符串，用反斜杠 "\" 分隔每一行
maze_str = "\\".join(CUSTOM_MAZE1)

try:
    register(
        id=env_name,
        entry_point='d4rl.pointmaze:MazeEnv',
        max_episode_steps=450, 
        kwargs={
            'maze_spec': maze_str,  # <--- 这里传入转换后的字符串
            'reward_type': 'dense',
            'reset_target': False,
            'ref_min_score': 0.0, 
            'ref_max_score': 1.0, 
            'dataset_url': ''
        }
    )
except gym.error.Error:
    pass # 如果已经注册过就跳过

# 跑随机策略
def get_random_score(env_name, num_episodes=10000):
    print(f"正在加载环境: {env_name} ...")
    env = gym.make(env_name)
    env.seed(0)
    
    scores = []
    
    print(f"🚀 开始计算 Random Score (运行 {num_episodes} 轮)...")
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 随机动作
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # 防止死循环（有些环境done判定有问题）
            if info.get('TimeLimit.truncated', False):
                done = True
                
        scores.append(total_reward)
        if (i+1) % 10 == 0:
            print(f"  Episode {i+1}/{num_episodes}: Reward = {total_reward:.4f}")
            
    avg_score = np.mean(scores)
    print(f"\n✅ ref_min_score (Random) 计算结果: {avg_score:.5f}")
    return avg_score

if __name__ == "__main__":
    get_random_score(env_name)