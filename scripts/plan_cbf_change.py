'''
/home/lqz27/anaconda3/envs/diffuser_to_use/bin/python /home/lqz27/dyx_ws/SafeDiffuser/scripts/plan_cdf.py \
    --dataset maze2d-large-v1 \
    --horizon 384 \
    --n_diffusion_steps 256 \
    --diffusion_epoch latest
'''
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
import json
import numpy as np
from os.path import join
import pdb
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch
from diffuser.models.cbf_adapter import NeuralBarrierAdapter

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-515
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

# ... (保留文件头部的 import 和 args = Parser... 以及 env 加载部分) ...

# ---------------------------------- Setup ---------------------------------- #
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# ---------------------------------- CBF Setup ---------------------------------- #
print("\n🚀 [System] 正在初始化 CBF 安全护盾...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adapter = NeuralBarrierAdapter(model_filename='cbf_maze2d.pth', device=device)

# 注入墙壁信息
maze_env = env.unwrapped 
if hasattr(maze_env, 'maze_arr'):
    maze_arr = maze_env.maze_arr
    h, w = maze_arr.shape
    real_walls = []
    for r in range(h):
        for c in range(w):
            if maze_arr[r, c] == 10: 
                real_walls.append([float(c) + 1.0, float(r) + 1.0])
    adapter.wall_centers_tensor = torch.tensor(real_walls, dtype=torch.float32, device=device)
    
    # 【关键修复】确保 CBF 挂载到真正的模型核心上 (穿透 EMA Wrapper)
    if hasattr(diffusion, 'model'):
        diffusion.model.neural_cbf = adapter
        print("🛡️ [CBF] 已挂载到 diffusion.model (内部核心)")
    else:
        diffusion.neural_cbf = adapter
        print("🛡️ [CBF] 已挂载到 diffusion (最外层)")
else:
    print("⚠️ [CBF] 警告：无法获取墙壁信息！")

policy = Policy(diffusion, dataset.normalizer)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# ---------------------------------- 自动调参锦标赛 ---------------------------------- #

import time
import numpy as np

# 定义选手 (参数组合)
# param_grid = [
#     {'name': '温和派',   'alpha': 0.01, 'clip': 0.01, 'threshold': 0.05},
#     {'name': '默认派',   'alpha': 0.05, 'clip': 0.01, 'threshold': 0.05},
#     {'name': '激进派',   'alpha': 0.10, 'clip': 0.02, 'threshold': 0.05},
#     {'name': '早鸟派',   'alpha': 0.05, 'clip': 0.01, 'threshold': 0.15}, # 提早介入
#     {'name': '强力派',   'alpha': 0.20, 'clip': 0.03, 'threshold': 0.05},
# ]
# 修改 plan_cbf.py 的 param_grid
# 🏅 ============ 最终排行榜 (按成功率排序) ============
# Name       | Alpha  | Clip   | Thres  | Success  | Score 
# -----------------------------------------------------------------
# 默认派        | 0.05   | 0.01   | 0.05   | 60%     | 0.88
# 温和派        | 0.01   | 0.01   | 0.05   | 60%     | 0.78
# 早鸟派        | 0.05   | 0.01   | 0.15   | 40%     | 0.58
# 强力派        | 0.2    | 0.03   | 0.05   | 40%     | 0.56
# 激进派        | 0.1    | 0.02   | 0.05   | 40%     | 0.46
param_grid = [
    # 原来的默认派 (作为基准)
    {'name': '默认派',   'alpha': 0.05, 'clip': 0.01,  'threshold': 0.05},
    
    # 新选手：手术刀派 (专门钻窄缝)
    # Threshold=0.02: 允许贴到极近 (胆大)
    # Clip=0.005: 每次只修一点点，防止撞到对面的墙 (心细)
    # Alpha=0.08: 稍微加点力，保证贴墙时能撑住
    {'name': '手术刀',   'alpha': 0.08, 'clip': 0.005, 'threshold': 0.02},
    
    # 新选手：动态衰减派 (模拟时间变化)
    # 这一组我们很难直接在这里写死，通常需要改代码逻辑
    # 但可以试一组 Alpha 很大但 Threshold 极小的“极限操作”
    {'name': '极限派',   'alpha': 0.15, 'clip': 0.01,  'threshold': 0.01},
]
# 🏅 ============ 最终排行榜 (按成功率排序) ============
# Name       | Alpha  | Clip   | Thres  | Success  | Score 
# -----------------------------------------------------------------
# 极限派        | 0.15   | 0.01   | 0.01   | 80%     | 1.28
# 默认派        | 0.05   | 0.01   | 0.05   | 70%     | 0.73
# 手术刀        | 0.08   | 0.005  | 0.02   | 50%     | 0.58
results = [] 
n_episodes_per_config = 10 

print(f"\n🏆 开始自动调参锦标赛！共有 {len(param_grid)} 位选手参赛。")
print(f"每位选手测试 {n_episodes_per_config} 次。\n")

for p_idx, config in enumerate(param_grid):
    config_name = config['name']
    print(f"👉 [选手 {p_idx+1}/{len(param_grid)}]: {config_name} | 参数: {config}")
    
    # 【关键修复】将参数注入到真正的模型核心 (穿透 EMA Wrapper)
    # 无论是外壳还是内核，全都赋上值，确保万无一失
    diffusion.cbf_config = config
    if hasattr(diffusion, 'model'):
        diffusion.model.cbf_config = config
    
    success_count = 0
    scores = []
    
    for iter in range(n_episodes_per_config):
        # 1. 环境重置
        observation = env.reset()    
        # 强制设置起点
        observation = np.array([0.94875744,  2.93648809, -0.01347715,  0.06358764])   
        env.set_state(observation[0:2], observation[2:4]) 

        # 2. 目标点设置 (保持和你之前能跑通的代码一致)
        if args.conditional:
            env.set_target() # 确保环境内部目标也重置
        
        target = env._target
        cond = {
            diffusion.horizon - 1: np.array([*target, 0, 0]),
        }
        
        # 3. 开始 Episode
        total_reward = 0
        terminal = False
        
        # 预先生成轨迹
        cond[0] = observation
        # 注意：这里接收参数用下划线忽略不需要的返回值，防止报错
        _, samples, _, _, _, _ = policy(cond, batch_size=args.batch_size)
        sequence = samples.observations[0]
        
        # 4. 执行控制
        for t in range(env.max_episode_steps):
            if t < len(sequence) - 1:
                next_waypoint = sequence[t+1]
            else:
                next_waypoint = sequence[-1].copy()
                next_waypoint[2:] = 0
            
            # 简单的 PD Controller
            action = next_waypoint[:2] - env.state_vector()[:2] + (next_waypoint[2:] - env.state_vector()[2:])
            
            next_observation, reward, terminal, _ = env.step(action)
            total_reward += reward
            
            if terminal:
                break
        
        # 5. 结算
        score = env.get_normalized_score(total_reward)
        scores.append(score)
        
        # 判定成功 (距离目标非常近)
        if reward > 0.95:
            success_count += 1
            
        print(".", end="", flush=True)

    avg_score = np.mean(scores)
    success_rate = success_count / n_episodes_per_config
    
    print(f"\n   🏁 结果: 成功率 {success_rate*100:.0f}% | 平均分 {avg_score:.2f}")
    
    results.append({
        'Name': config_name,
        'Alpha': config['alpha'],
        'Clip': config['clip'],
        'Thres': config['threshold'],
        'Success': success_rate,
        'Score': avg_score
    })
    print("-" * 50)

# ---------------------------------- 打印最终排行榜 ---------------------------------- #
print("\n🏅 ============ 最终排行榜 (按成功率排序) ============")
sorted_results = sorted(results, key=lambda x: (x['Success'], x['Score']), reverse=True)

print(f"{'Name':<10} | {'Alpha':<6} | {'Clip':<6} | {'Thres':<6} | {'Success':<8} | {'Score':<6}")
print("-" * 65)
for r in sorted_results:
    print(f"{r['Name']:<10} | {r['Alpha']:<6} | {r['Clip']:<6} | {r['Thres']:<6} | {r['Success']*100:.0f}%     | {r['Score']:.2f}")

# 脚本结束，不再执行后面的旧代码
import sys
sys.exit(0)