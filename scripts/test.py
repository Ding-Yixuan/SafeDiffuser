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
import time

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch
from diffuser.models.cbf_adapter import NeuralBarrierAdapter

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## enable CBF
USE_CBF = True
if USE_CBF:
    print("\n🚀 [System] 正在启动 CBF 安全护盾...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 初始化适配器 (保持原样，尊重 +1 偏移)
    adapter = NeuralBarrierAdapter(model_filename='cbf_maze2d.pth', device=device)
    
    # 2. 从当前物理环境提取墙壁 (用于 Adapter)
    maze_env = env.unwrapped 
    if hasattr(maze_env, 'maze_arr'):
        maze_arr = maze_env.maze_arr
        h, w = maze_arr.shape
        real_walls = []
        for r in range(h):
            for c in range(w):
                if maze_arr[r, c] == 10: 
                    # Adapter 需要 +1.0
                    real_walls.append([float(c) + 1.0, float(r) + 1.0])
        
        adapter.wall_centers_tensor = torch.tensor(
            real_walls, dtype=torch.float32, device=device
        )
        print(f"✅ [CBF] 已同步环境中的 {len(real_walls)} 个墙壁坐标 (Offset +1.0 retained)！")
        
        diffusion.neural_cbf = adapter
        print("🛡️ [CBF] 避障模块已挂载，物理路测准备就绪！\n")
    else:
        print("⚠️ [CBF] 警告：无法从环境获取 maze_arr，避障可能失效！")
else:
    if hasattr(diffusion, 'neural_cbf'):
        del diffusion.neural_cbf
    print("\n💀 [System] CBF 避障已关闭 (Baseline 模式)\n")

# ==========================================================

policy = Policy(diffusion, dataset.normalizer)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#---------------------------------- main loop ----------------------------------#
safe1_batch, safe2_batch = [], []
score_batch = []
comp_time = []
elbo_batch = []
success = 0
num = 10
runs_summary = []

# ==========================================
# 准备碰撞检测用的墙壁 (Referees)
# ==========================================
print("正在构建碰撞检测用的物理墙壁坐标 (Referees)...")
referee_walls = []

maze_env = env.unwrapped
if hasattr(maze_env, 'maze_arr'):
    hmap = maze_env.maze_arr
    hh, ww = hmap.shape
    for rr in range(hh):
        for cc in range(ww):
            if hmap[rr, cc] == 10: 
                # 🔥🔥🔥 终极修正：Row是X，Col是Y，且不加偏移 🔥🔥🔥
                # observation[0] = Row (rr)
                # observation[1] = Col (cc)
                
                phys_x = float(rr)
                phys_y = float(cc)
                
                referee_walls.append([phys_x, phys_y]) 
else:
    # 备用方案：如果必须从 Adapter 拿
    if USE_CBF and hasattr(diffusion, 'neural_cbf'):
        print("⚠️ 警告：正在从 Adapter 转换坐标 (Swap XY)...")
        adapter_walls = diffusion.neural_cbf.wall_centers_tensor.detach().cpu().numpy()
        # Adapter 存的是 [col+1, row+1]。我们需要 [row, col]
        # 所以先减 1，再交换列
        # Adapter: [x_adapt, y_adapt] = [c+1, r+1]
        # Target:  [x_phys, y_phys]   = [r, c]
        # So: x_phys = y_adapt - 1.0
        #     y_phys = x_adapt - 1.0
        referee_walls = adapter_walls[:, [1, 0]] - 1.0

walls_np = np.array(referee_walls, dtype=np.float32)
print(f"🛑 [监控开启] 已加载 {len(walls_np)} 个物理墙壁坐标 (已校准: Row->X, Col->Y)")

# ==========================================
# 主循环开始
# ==========================================
for iter in range(num):
    print(f"step: {iter}/{num}")

    observation = env.reset()
    observation = np.array([0.94875744, 8.93648809, -0.01347715, 0.06358764]) 
    env.set_state(observation[0:2], observation[2:4])

    if args.conditional:
        env.set_target()
    
    target = env._target
    print(f"目标点 (Target) 坐标: {target}")
    
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    rollout = [observation.copy()]
    total_reward = 0
    
    per_step_collisions = [] 
    per_step_min_d = []
    collided_flag = False
    min_dist_overall = float('inf')
    
    COLLISION_RADIUS = 0.60 

    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t == 0:
            cond[0] = observation
            start = time.time()
            action, samples, diffusion_paths, _, _, elbo = policy(cond, batch_size=args.batch_size)
            end = time.time()
            comp_time.append(end-start)
            elbo_batch.append(elbo)
            
            actions = samples.actions[0]
            sequence = samples.observations[0]
            diffusion_paths = diffusion_paths[0]

            if iter == num - 1:
                print("正在保存最后一次运行的可视化结果...")
                fullpath = join(args.savepath, f'final_plan_{iter}.png')
                renderer.composite(fullpath, samples.observations, ncol=1)
                diffusion_sm = diffusion_paths
                renderer.render_diffusion(join(args.savepath, f'final_diffusion.mp4'), diffusion_sm)
                diff_step = diffusion_sm.shape[0]  
                png_dir = join(args.savepath, 'final_png_sequence')
                makedirs(png_dir)

        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        
        # --- 碰撞检测 ---
        pos_xy = next_observation[:2].copy()
        
        if len(walls_np) > 0:
            # ✅ 使用修复后的变量名 walls_np
            dists = np.linalg.norm(walls_np - pos_xy, axis=1)
            min_d = float(np.min(dists))
        else:
            min_d = float('inf')

        per_step_min_d.append(min_d)
        
        if min_d < COLLISION_RADIUS:
            per_step_collisions.append(True)
            collided_flag = True
        else:
            per_step_collisions.append(False)

        if min_d < min_dist_overall:
            min_dist_overall = min_d

        rollout.append(next_observation.copy())
        if terminal:
            break

        observation = next_observation

    # 结算
    is_success = False
    if reward > 0.95:
        success = success + 1
        is_success = True
    
    score = env.get_normalized_score(total_reward)
    score_batch.append(score)

    status_icon = "✅" if (is_success and not collided_flag) else "❌"
    print(f"{status_icon} [Round {iter+1}/{num}] "
          f"Goal: {is_success} | "
          f"Safe: {'✅' if not collided_flag else '💥'} | "
          f"MinDist: {min_dist_overall:.3f}m | "
          f"Score: {score:.4f}")

    makedirs(args.savepath)
    run_diag = {
        'run': int(iter),
        'reached_goal': bool(is_success),
        'collided': bool(collided_flag),
        'min_distance_overall': float(min_dist_overall) if min_dist_overall != float('inf') else None,
        'score': float(score)
    }
    runs_summary.append(run_diag)

# 最终统计
elbo_batch = np.array(elbo_batch)
score_batch = np.array(score_batch)
comp_time = np.array(comp_time)

print("-" * 30)
print(f"Mode: {'SafeDiffuser (CBF On)' if USE_CBF else 'Baseline (CBF Off)'}")
print(f"Success Rate: {success}/{num}")
print(f"Average Score: {np.mean(score_batch):.4f}")
print("-" * 30)

try:
    makedirs(args.savepath)
    json.dump(runs_summary, open(join(args.savepath, 'runs_summary.json'), 'w'), indent=2)
except Exception as e:
    pass

json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)