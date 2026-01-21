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

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## enable CBF
USE_CBF = True
if USE_CBF:
    print("\n启动 CBF")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 初始化适配器
    adapter = NeuralBarrierAdapter(model_filename='cbf_maze2d.pth', device=device)
    
    # 2. 从当前物理环境提取墙壁 
    maze_env = env.unwrapped 
    if hasattr(maze_env, 'maze_arr'):
        maze_arr = maze_env.maze_arr
        h, w = maze_arr.shape
        real_walls = []
        
        # 解析墙壁坐标
        for r in range(h):
            for c in range(w):
                if maze_arr[r, c] == 10: # 10 代表墙
                    # 坐标变换：(col + 1.0, row + 1.0)
                    real_walls.append([float(c) + 1.0, float(r) + 1.0])
        
        # 3. 注入墙壁数据到 CBF
        adapter.wall_centers_tensor = torch.tensor(
            real_walls, dtype=torch.float32, device=device
        )
        print(f"同步环境中的 {len(real_walls)} 个墙壁坐标")
        
        # 4. 挂载到 Diffusion 模型上
        diffusion.neural_cbf = adapter
        print("CBF已挂载\n")
    else:
        print("无法从环境获取 maze_arr")
else:
    # 确保清空，防止意外残留
    if hasattr(diffusion, 'neural_cbf'):
        del diffusion.neural_cbf
    print("\nCBF关闭\n")
# ==========================================================

policy = Policy(diffusion, dataset.normalizer)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def smooth(diffusion):
    steps, horizon = diffusion.shape[0], diffusion.shape[1]
    diffusion_copy = diffusion.copy()
    for i in range(steps - 20, steps, 1):
        for j in range(5, horizon, 1):
            diffusion_copy[i,j,0:2] = np.mean(diffusion[i, j-5:j, 0:2], axis=0)
    
    return diffusion_copy

#---------------------------------- main loop ----------------------------------#
score_batch = []
comp_time = []
elbo_batch = []
success = 0
import time
num = 30
best_safe_margin = -1.0
best_score = -float('inf')
best_trajectory = None
runs_summary = []
# ==============================================================================
# 2. 碰撞检测
# ==============================================================================
print("正在构建碰撞检测用的物理墙壁坐标")
referee_walls = []

maze_env = env.unwrapped
if hasattr(maze_env, 'maze_arr'):
    hmap = maze_env.maze_arr
    hh, ww = hmap.shape
    for rr in range(hh):
        for cc in range(ww):
            if hmap[rr, cc] == 10: 
                # Observation X = Map Row, Observation Y = Map Col
                phys_x = float(rr)
                phys_y = float(cc)
                referee_walls.append([phys_x, phys_y]) 
else:

    if USE_CBF and hasattr(diffusion, 'neural_cbf'):
        print("正在从 Adapter 转换坐标 (Swap XY)...")
        adapter_walls = diffusion.neural_cbf.wall_centers_tensor.detach().cpu().numpy()
        # Adapter: [c+1, r+1] -> 减1 -> [c, r] -> 交换 -> [r, c]
        referee_walls = adapter_walls[:, [1, 0]] - 1.0

walls_np = np.array(referee_walls, dtype=np.float32)
print(f"{len(walls_np)} 个物理墙壁坐标 (Row->X, Col->Y)")



for iter in range(num):   # num of testing runs
    print("step: ", iter, "/100")

    observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    observation = np.array([0.94875744,  8.93648809, -0.01347715,  0.06358764])   # fix the initial position and final destination for comparison (not needed for general testing)
    env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    # target = np.array([7.0, 1.0])
    print(f"目标点 (Target) 坐标: {target}")
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0

    # --- 诊断变量初始化 ---
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
            current_trajectory = diffusion_paths[0]
            current_samples = samples.observations
            actions = samples.actions[0]
            sequence = samples.observations[0]
            diffusion_paths = diffusion_paths[0]

            # 添加了10次循环的保存的逻辑
            if iter == num - 1:
                print("正在保存最后一次运行的可视化结果...")
                
                # 保存规划的轨迹图
                fullpath = join(args.savepath, f'final_plan_{iter}.png')
                renderer.composite(fullpath, samples.observations, ncol=1)
                
                # 保存视频
                diffusion_sm = diffusion_paths
                renderer.render_diffusion(join(args.savepath, f'final_diffusion.mp4'), diffusion_sm)

                # 保存每一帧
                diff_step = diffusion_sm.shape[0]  
                png_dir = join(args.savepath, 'final_png_sequence')
                makedirs(png_dir)
                for kk in range(diff_step):
                    imgpath = join(png_dir, f'{kk}.png')
                    renderer.composite(imgpath, diffusion_sm[kk:kk+1], ncol=1)

        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward


        # 碰撞检测
        pos_xy = next_observation[:2].copy() # 机器人的真实物理坐标
        
        if len(walls_np) > 0:
            # 计算距离：使用校准后的 referee walls (walls_np)
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


        score = env.get_normalized_score(total_reward)
        rollout.append(next_observation.copy())
    
        if terminal:
            break

        observation = next_observation


#     print(f"Iter {iter}: Score = {score}")


#     # 如果当前分数比历史最高分高，或者这是第一次运行
#     if score > best_score:
#         print(f"🌟 发现更好的路径！分数从 {best_score} 提升到 {score}，正在保存...")
#         best_score = score
        
#         # 保存这个最好的结果（覆盖写入，始终保留最好的）
#         fullpath = join(args.savepath, 'best_plan.png')
#         renderer.composite(fullpath, current_samples, ncol=1)
        
#         renderer.render_diffusion(join(args.savepath, 'best_diffusion.mp4'), current_trajectory)
#         print("最佳结果已保存")

# # 本轮总结
#     is_success = False
#     if reward > 0.95:
#         success = success + 1
#         is_success = True

    
    
#     score_batch.append(score)

#     # 打印本轮结果
#     status_icon = "✅" if (is_success and not collided_flag) else "❌"
#     print(f"{status_icon} [Round {iter+1}/{num}] "
#           f"Goal: {is_success} | "
#           f"Safe: {'✅' if not collided_flag else '❌'} | "
#           f"MinDist: {min_dist_overall:.3f}m | "
#           f"Score: {score:.4f}")

#     # 保存单轮诊断数据
#     makedirs(args.savepath)
#     run_diag = {
#         'run': int(iter),
#         'reached_goal': bool(is_success),
#         'collided': bool(collided_flag),
#         'collision_steps': [int(i) for i, v in enumerate(per_step_collisions) if v] if len(per_step_collisions) > 0 else [],
#         'min_distance_overall': float(min_dist_overall) if min_dist_overall != float('inf') else None,
#         'score': float(score)
#     }
#     runs_summary.append(run_diag)

# 1. 先判断本轮是否成功 (提到保存逻辑之前)
    is_success = False
    if reward > 0.95:
        is_success = True
    all_runs_dir = join(args.savepath, 'all_runs_vis')
    makedirs(all_runs_dir)
    status_str = "OK" if is_success else "FAIL"
    img_filename = f'run_{iter:03d}_{status_str}_score_{score:.2f}.png'
    
    # 保存图片 (current_samples 是扩散模型生成的规划路径)
    renderer.composite(join(all_runs_dir, img_filename), current_samples, ncol=1)

    # 2. [核心修改] 保存逻辑：优先 Success，其次 Safety (MinDist)
    # 逻辑：必须成功，且 (当前的最小距离 > 历史最好的最小距离)
    if is_success:
        if min_dist_overall > best_safe_margin:
            print(f"🌟 发现更安全的成功路径！Run {iter}: MinDist 从 {best_safe_margin:.4f}m 提升到 {min_dist_overall:.4f}m (Score: {score:.4f})")
            best_safe_margin = min_dist_overall
            
            # 保存结果
            fullpath = join(args.savepath, 'best_safe_plan.png') # 改个名区分
            renderer.composite(fullpath, current_samples, ncol=1)
            
            renderer.render_diffusion(join(args.savepath, 'best_safe_diffusion.mp4'), current_trajectory)
            print("最佳安全结果已保存")
    else:
        # 如果没成功，即使距离很远也不保存（或者你可以保留一个 best_score 的备选逻辑，但为了纯粹性这里不加）
        pass

    print(f"Iter {iter}: Score = {score}, Success = {is_success}, MinDist = {min_dist_overall:.4f}")

    # 3. 统计计数 (修复了之前的重复计数问题)
    if is_success:
        success = success + 1
    
    score_batch.append(score)

    # 打印本轮详细状态
    status_icon = "✅" if (is_success and not collided_flag) else "❌"
    print(f"{status_icon} [Round {iter+1}/{num}] "
          f"Goal: {is_success} | "
          f"Safe: {'✅' if not collided_flag else '❌'} | "
          f"MinDist: {min_dist_overall:.3f}m | "
          f"Score: {score:.4f}")

    # 保存单轮诊断数据
    makedirs(args.savepath)
    run_diag = {
        'run': int(iter),
        'reached_goal': bool(is_success),
        'collided': bool(collided_flag),
        'collision_steps': [int(i) for i, v in enumerate(per_step_collisions) if v] if len(per_step_collisions) > 0 else [],
        'min_distance_overall': float(min_dist_overall) if min_dist_overall != float('inf') else None,
        'score': float(score)
    }
    runs_summary.append(run_diag)

elbo_batch = np.array(elbo_batch)
print("elbo mean: ", np.mean(elbo_batch))
print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
comp_time = np.array(comp_time)

print("score mean: ", np.mean(score_batch))
print("score std: ", np.std(score_batch))
print("computation time: ", np.mean(comp_time))
print("success rate: ", success)
if best_safe_margin > 0:
    print(f"Best Safe Margin (in successful runs): {best_safe_margin:.4f}m")
else:
    print("No successful runs recorded.")
exit()


import pdb; pdb.set_trace()

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

print("-" * 30)
print(f"Mode: {'SafeDiffuser (CBF On)' if USE_CBF else 'Baseline (CBF Off)'}")
print(f"Success Rate: {success}/{iter+1}")
print(f"Average Score: {np.mean(score_batch):.4f}")
print("-" * 30)