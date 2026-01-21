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
num=10
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
        




        score = env.get_normalized_score(total_reward)
        rollout.append(next_observation.copy())
    
        if terminal:
            break

        observation = next_observation

    if reward > 0.95:
        success = success + 1

    
    
    score_batch.append(score)


elbo_batch = np.array(elbo_batch)
print("elbo mean: ", np.mean(elbo_batch))
print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
comp_time = np.array(comp_time)

print("score mean: ", np.mean(score_batch))
print("score std: ", np.std(score_batch))
print("computation time: ", np.mean(comp_time))
print("success rate: ", success)

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