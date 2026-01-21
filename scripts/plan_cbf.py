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

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## enable CBF
USE_CBF = True
if USE_CBF:
    print("\n🚀 [System] 正在启动 CBF 安全护盾...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 初始化适配器
    # 确保 'cbf_maze2d.pth' 在正确路径，或者写绝对路径
    adapter = NeuralBarrierAdapter(model_filename='cbf_maze2d.pth', device=device)
    
    # 2. 从当前物理环境提取墙壁 (完全复用 compare_cbf 的逻辑)
    # env.unwrapped 确保拿到最底层的 Maze 环境
    maze_env = env.unwrapped 
    if hasattr(maze_env, 'maze_arr'):
        maze_arr = maze_env.maze_arr
        h, w = maze_arr.shape
        real_walls = []
        
        # 解析墙壁坐标 (这里的逻辑必须和训练/画图时完全一致)
        for r in range(h):
            for c in range(w):
                if maze_arr[r, c] == 10: # 10 代表墙
                    # 坐标变换：(col + 1.0, row + 1.0)
                    real_walls.append([float(c) + 1.0, float(r) + 1.0])
        
        # 3. 注入墙壁数据到 CBF
        adapter.wall_centers_tensor = torch.tensor(
            real_walls, dtype=torch.float32, device=device
        )
        print(f"✅ [CBF] 已同步环境中的 {len(real_walls)} 个墙壁坐标！")
        
        # 4. [最关键的一步] 挂载到 Diffusion 模型上
        # 只要这一步做了，diffusion.p_sample_loop 里就会自动调用它
        diffusion.neural_cbf = adapter
        print("🛡️ [CBF] 避障模块已挂载，物理路测准备就绪！\n")
    else:
        print("⚠️ [CBF] 警告：无法从环境获取 maze_arr，避障可能失效！")
else:
    # 确保清空，防止意外残留
    if hasattr(diffusion, 'neural_cbf'):
        del diffusion.neural_cbf
    print("\n💀 [System] CBF 避障已关闭 (Baseline 模式)\n")
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
safe1_batch, safe2_batch = [], []
score_batch = []
comp_time = []
elbo_batch = []
success = 0
import time
num=10
runs_summary = []
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

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:

            cond[0] = observation
            start = time.time()
            action, samples, diffusion_paths, _, _, elbo = policy(cond, batch_size=args.batch_size)
            end = time.time()
            comp_time.append(end-start)
            elbo_batch.append(elbo)
            
    #############################       single test
            # cond[0] = observation
            # action, samples, diffusion_paths, safe1, safe2 = policy(cond, batch_size=args.batch_size)  #policy.normalizer.normalizers['observations'].mins
            actions = samples.actions[0]
            sequence = samples.observations[0]
            diffusion_paths = diffusion_paths[0]

            # ------------------- prepare collision detection -------------------
            # build wall centers tensor if available (adapter) or from maze_arr
            if USE_CBF and hasattr(diffusion, 'neural_cbf') and getattr(diffusion.neural_cbf, 'wall_centers_tensor', None) is not None:
                wall_centers = diffusion.neural_cbf.wall_centers_tensor.detach().cpu().numpy()
            else:
                # fallback: build from maze_arr if present
                wall_centers = []
                if hasattr(maze_env, 'maze_arr'):
                    hmap = maze_env.maze_arr
                    hh, ww = hmap.shape
                    for rr in range(hh):
                        for cc in range(ww):
                            if hmap[rr, cc] == 10 or hmap[rr, cc] == 11:
                                wall_centers.append([float(cc) + 1.0, float(rr) + 1.0])
                if len(wall_centers) > 0:
                    wall_centers = np.array(wall_centers, dtype=np.float32)
                else:
                    wall_centers = np.zeros((0,2), dtype=np.float32)

            # collision threshold (meters in same physical units as env positions)
            COLLISION_RADIUS = 0.5

            # per-step diagnostics containers for this run
            per_step_collisions = []  # bool per timestep
            per_step_min_d = []
            per_step_positions = []
            collided_flag = False
            min_dist_overall = float('inf')

            
            # ##################################################save videos/images
            # fullpath = join(args.savepath, f'{iter}.png')
            # renderer.composite(fullpath, samples.observations, ncol=1)
            # #########################################s################# 8/3/2023
            # # diffusion_sm = smooth(diffusion_paths)    # smooth the generated traj.
            # diffusion_sm = diffusion_paths            # do not smooth the generated traj.
            # renderer.render_diffusion(join(args.savepath, f'diffusion.mp4'), diffusion_sm)

            # # makedirs(join(args.savepath, 'trap'))
            # # fullpath = join(args.savepath, f'trap/{iter}.png')
            # # renderer.composite(fullpath, samples.observations, ncol=1)

            # diff_step = diffusion_sm.shape[0]  
            # makedirs(join(args.savepath, 'png'))
            # for kk in range(diff_step):
            #     imgpath = join(args.savepath, f'png/{kk}.png')
            #     renderer.composite(imgpath, diffusion_sm[kk:kk+1], ncol=1)
            # ##################################################end saving videos/images

            # 添加了10次循环的保存的逻辑
            if iter == num - 1:
                print("正在保存最后一次运行的可视化结果...")
                
                # 保存规划的轨迹图
                fullpath = join(args.savepath, f'final_plan_{iter}.png')
                renderer.composite(fullpath, samples.observations, ncol=1)
                
                # 保存视频
                diffusion_sm = diffusion_paths
                renderer.render_diffusion(join(args.savepath, f'final_diffusion.mp4'), diffusion_sm)

                # 保存每一帧 (如果觉得不需要可以把下面这几行也注释掉)
                diff_step = diffusion_sm.shape[0]  
                png_dir = join(args.savepath, 'final_png_sequence')
                makedirs(png_dir)
                for kk in range(diff_step):
                    imgpath = join(png_dir, f'{kk}.png')
                    renderer.composite(imgpath, diffusion_sm[kk:kk+1], ncol=1)

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        
        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ###############################################################################################
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        # if 'maze2d' in args.dataset:
        #     xy = next_observation[:2]
        #     goal = env.unwrapped._target
        #     print(
        #         f'maze | pos: {xy} | goal: {goal}'
        #     )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        ###############################################################################################
        # if t % args.vis_freq == 0 or terminal:
        #     fullpath = join(args.savepath, f'{t}.png')

        #     if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        #     # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        #     ## save rollout thus far
        #     renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)   ## debug

        #     # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        #     # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

        # ---------------- record collision diagnostics for this new observation ----------------
        # observation[:2] is the (x,y) position in physical coords
        pos_xy = observation[:2].copy()
        # save per-step position for later grid-based checks / plotting
        per_step_positions.append([float(pos_xy[0]), float(pos_xy[1])])
        # compute min distance to walls
        if wall_centers.shape[0] > 0:
            # wall_centers are in (col+1, row+1) -> x,y ordering same as pos_xy
            dists = np.linalg.norm(wall_centers - pos_xy, axis=1)
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

    if reward > 0.95:
        success = success + 1

    # score = 0

    # ----------------- save per-run diagnostics -----------------
    makedirs(args.savepath)
    run_diag = {
        'run': int(iter),
        'reached_goal': bool(reward > 0.95),
        'collided': bool(collided_flag),
        'collision_steps': [int(i) for i, v in enumerate(per_step_collisions) if v] if len(per_step_collisions) > 0 else [],
        'min_distance_overall': float(min_dist_overall) if min_dist_overall != float('inf') else None,
        'per_step_min_d': per_step_min_d,
        'per_step_positions': per_step_positions,
        'score': float(score)
    }
    json.dump(run_diag, open(join(args.savepath, f'run_{iter}_diag.json'), 'w'), indent=2)
    runs_summary.append(run_diag)

    # safe1_batch.append(torch.cat([safe1[-1].unsqueeze(0).unsqueeze(0), torch.tensor(score).unsqueeze(0).unsqueeze(0).to(safe1.device)], dim = 1))
    # safe2_batch.append(torch.cat([safe2[-1].unsqueeze(0).unsqueeze(0), torch.tensor(score).unsqueeze(0).unsqueeze(0).to(safe2.device)], dim = 1))
    
    score_batch.append(score)
    # logger.finish(t, env.max_episode_steps, score=score, value=0)

elbo_batch = np.array(elbo_batch)
print("elbo mean: ", np.mean(elbo_batch))
print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
# safe1_batch = torch.cat(safe1_batch, dim=0)
# safe2_batch = torch.cat(safe2_batch, dim=0)
comp_time = np.array(comp_time)
# print("safe1: ", torch.min(safe1_batch[:,0]).cpu().numpy())
# print("safe2: ", torch.min(safe2_batch[:,0]).cpu().numpy())
print("score mean: ", np.mean(score_batch))
print("score std: ", np.std(score_batch))
print("computation time: ", np.mean(comp_time))
print("success rate: ", success)

# save aggregate runs summary
try:
    makedirs(args.savepath)
    json.dump(runs_summary, open(join(args.savepath, 'runs_summary.json'), 'w'), indent=2)
    print(f"Saved runs summary to {join(args.savepath, 'runs_summary.json')}")
except Exception as e:
    print("Warning: failed to save runs_summary.json:", e)

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