
'''
PYTHONPATH=. python scripts/plan_safe.py \
    --config config.maze2d \
    --dataset maze2d-custom-v1 \
    --horizon 256 \
    --n_diffusion_steps 128 \
    --diffusion_epoch 0
'''
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
import pickle
import argparse
import os
os.environ["EINOPS_BACKEND"] = "torch"
import einops
einops._backends._loaded_backends = {}
import sys

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
import json
import numpy as np
from os.path import join
import pdb
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import imageio
import einops
from diffuser.utils.rendering import MAZE_BOUNDS, plot2img

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
print(f"\n 结果保存在: {args.savepath} ===\n")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = datasets.load_environment(args.dataset)


#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## enable CBF
USE_CBF = False
if USE_CBF:
    print("\n启动 CBF (Neural Barrier)")
    
    adapter = NeuralBarrierAdapter(device=device)
    diffusion.neural_cbf = adapter
    print(f"CBF已挂载: {adapter.summary()}\n")

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

#---------------------------------- dynamic safe boundary ----------------------------------#
def normalize_maze_xy(xy, env_name):
    bounds = MAZE_BOUNDS[env_name]
    xy = xy + 0.5
    if len(bounds) == 2:
        _, scale = bounds
        xy[:, 0] /= scale
        xy[:, 1] /= scale
        return xy, scale, scale
    if len(bounds) == 4:
        _, iscale, _, jscale = bounds
        xy[:, 0] /= iscale
        xy[:, 1] /= jscale
        return xy, iscale, jscale
    raise RuntimeError(f"Unrecognized bounds for {env_name}: {bounds}")


def plot_barrier_boundary_on_maze(
    model,
    domain,
    env_name,
    plot_len=(300, 300),
    width=0.1,
    norm_eps=1e-6,
    ax=None,
    device=None,
):
    if ax is None:
        ax = plt.gca()

    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    nx, ny = int(plot_len[0]), int(plot_len[1])
    (xmin, xmax), (ymin, ymax) = domain

    xs = torch.linspace(xmin, xmax, nx, device=device, dtype=dtype)
    ys = torch.linspace(ymin, ymax, ny, device=device, dtype=dtype)

    Y, X = torch.meshgrid(ys, xs)
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    model.eval()
    prev_req = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)

    pts.requires_grad_(True)
    B = model(pts)
    if B.dim() > 1 and B.shape[1] != 1:
        B = B[:, :1]
    B = B.reshape(-1, 1)

    grad = torch.autograd.grad(
        outputs=B.sum(),
        inputs=pts,
        create_graph=False,
        retain_graph=False,
        allow_unused=False
    )[0]

    pts.requires_grad_(False)

    grad_norm = torch.clamp(torch.linalg.vector_norm(grad, dim=1), min=norm_eps)
    normed_B = (B.squeeze(1) / grad_norm)

    Z = normed_B.detach().cpu().numpy().reshape(ny, nx)

    X_world = X.detach().cpu().numpy()
    Y_world = Y.detach().cpu().numpy()
    grid = np.stack([X_world.reshape(-1), Y_world.reshape(-1)], axis=1)
    grid_norm, iscale, jscale = normalize_maze_xy(grid, env_name)
    X_norm = grid_norm[:, 0].reshape(ny, nx)
    Y_norm = grid_norm[:, 1].reshape(ny, nx)

    min_val = float(np.min(Z))
    max_val = float(np.max(Z))
    levels = [-width, 0.0, width]
    use_default_labels = True
    if max_val - min_val < 1e-6:
        return None
    if min_val > levels[0] or max_val < levels[-1]:
        levels = [min_val, 0.5 * (min_val + max_val), max_val]
        use_default_labels = False

    contour = ax.contour(
        Y_norm, X_norm, Z,
        levels=levels,
        linestyles=["dotted", "solid", "dotted"],
        linewidths=[1.5, 2.5, 1.5],
        colors=["red", "blue", "green"],
        zorder=30,
    )
    if use_default_labels:
        ax.clabel(contour, inline=True, fontsize=10, fmt={-width:'Dangerous', 0.0:'Boundary', width:'Safe'})

    for p, r in zip(model.parameters(), prev_req):
        p.requires_grad_(r)

    ax.set_xlim(0, 1)  # 归一化后的X范围（对应0-7）
    ax.set_ylim(0, 1)  # 归一化后的Y范围（对应0-7）
    ax.set_aspect('equal')  # 保持比例，避免变形

    return contour


class SafetyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


class VelocityWrapper(nn.Module):
    def __init__(self, base_model, velocity):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('velocity', velocity)

    # def forward(self, pts_2d):
    #     batch_size = pts_2d.shape[0]
    #     v_expanded = self.velocity.unsqueeze(0).expand(batch_size, -1)
    #     inputs = torch.cat([pts_2d, v_expanded], dim=1)
    #     return self.base_model(inputs)

def forward(self, pts_2d):
        # pts_2d 是绝对坐标 [x, y]，直接拼上 [vx, vy] 喂给模型
        batch_size = pts_2d.shape[0]
        v_expanded = self.velocity.unsqueeze(0).expand(batch_size, -1)
        return self.base_model(torch.cat([pts_2d, v_expanded], dim=1))



# class CenteredVelocityWrapper(nn.Module):
#     def __init__(self, base_model, center, velocity):
#         super().__init__()
#         self.base_model = base_model
#         self.register_buffer('center', torch.as_tensor(center, dtype=velocity.dtype, device=velocity.device))
#         self.register_buffer('velocity', velocity)

#     def forward(self, pts_2d):
#         batch_size = pts_2d.shape[0]
#         v_expanded = self.velocity.unsqueeze(0).expand(batch_size, -1)
#         rel_pts = pts_2d - self.center.unsqueeze(0)
#         inputs = torch.cat([rel_pts, v_expanded], dim=1)
#         return self.base_model(inputs)

class VelocityWrapper(nn.Module):
    def __init__(self, base_model, velocity):
        super().__init__()
        self.base_model = base_model
        # 将速度作为模型的固定缓冲区
        self.register_buffer('velocity', velocity)

    def forward(self, pts_2d):
        # pts_2d 是网格上的绝对坐标 [N, 2]
        # 复制速度，使其与网格点数量一致
        batch_size = pts_2d.shape[0]
        v_expanded = self.velocity.unsqueeze(0).expand(batch_size, -1)
        
        # 拼接成 [N, 4] -> [x, y, vx, vy]，直接喂给绝对坐标模型
        inputs = torch.cat([pts_2d, v_expanded], dim=1)
        return self.base_model(inputs)

def get_boundary_domain(center, half_extents, padding=2.0):
    return [
        (center[0] - half_extents[0] - padding, center[0] + half_extents[0] + padding),
        (center[1] - half_extents[1] - padding, center[1] + half_extents[1] + padding),
    ]


def save_dynamic_boundary_plot(model, velocity, domain, wall_center, wall_half_extents, save_path):
    if model is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    model_param = next(model.parameters())
    v_tensor = torch.tensor(velocity, device=model_param.device, dtype=model_param.dtype)
    wrapper = VelocityWrapper(model, v_tensor)

    plot_barrier_boundary_2d(wrapper, domain, ax=ax, width=0.1)

    rect = patches.Rectangle(
                (col_min, row_min),        # 起点: (Col, Row)
                col_max - col_min,         # 宽度: Col 跨度
                row_max - row_min,         # 高度: Row 跨度
                fill=False,
                color='orange',
                linewidth=2,
                label='Real Wall'
            )
    plt.gca().add_patch(rect)
    ax.add_patch(rect)

    ax.arrow(0, 0, 0.5 * velocity[0], 0.5 * velocity[1], head_width=0.1, color='black', label='Velocity')
    ax.set_title(f"Dynamic Safety Boundary (v=[{velocity[0]:.2f}, {velocity[1]:.2f}])")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



# -----------------------------------------------------------------------------#
# 全图多墙可视化函数
# -----------------------------------------------------------------------------#
def save_runs_with_boundary(
    savepath,
    paths,
    renderer,
    env_name,
    boundary_model,
    velocity,
    domain,
    all_obstacles,  # <--- 传入所有的墙
    ncol=1,
):
    assert len(paths) % ncol == 0
    images = []

    for path in paths:
        path = np.array(path)
        if path.ndim > 2: path = path.squeeze(0)

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        # 渲染底图
        plt.imshow(renderer._background * .5, extent=renderer._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        # 画轨迹点
        obs_xy = path[:, :2].copy()
        obs_norm, iscale, jscale = normalize_maze_xy(obs_xy, env_name)
        plt.plot(obs_norm[:, 1], obs_norm[:, 0], c='black', zorder=10)
        colors = plt.cm.jet(np.linspace(0, 1, len(obs_norm)))
        plt.scatter(obs_norm[:, 1], obs_norm[:, 0], c=colors, zorder=20)

        # 画全图动态安全边界
        if boundary_model is not None:
            v_tensor = torch.tensor(velocity, device=next(boundary_model.parameters()).device, dtype=next(boundary_model.parameters()).dtype)
            # 【关键】使用直接拼凑的 Wrapper，不再减去中心
            wrapper = VelocityWrapper(boundary_model, v_tensor)
            plot_barrier_boundary_on_maze(wrapper, domain, env_name, ax=plt.gca(), width=0.1)

            # 画出所有的真实物理墙壁 (黄框)
            for obs in all_obstacles:
                c = obs["center"]
                # print(f"墙中心: {c}")
                hw = obs["half_extents"]
                rect_pts = np.array([
                    [c[0] - hw[0], c[1] - hw[1]], # 左下
                    [c[0] + hw[0], c[1] + hw[1]]  # 右上
                ])
                rect_norm, _, _ = normalize_maze_xy(rect_pts, env_name)
                
                rect = patches.Rectangle(
                    (rect_norm[0, 1], rect_norm[0, 0]), # matplotlib 的矩形参数是 (Y, X)
                    rect_norm[1, 1] - rect_norm[0, 1],
                    rect_norm[1, 0] - rect_norm[0, 0],
                    fill=False, color='orange', linewidth=2, label='Real Wall' if obs is all_obstacles[0] else ""
                )
                plt.gca().add_patch(rect)
        ax = plt.gca()
        ax.set_xlim(renderer._extent[0], renderer._extent[1])  # 底图的X范围
        ax.set_ylim(renderer._extent[2], renderer._extent[3])  # 底图的Y范围
        ax.set_aspect('equal')  # 保持比例
        plt.axis('off')
        img = plot2img(fig, remove_margins=renderer._remove_margins)
        images.append(img)

    images = np.stack(images, axis=0)
    images = einops.rearrange(images, '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=len(images)//ncol, ncol=ncol)
    imageio.imsave(savepath, images)

#---------------------------------- main loop ----------------------------------#
score_batch = []
comp_time = []
elbo_batch = []
success = 0
safe_count = 0

import time
num = 200
best_safe_margin = -1.0
best_score = -float('inf')
best_trajectory = None
runs_summary = []
# ==============================================================================
# 2. 碰撞检测
# ==============================================================================
# print("正在配置目标障碍物碰撞检测...")

# # 我们关注的障碍物中心和尺寸 (与训练时一致)
# TARGET_CENTER = np.array([1.5, 5.0])
# HALF_EXTENTS = np.array([1.0, 0.5])  # 宽2米, 高1米的横向墙壁
# BOUNDARY_DOMAIN = get_boundary_domain(TARGET_CENTER, HALF_EXTENTS, padding=2.0)
BOUNDARY_VELOCITY_OVERRIDE = np.array([0.0, 0.0])  # 例如设置为 np.array([0.0, 0.0])

print("正在配置全图障碍物与可视化...")

# 1. 重新解析全图的墙壁 (用于画黄框)
# MAZE_MAP_LARGE = [
#     "OOOOOOOOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOO#O#OOOO",
#     "OOOOOOO#OOOO",
#     "OOOOO#O#OOOO",
#     "OOOOO#OOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
# ]
# MAZE_MAP_LARGE = [
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOO#OOOO",
#     "OOOOOOO#OOOO",
#     "OOOOOOO#OOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
# ]

# MAZE_MAP_LARGE = [
#     "OOOOOOOOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOO#O#OOOO",
#     "OOOOOOO#OOOO",
#     "OOOOOOO#OOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
# ]

MAZE_MAP_LARGE = [
    "OOOOOOO",
    "OOOOOOO",
    "OO#OOOO",
    "OOOOOOO",
    "OOOO#OO",
    "OOOO#OO",
    "OOOOOOO"
]
def parse_maze_map(map_lines):
    rows = len(map_lines); cols = len(map_lines[0])
    grid = np.array([[c == '#' for c in line] for line in map_lines], dtype=bool)
    visited = np.zeros_like(grid, dtype=bool)
    obstacles = []
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c] or visited[r, c]: continue
            stack = [(r, c)]; visited[r, c] = True; cells = []
            while stack:
                cr, cc = stack.pop(); cells.append((cr, cc))
                for nr, nc in [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]:
                    if 0<=nr<rows and 0<=nc<cols and grid[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True; stack.append((nr, nc))
            rows_idx = [cl[0] for cl in cells]
            cols_idx = [cl[1] for cl in cells]
            
            # X 对应 Row，Y 对应 Col，不加 1
            center_x = (min(rows_idx) + max(rows_idx)) / 2.0  # X 是行
            center_y = (min(cols_idx) + max(cols_idx)) / 2.0 # Y 是列
            half_x = (max(rows_idx) - min(rows_idx) + 1) / 2.0
            half_y = (max(cols_idx) - min(cols_idx) + 1) / 2.0
            
            obstacles.append({
                "center": np.array([center_x, center_y], dtype=np.float32),
                "half_extents": np.array([half_x, half_y], dtype=np.float32)
            })
    return obstacles

ALL_OBSTACLES = parse_maze_map(MAZE_MAP_LARGE)

# 2. 定义全图渲染范围 X:[0, 12], Y:[0, 10]
FULL_MAZE_DOMAIN = [(0.0, 7.0), (0.0, 7.0)] 


DRAW_DYNAMIC_BOUNDARY = False
BOUNDARY_MODEL_PATH = join(root_dir, 'ttc_model_small_2ob.pth')
boundary_model = None
if DRAW_DYNAMIC_BOUNDARY and os.path.exists(BOUNDARY_MODEL_PATH):
    boundary_model = SafetyNetwork().to(device)
    boundary_model.load_state_dict(torch.load(BOUNDARY_MODEL_PATH, map_location=device))
    boundary_model.eval()
    print(f"动态安全边界模型已加载: {BOUNDARY_MODEL_PATH}")

def get_closest_box_distance(pos):
    """计算机器人到全图任意一块墙的最近距离"""
    dists = []
    for obs in ALL_OBSTACLES:
        d = np.abs(pos - obs["center"]) - obs["half_extents"]
        dists.append(np.linalg.norm(np.maximum(d, 0)) + np.minimum(np.max(d), 0))
    min_dist = np.min(dists)
    closest_wall_idx = np.argmin(dists) # 找出是第几块墙
    if min_dist < 0.5:
        closest_obs = ALL_OBSTACLES[closest_wall_idx]
        obs_c = closest_obs["center"]
        obs_hw = closest_obs["half_extents"]
        
        # 墙的边界范围
        r_min, r_max = obs_c[0] - obs_hw[0], obs_c[0] + obs_hw[0]
        c_min, c_max = obs_c[1] - obs_hw[1], obs_c[1] + obs_hw[1]
        
        # print(f"[测距] 小车(Row:{pos[0]:.2f}, Col:{pos[1]:.2f}) "
        #       f"逼近墙#{closest_wall_idx} (Row:{r_min:.1f}~{r_max:.1f}, Col:{c_min:.1f}~{c_max:.1f}) "
        #       f"| 实际距离 min_d = {min_dist:.3f}m")
    return min_dist, closest_wall_idx

# for iter in range(num):   # num of testing runs
#     print("step: ", iter, "/100")

#     observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
#     observation = np.array([ 2, 1.3, 0, 0])   # fix the initial position and final destination for comparison (not needed for general testing)
#     env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line
#     run_velocity = observation[2:4].copy()

#     if args.conditional:
#         print('Resetting target')
#         env.set_target()

#     ## set conditioning xy position to be the goal
#     # target = env._target
#     target = np.array([5, 4.7])
#     env.set_target(target)


# ==============================================================================
# 1. 定义障碍物检测与随机生成逻辑
# ==============================================================================

# 定义障碍物区域 (Row_center, Col_center, Radius_row, Radius_col)
# 对应你的 maze2d-custom-v1 (7x7)
# OBSTACLES_DEF = [
#     {'c': [2.5, 2.5], 'r': [0.55, 0.55]},  # Wall 0 (加 0.05余量)
#     {'c': [5.0, 4.5], 'r': [1.05, 0.55]}   # Wall 1 (加 0.05余量)
# ]

# def is_valid_point(pos):
#     """检查点 pos=[row, col] 是否在地图内且不在障碍物中"""
#     r, c = pos[0], pos[1]
    
#     # 1. 检查边界 (0.5 ~ 6.5) 留出一点边缘
#     if not (0.5 < r < 6.5 and 0.5 < c < 6.5):
#         return False
        
#     # 2. 检查障碍物
#     for obs in OBSTACLES_DEF:
#         c_r, c_c = obs['c']
#         r_r, r_c = obs['r']
#         # 简单的 AABB 碰撞检测
#         if abs(r - c_r) < r_r and abs(c - c_c) < r_c:
#             return False
#     return True

# def generate_test_cases(num_cases, seed=42):
#     """生成固定的测试用例，保证不同实验间的一致性"""
#     np.random.seed(seed) # 🔒 锁死种子，保证 Diffuser/SafeDiffuser/Ours 面对的是同样的考题
#     cases = []
#     while len(cases) < num_cases:
#         # 随机采样起点和终点
#         start = np.random.uniform(1.0, 6.0, 2)
#         goal = np.random.uniform(1.0, 6.0, 2)
        
#         # 验证有效性：
#         # 1. 起点终点都不在墙里
#         # 2. 起点终点距离足够远 (比如 > 3.0)，太近了没测试意义
#         if is_valid_point(start) and is_valid_point(goal) and np.linalg.norm(start - goal) > 3.0:
#             cases.append((start, goal))
#             # print(f"生成测试用例 {len(cases)}: Start={start.round(2)} -> Goal={goal.round(2)}")
#     return cases

# # ==============================================================================
# # 2. 配置实验循环
# # ==============================================================================

# NUM_CASES = 5        # 有几组不同的起点终点 (例如 5 组)
# REPEATS_PER_CASE = 10 # 每组跑几次 (例如 10 次)
# TOTAL_RUNS = NUM_CASES * REPEATS_PER_CASE

# # 生成固定的考题
# test_cases = generate_test_cases(NUM_CASES, seed=2024) 
OBSTACLES_DEF = [
    # Wall 0 (小方块): 物理中心 (2.5, 2.5), 物理半径 0.5
    # 范围: Row [2.0, 3.0], Col [2.0, 3.0]
    {'c': [2.5, 2.5], 'r': [0.55, 0.55]},  
    
    # Wall 1 (长条墙): 物理中心 (5.0, 4.5), 物理半径 Row=1.0, Col=0.5
    # 范围: Row [4.0, 6.0], Col [4.0, 5.0]
    {'c': [5.0, 4.5], 'r': [1.05, 0.55]}   
]

# def is_valid_point(pos):
#     """
#     严格检查: 点是否在地图内，且【绝对不】在障碍物的几何矩形内。
#     pos = [row, col]
#     """
#     r, c = pos[0], pos[1]
    
#     # 1. 检查地图边界 (0 ~ 7)
#     # 我们稍微留 0.1 的余量防止生成在地图最外面的边框上，导致一出生就撞世界边界
#     if not (0.1 < r < 6.9 and 0.1 < c < 6.9):
#         return False
        
#     # 2. 检查障碍物 (严格几何判定)
#     for obs in OBSTACLES_DEF:
#         c_r, c_c = obs['c']
#         r_r, r_c = obs['r']
        
#         # 逻辑：如果 (行距离 < 半径) 且 (列距离 < 半径)，说明点在矩形内部 -> 无效
#         # 只要有一项 >= 半径，说明在矩形外面 -> 有效
#         if abs(r - c_r) < r_r and abs(c - c_c) < r_c:
#             return False
            
#     return True
def is_valid_point(pos):
    """
    严格检查: 考虑小车物理半径的安全生成
    """
    r, c = pos[0], pos[1]
    
    # 🤖 小车物理半径缓冲 (安全气囊)
    # 小车半径约 0.15，我们设 0.25 保证绝对不擦边
    ROBOT_PADDING = 0.2 
    
    # 1. 检查地图边界 (0 ~ 7)
    # 以前是 0.1，太近了！改成 0.25 防止出生在地图边缘墙里
    if not (ROBOT_PADDING < r < (7.0 - ROBOT_PADDING) and 
            ROBOT_PADDING < c < (7.0 - ROBOT_PADDING)):
        return False
        
    # 2. 检查障碍物 (考虑半径扩张)
    for obs in OBSTACLES_DEF:
        c_r, c_c = obs['c']
        # 障碍物半径 + 小车安全半径
        r_r = obs['r'][0] + ROBOT_PADDING
        r_c = obs['r'][1] + ROBOT_PADDING
        
        # 如果落在 (障碍物+小车半径) 的范围内，就是非法
        if abs(r - c_r) < r_r and abs(c - c_c) < r_c:
            return False
            
    return True

def generate_test_cases(num_cases, seed=42):
    """生成固定的测试用例"""
    np.random.seed(seed) 
    cases = []
    while len(cases) < num_cases:
        # 随机采样 [0.5, 6.5] 范围内的点
        start = np.random.uniform(1, 6, 2)
        goal = np.random.uniform(1, 6, 2)
        
        # 验证有效性
        if is_valid_point(start) and is_valid_point(goal) and np.linalg.norm(start - goal) > 3.0:
            cases.append((start, goal))
    return cases

# ==============================================================================
# 2. 配置实验循环
# ==============================================================================

# 1. 设定基准测试点
BASE_START = np.array([3, 1])
BASE_GOAL  = np.array([4, 5])

def is_valid_point(pos):
    """
    严格检查 pos=[row, col] 是否在地图内，且【绝对不】在障碍物内。
    使用全局的 ALL_OBSTACLES (由 parse_maze_map 生成) 作为真理标准。
    """
    r, c = pos[0], pos[1]
    
    # 🤖 小车物理半径缓冲 (安全气囊)
    # 车身半径 0.1 + 额外安全余量 0.1 = 0.2
    ROBOT_PADDING = 0.2 
    
    # 1. 检查地图边界 (0.0 ~ 7.0)
    # 确保不贴着地图最外圈的墙
    if not (ROBOT_PADDING < r < (7.0 - ROBOT_PADDING) and 
            ROBOT_PADDING < c < (7.0 - ROBOT_PADDING)):
        # print(f"❌ 点 {pos.round(2)} 超出地图边界")
        return False
        
    # 2. 检查所有障碍物 (使用 ALL_OBSTACLES)
    for i, obs in enumerate(ALL_OBSTACLES):
        c_r, c_c = obs['center']        # 墙中心
        h_r, h_c = obs['half_extents']  # 墙半长/半宽
        
        # 判定范围 = 墙的几何范围 + 机器人的安全半径
        # 只要行、列两个方向都“撞”上了，那就是真撞了
        if (abs(r - c_r) < (h_r + ROBOT_PADDING)) and \
           (abs(c - c_c) < (h_c + ROBOT_PADDING)):
            # print(f"❌ 点 {pos.round(2)} 落在墙#{i}内部或太近 (Dist < {ROBOT_PADDING})")
            return False
            
    return True

def generate_custom_cases(num_neighbors=5, seed=2024):
    """生成基准点和周边随机点，保证绝对安全"""
    np.random.seed(seed)
    cases = []
    
    # --- Case 0: 绝对基准 (先检查基准点本身是否安全) ---
    if is_valid_point(BASE_START) and is_valid_point(BASE_GOAL):
        cases.append((BASE_START, BASE_GOAL))
        print(f"✅ 添加基准用例: {BASE_START} -> {BASE_GOAL}")
    else:
        print(f"⚠️ 警告：基准点本身就在障碍物里！请修改 BASE_START/GOAL。")
        # 强制修正到一个安全位置 (可选)
        # BASE_START = np.array([3, 3]) 
    
    # --- Case 1~N: 周边随机 (死循环直到找到有效点) ---
    count = 0
    max_attempts = 1000 # 防止死循环
    total_attempts = 0
    
    print(f"正在生成 {num_neighbors} 组周边随机测试点...")
    
    while count < num_neighbors and total_attempts < max_attempts:
        total_attempts += 1
        
        # 随机偏移量 (范围 ±0.5m)
        noise_start = np.random.uniform(-0.5, 0.5, 2)
        noise_goal  = np.random.uniform(-0.5, 0.5, 2)
        
        s = BASE_START + noise_start
        g = BASE_GOAL + noise_goal
        
        # 只有当起点和终点都有效时才通过
        if is_valid_point(s) and is_valid_point(g):
            cases.append((s, g))
            print(f"  [#{count+1}] ✅ 有效: {s.round(2)} -> {g.round(2)}")
            count += 1
        # else:
            # print(f"  [尝试{total_attempts}] ❌ 无效，重试...")
            
    if total_attempts >= max_attempts:
        print("⚠️ 警告：尝试次数过多，未能生成足够的有效点。建议检查基准点是否离墙太近。")
            
    return cases

# 3. 配置运行参数
REPEATS_PER_CASE = 5  # 每个 Case 跑 10 次
# 生成周边的情况
# test_cases = generate_custom_cases(num_neighbors=5) # 1个基准 + 5个周边 = 6个 Case
# NUM_CASES = len(test_cases)

# 随机生成的情况
NUM_CASES = 200
test_cases = generate_test_cases(NUM_CASES, seed=2024) 
TOTAL_RUNS = NUM_CASES * REPEATS_PER_CASE
all_raw_results = []
print(f"\n🚀 开始评测: 共 {NUM_CASES} 组路径, 每组重复 {REPEATS_PER_CASE} 次, 总计 {TOTAL_RUNS} 次运行")

all_run_images = []
# 全局计数器
global_iter = 0
success = 0
safe_count = 0
score_batch = []
elbo_batch = []
comp_time = []
runs_summary = []
best_safe_margin = -float('inf')

# --- 外层循环：遍历不同的地形考题 ---
for case_idx, (start_pos, goal_pos) in enumerate(test_cases):
    
    print(f"\n>>> [Case {case_idx+1}/{NUM_CASES}] 起点:{start_pos.round(2)} -> 终点:{goal_pos.round(2)}")
    
    run_idx = 0
    while run_idx < REPEATS_PER_CASE:
        # 临时增加全局计数（仅用于显示，如果失败会回退）
        current_global_step = global_iter + 1
        
        print(f"    Run {run_idx+1}/{REPEATS_PER_CASE} (Total Valid: {global_iter}) - Attempting...")

        sim_crashed = False # 标记本局是否崩溃

        # 1. 环境初始化 (带保护)
        try:
            env.reset()
            
            # 设置起点 (Start)
            init_state = np.array([start_pos[0], start_pos[1], 0, 0]) 
            env.set_state(init_state[0:2], init_state[2:4])
            
            # 必须把 observation 定义出来
            observation = init_state.copy() 
            run_velocity = init_state[2:4].copy()

            # 设置终点 (Target)
            target = goal_pos
            env.set_target(target)
            
            # 设置 Diffusion 条件
            cond = {
                diffusion.horizon - 1: np.array([*target, 0, 0]),
            }
            
            # 初始化 rollout 和统计变量
            rollout = [observation.copy()]
            total_reward = 0
            per_step_collisions = [] 
            per_step_min_d = []
            collided_flag = False
            min_dist_overall = float('inf')
            COLLISION_RADIUS = 0.15 # 保持较大的 Padding 防止 NaN

        except Exception as e:
            print(f"⚠️ [初始化失败] Case {case_idx} 重置环境时出错: {e} -> 自动重试")
            continue # 直接跳过，重试这一局

        # --- 单次运行的主循环 (Time Steps) ---
        for t in range(env.max_episode_steps):
            state = env.state_vector().copy()

            # Diffusion 规划部分 (通常不会炸，但也包起来比较安全)
            try:
                if t == 0:
                    cond[0] = observation
                    start_time = time.time()
                    action, samples, diffusion_paths, _, _, elbo = policy(cond, batch_size=args.batch_size)
                    end_time = time.time()
                    comp_time.append(end_time - start_time)
                    elbo_batch.append(elbo)
                    
                    current_trajectory = diffusion_paths[0]
                    current_samples = samples.observations
                    sequence = samples.observations[0]
                    
                    if sequence.shape[0] > 0:
                        seq_pos = sequence[:, :2]
                        dist_to_box = np.array([get_closest_box_distance(p)[0] for p in seq_pos])
                        closest_idx = int(np.argmin(dist_to_box))
                        run_velocity = sequence[closest_idx, 2:4].copy()
                    
                    if BOUNDARY_VELOCITY_OVERRIDE is not None:
                        run_velocity = np.array(BOUNDARY_VELOCITY_OVERRIDE, dtype=np.float32)

                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0
                
                # 计算 Action
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            except Exception as e:
                print(f"⚠️ [规划失败] Diffusion 计算出错: {e}")
                sim_crashed = True
                break

            # -----------------------------------------------------------
            # 🛡️ [核心保护] 物理引擎步进 (防止 NaN 崩溃)
            # -----------------------------------------------------------
            try:
                next_observation, reward, terminal, _ = env.step(action)
            except Exception as e:
                print(f"💥 [严重崩溃] MuJoCo NaN 错误 (Step {t}): {e}")
                print(f"🔄 正在丢弃本局数据，重新生成 Case {case_idx} 的第 {run_idx+1} 条轨迹...")
                sim_crashed = True
                break # 跳出 for 循环，触发外层的 continue
            # -----------------------------------------------------------

            total_reward += reward

            # --- 碰撞检测 ---
            pos_xy = next_observation[:2].copy()
            min_d, wall_idx = get_closest_box_distance(pos_xy)
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
        
        # === 本局结束后的判断 ===
        if sim_crashed:
            # 如果是因为崩溃退出的，不保存，直接重试 (run_idx 不增加)
            continue

        # --- 只有正常跑完才会执行以下保存逻辑 ---
        
        # 1. 判断成功与否
        is_success = False
        if reward > 0.95: 
            is_success = True
        
        status_str = "OK" if is_success else "FAIL"
        
        # # 2. 图片保存 (仅保存每个 Case 的第一次成功运行，节省空间)
        # if run_idx == 0:
        #     all_runs_dir = join(args.savepath, 'all_runs_vis_small_Res2')
        #     makedirs(all_runs_dir)
        #     img_filename = f'Case_{case_idx}_Run_{run_idx}_{status_str}_score_{score:.2f}.png'
        #     save_path_full = join(all_runs_dir, img_filename)

        #     if DRAW_DYNAMIC_BOUNDARY and boundary_model is not None:
        #         save_runs_with_boundary(
        #             save_path_full, current_samples, renderer, args.dataset,
        #             boundary_model, run_velocity, FULL_MAZE_DOMAIN, ALL_OBSTACLES, ncol=1
        #         )
        #     else:
        #         renderer.composite(save_path_full, current_samples, ncol=1)
        #     print(f"  📸 已保存示例截图: {img_filename}")

        # 3. 更新最佳记录
        # 2. 保存图片逻辑 (确保每张都保存)
        # 文件夹路径：logs/..../all_runs_vis_small_lagcbf/
        all_runs_dir = join(args.savepath, 'runtest')
        makedirs(all_runs_dir)
        
        # 文件名：run_C{Case号}_R{Run号}_{状态}_score_{分数}.png
        # 这样每个 Case 的每次 Run 都会是一个独立的文件，不会被覆盖
        img_filename = f'run_C{case_idx}_R{run_idx}_{status_str}_score_{score:.2f}.png'
        save_path_full = join(all_runs_dir, img_filename)

        if DRAW_DYNAMIC_BOUNDARY and boundary_model is not None:
            save_runs_with_boundary(
                save_path_full,
                current_samples, # 这里用 diffusion 生成的样本轨迹
                renderer,
                args.dataset,
                boundary_model,
                run_velocity,
                FULL_MAZE_DOMAIN, 
                ALL_OBSTACLES,    
                ncol=1,
            )
        else:
            # 如果不画边界，就用普通的 render
            # 注意：renderer.composite 通常画的是 diffusion 的 plan (current_samples)
            # 如果你想画实际走出来的轨迹 (rollout)，需要转换一下格式
            # 这里保持原逻辑，画 current_samples (模型规划出的轨迹)
            renderer.composite(save_path_full, current_samples, ncol=1)

        try:
            # 直接读取刚刚保存好的这一张图
            img_data = imageio.imread(save_path_full)
            
            # 确保图片没有 Alpha 通道 (如果是4通道转3通道，防止拼接报错)
            if img_data.shape[-1] == 4:
                img_data = img_data[..., :3]
                
            all_run_images.append(img_data)
        except Exception as e:
            print(f"⚠️ 收集拼图失败: {e}")
        if is_success:
            if min_dist_overall > best_safe_margin:
                # print(f"发现更安全的成功路径！MinDist: {min_dist_overall:.4f}m")
                best_safe_margin = min_dist_overall
                fullpath = join(args.savepath, 'best_safe_plan_global.png') 
                renderer.composite(fullpath, current_samples, ncol=1)
                renderer.render_diffusion(join(args.savepath, f'best_safe_diffusion_global.mp4'), current_trajectory)

        # 4. 统计计数
        if is_success:
            success = success + 1
        if not collided_flag:
            safe_count = safe_count + 1
        score_batch.append(score)

        # 5. 收集数据 (PKL)
        traj_stack = np.stack(rollout) 
        traj_xy = traj_stack[:, :2]
        diffs = traj_xy[1:] - traj_xy[:-1]
        dists = np.linalg.norm(diffs, axis=1)
        path_length = np.sum(dists)
        n_collision_steps = sum(per_step_collisions)

        run_data = {
            'algorithm': "safehardlarge", # 按需修改算法名
            'case_idx': case_idx,
            'run_idx': run_idx,
            'start_point': start_pos,
            'goal_point': goal_pos,
            'is_collision': 1 if collided_flag else 0,
            'collision_steps': n_collision_steps,
            'path_length': path_length,
            'is_success': 1 if is_success else 0,
            'score': score,
        }
        all_raw_results.append(run_data)

        # 打印进度
        safe_icon = "✅" if not collided_flag else "❌"
        goal_icon = "✅" if is_success else "❌"
        # 使用 global_iter + 1 因为这一轮已经确定有效了
        print(f"Result [Total {global_iter + 1}/{TOTAL_RUNS}] Goal: {goal_icon} | Safe: {safe_icon} | MinDist: {min_dist_overall:.3f}m | Score: {score:.4f}")

        # 保存单轮诊断数据 (JSON)
        makedirs(args.savepath)
        run_diag = {
            'run': int(global_iter + 1),
            'case_idx': int(case_idx),
            'run_idx': int(run_idx),
            'reached_goal': bool(is_success),
            'collided': bool(collided_flag),
            'min_distance_overall': float(min_dist_overall) if min_dist_overall != float('inf') else None,
            'score': float(score)
        }
        runs_summary.append(run_diag)

        # ✅ 这一轮顺利完成，计数器加 1
        run_idx += 1
        global_iter += 1

# ==================== 【修改】生成 Case x Repeat 的矩阵拼图 ====================
print("\n正在生成最终汇总拼图...")
if len(all_run_images) > 0:
    import einops
    
    # 1. 堆叠成 numpy 数组 (Total, H, W, C)
    stack_imgs = np.stack(all_run_images) 
    
    # 2. 动态设定行列
    # 行数 = 你的 Case 数量 (6)
    # 列数 = 你的重复次数 (10)
    # 注意：这两个变量应该在你的脚本开头定义过
    target_rows = NUM_CASES        
    target_cols = REPEATS_PER_CASE 
    
    expected_count = target_rows * target_cols
    real_count = len(stack_imgs)
    
    if real_count == expected_count:
        # ✅ 完美情况：图片数量正好等于 6 * 10
        # 逻辑: (Case Run) 高 宽 通道 -> (Case 高) (Run 宽) 通道
        grid_image = einops.rearrange(
            stack_imgs, 
            '(case run) h w c -> (case h) (run w) c', 
            case=target_rows, 
            run=target_cols
        )
        
        # filename = f'Summary_Matrix_{target_rows}Cases_x_{target_cols}Runs.png'
        filename = f'safehardlarge.png'
        
    else:
        # ⚠️ 异常情况：可能中间有些 Run 崩溃了没存下来
        print(f"⚠️ 图片数量 ({real_count}) 与实验设定 ({target_rows}x{target_cols}={expected_count}) 不匹配，回退到自动布局。")
        # 自动算一个近似正方形的布局
        target_cols = int(np.ceil(np.sqrt(real_count)))
        target_rows = int(np.ceil(real_count / target_cols))
        
        # 补黑帧
        pad_num = target_rows * target_cols - real_count
        if pad_num > 0:
            padding = np.zeros((pad_num, *stack_imgs.shape[1:]), dtype=stack_imgs.dtype)
            stack_imgs = np.concatenate([stack_imgs, padding], axis=0)
            
        grid_image = einops.rearrange(
            stack_imgs, 
            '(rows cols) h w c -> (rows h) (cols w) c', 
            rows=target_rows, 
            cols=target_cols
        )
        # filename = f'Summary_Grid_Fallback_{target_rows}x{target_cols}.png'
        filename = f'safehardlarge.png'

    # 4. 保存大图
    grid_path = join(args.savepath, filename)
    imageio.imsave(grid_path, grid_image)
    print(f"✅ 最终拼图已保存: {grid_path}")
    
    # 💡 提示：如果图片太大看不清，可以单独去文件夹里看 all_runs_vis_small_xxx 目录
else:
    print("❌ 没有收集到图片，无法拼图")
# ========================================================================

elbo_batch = np.array(elbo_batch)
print("elbo mean: ", np.mean(elbo_batch))
print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
comp_time = np.array(comp_time)

print("score mean: ", np.mean(score_batch))
print("score std: ", np.std(score_batch))
print("computation time: ", np.mean(comp_time))
print("success rate: ", success)
print("safe rate: ", safe_count)
if best_safe_margin > 0:
    print(f"Best Safe Margin (in successful runs): {best_safe_margin:.4f}m")
else:
    print("No successful runs recorded.")



# ==================== 💾 [新增] 保存所有数据到 PKL ====================
# 文件名建议带上算法名字和时间戳，防止覆盖
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# ⚠️ 记得修改这里的 filename 前缀，对应你当前跑的算法
algo_name = "safehardlarge" 
pkl_filename = f"RawData_{algo_name}_{timestamp}.pkl"
pkl_path = join(args.savepath, pkl_filename)

with open(pkl_path, 'wb') as f:
    pickle.dump(all_raw_results, f)

print(f"\n✅ 原始数据已保存至: {pkl_path}")
print(f"包含 {len(all_raw_results)} 条轨迹数据")
# ==================================================================

exit()
