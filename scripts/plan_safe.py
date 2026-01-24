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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## enable CBF
USE_CBF = True
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

    # rect = patches.Rectangle(
    #     (wall_center[0] - wall_half_extents[0], wall_center[1] - wall_half_extents[1]),
    #     2.0 * wall_half_extents[0],
    #     2.0 * wall_half_extents[1],
    #     fill=False,
    #     color='orange',
    #     linewidth=2,
    #     label='Real Wall'
    # )
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


# def save_runs_with_boundary(
#     savepath,
#     paths,
#     renderer,
#     env_name,
#     boundary_model,
#     velocity,
#     domain,
#     wall_center,
#     wall_half_extents,
#     ncol=1,
# ):
#     assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'
#     images = []

#     for path in paths:
#         path = np.array(path)
#         if path.ndim > 2:
#             path = path.squeeze(0)

#         plt.clf()
#         fig = plt.gcf()
#         fig.set_size_inches(5, 5)
#         plt.imshow(renderer._background * .5,
#             extent=renderer._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

#         obs_xy = path[:, :2].copy()
#         obs_norm, iscale, jscale = normalize_maze_xy(obs_xy, env_name)
#         plt.plot(obs_norm[:, 1], obs_norm[:, 0], c='black', zorder=10)
#         colors = plt.cm.jet(np.linspace(0, 1, len(obs_norm)))
#         plt.scatter(obs_norm[:, 1], obs_norm[:, 0], c=colors, zorder=20)

#         if boundary_model is not None:
#             v_tensor = torch.tensor(velocity, device=next(boundary_model.parameters()).device,
#                                     dtype=next(boundary_model.parameters()).dtype)
#             wrapper = CenteredVelocityWrapper(boundary_model, wall_center, v_tensor)
#             plot_barrier_boundary_on_maze(wrapper, domain, env_name, ax=plt.gca(), width=0.1)

#             x_min = wall_center[0] - wall_half_extents[0]
#             x_max = wall_center[0] + wall_half_extents[0]
#             y_min = wall_center[1] - wall_half_extents[1]
#             y_max = wall_center[1] + wall_half_extents[1]
#             rect_pts = np.array([[x_min, y_min], [x_max, y_max]])
#             rect_norm, _, _ = normalize_maze_xy(rect_pts, env_name)
#             rect_x_min = rect_norm[0, 0]
#             rect_y_min = rect_norm[0, 1]
#             rect_x_max = rect_norm[1, 0]
#             rect_y_max = rect_norm[1, 1]

#             rect = patches.Rectangle(
#                 (rect_y_min, rect_x_min),
#                 rect_y_max - rect_y_min,
#                 rect_x_max - rect_x_min,
#                 fill=False,
#                 color='orange',
#                 linewidth=2,
#                 label='Real Wall'
#             )
#             plt.gca().add_patch(rect)

#         plt.axis('off')
#         img = plot2img(fig, remove_margins=renderer._remove_margins)
#         images.append(img)

#     images = np.stack(images, axis=0)
#     nrow = len(images) // ncol
#     images = einops.rearrange(images,
#         '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
#     imageio.imsave(savepath, images)

# -----------------------------------------------------------------------------#
# 修改 2: 全图多墙可视化函数
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
num = 30
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

# DRAW_DYNAMIC_BOUNDARY = True
# BOUNDARY_MODEL_PATH = join(root_dir, 'ttc_model_dataset.pth')
# boundary_model = None
# if DRAW_DYNAMIC_BOUNDARY:
#     if os.path.exists(BOUNDARY_MODEL_PATH):
#         boundary_model = SafetyNetwork().to(device)
#         boundary_model.load_state_dict(torch.load(BOUNDARY_MODEL_PATH, map_location=device))
#         boundary_model.eval()
#         print(f"动态安全边界模型已加载: {BOUNDARY_MODEL_PATH}")
#     else:
#         print(f"未找到动态边界模型，已跳过: {BOUNDARY_MODEL_PATH}")
#         DRAW_DYNAMIC_BOUNDARY = False

# def get_target_box_distance(pos):
#     """计算机器人到目标矩形表面的最短距离"""
#     rel_pos = pos - TARGET_CENTER
#     d = np.abs(rel_pos) - HALF_EXTENTS
#     # 外部距离
#     outside_dist = np.linalg.norm(np.maximum(d, 0))
#     # 内部距离 (撞进去了就是负数)
#     inside_dist = np.minimum(np.max(d), 0)
#     return outside_dist + inside_dist

print("正在配置全图障碍物与可视化...")

# 1. 重新解析全图的墙壁 (用于画黄框)
MAZE_MAP_LARGE = [
    "OOOOOOOOOOOO", 
    "OOO#OOOOOOOO", 
    "OOO#OOOOOOOO", 
    "OOOOOOOOOOOO",
    "OOO#OOOOOOOO", 
    "OOO#OOOOOOOO", 
    "OOO#OOOOOOOO", 
    "OOOOOOOOOOOO", 
    "OOOOOOOOOOOO",
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
            # rows_idx = [cl[0] for cl in cells]; cols_idx = [cl[1] for cl in cells]
            # obstacles.append({
            #     "center": np.array([(min(cols_idx)+max(cols_idx)+1)/2.0 + 1.0, (min(rows_idx)+max(rows_idx)+1)/2.0 + 1.0], dtype=np.float32),
            #     "half_extents": np.array([(max(cols_idx)-min(cols_idx)+1)/2.0, (max(rows_idx)-min(rows_idx)+1)/2.0], dtype=np.float32)
            # })
            rows_idx = [cl[0] for cl in cells]
            cols_idx = [cl[1] for cl in cells]
            
            # # 【修改】：和 adapter 里一样，X 对应 Col，Y 对应 Row
            # center_x = (min(cols_idx) + max(cols_idx)) / 2.0
            # center_y = (min(rows_idx) + max(rows_idx)) / 2.0
            # half_x = (max(cols_idx) - min(cols_idx) + 1) / 2.0
            # half_y = (max(rows_idx) - min(rows_idx) + 1) / 2.0
            
            # obstacles.append({
            #     "center": np.array([center_x, center_y], dtype=np.float32),
            #     "half_extents": np.array([half_x, half_y], dtype=np.float32)
            # })
            # 【核心修改】：X 对应 Row，Y 对应 Col，不加 1
            center_x = (min(rows_idx) + max(rows_idx)) / 2.0  # X 是行
            center_y = (min(cols_idx) + max(cols_idx)) / 2.0+2  # Y 是列
            half_x = (max(rows_idx) - min(rows_idx) + 1) / 2.0
            half_y = (max(cols_idx) - min(cols_idx) + 1) / 2.0
            
            obstacles.append({
                "center": np.array([center_x, center_y], dtype=np.float32),
                "half_extents": np.array([half_x, half_y], dtype=np.float32)
            })
    return obstacles
# def parse_maze_map(map_lines):
#     rows = len(map_lines); cols = len(map_lines[0])
#     grid = np.array([[c == '#' for c in line] for line in map_lines], dtype=bool)
#     visited = np.zeros_like(grid, dtype=bool)
#     obstacles = []
    
#     for r in range(rows):
#         for c in range(cols):
#             if not grid[r, c] or visited[r, c]: continue
#             stack = [(r, c)]; visited[r, c] = True; cells = []
#             while stack:
#                 cr, cc = stack.pop(); cells.append((cr, cc))
#                 for nr, nc in [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]:
#                     if 0<=nr<rows and 0<=nc<cols and grid[nr, nc] and not visited[nr, nc]:
#                         visited[nr, nc] = True; stack.append((nr, nc))
            
#             rows_idx = [cl[0] for cl in cells]
#             cols_idx = [cl[1] for cl in cells]
            
#             # 1. X 轴对应 列 (Col)
#             center_x = (min(cols_idx) + max(cols_idx) + 1) / 2.0
            
#             # 2. 【核心修改】：Y 轴对应 行 (Row)，且必须上下翻转！
#             raw_center_row = (min(rows_idx) + max(rows_idx) + 1) / 2.0
#             total_rows = len(map_lines)
#             center_y = total_rows - raw_center_row  # <--- 翻转在这里！
            
#             # 3. 宽高保持不变
#             half_x = (max(cols_idx) - min(cols_idx) + 1) / 2.0
#             half_y = (max(rows_idx) - min(rows_idx) + 1) / 2.0
            
#             obstacles.append({
#                 "center": np.array([center_x, center_y], dtype=np.float32),
#                 "half_extents": np.array([half_x, half_y], dtype=np.float32)
#             })
#     return obstacles
ALL_OBSTACLES = parse_maze_map(MAZE_MAP_LARGE)

# 2. 定义全图渲染范围 X:[0, 12], Y:[0, 10]
FULL_MAZE_DOMAIN = [(0.0, 12.0), (0.0, 10.0)] 

DRAW_DYNAMIC_BOUNDARY = True
BOUNDARY_MODEL_PATH = join(root_dir, 'ttc_model_dataset_new.pth')
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
    return np.min(dists)

for iter in range(num):   # num of testing runs
    print("step: ", iter, "/100")

    observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    observation = np.array([ 0.94875744,  2.93648809, -0.01347715,  0.06358764])   # fix the initial position and final destination for comparison (not needed for general testing)
    env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line
    run_velocity = observation[2:4].copy()

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    # target = np.array([1.0, 10.0])
    # env.set_target(target)
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
            if sequence.shape[0] > 0:
                seq_pos = sequence[:, :2]
                # dist_to_box = np.array([get_target_box_distance(p) for p in seq_pos])
                dist_to_box = np.array([get_closest_box_distance(p) for p in seq_pos])
                closest_idx = int(np.argmin(dist_to_box))
                run_velocity = sequence[closest_idx, 2:4].copy()
            if BOUNDARY_VELOCITY_OVERRIDE is not None:
                run_velocity = np.array(BOUNDARY_VELOCITY_OVERRIDE, dtype=np.float32)
            diffusion_paths = diffusion_paths[0]

            # 动态边界只叠加到 all_runs_vis 中，不再单独输出 boundary_run 图

            # # 添加了10次循环的保存的逻辑
            # if iter == num - 1:
            #     print("正在保存最后一次运行的可视化结果...")
                
            #     # 保存规划的轨迹图
            #     fullpath = join(args.savepath, f'final_plan_{iter}.png')
            #     renderer.composite(fullpath, samples.observations, ncol=1)
                
            #     # 保存视频
            #     diffusion_sm = diffusion_paths
            #     renderer.render_diffusion(join(args.savepath, f'final_diffusion.mp4'), diffusion_sm)

            #     # 保存每一帧
            #     diff_step = diffusion_sm.shape[0]  
            #     png_dir = join(args.savepath, 'final_png_sequence')
            #     makedirs(png_dir)
            #     for kk in range(diff_step):
            #         imgpath = join(png_dir, f'{kk}.png')
            #         renderer.composite(imgpath, diffusion_sm[kk:kk+1], ncol=1)

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
        
        # min_d = get_target_box_distance(pos_xy)
        min_d = get_closest_box_distance(pos_xy)

        per_step_min_d.append(min_d)
        
        # 判定是否发生碰撞
        # 注意：机器人的物理半径大约是 0.15m。
        # 如果距离墙壁表面小于 0.15m，就认为是物理碰撞了。
        COLLISION_RADIUS = 0.15 
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
    # if DRAW_DYNAMIC_BOUNDARY and boundary_model is not None:
    #     save_runs_with_boundary(
    #         join(all_runs_dir, img_filename),
    #         current_samples,
    #         renderer,
    #         args.dataset,
    #         boundary_model,
    #         run_velocity,
    #         BOUNDARY_DOMAIN,
    #         TARGET_CENTER,
    #         HALF_EXTENTS,
    #         ncol=1,
    #     )
    # else:
    #     renderer.composite(join(all_runs_dir, img_filename), current_samples, ncol=1)
    if DRAW_DYNAMIC_BOUNDARY and boundary_model is not None:
        save_runs_with_boundary(
            join(all_runs_dir, img_filename),
            current_samples,
            renderer,
            args.dataset,
            boundary_model,
            run_velocity,
            FULL_MAZE_DOMAIN, # <--- 改成全图范围
            ALL_OBSTACLES,    # <--- 改成全部墙壁列表
            ncol=1,
        )
    else:
        renderer.composite(join(all_runs_dir, img_filename), current_samples, ncol=1)

    # 2. [核心修改] 保存逻辑：优先 Success，其次 Safety (MinDist)
    # 逻辑：必须成功，且 (当前的最小距离 > 历史最好的最小距离)
    if is_success:
        if min_dist_overall > best_safe_margin:
            print(f"发现更安全的成功路径！Run {iter}: MinDist 从 {best_safe_margin:.4f}m 提升到 {min_dist_overall:.4f}m (Score: {score:.4f})")
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

    # 3. 统计计数
    if is_success:
        success = success + 1
    if not collided_flag:
        safe_count = safe_count + 1
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
print("safe rate: ", safe_count)
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