import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (SafeDiffuser)
root_dir = os.path.dirname(current_dir)
# 把根目录加到 path 第一位，确保优先读取本地代码
sys.path.insert(0, root_dir)

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from diffuser.utils import Parser, load_diffusion, load_environment
from diffuser.models.cbf_adapter import NeuralBarrierAdapter

# =========================================================
# 1. 设置与加载
# =========================================================
class CompareParser(Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = CompareParser().parse_args('plan')

# 强制使用 CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Loading diffusion model from: {args.logbase}...")
diffusion_experiment = load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
env = diffusion_experiment.renderer.env # 获取环境以读取地图

# 确保模型在 GPU
diffusion.to(device)

# =========================================================
# 稳健地注入归一化参数
# =========================================================
print("🔧 正在注入归一化参数给 Diffusion 模型...")
normalizer = dataset.normalizer

# 尝试从 observation 的归一化器中提取参数
try:
    # 路径 A: 它是 DatasetNormalizer 容器，里面装着 'observations' 的 normalizer
    if hasattr(normalizer, 'normalizers'):
        obs_norm = normalizer.normalizers['observations']
        diffusion.norm_mins = obs_norm.mins
        diffusion.norm_maxs = obs_norm.maxs
    # 路径 B: 它直接就是 LimitNormalizer
    elif hasattr(normalizer, 'mins'):
        diffusion.norm_mins = normalizer.mins
        diffusion.norm_maxs = normalizer.maxs
    else:
        # 如果都找不到，打印出来看看它是何方神圣
        print(f"❌ 错误: 无法找到 mins/maxs, normalizer 包含属性: {dir(normalizer)}")
        raise AttributeError("Normalizer structure unknown")

    print(f"   -> Mins shape: {diffusion.norm_mins.shape}, Values: {diffusion.norm_mins}")
    print(f"   -> Maxs shape: {diffusion.norm_maxs.shape}, Values: {diffusion.norm_maxs}")

except Exception as e:
    print(f"⚠️ 警告: 归一化参数注入失败: {e}")
    # 防止后面 crash，给个假的兜底（虽然效果会不对，但至少能跑）
    diffusion.norm_mins = np.array([-1]*6)
    diffusion.norm_maxs = np.array([1]*6)

# 将 numpy 转为 tensor (为了保险)
if not isinstance(diffusion.norm_mins, torch.Tensor):
    diffusion.norm_mins = torch.tensor(diffusion.norm_mins, device=device, dtype=torch.float32)
    diffusion.norm_maxs = torch.tensor(diffusion.norm_maxs, device=device, dtype=torch.float32)
# =========================================================

# =========================================================
# 2. 定义对比函数
# =========================================================

def generate_trajectory(cond, use_cbf=False):
    """
    生成一条轨迹
    use_cbf: True=开启避障, False=关闭避障
    """
    # 1. 设置 CBF 状态
    if use_cbf:
        if not hasattr(diffusion, 'neural_cbf') or diffusion.neural_cbf is None:
            diffusion.neural_cbf = NeuralBarrierAdapter(device=device)
            
            # [关键修复] 强制把环境里的真实墙壁同步给 CBF
            # 既然画图是对的，我们就用画图的逻辑来生成墙壁坐标
            maze_arr = env.maze_arr
            h, w = maze_arr.shape
            real_walls = []
            
            for r in range(h):
                for c in range(w):
                    if maze_arr[r, c] == 10: # 10 是墙
                        # 使用和画图一模一样的坐标变换
                        # 画图逻辑: x = c, y = h - 1 - r
                        wall_x = float(c)
                        wall_y = float(h - 1 - r)
                        real_walls.append([wall_x, wall_y])
            
            # 覆盖 CBF 里的旧墙壁数据
            diffusion.neural_cbf.wall_centers_tensor = torch.tensor(
                real_walls, dtype=torch.float32, device=device
            )
            print(f"✅ 已同步环境中的 {len(real_walls)} 个墙壁坐标到 CBF！")
            
        print("⚡ [Mode] CBF 避障已开启")
    else:
        if hasattr(diffusion, 'neural_cbf'):
            saved_adapter = diffusion.neural_cbf
            del diffusion.neural_cbf 
        print("💀 [Mode] CBF 避障已关闭 (Baseline)")

    # 2. 数据预处理：归一化 + 转 Tensor
    cond_batch = {}
    for k, v in cond.items():
        # v 是物理坐标 (numpy)，先进行归一化
        v_norm = dataset.normalizer.normalize(v, 'observations')
        v_norm_batch = v_norm[None, ...]
        cond_batch[k] = torch.tensor(v_norm_batch, dtype=torch.float32, device=device)
    
    # 3. 计算正确的轨迹总维度 (Action + Observation)
    # 修复点：不能只用 observation_dim (4)，要加上 action_dim (2)，总共是 6
    transition_dim = diffusion.observation_dim + diffusion.action_dim

    # 4. 运行扩散采样
    samples = diffusion.p_sample_loop(
        shape=(1, diffusion.horizon, transition_dim), # <--- 修复了这里的形状
        cond=cond_batch
    )
    
    # 恢复 CBF
    if not use_cbf and 'saved_adapter' in locals():
        diffusion.neural_cbf = saved_adapter

    # 5. 返回数据
    # samples 的形状是 (1, Horizon, 6) -> [Action(2), Observation(4)]
    # 我们只需要 Observation 部分，所以从 action_dim 开始切片
    traj_normalized = samples[0, :, diffusion.action_dim:].detach().cpu().numpy()
    
    return traj_normalized

# =========================================================
# 3. 运行对比实验
# =========================================================

# --- A. 设定起点和终点 ---
# 我们找一个经典的“穿墙”场景
# 在 Maze2D Large 中，(2,2) 到 (4,2) 中间通常有墙
start_pos = np.array([1.0, 1.0]) 
target_pos = np.array([5.0, 6.0]) # 你可以改这个坐标测试不同的墙

# 构造条件
cond = {
    0: np.concatenate([start_pos, [0,0]]), # 起点，速度0
    diffusion.horizon - 1: np.concatenate([target_pos, [0,0]]) # 终点
}

# --- B. 运行两次 ---
print("\n=== 开始生成 Baseline 轨迹 (无避障) ===")
traj_unsafe = generate_trajectory(cond, use_cbf=False)

print("\n=== 开始生成 Safe 轨迹 (有避障) ===")
traj_safe = generate_trajectory(cond, use_cbf=True)

# =========================================================
# 4. 像 plan_maze2d 一样直接画矩阵
# =========================================================
print("\n🎨 正在绘制最终对比图 (官方对齐逻辑)...")

# 1. 获取迷宫矩阵 (真理之源)
# maze_arr: 0是路，10或11是墙
maze_arr = env.maze_arr 

# 2. 准备画布
# 这里的 figsize 可以随意，不影响对齐，只影响清晰度
fig, ax = plt.subplots(figsize=(12, 12))

# 3. 画迷宫背景 (模仿 renderer 的核心逻辑)
# map_mask: 墙是0(黑)，路是1(白) -> 或者是反过来，看 cmap
# 官方 renderer 用的是 cmap='gray_r' (Reverse Gray)，也就是数值大的是黑，小的是白
# maze_arr 里墙是 10，路是 0。
# 所以 'gray_r' 会让 10(大数) 变黑，0(小数) 变白。完美！
ax.imshow(maze_arr, cmap='gray', origin='upper')

# 4. 准备轨迹数据
# Unnormalize 拿到物理坐标
traj_unsafe_phys = dataset.normalizer.unnormalize(traj_unsafe, 'observations')
traj_safe_phys = dataset.normalizer.unnormalize(traj_safe, 'observations')

# 5. [核心对齐] 坐标映射
# Maze2D 的物理坐标定义：
# observation[0] = Vertical (垂直方向/行号/Row) = X
# observation[1] = Horizontal (水平方向/列号/Col) = Y

# Matplotlib 的 plot(x, y) 定义：
# 第一个参数是 横轴坐标 (Horizontal)
# 第二个参数是 纵轴坐标 (Vertical)

# ---> 所以，我们必须把 observation[1] 放前面，observation[0] 放后面！ <---

unsafe_row = traj_unsafe_phys[:, 0] # 物理X (行)
unsafe_col = traj_unsafe_phys[:, 1] # 物理Y (列)
safe_row = traj_safe_phys[:, 0]
safe_col = traj_safe_phys[:, 1]

# 6. 画轨迹 (Col, Row)
ax.plot(unsafe_col, unsafe_row, color='red', linestyle='--', linewidth=3, label='Original (Unsafe)')
ax.plot(safe_col, safe_row, color='#00FF00', linewidth=3, label='SafeDiffuser (CBF)')

# 7. 画起点终点 (Col, Row)
# 起点 (1.0, 1.0) -> Col=1.0, Row=1.0
ax.scatter(start_pos[1], start_pos[0], color='blue', s=400, marker='*', label='Start', zorder=10)
ax.scatter(target_pos[1], target_pos[0], color='gold', s=400, marker='X', label='Target', zorder=10)

# 8. 装饰
ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
ax.set_title("SafeDiffuser Comparison (Perfect Alignment)", fontsize=16)

# 9. 锁定坐标轴比例
# 这步很重要，保证迷宫是正方形格子，不会被拉扁
ax.set_aspect('equal')

# 10. 隐藏刻度 (可选)
ax.axis('off')

# 保存
save_path = "final_comparison_perfect.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✅ 完美对齐图已生成: {save_path}")

# # =========================================================
# # 4. [科学评估版] 自动检测收敛 + 虚实结合画图 + 正确黑白
# # =========================================================
# print("\n🎨 正在绘制最终对比图 (科学评估版)...")

# # 1. 获取迷宫矩阵
# maze_arr = env.maze_arr 

# # 2. 准备画布
# fig, ax = plt.subplots(figsize=(12, 12))

# # 3. [背景修正] 强制黑墙白路
# ax.imshow(maze_arr, cmap='gray', origin='upper')

# # 4. 准备轨迹数据
# traj_unsafe_phys = dataset.normalizer.unnormalize(traj_unsafe, 'observations')
# traj_safe_phys = dataset.normalizer.unnormalize(traj_safe, 'observations')

# # 坐标提取 (注意：Index 0=Row/Height, Index 1=Col/Width)
# unsafe_row = traj_unsafe_phys[:, 0]
# unsafe_col = traj_unsafe_phys[:, 1]
# safe_row = traj_safe_phys[:, 0]
# safe_col = traj_safe_phys[:, 1]

# # =========================================================
# # 5. 定义收敛检测函数 (解决“回头路”视觉问题)
# # =========================================================
# def split_trajectory_by_convergence(traj_col, traj_row, target, dist_thr=0.5):
#     """
#     找到轨迹最后一次进入目标圈(dist_thr)并不再出来的时刻。
#     """
#     points = np.stack([traj_col, traj_row], axis=1) # (N, 2)
#     # 注意：target_pos 是 [Row, Col]，这里我们要跟 points 里的 [Col, Row] 对齐
#     # 所以 target 传入时应该是 [Target_Col, Target_Row]
#     target_point = np.array(target)
    
#     # 1. 计算距离
#     dists = np.linalg.norm(points - target_point, axis=1)
    
#     # 2. 判定入圈
#     in_zone = dists < dist_thr
    
#     # 3. 倒着找第一个“出圈”的点
#     out_of_zone_indices = np.where(~in_zone)[0]
    
#     if len(out_of_zone_indices) == 0:
#         return 0 # 一直在终点
#     elif len(out_of_zone_indices) == len(traj_col):
#         return len(traj_col) # 从未收敛
#     else:
#         # 收敛点是最后一个出圈点的下一个点
#         return out_of_zone_indices[-1] + 1

# # 目标坐标 (用于画图和计算距离，必须是 [Col, Row])
# target_plot = [target_pos[1], target_pos[0]]

# # 计算截断点
# idx_unsafe = split_trajectory_by_convergence(unsafe_col, unsafe_row, target_plot)
# idx_safe = split_trajectory_by_convergence(safe_col, safe_row, target_plot)

# # =========================================================
# # 6. 画轨迹 (实线=赶路, 虚线=磨蹭)
# # =========================================================

# # --- A. Baseline (红色) ---
# # 实线：有效赶路阶段
# ax.plot(unsafe_col[:idx_unsafe], unsafe_row[:idx_unsafe], 
#         color='red', linestyle='--', linewidth=3, label='Original (Active)')
# # 虚线：到达后徘徊阶段 (透明度低)
# if idx_unsafe < len(unsafe_col):
#     ax.plot(unsafe_col[idx_unsafe:], unsafe_row[idx_unsafe:], 
#             color='red', linestyle=':', linewidth=1, alpha=0.3)

# # --- B. SafeDiffuser (绿色) ---
# # 实线：有效赶路阶段
# ax.plot(safe_col[:idx_safe], safe_row[:idx_safe], 
#         color='#00FF00', linewidth=3, label='SafeDiffuser (Active)')
# # 虚线：到达后徘徊阶段
# if idx_safe < len(safe_col):
#     ax.plot(safe_col[idx_safe:], safe_row[idx_safe:], 
#             color='#00FF00', linestyle='-', linewidth=1, alpha=0.2)
#     # 画一个圈标记“停车点”
#     stop_idx = min(idx_safe, len(safe_col)-1)
#     ax.scatter(safe_col[stop_idx], safe_row[stop_idx], 
#                color='#00FF00', s=80, marker='o', edgecolors='white', zorder=5, label='Settled')

# # 7. 画起点终点 (Col, Row)
# ax.scatter(start_pos[1], start_pos[0], color='blue', s=400, marker='*', label='Start', zorder=10)
# ax.scatter(target_pos[1], target_pos[0], color='gold', s=400, marker='X', label='Target', zorder=10)

# # 8. 装饰
# ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
# ax.set_title("SafeDiffuser Evaluation (Solid=Active, Fade=Settled)", fontsize=16)
# ax.set_aspect('equal')
# ax.axis('off')

# # 保存
# save_path = "final_comparison_scientific.png"
# plt.savefig(save_path, dpi=150, bbox_inches='tight')
# print(f"✅ 科学评估图已生成: {save_path}")