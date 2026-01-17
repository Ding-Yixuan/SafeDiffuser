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
# [关键补丁 V2] 稳健地注入归一化参数
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
# def generate_trajectory(cond, use_cbf=False):
#     """
#     生成一条轨迹
#     use_cbf: True=开启避障, False=关闭避障
#     """
#     # 临时开关 CBF
#     if use_cbf:
#         # 如果需要开启，确保 adapter 已加载
#         if not hasattr(diffusion, 'neural_cbf') or diffusion.neural_cbf is None:
#             diffusion.neural_cbf = NeuralBarrierAdapter(device=device)
#         print("⚡ [Mode] CBF 避障已开启")
#     else:
#         # 如果需要关闭，临时保存并移除
#         if hasattr(diffusion, 'neural_cbf'):
#             saved_adapter = diffusion.neural_cbf
#             del diffusion.neural_cbf # 临时删除
#         print("💀 [Mode] CBF 避障已关闭 (Baseline)")

#     # 运行扩散采样 (p_sample_loop)
#     # 构造 batch (size=1)
#     cond_batch = {k: v[None, ...] for k, v in cond.items()}
    
#     # 采样
#     samples = diffusion.p_sample_loop(shape=(1, diffusion.horizon, diffusion.observation_dim), cond=cond_batch)
    
#     # 恢复 CBF (为了不影响下一次运行)
#     if not use_cbf and 'saved_adapter' in locals():
#         diffusion.neural_cbf = saved_adapter

#     # 返回物理坐标 (Batch, Horizon, Dim) -> (Horizon, Dim)
#     return samples[0].cpu().numpy()

def generate_trajectory(cond, use_cbf=False):
    """
    生成一条轨迹
    use_cbf: True=开启避障, False=关闭避障
    """
    # 1. 设置 CBF 状态
    if use_cbf:
        if not hasattr(diffusion, 'neural_cbf') or diffusion.neural_cbf is None:
            diffusion.neural_cbf = NeuralBarrierAdapter(device=device)
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
target_pos = np.array([5.0, 5.0]) # 你可以改这个坐标测试不同的墙

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
# 4. 画图对比
# =========================================================
print("\n🎨 正在绘制对比图...")

# 获取反归一化后的物理坐标 (假设 diff 输出是归一化的)
# 如果你的 dataset.normalizer 已经处理了，这里可能需要 unnormalize
# 通常 p_sample_loop 返回的是 Normalized 数据
traj_unsafe_phys = dataset.normalizer.unnormalize(traj_unsafe, 'observations')
traj_safe_phys = dataset.normalizer.unnormalize(traj_safe, 'observations')

# 提取 X, Y
# 根据之前的分析，index 0=x, 1=y (或 2,3 取决于是否包含 action，但 unnormalize 后通常是 obs)
# Maze2D obs 通常是 [x, y, vx, vy]
path_unsafe_x = traj_unsafe_phys[:, 0]
path_unsafe_y = traj_unsafe_phys[:, 1]
path_safe_x = traj_safe_phys[:, 0]
path_safe_y = traj_safe_phys[:, 1]

# 创建画布
fig, ax = plt.subplots(figsize=(10, 10))

# --- 画地图背景 ---
# 解析迷宫结构
maze_arr = env.maze_arr
h, w = maze_arr.shape
# Maze2D 的物理坐标系转换: grid (r, c) -> phys (c+1, r+1)
# 我们直接画格子
for r in range(h):
    for c in range(w):
        if maze_arr[r, c] == 10: # 10 代表墙
            # 画墙壁 (物理坐标)
            # x = c+0.5 到 c+1.5 (中心是 c+1) -> 这里的 matplotlib rect 是 (left, bottom)
            # 物理中心 (c+1, r+1), 宽1, 高1 -> 左下角 (c+0.5, r+0.5)
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0, facecolor='black')
            ax.add_patch(rect)

# --- 画轨迹 ---
# 1. Baseline (红色虚线)
ax.plot(path_unsafe_x, path_unsafe_y, color='red', linestyle='--', linewidth=2, label='Original Diffuser (Unsafe)', alpha=0.7)
ax.scatter(path_unsafe_x, path_unsafe_y, color='red', s=10, alpha=0.3)

# 2. Safe (绿色实线)
ax.plot(path_safe_x, path_safe_y, color='#00FF00', linewidth=3, label='SafeDiffuser (With CBF)')
ax.scatter(path_safe_x, path_safe_y, color='green', s=15, alpha=0.5)

# --- 画起点终点 ---
ax.scatter(start_pos[0], start_pos[1], color='blue', s=200, marker='*', label='Start', zorder=10)
ax.scatter(target_pos[0], target_pos[1], color='gold', s=200, marker='X', label='Target', zorder=10)

# 设置图形属性
ax.set_aspect('equal')
ax.set_xlim(0, 10) # 根据 Maze Large 调整
ax.set_ylim(0, 12)
ax.legend(loc='upper right', fontsize=12)
ax.set_title("Comparison: Original vs SafeDiffuser", fontsize=16)
ax.grid(True, linestyle=':', alpha=0.3)

# 保存
save_path = "comparison_result.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✅ 对比图已生成: {save_path}")
print("快打开看看红色和绿色的区别！")