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

# 1. 设置与加载
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

# 稳健地注入归一化参数
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
        # 如果都找不到
        print(f"无法找到 mins/maxs, normalizer 包含属性: {dir(normalizer)}")
        raise AttributeError("Normalizer structure unknown")

    print(f"   -> Mins shape: {diffusion.norm_mins.shape}, Values: {diffusion.norm_mins}")
    print(f"   -> Maxs shape: {diffusion.norm_maxs.shape}, Values: {diffusion.norm_maxs}")

except Exception as e:
    print(f"归一化参数注入失败: {e}")


# 将 numpy 转为 tensor
if not isinstance(diffusion.norm_mins, torch.Tensor):
    diffusion.norm_mins = torch.tensor(diffusion.norm_mins, device=device, dtype=torch.float32)
    diffusion.norm_maxs = torch.tensor(diffusion.norm_maxs, device=device, dtype=torch.float32)



# 2. 定义对比函数
def generate_trajectory(cond, use_cbf=False):
    """
    生成一条轨迹
    use_cbf: True=开启避障, False=关闭避障
    """
    # 1. 设置 CBF 状态
    if use_cbf:
        if not hasattr(diffusion, 'neural_cbf') or diffusion.neural_cbf is None:
            diffusion.neural_cbf = NeuralBarrierAdapter(device=device)

            # 生成墙壁坐标
            maze_arr = env.maze_arr
            h, w = maze_arr.shape
            real_walls = []
            
            for r in range(h):
                for c in range(w):
                    if maze_arr[r, c] == 10: # 10 是墙
                        # x = c, y = h - 1 - r
                        # wall_x = float(c)
                        # wall_y = float(h - 1 - r)
                        # real_walls.append([wall_x, wall_y])
                        # 为与 NeuralBarrierAdapter._parse_maze 保持一致，使用 (col + 1.0, row + 1.0)
                        # adapter 默认解析时使用的是 1-based 的坐标: [w+1.0, h+1.0]
                        real_walls.append([float(c) + 1.0, float(r) + 1.0])
            
            # 使用动态的墙壁数据
            diffusion.neural_cbf.wall_centers_tensor = torch.tensor(
                real_walls, dtype=torch.float32, device=device
            )
            print(f"已同步真实环境 {len(real_walls)} 个墙壁坐标到 CBF！")
            
        print("CBF 避障开启")
    else:
        if hasattr(diffusion, 'neural_cbf'):
            saved_adapter = diffusion.neural_cbf
            del diffusion.neural_cbf 
        print("CBF 避障已关闭 (Baseline)")

    # 2. 数据预处理：归一化 + 转 Tensor
    cond_batch = {}
    for k, v in cond.items():
        # v 是物理坐标 (numpy)，先进行归一化
        v_norm = dataset.normalizer.normalize(v, 'observations')
        v_norm_batch = v_norm[None, ...]
        cond_batch[k] = torch.tensor(v_norm_batch, dtype=torch.float32, device=device)
    
    # 3. 计算正确的轨迹总维度 (Action + Observation)
    transition_dim = diffusion.observation_dim + diffusion.action_dim

    # 4. 运行扩散采样
    samples = diffusion.p_sample_loop(
        shape=(1, diffusion.horizon, transition_dim), 
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


# 3. 运行对比实验


# --- A. 设定起点和终点 ---
start_pos = np.array([7.0, 1.0]) 
target_pos = np.array([7.0, 9.0]) # 你可以改这个坐标测试不同的墙

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


# 4. 像 plan_maze2d 一样直接画矩阵
print("\n绘制最终对比图")

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

# --- 诊断：计算轨迹到墙的最小距离、碰撞统计，并采样 h/grad ---
def trajectory_collision_and_cbf_stats(traj_phys, adapter, threshold=0.5, sample_n=8):
    """
    traj_phys: (T, 2) array with [row, col]
    adapter: diffusion.neural_cbf (NeuralBarrierAdapter)
    threshold: 距离阈值，低于认为碰撞（单位为格子单位）
    返回: dict with min_dists, collision_mask, sample_h_grad
    """
    # wall_centers_tensor 存储为 (col+1, row+1)
    wall_centers = adapter.wall_centers_tensor.detach().cpu().numpy()  # (N,2) as (col+1, row+1)

    # 将 traj_phys ([row, col]) 转换为 adapter 坐标系: (col+1, row+1)
    traj_adapter = np.stack([traj_phys[:, 1] + 1.0, traj_phys[:, 0] + 1.0], axis=1)

    # 计算每个轨迹点到所有墙心的最小距离
    dists = np.sqrt(((traj_adapter[:, None, :] - wall_centers[None, :, :]) ** 2).sum(axis=2))  # (T, N)
    min_dists = dists.min(axis=1)
    collisions = min_dists < threshold

    # 采样若干点计算 h 和 grad（在 adapter 坐标系下构造 phys_state）
    T = traj_adapter.shape[0]
    idxs = np.linspace(0, T - 1, min(sample_n, T)).astype(int)
    sample_h = []
    sample_grad = []
    with torch.no_grad():
        for i in idxs:
            pos = traj_adapter[i]
            state = torch.tensor([[pos[0], pos[1], 0.0, 0.0]], dtype=torch.float32, device=adapter.device)
            h_val, grad = adapter.get_correction_gradient(state)
            sample_h.append(float(h_val.cpu().numpy().ravel()[0]))
            sample_grad.append(grad.cpu().numpy().ravel().tolist())

    return {
        'min_dists': min_dists,
        'collisions': collisions,
        'collision_count': int(collisions.sum()),
        'sample_idxs': idxs.tolist(),
        'sample_h': sample_h,
        'sample_grad': sample_grad,
    }


print("\n--- 运行碰撞与 CBF 诊断 ---")
if hasattr(diffusion, 'neural_cbf') and diffusion.neural_cbf is not None:
    adapter = diffusion.neural_cbf
    stats_safe = trajectory_collision_and_cbf_stats(traj_safe_phys[:, :2], adapter)
    stats_unsafe = trajectory_collision_and_cbf_stats(traj_unsafe_phys[:, :2], adapter)

    print(f"Safe traj collisions: {stats_safe['collision_count']} / {len(stats_safe['min_dists'])}")
    print(f"Unsafe traj collisions: {stats_unsafe['collision_count']} / {len(stats_unsafe['min_dists'])}")
    print("Safe sample h/grad:")
    for i, h in enumerate(stats_safe['sample_h']):
        print(f" idx {stats_safe['sample_idxs'][i]}: h={h:.4f}, grad={stats_safe['sample_grad'][i]}")

    # 额外诊断：哪些碰撞点实际上对应的地图像素不是墙（灰色区域）？打印详细信息
    bad_collision_idxs = []
    for i in np.where(stats_safe['collisions'])[0]:
        r = int(round(traj_safe_phys[i, 0]))
        c = int(round(traj_safe_phys[i, 1]))
        # 防越界
        if r < 0 or r >= maze_arr.shape[0] or c < 0 or c >= maze_arr.shape[1]:
            continue
        if maze_arr[r, c] != 10:
            bad_collision_idxs.append(i)

    if len(bad_collision_idxs) > 0:
        print("\n被判为碰撞但地图像素并非墙 (maze_arr != 10) 的轨迹索引与详情：")
        wc = adapter.wall_centers_tensor.detach().cpu().numpy()
        for i in bad_collision_idxs:
            r = int(round(traj_safe_phys[i, 0]))
            c = int(round(traj_safe_phys[i, 1]))
            pos_adapter = np.array([c + 1.0, r + 1.0])
            dists_to_wc = np.linalg.norm(wc - pos_adapter[None, :], axis=1)
            nearest_idx = int(np.argmin(dists_to_wc))
            nearest_wc = wc[nearest_idx]
            nearest_dist = float(dists_to_wc[nearest_idx])
            print(f" idx={i}, traj_pos=(row={r},col={c}), maze_val={maze_arr[r,c]}, min_dist={stats_safe['min_dists'][i]:.3f}")
            print(f"   nearest_wall_center_idx={nearest_idx}, center(col+1,row+1)={nearest_wc.tolist()}, dist_to_center={nearest_dist:.3f}")
            # 打印附近地图小窗口
            r0 = max(0, r-2); r1 = min(maze_arr.shape[0], r+3)
            c0 = max(0, c-2); c1 = min(maze_arr.shape[1], c+3)
            print("   nearby maze window (rows %d:%d, cols %d:%d):" % (r0, r1, c0, c1))
            print(maze_arr[r0:r1, c0:c1])
    else:
        print("\n没有发现“灰色区域却被判碰撞”的点（maze_arr != 10 的碰撞点）。")

    # 保存带墙心与碰撞标记的诊断图
    fig2, ax2 = plt.subplots(figsize=(12, 12))
    ax2.imshow(maze_arr, cmap='gray', origin='upper')

    # 画轨迹
    ax2.plot(traj_unsafe_phys[:, 1], traj_unsafe_phys[:, 0], color='red', linestyle='--', linewidth=3, label='Original (Unsafe)')
    ax2.plot(traj_safe_phys[:, 1], traj_safe_phys[:, 0], color='#00FF00', linewidth=3, label='SafeDiffuser (CBF)')

    # 画墙心（还原到像素坐标：col+1,row+1 -> col,row）
    wc = adapter.wall_centers_tensor.detach().cpu().numpy()
    if wc.shape[0] > 0:
        ax2.scatter(wc[:, 0] - 1.0, wc[:, 1] - 1.0, c='yellow', s=10, alpha=0.6, label='wall_centers')

    # 标出碰撞点（safe）
    coll_idxs = np.where(stats_safe['collisions'])[0]
    if coll_idxs.size > 0:
        ax2.scatter(traj_safe_phys[coll_idxs, 1], traj_safe_phys[coll_idxs, 0], c='magenta', s=50, marker='x', label='collisions')

    ax2.scatter(start_pos[1], start_pos[0], color='blue', s=400, marker='*', label='Start', zorder=10)
    ax2.scatter(target_pos[1], target_pos[0], color='gold', s=400, marker='X', label='Target', zorder=10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.legend(loc='upper right')
    diag_path = 'final_comparison_diagnostics.png'
    plt.savefig(diag_path, dpi=150, bbox_inches='tight')
    print(f"诊断图已保存: {diag_path}")
else:
    print("未检测到 diffusion.neural_cbf，跳过 CBF 诊断")

# --- 诊断结束 ---

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

