import numpy as np
import matplotlib.pyplot as plt
import diffuser.utils as utils
import os

# ================= 配置区域 =================
# 1. 障碍物中心坐标 (根据 ASCII 地图估算)
# 注意：你需要看图确认这个红叉是不是正好在“墙”的位置（也就是轨迹空白处）
OBSTACLE_CENTER = np.array([1.5, 5.0]) 

# 2. 关注半径 (Region of Interest)
# 只保留距离障碍物中心这么远以内的数据
ROI_RADIUS = 2.0 

# 3. 输出文件名
OUTPUT_FILE = "ttc_training_data_walls.npy"
# ===========================================

# 1. 加载数据集
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')

# 确保路径存在
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

print(f"正在加载数据集 {args.dataset} ...")
dataset = dataset_config()
all_obs = dataset.fields['observations'] # (N_episodes, T, 4)
all_terminals = dataset.fields['terminals']
all_timeouts = dataset.fields['timeouts']

# 辅助函数：获取有效长度
def get_valid_length(episode_idx):
    term_idxs = np.where(all_terminals[episode_idx] > 0.5)[0]
    time_idxs = np.where(all_timeouts[episode_idx] > 0.5)[0]
    if len(term_idxs) > 0: return term_idxs[0] + 1
    elif len(time_idxs) > 0: return time_idxs[0] + 1
    obs = all_obs[episode_idx]
    non_zero_idxs = np.nonzero(np.sum(np.abs(obs), axis=1))[0]
    if len(non_zero_idxs) > 0: return non_zero_idxs[-1] + 1
    return 0

# -----------------------------------------------------------------------------#
# 2. 核心逻辑：筛选与转换
# -----------------------------------------------------------------------------#
print(f"正在筛选距离 {OBSTACLE_CENTER} 半径 {ROI_RADIUS} 内的数据...")

processed_data = [] # 用于存放 [rel_x, rel_y, vx, vy]
vis_segments = []   # 用于画图验证 (原始坐标)

total_points = 0
selected_points = 0

for i in range(all_obs.shape[0]):
    valid_len = get_valid_length(i)
    if valid_len < 2: continue
    
    # 取出整条轨迹: [x, y, vx, vy]
    traj = all_obs[i, :valid_len, :]
    pos = traj[:, :2] # (T, 2)
    vel = traj[:, 2:4] # (T, 2)
    
    # 计算距离障碍物中心的距离
    # dist shape: (T,)
    dist = np.linalg.norm(pos - OBSTACLE_CENTER, axis=1)
    
    # 生成掩码：哪些点在半径内
    mask = dist < ROI_RADIUS
    
    if np.sum(mask) > 0:
        # 1. 提取符合条件的点
        selected_pos = pos[mask]
        selected_vel = vel[mask]
        
        # 2. 坐标变换：绝对坐标 -> 相对坐标
        # relative_pos = [px - ox, py - oy]
        relative_pos = selected_pos - OBSTACLE_CENTER
        
        # 3. 拼接数据 [rel_x, rel_y, vx, vy]
        # 注意：速度不需要减去障碍物速度（障碍物是静止的），所以保持绝对速度即可
        # 除非你想让速度也变成相对于障碍物的方向（通常不需要，笛卡尔坐标系够用了）
        segment_data = np.concatenate([relative_pos, selected_vel], axis=1)
        
        processed_data.append(segment_data)
        vis_segments.append(selected_pos) # 存原始坐标用于画图
        
        selected_points += np.sum(mask)
    
    total_points += valid_len

# 拼接所有片段成一个大数组
if len(processed_data) > 0:
    final_dataset = np.concatenate(processed_data, axis=0)
    print(f"\n筛选完成！")
    print(f"原始总点数: {total_points}")
    print(f"选中点数:   {selected_points} (占比 {selected_points/total_points*100:.2f}%)")
    print(f"最终数据集形状: {final_dataset.shape}")
    
    # 保存数据
    np.save(OUTPUT_FILE, final_dataset)
    print(f"数据已保存至: {OUTPUT_FILE}")
else:
    print("错误：没有筛选到任何数据！请检查 OBSTACLE_CENTER 坐标是否正确。")
    exit()

# -----------------------------------------------------------------------------#
# 3. 可视化验证 (这一步非常重要)
# -----------------------------------------------------------------------------#
print("\n正在生成验证图片 check_roi.png ...")
plt.figure(figsize=(10, 10))

# 画背景轨迹 (灰色) - 只画前200条避免太乱
for i in range(min(all_obs.shape[0], 200)):
    valid_len = get_valid_length(i)
    if valid_len < 2: continue
    plt.plot(all_obs[i, :valid_len, 0], all_obs[i, :valid_len, 1], 
             color='lightgray', alpha=0.3, zorder=0)

# 画选中的片段 (蓝色点) - 降采样一下，不然点太多
vis_concat = np.concatenate(vis_segments, axis=0)
# 只画前 5000 个点用于示意
if len(vis_concat) > 5000:
    indices = np.random.choice(len(vis_concat), 5000, replace=False)
    vis_subset = vis_concat[indices]
else:
    vis_subset = vis_concat

plt.scatter(vis_subset[:, 0], vis_subset[:, 1], s=5, c='blue', alpha=0.5, label='Selected Data')

# 画障碍物中心 (红色大叉)
plt.scatter(OBSTACLE_CENTER[0], OBSTACLE_CENTER[1], s=300, c='red', marker='x', linewidth=3, label='Obstacle Center', zorder=5)

# 画关注区域圆圈
circle = plt.Circle(OBSTACLE_CENTER, ROI_RADIUS, color='red', fill=False, linestyle='--', label='ROI Radius')
plt.gca().add_patch(circle)

plt.title(f'TTC Data Selection Verification\nCenter:{OBSTACLE_CENTER}, Radius:{ROI_RADIUS}')
plt.xlabel('X (Absolute)')
plt.ylabel('Y (Absolute)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig("check_roi.png", dpi=100)
print("验证图片已保存。请查看 check_roi.png 确认红叉是否在墙壁位置！")