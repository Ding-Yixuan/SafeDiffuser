import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import diffuser.utils as utils

# ================= 配置区域 =================
MAZE_MAP_LARGE = [
    "OOOOOOOOOOOO",
    "OOOOO#OOOOOO",
    "OOOOO#OOOOOO",
    "OOOOOOOOOOOO",
    "OOOOO#OOOOOO",
    "OOOOO#OOOOOO",
    "OOOOO#OOOOOO",
    "OOOOOOOOOOOO",
    "OOOOOOOOOOOO",
]

ROI_RADIUS = 4.0       
NUM_TRAJ_TO_PLOT = 50  

# ================= 1. 辅助函数定义 =================
def get_box_distance(rel_pos, half_size):
    d = np.abs(rel_pos) - half_size
    outside_dist = np.linalg.norm(np.maximum(d, 0), axis=1)
    inside_dist = np.minimum(np.max(d, axis=1), 0)
    return outside_dist + inside_dist

def parse_maze_map_to_obstacles(map_lines):
    rows = len(map_lines)
    cols = len(map_lines[0]) if rows > 0 else 0
    grid = np.array([[c == '#' for c in line] for line in map_lines], dtype=bool)
    visited = np.zeros_like(grid, dtype=bool)
    obstacles = []

    def neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    for r in range(rows):
        for c in range(cols):
            if not grid[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in neighbors(cr, cc):
                    if grid[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows_idx = [cell[0] for cell in cells]
            cols_idx = [cell[1] for cell in cells]
            r_min, r_max = min(rows_idx), max(rows_idx)
            c_min, c_max = min(cols_idx), max(cols_idx)

            center_row = (r_min + r_max) / 2.0
            center_col = (c_min + c_max) / 2.0
            half_h = (r_max - r_min + 1) / 2.0 # 行的高度
            half_w = (c_max - c_min + 1) / 2.0 # 列的宽度

            # 【关键修改】：XY轴互换！
            # 之前：X = Col, Y = Row
            # 现在：X = Row, Y = Col
            obstacles.append({
                "center": np.array([center_row, center_col], dtype=np.float32),
                "half_extents": np.array([half_h, half_w], dtype=np.float32), 
            })
    return obstacles

def get_valid_length(episode_idx, all_obs, all_terminals, all_timeouts):
    term_idxs = np.where(all_terminals[episode_idx] > 0.5)[0]
    time_idxs = np.where(all_timeouts[episode_idx] > 0.5)[0]
    if len(term_idxs) > 0: return term_idxs[0] + 1
    elif len(time_idxs) > 0: return time_idxs[0] + 1
    obs = all_obs[episode_idx]
    non_zero_idxs = np.nonzero(np.sum(np.abs(obs), axis=1))[0]
    if len(non_zero_idxs) > 0: return non_zero_idxs[-1] + 1
    return 0

# ================= 2. 加载数据 =================
print("正在加载数据集...")
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')
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
dataset = dataset_config()
all_obs = dataset.fields['observations']
all_terminals = dataset.fields['terminals']
all_timeouts = dataset.fields['timeouts']

OBSTACLES = parse_maze_map_to_obstacles(MAZE_MAP_LARGE)
print(f"成功加载，障碍物数量: {len(OBSTACLES)}")

# ================= 3. 绘图准备 =================
plt.figure(figsize=(10, 12)) # 因为XY互换，图片比例也跟着倒过来

for idx, obs in enumerate(OBSTACLES):
    c = obs["center"]
    hw = obs["half_extents"]
    
    rect = patches.Rectangle((c[0]-hw[0], c[1]-hw[1]), hw[0]*2, hw[1]*2,
                             fill=True, color='yellow', alpha=0.5, edgecolor='black', linewidth=2)
    plt.gca().add_patch(rect)
    plt.plot(c[0], c[1], 'k+', markersize=15, markeredgewidth=2) 
    
    roi_rect = patches.FancyBboxPatch(
        (c[0]-hw[0], c[1]-hw[1]), hw[0]*2, hw[1]*2,
        boxstyle=f"round,pad={ROI_RADIUS}",
        fill=False, edgecolor='green', linestyle='--', linewidth=1.5, alpha=0.7
    )
    plt.gca().add_patch(roi_rect)

plt.scatter([], [], color='red', s=10, label=f'Selected (Dist < {ROI_RADIUS}m)')
plt.scatter([], [], color='gray', s=10, alpha=0.3, label=f'Discarded')
plt.plot([], [], 'g--', label=f'ROI Boundary ({ROI_RADIUS}m)')

# ================= 4. 遍历轨迹并分类画点 =================
print(f"正在处理并绘制前 {NUM_TRAJ_TO_PLOT} 条轨迹...")

total_selected = 0
total_discarded = 0

for i in range(min(NUM_TRAJ_TO_PLOT, all_obs.shape[0])):
    valid_len = get_valid_length(i, all_obs, all_terminals, all_timeouts)
    if valid_len < 2: continue
    
    pos = all_obs[i, :valid_len, :2]
    
    # 距离计算逻辑不变，因为 obs 的中心也互换了
    dists_now = [get_box_distance(pos - obs["center"], obs["half_extents"]) for obs in OBSTACLES]
    min_dist_now = np.min(np.stack(dists_now, axis=0), axis=0)
    
    mask = min_dist_now < ROI_RADIUS
    
    selected_pos = pos[mask]
    discarded_pos = pos[~mask]
    
    total_selected += len(selected_pos)
    total_discarded += len(discarded_pos)

    plt.plot(pos[:, 0], pos[:, 1], color='black', alpha=0.05, linewidth=0.5)
    
    if len(discarded_pos) > 0:
        plt.scatter(discarded_pos[:, 0], discarded_pos[:, 1], color='gray', s=5, alpha=0.3)
        
    if len(selected_pos) > 0:
        plt.scatter(selected_pos[:, 0], selected_pos[:, 1], color='red', s=5, alpha=0.8)

# ================= 5. 图形收尾 =================
plt.title(f"Trajectory Selection (X/Y Swapped)\n"
          f"Selected: {total_selected} | Discarded: {total_discarded}")

# 【关键修改】：坐标轴标签和范围互换
plt.xlabel("X (Row in Array)")
plt.ylabel("Y (Col in Array)")
plt.xlim(0, 10) # 之前的 Y 范围
plt.ylim(0, 12) # 之前的 X 范围
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.axis('equal') 

plt.tight_layout()
plt.savefig("trajectory_roi_swapped.png", dpi=200)
print("✅ 完成！X/Y 互换后的图已保存为 trajectory_roi_swapped.png")