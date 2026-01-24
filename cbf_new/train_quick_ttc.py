import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import patches
import diffuser.utils as utils
import os

# ================= 配置区域 =================
MAZE_MAP_LARGE = [
    "OOOOOOOOOOOO",
    "OOO#OOOOOOOO",
    "OOO#OOOOOOOO",
    "OOOOOOOOOOOO",
    "OOOOO#OOOOOO",
    "OOOOO#OOOOOO",
    "OOOOO#OOOOOO",
    "OOOOOOOOOOOO",
    "OOOOOOOOOOOO",
]

USE_MAZE_MAP_OBSTACLES = True

# 安全余量
SAFETY_BUFFER = 0.15                     

# 【修改点1】加速参数与全图范围
ROI_RADIUS = 4.0       # 扩大范围以便覆盖全图
TTC_LOOKAHEAD = 1.0    
BATCH_SIZE = 4096      # 【加速】加大 Batch Size
LR = 2e-3              # 调大学习率
EPOCHS = 20            # 【加速】20 轮足够看清全图了
MAX_TRAJECTORIES = 300 # 【加速】只用前 300 条轨迹
# ===========================================

# -----------------------------------------------------------------------------#
# 1. 加载并筛选数据 
# -----------------------------------------------------------------------------#
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')

# 加载 Dataset
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

def get_valid_length(episode_idx):
    term_idxs = np.where(all_terminals[episode_idx] > 0.5)[0]
    time_idxs = np.where(all_timeouts[episode_idx] > 0.5)[0]
    if len(term_idxs) > 0: return term_idxs[0] + 1
    elif len(time_idxs) > 0: return time_idxs[0] + 1
    obs = all_obs[episode_idx]
    non_zero_idxs = np.nonzero(np.sum(np.abs(obs), axis=1))[0]
    if len(non_zero_idxs) > 0: return non_zero_idxs[-1] + 1
    return 0

# 计算矩形距离 (SDF)
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
            if not grid[r, c] or visited[r, c]: continue
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

            # Maze2D 的物理坐标一般是 grid 索引 + 1.0，而且 XY对应行列有区别
            # 这里统一用绝对坐标计算
            center_x = (c_min + c_max + 1) / 2.0 + 1.0
            center_y = (r_min + r_max + 1) / 2.0 + 1.0
            half_w = (c_max - c_min + 1) / 2.0
            half_h = (r_max - r_min + 1) / 2.0

            obstacles.append({
                "center": np.array([center_x, center_y], dtype=np.float32),
                "half_extents": np.array([half_w, half_h], dtype=np.float32),
            })

    return obstacles

OBSTACLES = parse_maze_map_to_obstacles(MAZE_MAP_LARGE)
print(f"✅ 使用墙数量: {len(OBSTACLES)}")

print(f"正在全图提取数据...")
training_data = []
training_labels = []

# 【修改点2】限制数据量，只取前 MAX_TRAJECTORIES 条轨迹加速
for i in range(min(MAX_TRAJECTORIES, all_obs.shape[0])):
    valid_len = get_valid_length(i)
    if valid_len < 2: continue
    
    pos = all_obs[i, :valid_len, :2]
    vel = all_obs[i, :valid_len, 2:4]
    
    # 预测未来位置
    future_pos = pos + vel * TTC_LOOKAHEAD
    
    # 计算到所有墙的 Box 距离 (使用绝对坐标)
    dists_now = [get_box_distance(pos - obs["center"], obs["half_extents"]) for obs in OBSTACLES]
    dists_future = [get_box_distance(future_pos - obs["center"], obs["half_extents"]) for obs in OBSTACLES]
    
    # 取最危险的距离 (离任意墙最近的距离)
    min_dist_now = np.min(np.stack(dists_now, axis=0), axis=0)
    min_dist_future = np.min(np.stack(dists_future, axis=0), axis=0)
    
    # 粗筛选：只保留离墙较近的数据点
    mask = min_dist_now < ROI_RADIUS
    if np.sum(mask) == 0: continue
    
    # 标签：当前和未来最危险的时刻，减去安全余量
    labels = np.minimum(min_dist_now[mask], min_dist_future[mask]) - SAFETY_BUFFER
    
    # 【核心修改】Inputs 必须是绝对坐标 pos，不能减 center！
    inputs = np.concatenate([pos[mask], vel[mask]], axis=1)
    
    training_data.append(inputs)
    training_labels.append(labels)

X_np = np.concatenate(training_data, axis=0)
y_np = np.concatenate(training_labels, axis=0)

print(f"筛选完成！数据集大小: {X_np.shape} (数据量变小了，速度快了！)")
print(f"危险样本数: {np.sum(y_np < 0)} / {len(y_np)} ({np.sum(y_np < 0)/len(y_np)*100:.2f}%)")

# -----------------------------------------------------------------------------#
# 2. 定义 TTC 网络 
# -----------------------------------------------------------------------------#
class SafetyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 【修改点3】稍微把网络变宽一点 (128->256)，因为它需要记全图两个框的位置
        self.net = nn.Sequential(
            nn.Linear(4, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) 
        )
    
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------#
# 3. 开始训练
# -----------------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SafetyNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss() 

dataset_torch = TensorDataset(torch.FloatTensor(X_np), torch.FloatTensor(y_np).unsqueeze(1))
train_loader = DataLoader(dataset_torch, batch_size=BATCH_SIZE, shuffle=True)

print(f"\n🚀 开始快速训练全图模型 (Device: {device})...")
losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "ttc_model_dataset.pth")
print("✅ 模型已保存为 ttc_model_dataset.pth")

# -----------------------------------------------------------------------------#
# 4. 【核心修改】验证可视化：画出全图的两个墙！
# -----------------------------------------------------------------------------#
print("正在生成全图验证图...")
model.eval()
with torch.no_grad():
    # 【修改点4】网格范围改为整个迷宫的范围 X:[0,12], Y:[0,10]
    x = np.linspace(0, 12, 150)
    y = np.linspace(0, 10, 150)
    xx, yy = np.meshgrid(x, y)
    
    # 绝对坐标的静态安全场 (v=0)
    in_static = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())], axis=1)
    out_static = model(torch.FloatTensor(in_static).to(device)).cpu().numpy().reshape(xx.shape)

# 画单张大图，看全貌
plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, out_static, levels=20, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(label='Safety Score')
# 画出安全边界线 (Score = 0 的线)，此时应该是两个分开的框！
plt.contour(xx, yy, out_static, levels=[0], colors='black', linewidths=2, linestyles='--')

# 画出所有墙壁的黄色真实边框供对比
for obs in OBSTACLES:
    c = obs["center"]
    hw = obs["half_extents"]
    rect = patches.Rectangle((c[0]-hw[0], c[1]-hw[1]), hw[0]*2, hw[1]*2,
                             fill=False, color='yellow', linewidth=3)
    plt.gca().add_patch(rect)

plt.title("Global Maze Safety Field (Absolute Coords)\nYou should see TWO SEPARATE boxes!")
plt.xlabel("X (Absolute)")
plt.ylabel("Y (Absolute)")
plt.grid(True, alpha=0.3)
plt.axis('equal') # 保证地图长宽比例不失真

plt.savefig("ttc_verification_box.png", dpi=150)
print("✅ 验证图已保存为 ttc_verification_box.png，快去看看吧！")