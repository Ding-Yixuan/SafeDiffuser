# PYTHONPATH=. /home/lqz27/anaconda3/envs/diffuser_train_dyx/bin/python /home/lqz27/dyx_ws/SafeDiffuser/cbf_new/trainfinal.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import patches
import diffuser.utils as utils
import os

# ================= 1. 配置区域 =================
# 迷宫地图定义 (9行 x 12列)
# MAZE_MAP_LARGE = [
#     "OOOOOOOOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOO#OOOOOO",
#     "OOOOOOOOOOOO",
#     "OOOOOOOOOOOO",
# ]
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

SAFETY_BUFFER = 0.15      # 安全余量
ROI_RADIUS = 4.0          # 筛选数据的范围
TTC_LOOKAHEAD = 1.0       # 预测未来几秒
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 500
NUM_TRAJ_TO_PLOT = 50     # 可视化时画多少条轨迹

# 环境变量覆盖
EPOCHS = int(os.environ.get("TTC_EPOCHS", EPOCHS))
BATCH_SIZE = int(os.environ.get("TTC_BATCH_SIZE", BATCH_SIZE))
LR = float(os.environ.get("TTC_LR", LR))

# ================= 2. 辅助函数定义 =================
def get_box_distance(rel_pos, half_size):
    """计算点到矩形表面的距离 (SDF)"""
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

            # 【关键】X = Row, Y = Col (与数据集坐标系对齐)
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

# ================= 3. 加载数据集 =================
print("正在加载数据集...")
class Parser(utils.Parser):
    dataset: str = 'maze2d-custom-v1'
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
print(f"✅ 成功解析墙体数量: {len(OBSTACLES)}")

# ================= 4. 数据提取与筛选 =================
print(f"正在全图提取并筛选数据...")
training_data = []
training_labels = []

for i in range(all_obs.shape[0]):
    valid_len = get_valid_length(i, all_obs, all_terminals, all_timeouts)
    if valid_len < 2: continue
    
    pos = all_obs[i, :valid_len, :2]
    vel = all_obs[i, :valid_len, 2:4]
    
    future_pos = pos + vel * TTC_LOOKAHEAD
    
    dists_now = [get_box_distance(pos - obs["center"], obs["half_extents"]) for obs in OBSTACLES]
    dists_future = [get_box_distance(future_pos - obs["center"], obs["half_extents"]) for obs in OBSTACLES]
    
    min_dist_now = np.min(np.stack(dists_now, axis=0), axis=0)
    min_dist_future = np.min(np.stack(dists_future, axis=0), axis=0)
    
    mask = min_dist_now < ROI_RADIUS
    if np.sum(mask) == 0: continue
    
    labels = np.minimum(min_dist_now[mask], min_dist_future[mask]) - SAFETY_BUFFER
    inputs = np.concatenate([pos[mask], vel[mask]], axis=1)
    
    training_data.append(inputs)
    training_labels.append(labels)

X_np = np.concatenate(training_data, axis=0)
y_np = np.concatenate(training_labels, axis=0)

print(f"筛选完成！数据集大小: {X_np.shape}")
print(f"危险样本数: {np.sum(y_np < 0)} / {len(y_np)} ({np.sum(y_np < 0)/len(y_np)*100:.2f}%)")

# ================= 5. 定义 TTC 网络并训练 =================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SafetyNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)

criterion = nn.MSELoss()

dataset_torch = TensorDataset(torch.FloatTensor(X_np), torch.FloatTensor(y_np).unsqueeze(1))
train_loader = DataLoader(dataset_torch, batch_size=BATCH_SIZE, shuffle=True)

print(f"\n开始训练 (Device: {device})...")
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
    scheduler.step()
    losses.append(avg_loss)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "ttc_model_small_2ob.pth")

# 保存 Loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(losses) + 1), losses, color='blue')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("TTC Training Loss")
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0, top=losses[0])
plt.tight_layout()
plt.savefig("ttc_training_loss.png", dpi=150)

# ================= 6. 终极验证：模型等高线 + 实际轨迹 =================
print("\n正在生成融合验证图 (安全场 + 轨迹点)...")
model.eval()
with torch.no_grad():
    # 生成网格，注意 X 范围[0, 10]，Y 范围[0, 12]
    x = np.linspace(0, 10, 150)
    y = np.linspace(0, 12, 150)
    xx, yy = np.meshgrid(x, y)
    
    # 静态安全场 (v=0)
    in_static = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())], axis=1)
    out_static = model(torch.FloatTensor(in_static).to(device)).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(10, 12))

# 6.1 画模型的安全得分等高线
contour = plt.contourf(xx, yy, out_static, levels=20, cmap='RdBu', vmin=-1, vmax=1, alpha=0.6)
plt.colorbar(contour, label='Safety Score')
plt.contour(xx, yy, out_static, levels=[0], colors='black', linewidths=2, linestyles='--')

# 6.2 画墙壁实体 (黄框)
for obs in OBSTACLES:
    c = obs["center"]
    hw = obs["half_extents"]
    rect = patches.Rectangle((c[0]-hw[0], c[1]-hw[1]), hw[0]*2, hw[1]*2,
                             fill=True, color='yellow', alpha=0.5, edgecolor='black', linewidth=2)
    plt.gca().add_patch(rect)

# 6.3 画真实轨迹点 (散点)
total_selected = 0
for i in range(min(NUM_TRAJ_TO_PLOT, all_obs.shape[0])):
    valid_len = get_valid_length(i, all_obs, all_terminals, all_timeouts)
    if valid_len < 2: continue
    
    pos = all_obs[i, :valid_len, :2]
    
    # 计算当前轨迹点到墙的距离以区分颜色
    dists_now = [get_box_distance(pos - obs["center"], obs["half_extents"]) for obs in OBSTACLES]
    min_dist_now = np.min(np.stack(dists_now, axis=0), axis=0)
    mask = min_dist_now < ROI_RADIUS
    
    selected_pos = pos[mask]
    discarded_pos = pos[~mask]
    total_selected += len(selected_pos)

    # 画轨迹底纹线
    plt.plot(pos[:, 0], pos[:, 1], color='black', alpha=0.05, linewidth=0.5)
    
    # 画不在 ROI 的点 (灰色)
    if len(discarded_pos) > 0:
        plt.scatter(discarded_pos[:, 0], discarded_pos[:, 1], color='gray', s=5, alpha=0.15)
    # 画在 ROI 的点 (红色，也是网络实际训练使用的数据区域)
    if len(selected_pos) > 0:
        plt.scatter(selected_pos[:, 0], selected_pos[:, 1], color='red', s=5, alpha=0.8)

# 添加图例用的虚拟点
plt.scatter([], [], color='red', s=15, alpha=0.8, label=f'Trained Data (Dist < {ROI_RADIUS}m)')
plt.scatter([], [], color='gray', s=15, alpha=0.3, label='Ignored Data')
plt.plot([], [], color='black', linestyle='--', linewidth=2, label='Model Boundary (Score=0)')

# 6.4 图形收尾
plt.title(f"Global Maze Safety Field & Trajectories\nRed points show training data ({total_selected} steps)")
plt.xlabel("X (Row in Array)")
plt.ylabel("Y (Col in Array)")
plt.xlim(0, 10) 
plt.ylim(0, 12) 
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.axis('equal')

plt.tight_layout()
plt.savefig("ttc_verification_combined.png", dpi=200)
print("✅ 终极验证图已保存为 ttc_verification_combined.png！")