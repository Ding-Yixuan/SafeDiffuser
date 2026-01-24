# 
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
# 1. 障碍物参数
# 支持多墙：通过地图中的 # 自动生成墙块（按连通块合并成矩形）
MAZE_MAP_LARGE = [
    "OOOOOOOOOOOO",
    "OOOOO#OOOOOO",
    "OOOOO#O#OOOO",
    "OOOOOOO#OOOO",
    "OOOOO#O#OOOO",
    "OOOOO#OOOOOO",
    "OOOOO#OOOOOO",
    "OOOOOOOOOOOO",
    "OOOOOOOOOOOO",
]

USE_MAZE_MAP_OBSTACLES = True

# 作为兜底的单墙配置（当地图解析为空或关闭时）
#OBSTACLE_CENTER = np.array([1.5, 5.0])
#HALF_EXTENTS = np.array([1.0, 0.5])

SAFETY_BUFFER = 0.15                     # 安全余量

# 2. 训练参数
ROI_RADIUS = 4.0       # 筛选数据的范围 (只看墙周围3米的数据)
TTC_LOOKAHEAD = 1.0    # 预测未来几秒 (让速度v发挥作用的关键!)
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 300

# 可选：通过环境变量快速覆盖训练参数（便于观察收敛）
EPOCHS = int(os.environ.get("TTC_EPOCHS", EPOCHS))
BATCH_SIZE = int(os.environ.get("TTC_BATCH_SIZE", BATCH_SIZE))
LR = float(os.environ.get("TTC_LR", LR))
# ===========================================

# -----------------------------------------------------------------------------#
# 1. 加载并筛选数据 (带速度 v)
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
    """
    计算点到矩形表面的距离
    返回 > 0 表示在外部，< 0 表示在内部
    """
    d = np.abs(rel_pos) - half_size
    # 外部距离 (向量长度)
    outside_dist = np.linalg.norm(np.maximum(d, 0), axis=1)
    # 内部距离 (最近边距离，负数)
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

            # 坐标约定：pos[:,0] 是“行/y”，pos[:,1] 是“列/x”
            center_row = (r_min + r_max) / 2.0
            center_col = (c_min + c_max) / 2.0
            half_h = (r_max - r_min + 1) / 2.0
            half_w = (c_max - c_min + 1) / 2.0

            obstacles.append({
                "center": np.array([center_col, center_row], dtype=np.float32),
                "half_extents": np.array([half_w, half_h], dtype=np.float32),
            })

    return obstacles


# if USE_MAZE_MAP_OBSTACLES:
#     OBSTACLES = parse_maze_map_to_obstacles(MAZE_MAP_LARGE)
#     if len(OBSTACLES) == 0:
#         OBSTACLES = [{
#             "center": OBSTACLE_CENTER.astype(np.float32),
#             "half_extents": HALF_EXTENTS.astype(np.float32),
#         }]
# else:
#     OBSTACLES = [{
#         "center": OBSTACLE_CENTER.astype(np.float32),
#         "half_extents": HALF_EXTENTS.astype(np.float32),
#     }]

# print(f"使用墙数量: {len(OBSTACLES)}")

# print(f"正在筛选 ROI ({ROI_RADIUS}m) 内的数据...")
# training_data = []
# training_labels = []

# for i in range(all_obs.shape[0]):
#     valid_len = get_valid_length(i)
#     if valid_len < 2: continue
    
#     traj = all_obs[i, :valid_len, :]
#     pos = traj[:, :2]
#     vel = traj[:, 2:4] # 必须包含速度
    
#     # 1. 计算到任意墙中心的最小距离 (用于 ROI 筛选)
#     center_dists = []
#     for obs in OBSTACLES:
#         center_dists.append(np.linalg.norm(pos - obs["center"], axis=1))
#     dist_now = np.min(np.stack(center_dists, axis=0), axis=0)

#     # 2. 筛选在 ROI 内的数据
#     mask = dist_now < ROI_RADIUS
#     if np.sum(mask) == 0: continue
    
#     sel_pos = pos[mask]
#     sel_vel = vel[mask]
#     sel_dist_now = dist_now[mask]
    
#     # 3. 使用矩形距离生成标签 (多墙取最危险)
#     dist_list = []
#     for obs in OBSTACLES:
#         rel_pos = sel_pos - obs["center"]
#         dist_now_box = get_box_distance(rel_pos, obs["half_extents"])
#         future_rel_pos = rel_pos + sel_vel * TTC_LOOKAHEAD
#         dist_future_box = get_box_distance(future_rel_pos, obs["half_extents"])
#         dist_list.append(np.minimum(dist_now_box, dist_future_box))

#     min_dist = np.min(np.stack(dist_list, axis=0), axis=0)
#     labels = min_dist - SAFETY_BUFFER
    
#     # 拼接输入: [rx, ry, vx, vy] (对每个墙中心分别生成一份训练样本)
#     for obs in OBSTACLES:
#         rel_pos = sel_pos - obs["center"]
#         inputs = np.concatenate([rel_pos, sel_vel], axis=1)
#         training_data.append(inputs)
#         training_labels.append(labels)

# if len(training_data) == 0:
#     print("错误：没有筛选到数据！请检查坐标。")
#     exit()

OBSTACLES = parse_maze_map_to_obstacles(MAZE_MAP_LARGE)
print(f"✅ 使用墙数量: {len(OBSTACLES)}")

# -----------------------------------------------------------------------------#
# 2. 【核心修改】数据筛选 (使用绝对坐标)
# -----------------------------------------------------------------------------#
print(f"正在全图提取数据...")
training_data = []
training_labels = []

for i in range(all_obs.shape[0]):
    valid_len = get_valid_length(i)
    if valid_len < 2: continue
    
    pos = all_obs[i, :valid_len, :2]
    vel = all_obs[i, :valid_len, 2:4]
    
    # 预测未来位置
    future_pos = pos + vel * TTC_LOOKAHEAD
    
    # 计算到所有墙的 Box 距离
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
    
    # 【关键修改】：Inputs 必须是绝对坐标 pos，不能减 center！
    inputs = np.concatenate([pos[mask], vel[mask]], axis=1)
    
    training_data.append(inputs)
    training_labels.append(labels)

X_np = np.concatenate(training_data, axis=0)
y_np = np.concatenate(training_labels, axis=0)

# print(f"筛选完成！数据集大小: {X_np.shape}")
# print(f"标签统计: Min={y_np.min():.3f}, Max={y_np.max():.3f}")
# print(f"危险样本数 (Label < 0): {np.sum(y_np < 0)} / {len(y_np)} ({np.sum(y_np < 0)/len(y_np)*100:.2f}%)")
print(f"筛选完成！数据集大小: {X_np.shape}")
print(f"危险样本数: {np.sum(y_np < 0)} / {len(y_np)} ({np.sum(y_np < 0)/len(y_np)*100:.2f}%)")


# -----------------------------------------------------------------------------#
# 2. 定义 TTC 网络 (简单的 MLP)
# -----------------------------------------------------------------------------#
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

# -----------------------------------------------------------------------------#
# 3. 开始训练
# -----------------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SafetyNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss() # 回归任务

# 准备 DataLoader
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
    losses.append(avg_loss)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

# 保存模型
torch.save(model.state_dict(), "ttc_model_dataset_new.pth")
print("模型已保存为 ttc_model_dataset_new.pth")

# 保存 loss 曲线
# plt.figure(figsize=(6, 4))
# plt.plot(np.arange(1, len(losses) + 1), losses, color='blue')
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.title("TTC Training Loss")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("ttc_training_loss.png", dpi=150)
# print("训练损失曲线已保存为 ttc_training_loss.png")
# 保存 loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(losses) + 1), losses, color='blue')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("TTC Training Loss")
plt.grid(True, alpha=0.3)

# ===================== 新增/修改的核心代码 =====================
# 1. 获取第一个epoch的loss值作为y轴最大值
initial_loss = losses[0]  # 第一轮的loss值
min_loss = min(losses)    # 可选：获取最小loss值作为y轴下限

# 2. 设置y轴范围：上限=初始loss，下限=0（或最小loss）
# 方式1：下限设为0（推荐，对比更直观）
plt.ylim(bottom=0, top=initial_loss)
# 方式2：下限设为最小loss（更紧凑）
# plt.ylim(bottom=min_loss * 0.9, top=initial_loss)  # 乘以0.9留一点余量
# ==============================================================

plt.tight_layout()
plt.savefig("ttc_training_loss.png", dpi=150)
print("训练损失曲线已保存为 ttc_training_loss.png")
# # -----------------------------------------------------------------------------#
# # 4. 验证可视化 (证明 v 被学到了)
# # -----------------------------------------------------------------------------#
# print("正在生成验证图...")
# model.eval()
# with torch.no_grad():
#     x = np.linspace(-2, 2, 100)
#     y = np.linspace(-2, 2, 100)
#     xx, yy = np.meshgrid(x, y)
    
#     # Case 1: 速度为 0
#     in_static = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())], axis=1)
#     out_static = model(torch.FloatTensor(in_static).to(device)).cpu().numpy().reshape(xx.shape)
    
#     # Case 2: 速度冲向右上方
#     in_dynamic = np.stack([xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), 2.0), np.full_like(xx.ravel(), 2.0)], axis=1)
#     out_dynamic = model(torch.FloatTensor(in_dynamic).to(device)).cpu().numpy().reshape(xx.shape)

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.contourf(xx, yy, out_static, levels=20, cmap='RdBu', vmin=-1, vmax=1)
# plt.colorbar(label='Safety Score')
# plt.contour(xx, yy, out_static, levels=[0], colors='black', linewidths=2, linestyles='--')
# # 画出真实的矩形边界供对比
# ref_half_extents = OBSTACLES[0]["half_extents"]
# rect = patches.Rectangle((-ref_half_extents[0], -ref_half_extents[1]), ref_half_extents[0]*2, ref_half_extents[1]*2,
#                          fill=False, color='yellow', linewidth=2, label='Wall Boundary')
# plt.gca().add_patch(rect)
# plt.title(f"Static Safety Field (v=0)\n(Should match rectangle)")
# plt.xlabel("Rel X")
# plt.ylabel("Rel Y")

# plt.subplot(1, 2, 2)
# plt.contourf(xx, yy, out_dynamic, levels=20, cmap='RdBu', vmin=-1, vmax=1)
# plt.colorbar(label='Safety Score')
# plt.contour(xx, yy, out_dynamic, levels=[0], colors='black', linewidths=2, linestyles='--')
# plt.arrow(0, 0, 0.5, 0.5, head_width=0.1, color='yellow', label='Vel Direction')
# plt.title("Dynamic Safety Field (v=[2, 2])")
# plt.xlabel("Rel X")

# plt.savefig("ttc_verification_box.png")
# print("验证图已保存为 ttc_verification_box.png")

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