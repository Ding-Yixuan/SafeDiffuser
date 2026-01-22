# 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import diffuser.utils as utils
import os

# ================= 配置区域 =================
# 1. 障碍物参数
OBSTACLE_CENTER = np.array([1.5, 5.0])  # 修正后的中心
WALL_RADIUS = 0.5                       # 墙的物理半径 (大概半个格子)
SAFETY_BUFFER = 0.15                     # 安全余量
SAFE_THRESHOLD = WALL_RADIUS + SAFETY_BUFFER # 0.65m

# 2. 训练参数
ROI_RADIUS = 3.0       # 筛选数据的范围 (只看墙周围3米的数据)
TTC_LOOKAHEAD = 1.0    # 预测未来几秒 (让速度v发挥作用的关键!)
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
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

print(f"正在筛选 ROI ({ROI_RADIUS}m) 内的数据...")
training_data = []
training_labels = []

for i in range(all_obs.shape[0]):
    valid_len = get_valid_length(i)
    if valid_len < 2: continue
    
    traj = all_obs[i, :valid_len, :]
    pos = traj[:, :2]
    vel = traj[:, 2:4] # 必须包含速度
    
    # 1. 计算当前距离
    rel_pos = pos - OBSTACLE_CENTER
    dist_now = np.linalg.norm(rel_pos, axis=1)
    
    # 2. 筛选在 ROI 内的数据
    mask = dist_now < ROI_RADIUS
    if np.sum(mask) == 0: continue
    
    sel_rel_pos = rel_pos[mask]
    sel_vel = vel[mask]
    sel_dist_now = dist_now[mask]
    
    # 3. [关键] 生成动态标签 (让 v 发挥作用)
    # 预测未来位置: p_future = p_now + v * t
    future_rel_pos = sel_rel_pos + sel_vel * TTC_LOOKAHEAD
    dist_future = np.linalg.norm(future_rel_pos, axis=1)
    
    # 标签逻辑: 取当前和未来最危险的那一刻
    # Label = min(dist_now, dist_future) - SAFE_THRESHOLD
    # 结果 < 0 表示危险 (Unsafe), > 0 表示安全 (Safe)
    min_dist = np.minimum(sel_dist_now, dist_future)
    labels = min_dist - SAFE_THRESHOLD
    
    # 拼接输入: [rx, ry, vx, vy]
    inputs = np.concatenate([sel_rel_pos, sel_vel], axis=1)
    
    training_data.append(inputs)
    training_labels.append(labels)

if len(training_data) == 0:
    print("错误：没有筛选到数据！请检查坐标。")
    exit()

X_np = np.concatenate(training_data, axis=0)
y_np = np.concatenate(training_labels, axis=0)

print(f"筛选完成！数据集大小: {X_np.shape}")
print(f"标签统计: Min={y_np.min():.3f}, Max={y_np.max():.3f}")
print(f"危险样本数 (Label < 0): {np.sum(y_np < 0)} / {len(y_np)} ({np.sum(y_np < 0)/len(y_np)*100:.2f}%)")

# -----------------------------------------------------------------------------#
# 2. 定义 TTC 网络 (简单的 MLP)
# -----------------------------------------------------------------------------#
class SafetyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
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
torch.save(model.state_dict(), "ttc_model_dataset.pth")
print("模型已保存为 ttc_model_dataset.pth")

# -----------------------------------------------------------------------------#
# 4. 验证可视化 (证明 v 被学到了)
# -----------------------------------------------------------------------------#
print("正在生成验证图...")
model.eval()
with torch.no_grad():
    # 生成网格位置, 假设速度为 0 (静态安全场)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    xx, yy = np.meshgrid(x, y)
    
    # Case 1: 速度为 0
    in_static = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())], axis=1)
    out_static = model(torch.FloatTensor(in_static).to(device)).cpu().numpy().reshape(xx.shape)
    
    # Case 2: 速度冲向右上方 (vx=2, vy=2)
    # 这时候左下角的区域应该变得更危险 (红色区域扩大)
    in_dynamic = np.stack([xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), 2.0), np.full_like(xx.ravel(), 2.0)], axis=1)
    out_dynamic = model(torch.FloatTensor(in_dynamic).to(device)).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, out_static, levels=20, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(label='Safety Score')
plt.contour(xx, yy, out_static, levels=[0], colors='black', linewidths=2, linestyles='--')
plt.title(f"Static Safety Field (v=0)\nCenter={OBSTACLE_CENTER}")
plt.xlabel("Rel X")
plt.ylabel("Rel Y")

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, out_dynamic, levels=20, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(label='Safety Score')
plt.contour(xx, yy, out_dynamic, levels=[0], colors='black', linewidths=2, linestyles='--')
plt.arrow(0, 0, 0.5, 0.5, head_width=0.1, color='yellow', label='Vel Direction') # 画个箭头示意速度
plt.title("Dynamic Safety Field (v=[2, 2])\nNote: Danger zone should shift!")
plt.xlabel("Rel X")

plt.savefig("ttc_verification.png")
print("验证图已保存为 ttc_verification.png")