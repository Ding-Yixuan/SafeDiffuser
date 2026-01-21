import os
import sys

# 获取当前脚本所在目录 (cbf_new)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (SafeDiffuser)
root_dir = os.path.dirname(current_dir)
# 把根目录加到 path 的第一位
sys.path.insert(0, root_dir)
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from diffuser.models.cbf_network import CBFNetwork

class RobustTTCSampler:
    def __init__(self):
        self.wall_radius = 0.65 
        self.ttc_time = 1.0 
        
    def get_signed_distance(self, pos):
        """
        计算点到墙壁边缘的有向距离 (Signed Distance)
        负数表示在墙内，正数表示在墙外
        Box SDF 公式: max(abs(x) - r, abs(y) - r) (简化版)
        """
        # Box SDF 近似: 取 x 和 y 方向里离得更远的那个边界距离
        dist_x = np.abs(pos[:, 0]) - self.wall_radius
        dist_y = np.abs(pos[:, 1]) - self.wall_radius
        # 这是一个近似的 Box SDF，足够用于避障训练
        # 如果都在内部，取较大的负值(离边界最近的)；如果在外部，取较大的正值
        return np.maximum(dist_x, dist_y)

    def generate_batch(self, batch_size):
        """
        混合采样策略：
        1. 随机均匀采样 (Global Exploration)
        2. 边界增强采样 (Boundary Refinement)
        """
        n_uniform = int(batch_size * 0.4)
        n_boundary = int(batch_size * 0.4)
        n_ttc = batch_size - n_uniform - n_boundary
        
        obs_list = []
        labels_list = []

        # --- A. 均匀采样 (覆盖全图) ---
        pos_u = np.random.uniform(-2.0, 2.0, size=(n_uniform, 2))
        vel_u = np.random.uniform(-2.0, 2.0, size=(n_uniform, 2))
        
        # --- B. 边界采样 (专注于墙皮附近 +/- 0.2m) ---
        # 这种数据让 h=0 的位置超级精准
        pos_b = []
        for _ in range(n_boundary):
            # 随机选 x 边还是 y 边
            if np.random.rand() > 0.5:
                # 贴近 x 边界 (+/- 0.65 附近)
                px = (self.wall_radius + np.random.uniform(-0.2, 0.2)) * np.random.choice([-1, 1])
                py = np.random.uniform(-1.0, 1.0)
            else:
                # 贴近 y 边界
                px = np.random.uniform(-1.0, 1.0)
                py = (self.wall_radius + np.random.uniform(-0.2, 0.2)) * np.random.choice([-1, 1])
            pos_b.append([px, py])
        pos_b = np.array(pos_b)
        vel_b = np.random.uniform(-2.0, 2.0, size=(n_boundary, 2))

        # --- C. TTC 采样 (模拟高速撞击) ---
        pos_t = np.random.uniform(-1.5, 1.5, size=(n_ttc, 2))
        # 速度大一点
        vel_t = np.random.uniform(-3.0, 3.0, size=(n_ttc, 2))

        # 合并所有数据
        all_pos = np.concatenate([pos_u, pos_b, pos_t], axis=0)
        all_vel = np.concatenate([vel_u, vel_b, vel_t], axis=0)
        
        # ==========================================
        # 🌟 核心升级：计算连续标签 (Regression Target)
        # ==========================================
        # 1. 当前时刻的距离
        dist_now = self.get_signed_distance(all_pos)
        
        # 2. TTC 时刻的距离
        future_pos = all_pos + all_vel * self.ttc_time
        dist_future = self.get_signed_distance(future_pos)
        
        # 3. 标签逻辑：
        # 我们希望 h(x) 能够预测 "未来最危险的那一刻离墙有多远"
        # 所以取 min(当前距离, 未来距离)
        # 还要做一个平滑截断，太远了(>1.0)就别管了，太深了(<-1.0)也别管了，专注边界
        raw_label = np.minimum(dist_now, dist_future)
        labels = np.clip(raw_label, -1.0, 1.0)

        # 拼装 Observation
        batch_obs = np.concatenate([all_pos, all_vel], axis=1)
        
        return batch_obs, labels

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CBFNetwork(input_dim=4, output_dim=1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # LR 稍微调小一点，拟合更稳
    
    # 使用 MSELoss 进行回归训练
    # 相比 ReLU Loss，这能保证全域都有梯度
    criterion = nn.MSELoss()

    sampler = RobustTTCSampler()
    
    print("🚀 开始 TTC 距离场回归训练 (Regression Mode)...")
    
    steps = 20000 # 多训一点，回归比分类难
    batch_size = 256
    
    for step in range(steps):
        obs_np, labels_np = sampler.generate_batch(batch_size)
        
        b_obs = torch.FloatTensor(obs_np).to(device)
        b_labels = torch.FloatTensor(labels_np).to(device).unsqueeze(1)
        
        # 前向传播
        h_val = model(b_obs)
        
        # Loss: 让 h(x) 尽可能拟合真实的物理距离
        loss = criterion(h_val, b_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 2000 == 0:
            with torch.enable_grad():
                # 验证梯度是否存在 (关键！)
                t_check = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
                h_check = model(t_check)
                # 求导测试
                grad_check = torch.autograd.grad(h_check, t_check)[0]
                grad_norm = grad_check.norm().item()
                
                print(f"Step {step}: MSE={loss.item():.5f} | Center Grad Norm={grad_norm:.4f}")
                # Center Grad Norm 如果是 0，说明梯度消失了，这个版本保证它不是 0

    torch.save(model.state_dict(), "cbf_maze2d11.pth")
    print("✅ 距离场模型已保存 (cbf_maze2d11.pth)")

if __name__ == "__main__":
    train()