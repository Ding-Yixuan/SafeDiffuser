import os
import numpy as np
import torch
import torch.optim as optim
from diffuser.models.cbf_network import CBFNetwork


# 1. 定义数据采样器
class DataSampler:
    def __init__(self):
        # 墙壁定义：中心在 (0,0)，宽高 1.0 (即 x,y 范围 [-0.5, 0.5])
        # 加上机器人半径 0.15，实际上危险区域是 [-0.65, 0.65]
        self.unsafe_bound = 0.65 
        
    def get_batch(self, batch_size=256):
        obs_list = []
        label_list = [] # 1=Safe, -1=Unsafe
        
        half_batch = batch_size // 2
        
        # --- A. 采样 Unsafe 数据 (强制采样一半) ---
        # 直接在 [-0.65, 0.65] 范围内生成点
        # 这样网络必须学会输出负数
        unsafe_pos = np.random.uniform(-self.unsafe_bound, self.unsafe_bound, size=(half_batch, 2))
        unsafe_vel = np.random.uniform(-1, 1, size=(half_batch, 2))
        
        for i in range(half_batch):
            # 标记为 Unsafe (-1)
            # 输入向量: [rel_x, rel_y, vx, vy]
            obs = np.concatenate([unsafe_pos[i], unsafe_vel[i]])
            obs_list.append(obs)
            label_list.append(-1.0) # Unsafe target
            
        # --- B. 采样 Safe 数据 (采样另一半) ---
        # 在 [-2, 2] 范围内采样，但剔除掉中心区域
        count = 0
        while count < (batch_size - half_batch):
            pos = np.random.uniform(-2, 2, size=2)
            # 只有当它真的在安全区时才采用
            if np.max(np.abs(pos)) > self.unsafe_bound + 0.05: # 留点 buffer
                vel = np.random.uniform(-1, 1, size=2)
                obs = np.concatenate([pos, vel])
                obs_list.append(obs)
                label_list.append(1.0) # Safe target
                count += 1
                
        return np.array(obs_list), np.array(label_list)

# ==========================================
# 2. 训练循环
# ==========================================
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化网络
    model = CBFNetwork(input_dim=4, output_dim=1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # 稍微加大一点学习率

    sampler = DataSampler()
    
    print("🚀 开始强化训练 Maze2D CBF (V2)...")
    print("策略：50% 安全数据 vs 50% 撞墙数据")
    
    steps = 5000
    batch_size = 256
    
    for step in range(steps):
        # 1. 获取均衡的数据集
        obs_np, labels_np = sampler.get_batch(batch_size)
        
        b_obs = torch.FloatTensor(obs_np).to(device)
        b_labels = torch.FloatTensor(labels_np).to(device).unsqueeze(1) # (B, 1)
        
        # 2. 计算 Loss
        h_val = model(b_obs)
        
        # 如果 label 是 1 (Safe)，希望 h > 0.1
        # 如果 label 是 -1 (Unsafe)，希望 h < -0.1
        
        # Loss Safe: ReLU(0.1 - h) 当 label=1
        loss_safe = torch.relu(0.1 - h_val) * (b_labels > 0).float()
        
        # Loss Unsafe: ReLU(h + 0.1) 当 label=-1 (即希望 h < -0.1)
        loss_unsafe = torch.relu(h_val + 0.1) * (b_labels < 0).float()
        
        total_loss = loss_safe.mean() + loss_unsafe.mean()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            # 简单验证一下
            with torch.no_grad():
                test_safe = torch.tensor([[2.0, 2.0, 0, 0]], dtype=torch.float32).to(device) # 远
                test_unsafe = torch.tensor([[0.0, 0.0, 0, 0]], dtype=torch.float32).to(device) # 中心
                h_s = model(test_safe).item()
                h_u = model(test_unsafe).item()
            print(f"Step {step}: Loss={total_loss.item():.4f} | h(safe)={h_s:.3f}, h(unsafe)={h_u:.3f}")

    torch.save(model.state_dict(), "cbf_new/cbf_maze2d.pth")
    print("✅ 模型已重新保存为 cbf_maze2d.pth")

if __name__ == "__main__":
    train()