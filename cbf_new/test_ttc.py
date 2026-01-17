import os
import numpy as np
import torch
import torch.optim as optim
from diffuser.models.cbf_network import CBFNetwork

# ==========================================
# 1. 稳健的 TTC 采样器 (参考 SafetyGym)
# ==========================================
class RobustTTCSampler:
    def __init__(self):
        # 墙壁范围 (Maze2D wall 0.5 + Robot 0.15)
        self.wall_radius = 0.65 
        # 预测时间 (SafetyGym logic: lookahead)
        self.ttc_time = 1.0 
        
    def check_collision(self, pos):
        """检查是否在墙里 (Box 判定)"""
        in_x = np.abs(pos[:, 0]) < self.wall_radius
        in_y = np.abs(pos[:, 1]) < self.wall_radius
        return in_x & in_y

    def generate_unsafe_batch(self, n):
        """专门生成 '危险' 数据"""
        obs_list = []
        count = 0
        while count < n:
            # 策略 A: 直接生成在墙里的点 (50%)
            # 策略 B: 生成在墙附近，但速度冲向墙的点 (50% -> TTC case)
            
            # 先生成一批候选点
            candidates_pos = np.random.uniform(-1.5, 1.5, size=(n, 2))
            candidates_vel = np.random.uniform(-2.0, 2.0, size=(n, 2))
            
            # 预测未来位置
            future_pos = candidates_pos + candidates_vel * self.ttc_time
            
            # 判定
            now_crash = self.check_collision(candidates_pos)
            future_crash = self.check_collision(future_pos)
            
            # 只要满足任一条件，就是 Unsafe
            is_unsafe = now_crash | future_crash
            
            # 筛选出 unsafe 的
            valid_pos = candidates_pos[is_unsafe]
            valid_vel = candidates_vel[is_unsafe]
            
            for p, v in zip(valid_pos, valid_vel):
                if count >= n: break
                obs_list.append(np.concatenate([p, v]))
                count += 1
                
        return np.array(obs_list)

    def generate_safe_batch(self, n):
        """专门生成 '安全' 数据"""
        obs_list = []
        count = 0
        while count < n:
            # 生成在外面的点
            candidates_pos = np.random.uniform(-2.5, 2.5, size=(n*2, 2))
            candidates_vel = np.random.uniform(-1.0, 1.0, size=(n*2, 2))
            
            future_pos = candidates_pos + candidates_vel * self.ttc_time
            
            now_crash = self.check_collision(candidates_pos)
            future_crash = self.check_collision(future_pos)
            
            # 必须现在没撞，未来也没撞，才算 Safe
            is_safe = ~(now_crash | future_crash)
            
            valid_pos = candidates_pos[is_safe]
            valid_vel = candidates_vel[is_safe]
            
            for p, v in zip(valid_pos, valid_vel):
                if count >= n: break
                obs_list.append(np.concatenate([p, v]))
                count += 1
                
        return np.array(obs_list)

    def get_batch(self, batch_size=256):
        half = batch_size // 2
        
        # 分别生成，杜绝递归报错
        unsafe_data = self.generate_unsafe_batch(half)
        safe_data = self.generate_safe_batch(batch_size - half)
        
        # 拼起来
        batch_obs = np.concatenate([unsafe_data, safe_data], axis=0)
        
        # 生成标签 (-1: Unsafe, 1: Safe)
        labels = np.concatenate([
            -1 * np.ones(len(unsafe_data)), 
             1 * np.ones(len(safe_data))
        ])
        
        # 打乱顺序
        perm = np.random.permutation(len(labels))
        return batch_obs[perm], labels[perm]

# ==========================================
# 2. 训练循环
# ==========================================
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CBFNetwork(input_dim=4, output_dim=1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    sampler = RobustTTCSampler()
    
    print("🚀 开始 TTC 增强训练 (修复版)...")
    print(f"预测时间 (TTC Lookahead): {sampler.ttc_time}s")
    
    steps = 10000
    batch_size = 256
    
    for step in range(steps):
        # 1. 获取数据
        obs_np, labels_np = sampler.get_batch(batch_size)
        
        b_obs = torch.FloatTensor(obs_np).to(device)
        b_labels = torch.FloatTensor(labels_np).to(device).unsqueeze(1)
        
        # 2. 前向传播
        h_val = model(b_obs)
        
        # 3. Loss 计算
        # 这里的 margin 稍微放大一点，让边界更清晰
        # Safe: h > 0.5
        loss_safe = torch.relu(0.5 - h_val) * (b_labels > 0).float()
        
        # Unsafe: h < -0.5
        loss_unsafe = torch.relu(h_val + 0.5) * (b_labels < 0).float()
        
        # 增加一个正则项，防止 h 值爆炸
        loss_reg = 0.01 * (h_val ** 2).mean()
        
        total_loss = loss_safe.mean() + loss_unsafe.mean() + loss_reg
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            with torch.no_grad():
                # --- 验证案例 ---
                # 墙在 (0,0), 半径 0.65
                
                # Case 1: 绝对安全 (2.0, 2.0) 静止
                t_safe = torch.tensor([[2.0, 2.0, 0, 0]], device=device)
                
                # Case 2: 危险边缘 (1.0, 0) 静止 -> 离墙 0.35m
                # 虽然近，但是静止，应该是安全的(正数)
                t_close = torch.tensor([[1.0, 0.0, 0, 0]], device=device)
                
                # Case 3: TTC 危险 (1.5, 0) 冲向墙 v=(-2, 0)
                # 1.0s 后位置: 1.5 - 2.0 = -0.5 (撞) -> 应该是负数
                t_ttc = torch.tensor([[1.5, 0.0, -2.0, 0.0]], device=device)
                
                h_safe = model(t_safe).item()
                h_close = model(t_close).item()
                h_ttc = model(t_ttc).item()
                
            print(f"Step {step}: Loss={total_loss.item():.4f}")
            print(f"   Safe  (2.0, 0) : {h_safe:.3f}")
            print(f"   Close (1.0, 0) : {h_close:.3f} (期望 > 0)")
            print(f"   TTC   (1.5, -2): {h_ttc:.3f} (期望 < 0 !!)")

    torch.save(model.state_dict(), "cbf_maze2d.pth")
    print("✅ 模型已保存")

if __name__ == "__main__":
    train()