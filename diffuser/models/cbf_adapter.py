import torch
import torch.nn as nn
import numpy as np
import os

# ================= 配置区域 =================
# 障碍物中心 (必须和训练时一致)
TARGET_OBSTACLE = torch.tensor([1.5, 5.0]) 

# 模型文件名 (假设在同级目录或根目录)
MODEL_FILENAME = "ttc_model_dataset.pth"
# ===========================================

class SafetyNetwork(nn.Module):
    """
    TTC 网络结构 (4 -> 128 -> 128 -> 1)
    """
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

class NeuralBarrierAdapter:
    def __init__(self, device='cuda'):
        self.device = device
        
        # 1. 自动寻找模型路径 (兼容不同运行目录)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 尝试在当前目录找
        path1 = os.path.join(current_dir, MODEL_FILENAME)
        # 尝试在上一级目录找
        path2 = os.path.join(os.path.dirname(current_dir), MODEL_FILENAME)
        # 尝试在当前工作目录找
        path3 = MODEL_FILENAME
        
        if os.path.exists(path1): model_path = path1
        elif os.path.exists(path2): model_path = path2
        elif os.path.exists(path3): model_path = path3
        else:
            print(f"[NeuralBarrier] ❌ 严重错误: 找不到模型文件 {MODEL_FILENAME}")
            print(f"请检查路径: \n1.{path1}\n2.{path2}\n3.{path3}")
            model_path = None

        # 2. 初始化网络
        self.center = TARGET_OBSTACLE.to(device)
        self.model = SafetyNetwork().to(device)
        
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[NeuralBarrier] ✅ 成功加载 TTC 模型: {model_path}")
                print(f"[NeuralBarrier] 🎯 避障中心: {self.center.cpu().numpy()}")
            except Exception as e:
                print(f"[NeuralBarrier] ❌ 加载模型失败: {e}")
            
    def get_correction_gradient(self, phys_state):
        """
        Input: phys_state [Batch, 4] -> [x, y, vx, vy] (绝对坐标)
        Output: h_val, grad_pos
        """
        # 1. Clone并开启梯度
        x_in = phys_state.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            pos = x_in[:, :2]
            vel = x_in[:, 2:4]
            
            # 2. 坐标变换 (绝对 -> 相对)
            # Input: [px - ox, py - oy, vx, vy]
            rel_pos = pos - self.center
            net_input = torch.cat([rel_pos, vel], dim=1)
            
            # 3. 前向传播
            # h > 0: 安全, h < 0: 危险
            h_val = self.model(net_input)
            
            # 4. 计算梯度 (我们希望 h 变大/更安全)
            grads = torch.autograd.grad(
                outputs=h_val,
                inputs=x_in,
                grad_outputs=torch.ones_like(h_val),
                create_graph=False,
                retain_graph=False
            )[0]
            
            # 取出对位置 (x, y) 的梯度
            grad_pos = grads[:, :2]
            
        return h_val.detach(), grad_pos.detach()