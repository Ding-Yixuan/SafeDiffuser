import torch
import numpy as np
import os
from .cbf_network import CBFNetwork

# Maze2D-Large 地图 (来自 D4RL / SafeDiffuser PDF)
MAZE_MAP_LARGE = \
"############\\"+\
"#OOOO#OOOOO#\\"+\
"#O##O#O#O#O#\\"+\
"#OOOOOO#OOO#\\"+\
"#O####O###O#\\"+\
"#OO#O#OOOOO#\\"+\
"##O#O#O#O###\\"+\
"#OO#OOO#OGO#\\"+\
"############"

class NeuralBarrierAdapter:
    def __init__(self, model_filename='cbf_maze2d.pth', device='cuda'):
        self.device = device
        
        # 1. 自动定位同级目录下的模型文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_filename)
        
        # 2. 加载 TTC 网络
        # 注意：这里 input_dim=4 (rel_x, rel_y, vx, vy)
        self.model = CBFNetwork(input_dim=4, output_dim=1)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[NeuralBarrier] ✅ 成功加载 TTC 避障模型: {model_path}")
        else:
            print(f"[NeuralBarrier] ❌ 警告: 找不到模型文件 {model_path}，避障功能将失效！")
            
        self.model.to(device)
        self.model.eval()
        
        # 3. 解析地图
        self.wall_centers = self._parse_maze(MAZE_MAP_LARGE)
        self.wall_centers_tensor = torch.tensor(self.wall_centers, dtype=torch.float32, device=device)

    def _parse_maze(self, maze_str):
        """解析 ASCII 地图为墙壁中心坐标列表"""
        lines = maze_str.strip().split('\\')
        walls = []
        for h, row in enumerate(lines):
            for w, char in enumerate(row):
                if char == '#':
                    # 物理坐标转换: x = col + 1, y = row + 1
                    walls.append([w + 1.0, h + 1.0])
        return np.array(walls)

    # def get_correction_gradient(self, phys_state):
    #     """
    #     核心函数：计算基于 TTC 的修正梯度
    #     输入: phys_state (Batch, 4) -> [x, y, vx, vy] (必须是物理单位!)
    #     输出: h_val, grad_pos (Batch, 2)
    #     """
    #     # 必须 clone 并 detach，确保我们能对它求导
    #     x_in = phys_state.clone().detach().requires_grad_(True)
        
    #     # 分离位置和速度
    #     pos = x_in[:, :2] # (x, y)
    #     vel = x_in[:, 2:4] # (vx, vy)
        
    #     # --- A. 寻找最近的墙 ---
    #     # 计算每个点到所有墙的距离
    #     # dists shape: (Batch, N_walls)
    #     dists = torch.cdist(pos, self.wall_centers_tensor)
        
    #     # 找到最近墙的索引
    #     min_idxs = torch.argmin(dists, dim=1)
    #     nearest_walls = self.wall_centers_tensor[min_idxs]
        
    #     # --- B. 构造网络输入 (相对位置 + 绝对速度) ---
    #     rel_pos = pos - nearest_walls
        
    #     # 输入向量: [rel_x, rel_y, vx, vy]
    #     # 这就是我们在 train_ttc 里训练的格式
    #     net_input = torch.cat([rel_pos, vel], dim=1)
        
    #     # --- C. 前向传播 ---
    #     h_val = self.model(net_input)
        
    #     # --- D. 对输入位置求导 ---
    #     # 我们问网络："我该怎么移动位置(x,y)才能让 h 变大(变安全)？"
    #     # 虽然改变速度也能变安全，但直接改位置在 Diffusion 生成中更直接
    #     grads = torch.autograd.grad(
    #         outputs=h_val,
    #         inputs=x_in,
    #         grad_outputs=torch.ones_like(h_val),
    #         create_graph=False,
    #         retain_graph=False
    #     )[0]
        
    #     # 只取位置部分的梯度 [dh/dx, dh/dy]
    #     grad_pos = grads[:, :2]
        
    #     return h_val.detach(), grad_pos.detach()

    def get_correction_gradient(self, phys_state):
        """
        核心函数：计算基于 TTC 的修正梯度
        """
        # 1. 准备输入
        # x_in 必须设为 requires_grad=True
        x_in = phys_state.clone().detach().requires_grad_(True)
        
        # 2. [关键修复] 强制开启梯度计算
        # 即使外部是 no_grad 模式，这里也必须开启，否则无法求导
        with torch.enable_grad():
            
            # --- 下面的逻辑必须在这个缩进里 ---
            
            # 分离位置和速度
            pos = x_in[:, :2] # (x, y)
            vel = x_in[:, 2:4] # (vx, vy)
            
            # --- A. 寻找最近的墙 ---
            dists = torch.cdist(pos, self.wall_centers_tensor)
            min_idxs = torch.argmin(dists, dim=1)
            nearest_walls = self.wall_centers_tensor[min_idxs]
            
            # --- B. 构造网络输入 ---
            rel_pos = pos - nearest_walls
            net_input = torch.cat([rel_pos, vel], dim=1)
            
            # --- C. 前向传播 ---
            h_val = self.model(net_input)
            
            # --- D. 对输入位置求导 ---
            grads = torch.autograd.grad(
                outputs=h_val,
                inputs=x_in,
                grad_outputs=torch.ones_like(h_val),
                create_graph=False,
                retain_graph=False
            )[0]
            
            # 只取位置部分的梯度 [dh/dx, dh/dy]
            grad_pos = grads[:, :2]
        
        # 3. 返回 detach 的结果 (以免影响外部)
        return h_val.detach(), grad_pos.detach()