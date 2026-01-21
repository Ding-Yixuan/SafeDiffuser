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
            print(f"[NeuralBarrier]成功加载 TTC 避障模型: {model_path}")
        else:
            print(f"[NeuralBarrier]找不到模型文件 {model_path}，无法避障")
            
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
    #     """
    #     # 1. 准备输入
    #     x_in = phys_state.clone().detach().requires_grad_(True)
        
    #     # 开启梯度计算
    #     with torch.enable_grad():            
    #         # 分离位置和速度
    #         pos = x_in[:, :2] # (x, y)
    #         vel = x_in[:, 2:4] # (vx, vy)
            
    #         # --- A. 寻找最近的墙 ---
    #         dists = torch.cdist(pos, self.wall_centers_tensor)

            
    #         min_idxs = torch.argmin(dists, dim=1)
    #         nearest_walls = self.wall_centers_tensor[min_idxs]
            
    #         # --- B. 构造网络输入 ---
    #         rel_pos = pos - nearest_walls
    #         # 输入向量: [rel_x, rel_y, vx, vy]
    #         net_input = torch.cat([rel_pos, vel], dim=1)
            
    #         # --- C. 前向传播 ---
    #         h_val = self.model(net_input)
            
    #         # --- D. 对输入位置求导 ---
    #         grads = torch.autograd.grad(
    #             outputs=h_val,
    #             inputs=x_in,
    #             grad_outputs=torch.ones_like(h_val),
    #             create_graph=False,
    #             retain_graph=False
    #         )[0]
            
    #         # 只取位置部分的梯度 [dh/dx, dh/dy]
    #         grad_pos = grads[:, :2]
        
    #     # 3. 返回 detach 的结果 (以免影响外部)
    #     return h_val.detach(), grad_pos.detach()

    def get_correction_gradient(self, phys_state):
        """
        核心函数：计算基于 TTC 的修正梯度 (Multi-Wall Fusion 版)
        解决死锁问题的关键：同时计算最近 K 个墙的排斥力，并进行加权合成。
        """
        # 1. 准备输入
        x_in = phys_state.clone().detach().requires_grad_(True)
        
        # 开启梯度计算
        with torch.enable_grad():            
            # 分离位置和速度
            pos = x_in[:, :2] # (x, y)
            vel = x_in[:, 2:4] # (vx, vy)
            
            # --- A. 寻找最近的 K 个墙 ---
            # 计算到所有墙的距离
            dists = torch.cdist(pos, self.wall_centers_tensor)
            
            # 取最近的 3 个墙 (K=3)
            # topk_idxs: [Batch, 3] -> 每一行是该样本最近的3个墙的ID
            k = 3
            topk_vals, topk_idxs = torch.topk(dists, k=k, largest=False, dim=1)
            
            # 初始化累加器
            total_grad_pos = torch.zeros_like(pos)
            
            # 用于记录最危险的 h 值 (用于外部的 is_unsafe 判断)
            # 初始化为一个较大的安全值 (比如 10.0)
            min_h_vals = torch.ones((pos.shape[0], 1), device=pos.device) * 10.0
            
            # --- B. 循环处理每一个威胁 ---
            for i in range(k):
                # 1. 取出第 i 近的墙的坐标
                # topk_idxs[:, i] 是所有 batch 在第 i 近邻的墙索引
                wall_idx = topk_idxs[:, i] 
                target_walls = self.wall_centers_tensor[wall_idx]
                
                # 2. 构造网络输入 (相对坐标)
                rel_pos = pos - target_walls
                net_input = torch.cat([rel_pos, vel], dim=1)
                
                # 3. 前向传播
                h_val = self.model(net_input)
                
                # 4. 记录最危险的情况 (取最小值)
                min_h_vals = torch.min(min_h_vals, h_val)
                
                # 5. 计算单一墙壁的梯度
                # 注意：这里我们需要 retain_graph=True，因为我们要对同一个 x_in 求多次导
                grads = torch.autograd.grad(
                    outputs=h_val,
                    inputs=x_in,
                    grad_outputs=torch.ones_like(h_val),
                    create_graph=False,
                    retain_graph=True  # <--- 关键：保持计算图，以便下一次循环继续求导
                )[0]
                
                single_grad_pos = grads[:, :2]
                
                # 6. [核心逻辑] 风险加权融合 (Risk-Weighted Fusion)
                # 权重逻辑：h 越小(越危险)，权重越大。
                # 0.5 是我们在训练时设定的安全边界 (margin)。
                # 如果 h=0.6 (安全)，ReLU后权重为0，完全忽略。
                # 如果 h=0.0 (危险)，权重为0.5。
                # 如果 h=-0.5 (极度危险)，权重为1.0。
                danger_weight = torch.relu(0.5 - h_val)
                
                # 累加加权后的梯度
                # 这样，离得越近的墙，对机器人的推力贡献就越大
                total_grad_pos += single_grad_pos * danger_weight

        # 3. 返回最危险的 h 值 和 融合后的梯度
        return min_h_vals.detach(), total_grad_pos.detach()