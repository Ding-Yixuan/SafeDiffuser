import torch
import torch.nn as nn
import numpy as np
import os

# ================= 配置区域 =================
# 说明：当前 TTC 模型以“绝对坐标 + 速度”训练（多墙全图标签）
USE_ABSOLUTE_INPUT = True

# 用于打印验证（不参与推理）
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
# ===========================================

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
            half_h = (r_max - r_min + 1) / 2.0
            half_w = (c_max - c_min + 1) / 2.0

            obstacles.append({
                # 【修改】：X 是 col，Y 是 row
                "center": np.array([center_row, center_col], dtype=np.float32),
                # 【修改】：宽度是 w，高度是 h
                "half_extents": np.array([half_h, half_w], dtype=np.float32),
            })
    #        1. 计算以左上角为原点的中心
    #         raw_center_row = (r_min + r_max + 1) / 2.0
    #         center_col = (c_min + c_max + 1) / 2.0
            
    #         # 2. 【核心修正：翻转 Y 轴！】 
    #         # 物理 Y = 总行数 - 矩阵行数
    #         total_rows = len(map_lines) 
    #         center_y = total_rows - raw_center_row  # 这样 Row 0 (最上面) 就会变成最大的 Y 值！
            
    #         # X 轴不需要翻转，直接等于 Col
    #         center_x = center_col 
            
    #         half_h = (r_max - r_min + 1) / 2.0
    #         half_w = (c_max - c_min + 1) / 2.0

    #         obstacles.append({
    #             # X 是 col(center_x), Y 是翻转后的 row(center_y)
    #             "center": np.array([center_x, center_y], dtype=np.float32), 
    #             # 宽对应 X(w), 高对应 Y(h)
    #             "half_extents": np.array([half_w, half_h], dtype=np.float32),
    #         })
    return obstacles

# ================= 模型区域 =================
# 障碍物中心 (必须和训练时一致)
# TARGET_OBSTACLE = torch.tensor([1.5, 5.0]) 

# 模型文件名 (假设在同级目录或根目录)
MODEL_FILENAME = "ttc_model_small_2ob.pth"
# ===========================================

class SafetyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 【修改】：全部改为 256 维
        self.net = nn.Sequential(
            nn.Linear(4, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) 
        )
    
    def forward(self, x):
        return self.net(x)

class NeuralBarrierAdapter:
    def __init__(self, device='cuda'):
        self.device = device
        self.debug_every = 200
        self._dbg_calls = 0

        # 用于打印验证（多墙数量）
        self.obstacles = parse_maze_map_to_obstacles(MAZE_MAP_LARGE)
        self.obstacles_count = len(self.obstacles)

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
        # self.center = TARGET_OBSTACLE.to(device)
        self.model = SafetyNetwork().to(device)

        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[NeuralBarrier] ✅ 成功加载 TTC 模型: {model_path}")
                print(f"[NeuralBarrier] 🧱 多墙数量: {self.obstacles_count}")
                print(f"[NeuralBarrier] 📌 输入模式: {'absolute' if USE_ABSOLUTE_INPUT else 'relative'}")
            except Exception as e:
                print(f"[NeuralBarrier] ❌ 加载模型失败: {e}")

    def summary(self):
        return f"mode={'absolute' if USE_ABSOLUTE_INPUT else 'relative'}, walls={self.obstacles_count}"

    def get_correction_gradient(self, phys_state):
        """
        Input: phys_state [Batch, 4] -> [x, y, vx, vy] (绝对坐标)
        Output: h_val, grad_pos
        """
        x_in = phys_state.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            if USE_ABSOLUTE_INPUT:
                net_input = x_in[:, :4]
            else:
                raise ValueError("需启用 USE_ABSOLUTE_INPUT=True")

            h_val = self.model(net_input)

            grads = torch.autograd.grad(
                outputs=h_val,
                inputs=x_in,
                grad_outputs=torch.ones_like(h_val),
                create_graph=False,
                retain_graph=False
            )[0]

            grad_pos = grads[:, :2]

        # 轻量调试：证明 CBF 正在被调用/调整
        self._dbg_calls += 1
        if (self._dbg_calls % self.debug_every == 0) or (h_val.min().item() < 0):
            grad_norm = torch.linalg.vector_norm(grad_pos, dim=1).mean().item()
            min_h = h_val.min().item()
            # print(f"[NeuralBarrier] CBF调用#{self._dbg_calls} | min_h={min_h:.4f} | mean|grad_pos|={grad_norm:.4f}")

        return h_val.detach(), grad_pos.detach()