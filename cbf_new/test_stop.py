import torch
import numpy as np
import os
from diffuser.models.cbf_network import CBFNetwork
# 1. Maze2D-Large 地图
MAZE_MAP_LARGE = \
"############\\"+\
"#OOOO#OOOOO#\\"+\
"#O##O#O#O#O#\\"+\
"#OOOOOO#OOO#\\"+\
"#O#######O#\\"+\
"#OO#O#OOOOO#\\"+\
"##O#O#O#O###\\"+\
"#OO#OOO#OGO#\\"+\
"############"

class MiniAdapter:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载网络
        # 输入维度 4: [dx, dy, vx, vy]
        # 输出维度 1: h 值
        self.model = CBFNetwork(input_dim=4, output_dim=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 解析地图
        self.walls = self._parse_maze(MAZE_MAP_LARGE)
        self.walls_tensor = torch.tensor(self.walls, dtype=torch.float32, device=self.device)
        print(f"共找到 {len(self.walls)} 个墙壁方块")

    def _parse_maze(self, maze_str):
        # 解析墙壁坐标
        lines = maze_str.strip().split('\\')
        walls = []
        for h, row in enumerate(lines):
            for w, char in enumerate(row):
                if char == '#':
                    # 物理坐标转换: x = col + 1, y = row + 1
                    ## diffusion.py 里 x 是 col, y 是 row
                    walls.append([w + 1.0, h + 1.0])
        return np.array(walls)

    def predict(self, robot_x, robot_y):
        """
        输入机器人的物理坐标，返回 CBF 值和梯度建议
        """
        # 1. 准备数据
        pos = torch.tensor([[robot_x, robot_y]], dtype=torch.float32, device=self.device)
        vel = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device) # 假设静止测试
        
        pos.requires_grad_(True)
        
        # 2. 找最近的墙
        dists = torch.cdist(pos, self.walls_tensor)
        min_idx = torch.argmin(dists, dim=1)
        nearest_wall = self.walls_tensor[min_idx]
        
        # 3. 构造相对坐标输入 [dx, dy, vx, vy]
        rel_pos = pos - nearest_wall
        net_input = torch.cat([rel_pos, vel], dim=1)
        
        # 4. 预测
        h_val = self.model(net_input)
        
        # 5. 求梯度 (这就是我们要推力的方向)
        grads = torch.autograd.grad(h_val, pos, retain_graph=False)[0]
        
        return h_val.item(), grads.detach().cpu().numpy()[0], nearest_wall.cpu().numpy()[0]

# ==========================================
# 2. 运行测试用例
# ==========================================
def run_test():
    model_path = "cbf_new/cbf_maze2d.pth"
    if not os.path.exists(model_path):
        print("❌ 错误：找不到模型文件 cbf_maze2d.pth，请先运行训练脚本。")
        return

    adapter = MiniAdapter(model_path)
    print("-" * 50)

    # --- 测试案例 1: 安全位置 ---
    # 地图中 Row 1, Col 1 是空地 (O)，物理坐标 (2.0, 2.0)
    # 我们选一个中心点 (2.0, 2.0)，离周围墙壁至少有 1.0 的距离
    print("🧪 测试 1: 站在空地中心 (2.0, 2.0)")
    h, grad, wall = adapter.predict(2.0, 2.0)
    print(f"   -> 最近的墙在: {wall}")
    print(f"   -> 安全值 h(x): {h:.4f} (预期: 正数，表示安全)")
    
    print("-" * 30)

    # --- 测试案例 2: 危险位置 (贴脸) ---
    # 地图中 Row 1, Col 0 是墙 (#)，物理坐标 (1.0, 2.0)
    # 墙边缘在 1.5。我们站在 1.55 (离墙只有 0.05米)
    print("🧪 测试 2: 贴近墙壁边缘 (1.55, 2.0)")
    h, grad, wall = adapter.predict(1.55, 2.0)
    print(f"   -> 最近的墙在: {wall} (预期: [1. 2.])")
    print(f"   -> 安全值 h(x): {h:.4f} (预期: 接近0或负数)")
    print(f"   -> 梯度方向: {grad} (预期: x分量应该为正，把你往右推)")

    print("-" * 30)
    
    # --- 测试案例 3: 撞墙了 (在墙里) ---
    # 直接站在墙心 (1.0, 2.0)
    print("🧪 测试 3: 站在墙里 (1.0, 2.0)")
    h, grad, wall = adapter.predict(1.0, 2.0)
    print(f"   -> 安全值 h(x): {h:.4f} (预期: 很大的负数)")

if __name__ == "__main__":
    run_test()