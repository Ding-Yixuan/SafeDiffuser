import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 你的高阶绘图函数 (原封不动)
# ==========================================
def plot_barrier_boundary_2d(
    model,
    domain,          # e.g. [(xmin, xmax), (ymin, ymax)]
    plot_len=(300, 300),  # (nx, ny)
    width=0.1,       # WIDTH
    norm_eps=1e-6,   # NORM_EPS
    ax=None,
    device=None,
):
    """
    绘制 barrier NN 的边界线（兼容旧版 PyTorch）
    """
    if ax is None:
        ax = plt.gca()

    if device is None:
        device = next(model.parameters()).device

    nx, ny = int(plot_len[0]), int(plot_len[1])
    (xmin, xmax), (ymin, ymax) = domain

    # --- 1) 生成网格 ---
    xs = torch.linspace(xmin, xmax, nx, device=device)
    ys = torch.linspace(ymin, ymax, ny, device=device)
    
    # [关键修改] 兼容旧版 PyTorch 的写法
    # 旧版 meshgrid 默认是 'ij' (Matrix) 索引
    # 为了得到 'xy' (Cartesian) 效果，我们输入 (ys, xs)，接收 (Y, X)
    Y, X = torch.meshgrid(ys, xs)
    
    # 此时 X, Y 的 shape 都是 [ny, nx]，我们需要 flatten 后堆叠
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [nx*ny, 2]

    # --- 2) 前向 + 对输入求梯度 ---
    model.eval()

    prev_req = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)

    pts.requires_grad_(True)
    B = model(pts)
    
    if B.dim() > 1 and B.shape[1] != 1:
        B = B[:, :1]
    B = B.reshape(-1, 1)

    grad = torch.autograd.grad(
        outputs=B.sum(),
        inputs=pts,
        create_graph=False,
        retain_graph=False,
        allow_unused=False
    )[0]  # [N,2]

    pts.requires_grad_(False)

    grad_norm = torch.clamp(torch.linalg.vector_norm(grad, dim=1), min=norm_eps)  # [N]
    normed_B = (B.squeeze(1) / grad_norm)  # [N]

    # --- 3) reshape 回网格并画 contour ---
    # 注意：这里的 shape 要和 meshgrid 生成的一致 [ny, nx]
    Z = normed_B.detach().cpu().numpy().reshape(ny, nx)

    x_np = xs.detach().cpu().numpy()
    y_np = ys.detach().cpu().numpy()
    
    # Numpy 的 meshgrid 默认就是 'xy'，这里不需要改
    X_np, Y_np = np.meshgrid(x_np, y_np) 
    
    # 此时 X_np shape 是 [ny, nx]，Z 也是 [ny, nx]，直接画即可，不需要转置了
    
    contour = ax.contour(
        X_np, Y_np, Z,
        levels=[-width, 0.0, width],
        linestyles=["dotted", "solid", "dotted"],
        linewidths=[1.5, 2.5, 1.5],
        colors=["red", "blue", "green"]
    )
    ax.clabel(contour, inline=True, fontsize=10, fmt={-width:'Dangerous', 0.0:'Boundary', width:'Safe'})

    for p, r in zip(model.parameters(), prev_req):
        p.requires_grad_(r)

    return contour

# ==========================================
# 2. 模型定义与包装器
# ==========================================
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

# [关键] 这是一个包装器，把 2D 坐标变成 4D 输入 (注入速度)
class VelocityWrapper(nn.Module):
    def __init__(self, base_model, velocity):
        super().__init__()
        self.base_model = base_model
        # velocity: tensor [vx, vy]
        self.register_buffer('velocity', velocity) 

    def forward(self, pts_2d):
        # pts_2d: [N, 2] -> (x, y)
        # 我们需要拼接 velocity 变成 [N, 4] -> (x, y, vx, vy)
        
        batch_size = pts_2d.shape[0]
        # 扩展速度到 batch 大小
        v_expanded = self.velocity.unsqueeze(0).expand(batch_size, -1) # [N, 2]
        
        # 拼接
        inputs = torch.cat([pts_2d, v_expanded], dim=1)
        
        # 这里的输出就是 B(x)
        return self.base_model(inputs)

# ==========================================
# 3. 主程序：生成精确对比图
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = SafetyNetwork().to(device)
    model_path = "ttc_model_dataset.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")
    else:
        print("模型文件不存在，请先训练！")
        return

    # 2. 准备画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 定义绘图范围 (相对坐标系)
    domain = [(-2.0, 2.0), (-2.0, 2.0)]
    
    # --- Case 1: 静态 (Velocity = 0) ---
    print("正在绘制静态边界...")
    v_static = torch.tensor([0.0, 0.0]).to(device)
    wrapper_static = VelocityWrapper(model, v_static)
    
    ax0 = axes[0]
    plot_barrier_boundary_2d(wrapper_static, domain, ax=ax0, width=0.1)
    
    # 画真实的墙壁框 (Yellow)
    # 横向长方形: 宽2 (半宽1), 高1 (半高0.5)
    rect = plt.Rectangle((-1.0, -0.5), 2.0, 1.0, 
                         fill=False, color='orange', linewidth=2, label='Real Wall')
    ax0.add_patch(rect)
    ax0.set_title("Static Safety Boundary (v=0)")
    ax0.set_aspect('equal')
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # --- Case 2: 动态 (Velocity = [2, 2]) ---
    print("正在绘制动态边界...")
    v_dynamic = torch.tensor([0.8, 0.8]).to(device)
    wrapper_dynamic = VelocityWrapper(model, v_dynamic)
    
    ax1 = axes[1]
    plot_barrier_boundary_2d(wrapper_dynamic, domain, ax=ax1, width=0.1)
    


    rect1 = plt.Rectangle((-1.0, -0.5), 2.0, 1.0, 
                         fill=False, color='orange', linewidth=2, label='Real Wall')
    ax1.add_patch(rect1)
    
    # 画速度箭头
    ax1.arrow(0, 0, 0.5, 0.5, head_width=0.1, color='black', label='Velocity')
    ax1.set_title("Dynamic Safety Boundary (v=[0.8, 0.8])")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 保存
    save_name = "precise_boundary_vis.png"
    plt.savefig(save_name, dpi=150)
    print(f"高清边界图已保存为: {save_name}")

if __name__ == "__main__":
    main()