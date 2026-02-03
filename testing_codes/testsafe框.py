import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import diffuser.datasets as datasets
from diffuser.utils.rendering import MAZE_BOUNDS, plot2img
import imageio

# ==============================================================================
# 1. 障碍物定义 (这里我们定义两个框，看看画出来在哪里)
# ==============================================================================
ALL_OBSTACLES = [
    # 上方的小墙
    {"center": np.array([4.0, 3.0]), "half_extents": np.array([2.0, 1.0])},
    # 下方的大墙
    {"center": np.array([7.0, 7.0]), "half_extents": np.array([3.0, 1.5])}
]

# 尝试不同的偏移量！(X_offset, Y_offset)
OFFSET = (-0.5, 1.5)  

# ==============================================================================
# 2. 坐标转换函数 (来自你的渲染器)
# ==============================================================================
def normalize_maze_xy(xy, env_name):
    bounds = MAZE_BOUNDS[env_name]
    xy = xy + 0.5
    if len(bounds) == 2:
        _, scale = bounds
        xy[:, 0] /= scale
        xy[:, 1] /= scale
        return xy, scale, scale
    if len(bounds) == 4:
        _, iscale, _, jscale = bounds
        xy[:, 0] /= iscale
        xy[:, 1] /= jscale
        return xy, iscale, jscale
    raise RuntimeError(f"Unrecognized bounds for {env_name}: {bounds}")

# ==============================================================================
# 3. 极简画图测试
# ==============================================================================
def test_yellow_box():
    env_name = 'maze2d-large-v1'
    
    # 1. 加载一个空环境，只为了拿到 renderer 里的背景图
    env = datasets.load_environment(env_name)
    from diffuser.utils.rendering import Maze2dRenderer
    renderer = Maze2dRenderer(env_name)

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    
    # 2. 画出官方的迷宫底图
    plt.imshow(renderer._background * .5, extent=renderer._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

    # 3. 画出我们的黄框
    for obs in ALL_OBSTACLES:
        c = obs["center"]
        hw = obs["half_extents"]
        
        # 加上偏移量 (你刚才肉眼观察出来的 X+1.5, Y-1.0 是对的)
        shifted_cx = c[0] - 0.5 
        shifted_cy = c[1] + 1.5 

        # 计算四个物理角点 [X, Y]
        corners = np.array([
            [shifted_cx - hw[0], shifted_cy - hw[1]], # 左下
            [shifted_cx + hw[0], shifted_cy - hw[1]], # 右下
            [shifted_cx + hw[0], shifted_cy + hw[1]], # 右上
            [shifted_cx - hw[0], shifted_cy + hw[1]]  # 左上
        ])
        
        # 批量把四个角的物理坐标转换成图像的像素坐标
        corners_norm, _, _ = normalize_maze_xy(corners, env_name)

        # Matplotlib 的坐标系里，横轴是 cols (Y), 纵轴是 rows (X)
        # 所以我们需要把 (X, Y) 翻转为 (Y, X)
        polygon_pts = np.stack([corners_norm[:, 1], corners_norm[:, 0]], axis=1)

        # 直接画多边形，完美包裹这四个角
        poly = patches.Polygon(
            polygon_pts, 
            closed=True, 
            fill=False, 
            color='orange', 
            linewidth=3, 
            linestyle='--'
        )
        plt.gca().add_patch(poly)

    # 4. 保存
    img = plot2img(fig, remove_margins=False)
    imageio.imsave("verify_yellow_box.png", img)
    print("✅ 验证图已生成: verify_yellow_box.png")

if __name__ == "__main__":
    test_yellow_box()