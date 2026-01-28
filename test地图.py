# import gym
# import numpy as np
# import h5py
# import pygame
# import time
# from d4rl.pointmaze import maze_model

# # 1. 地图定义 (还是那个双障碍物地图)
# CUSTOM_MAZE = \
#     "#######\\" \
#     "#O    #\\" \
#     "# #   #\\" \
#     "#     #\\" \
#     "#  #  #\\" \
#     "#  # G#\\" \
#     "#######"

# env = maze_model.MazeEnv(CUSTOM_MAZE, reset_target=False, reward_type='dense')

# # 2. Pygame 初始化 (用于键盘控制和渲染)
# pygame.init()
# screen = pygame.display.set_mode((400, 400)) # 随便开个小窗口为了捕获按键
# pygame.display.set_caption("按 上下左右 移动小球，按 ESC 结束并保存")
# clock = pygame.time.Clock()

# # 3. 数据容器
# data = {
#     'observations': [], 'actions': [], 
#     'terminals': [], 'timeouts': []
# }

# obs = env.reset()
# done = False
# running = True

# print("🎮 游戏开始！请点击弹出的窗口，用方向键控制小球到达终点 (右下角)。")
# print("⚠️ 你的每一个动作都会被记录为训练数据！")

# # 4. 主循环：你就是控制策略 (Human-in-the-loop)
# while running:
#     # 渲染环境 (这里借用 env 自带的 render)
#     env.render()
    
#     # 监听键盘
#     action = np.zeros(2) # [x, y] 方向的力
#     force = 0.8 # 操控力度
    
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
#             running = False

#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_LEFT]:  action[0] = -force
#     if keys[pygame.K_RIGHT]: action[0] =  force
#     if keys[pygame.K_UP]:    action[1] =  force # MuJoCo 中 Y 轴朝上
#     if keys[pygame.K_DOWN]:  action[1] = -force

#     # 执行动作
#     next_obs, reward, done, info = env.step(action)
    
#     # 保存这一步 (只要你没撞墙，这就是绝对安全的专家数据)
#     data['observations'].append(obs)
#     data['actions'].append(action)
#     data['terminals'].append(done)
#     data['timeouts'].append(False)
    
#     obs = next_obs
#     if done:
#         print("🎉 到达终点！重置环境继续收集...")
#         obs = env.reset()

#     clock.tick(20) # 控制帧率，别太快

# # 5. 保存你创造的数据集
# pygame.quit()
# env.close()

# for k in data:
#     data[k] = np.array(data[k], dtype=np.float32)

# filename = "human_expert_maze.hdf5"
# with h5py.File(filename, 'w') as f:
#     for k in data:
#         f.create_dataset(k, data=data[k])

# print(f"✅ 保存成功！你亲自生成的专家数据集已保存至: {filename} (共 {len(data['observations'])} 步)")

import sys
import os

# 屏蔽 TensorFlow 相关的打印和警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 骗过 Python，让它认为 d4rl.carla 和 tensorflow 已经加载了，阻止其报错
class DummyModule:
    pass

sys.modules['tensorflow'] = DummyModule()
sys.modules['d4rl.carla'] = DummyModule()

# 现在可以安全导入了
import gym
import numpy as np
import h5py
import pygame
from d4rl.pointmaze import maze_model

# 1. 地图定义
CUSTOM_MAZE = \
    "#######\\" \
    "#O    #\\" \
    "#  #  #\\" \
    "#     #\\" \
    "#  #  #\\" \
    "#  # G#\\" \
    "#######"

env = maze_model.MazeEnv(CUSTOM_MAZE, reset_target=False, reward_type='dense')

pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("方向键控制 | 空格键:失控模式 | ESC:保存退出")
clock = pygame.time.Clock()

# 修改点 1：数据容器增加 'costs' 用于标记不安全状态
data = {
    'observations': [], 'actions': [], 
    'terminals': [], 'timeouts': [],
    'costs': [] # 新增：0 表示安全，1 表示不安全
}

obs = env.reset()
done = False
running = True

print("🎮 游戏开始！")
print("✅ 正常按键：收集【安全轨迹】")
print("⚠️ 按住 SPACE 键：注入随机噪声，模拟失控（用于收集【不安全轨迹】）")
print("💥 撞墙时会自动记录 cost=1")

# 获取 MuJoCo 的底层物理步长，用于检测碰撞
geom_names = env.sim.model.geom_names
wall_geoms = [i for i, name in enumerate(geom_names) if 'wall' in name]
ball_geom = geom_names.index('particle_geom')

while running:
    # 1. 让 MuJoCo 渲染成像素数组，而不是直接弹窗
    img = env.render(mode='rgb_array')
    
    # 2. 将图像调整为 Pygame 能够读取的格式 (宽, 高, 通道)
    img = np.transpose(img, (1, 0, 2))
    
    # 3. 把图像贴到我们的小窗口上
    surf = pygame.surfarray.make_surface(img)
    surf = pygame.transform.scale(surf, (400, 400)) # 缩放到窗口大小
    screen.blit(surf, (0, 0))
    pygame.display.flip() # 刷新屏幕
    
    # --- 下面继续写你原来的按键监听和 env.step 逻辑 ---
    action = np.zeros(2)
    force = 0.8
    cost = 0.0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()
    
    # 修改点 2：按住空格键，模拟控制失灵（随机瞎走）
    if keys[pygame.K_SPACE]:
        action = np.random.uniform(-force, force, size=2)
        # 也可以让人为的瞎走被标记为某种潜在风险，视情况而定
    else:
        if keys[pygame.K_LEFT]:  action[0] = -force
        if keys[pygame.K_RIGHT]: action[0] =  force
        if keys[pygame.K_UP]:    action[1] =  force 
        if keys[pygame.K_DOWN]:  action[1] = -force

    next_obs, reward, done, info = env.step(action)
    
    # 修改点 3：碰撞检测 (Collision Detection)
    # 检查 MuJoCo 物理引擎中球和墙是否产生了 contact
    in_collision = False
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        if (contact.geom1 == ball_geom and contact.geom2 in wall_geoms) or \
           (contact.geom2 == ball_geom and contact.geom1 in wall_geoms):
            in_collision = True
            break
            
    if in_collision:
        cost = 1.0 # 发生碰撞，记录为不安全
        print("💥 撞墙警报！当前动作被标记为不安全 (Cost=1)")

    # 保存数据
    data['observations'].append(obs)
    data['actions'].append(action)
    data['terminals'].append(done)
    data['timeouts'].append(False)
    data['costs'].append(cost) # 保存安全/不安全标签
    
    obs = next_obs
    if done:
        print("🎉 到达终点！重置环境继续收集...")
        obs = env.reset()

    clock.tick(20)

pygame.quit()
env.close()

# 保存数据集
for k in data:
    data[k] = np.array(data[k], dtype=np.float32)

filename = "human_safe_unsafe_maze.hdf5"
with h5py.File(filename, 'w') as f:
    for k in data:
        f.create_dataset(k, data=data[k])

total_steps = len(data['observations'])
unsafe_steps = np.sum(data['costs'])
print(f"✅ 保存成功！文件: {filename}")
print(f"📊 数据统计: 总步数 {total_steps}, 其中不安全(撞墙)步数 {int(unsafe_steps)}")