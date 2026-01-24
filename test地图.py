import gym
import numpy as np
import h5py
import pygame
import time
from d4rl.pointmaze import maze_model

# 1. 地图定义 (还是那个双障碍物地图)
CUSTOM_MAZE = \
    "#######\\" \
    "#O    #\\" \
    "#  #  #\\" \
    "#     #\\" \
    "#  #  #\\" \
    "#    G#\\" \
    "#######"

env = maze_model.MazeEnv(CUSTOM_MAZE, reset_target=False, reward_type='dense')

# 2. Pygame 初始化 (用于键盘控制和渲染)
pygame.init()
screen = pygame.display.set_mode((400, 400)) # 随便开个小窗口为了捕获按键
pygame.display.set_caption("按 上下左右 移动小球，按 ESC 结束并保存")
clock = pygame.time.Clock()

# 3. 数据容器
data = {
    'observations': [], 'actions': [], 
    'terminals': [], 'timeouts': []
}

obs = env.reset()
done = False
running = True

print("🎮 游戏开始！请点击弹出的窗口，用方向键控制小球到达终点 (右下角)。")
print("⚠️ 你的每一个动作都会被记录为训练数据！")

# 4. 主循环：你就是控制策略 (Human-in-the-loop)
while running:
    # 渲染环境 (这里借用 env 自带的 render)
    env.render()
    
    # 监听键盘
    action = np.zeros(2) # [x, y] 方向的力
    force = 0.8 # 操控力度
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:  action[0] = -force
    if keys[pygame.K_RIGHT]: action[0] =  force
    if keys[pygame.K_UP]:    action[1] =  force # MuJoCo 中 Y 轴朝上
    if keys[pygame.K_DOWN]:  action[1] = -force

    # 执行动作
    next_obs, reward, done, info = env.step(action)
    
    # 保存这一步 (只要你没撞墙，这就是绝对安全的专家数据)
    data['observations'].append(obs)
    data['actions'].append(action)
    data['terminals'].append(done)
    data['timeouts'].append(False)
    
    obs = next_obs
    if done:
        print("🎉 到达终点！重置环境继续收集...")
        obs = env.reset()

    clock.tick(20) # 控制帧率，别太快

# 5. 保存你创造的数据集
pygame.quit()
env.close()

for k in data:
    data[k] = np.array(data[k], dtype=np.float32)

filename = "human_expert_maze.hdf5"
with h5py.File(filename, 'w') as f:
    for k in data:
        f.create_dataset(k, data=data[k])

print(f"✅ 保存成功！你亲自生成的专家数据集已保存至: {filename} (共 {len(data['observations'])} 步)")