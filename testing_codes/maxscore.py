import os
# 设置无头模式
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import numpy as np
import d4rl.pointmaze
from gym.envs.registration import register

from d4rl.pointmaze import q_iteration
from d4rl.pointmaze.gridcraft import grid_env
from d4rl.pointmaze.gridcraft import grid_spec

# =======================================================
# 1. 官方 WaypointController 类
# =======================================================

ZEROS = np.zeros((2,), dtype=np.float32)
ONES = np.zeros((2,), dtype=np.float32)

class WaypointController(object):
    def __init__(self, maze_str, solve_thresh=0.1, p_gain=10.0, d_gain=-1.0):
        self.maze_str = maze_str
        self._target = -1000 * ONES

        self.p_gain = p_gain
        self.d_gain = d_gain
        self.solve_thresh = solve_thresh
        self.vel_thresh = 0.1

        self._waypoint_idx = 0
        self._waypoints = []
        self._waypoint_prev_loc = ZEROS

        self.env = grid_env.GridEnv(grid_spec.spec_from_string(maze_str))

    def current_waypoint(self):
        return self._waypoints[self._waypoint_idx]

    def get_action(self, location, velocity, target):
        if np.linalg.norm(self._target - np.array(self.gridify_state(target))) > 1e-3: 
            self._new_target(location, target)

        dist = np.linalg.norm(location - self._target)
        vel = self._waypoint_prev_loc - location
        vel_norm = np.linalg.norm(vel)
        task_not_solved = (dist >= self.solve_thresh) or (vel_norm >= self.vel_thresh)

        if task_not_solved:
            next_wpnt = self._waypoints[self._waypoint_idx]
        else:
            next_wpnt = self._target

        prop = next_wpnt - location
        action = self.p_gain * prop + self.d_gain * velocity

        dist_next_wpnt = np.linalg.norm(location - next_wpnt)
        if task_not_solved and (dist_next_wpnt < self.solve_thresh) and (vel_norm<self.vel_thresh):
            self._waypoint_idx += 1
            if self._waypoint_idx == len(self._waypoints)-1:
                assert np.linalg.norm(self._waypoints[self._waypoint_idx] - self._target) <= self.solve_thresh

        self._waypoint_prev_loc = location
        action = np.clip(action, -1.0, 1.0)
        return action, (not task_not_solved)

    def gridify_state(self, state):
        return (int(round(state[0])), int(round(state[1])))

    def _new_target(self, start, target):
        start = self.gridify_state(start)
        start_idx = self.env.gs.xy_to_idx(start)
        target = self.gridify_state(target)
        target_idx = self.env.gs.xy_to_idx(target)
        self._waypoint_idx = 0

        self.env.gs[target] = grid_spec.REWARD
        q_values = q_iteration.q_iteration(env=self.env, num_itrs=50, discount=0.99)
        
        max_ts = 100
        s = start_idx
        waypoints = []
        for i in range(max_ts):
            a = np.argmax(q_values[s])
            new_s, reward = self.env.step_stateless(s, a)

            waypoint = self.env.gs.idx_to_xy(new_s)
            if new_s != target_idx:
                waypoint = waypoint - np.random.uniform(size=(2,))*0.2
            waypoints.append(waypoint)
            s = new_s
            if new_s == target_idx:
                break
        self.env.gs[target] = grid_spec.EMPTY
        self._waypoints = waypoints
        self._waypoint_prev_loc = start
        self._target = target


# =======================================================
# 2. 地图配置与环境注册
# =======================================================
CUSTOM_MAZE1 = [
    "#######",
    "#OOOOO#",
    "#O#OOO#",
    "#OOOOO#",
    "#OOO#O#",
    "#OOO#G#",
    "#######"
]

# 转换为 D4RL 需要的字符串格式
maze_str = "\\".join(CUSTOM_MAZE1)
ENV_NAME = 'maze2d-custom-official-expert-v0'

try:
    register(
        id=ENV_NAME,
        entry_point='d4rl.pointmaze:MazeEnv',
        max_episode_steps=450,
        kwargs={
            'maze_spec': maze_str,
            'reward_type': 'dense',
            'reset_target': False,
            'ref_min_score': 0.0,
            'ref_max_score': 1.0, 
            'dataset_url': ''
        }
    )
except gym.error.Error:
    pass

# =======================================================
# 3. 主逻辑
# =======================================================

def calculate_official_ref_max_score():
    print("🚀 初始化官方 WaypointController (Q-Iteration + PD)...")
    controller = WaypointController(maze_str)
    
    env = gym.make(ENV_NAME)
    env.seed(100)
    np.random.seed(293)

    num_episodes = 10000
    scores = []

    print(f"📊 开始运行 {num_episodes} 轮专家评估...")

    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        # 🟢 [修复点] 直接使用 env._target 获取目标坐标
        # D4RL Maze 环境在 reset 后会将当前目标存放在 self._target 中
        # 如果 env 被 wrapper 包裹，可能需要解包：env.unwrapped._target
        try:
            current_target = np.array(env.unwrapped._target)
        except AttributeError:
            # 备用方案：如果实在没有 _target，尝试从 obs 里反推（仅适用于特定环境设置，通常 _target 都有）
            # 或者打印 env.__dict__.keys() 看看
            print("Error: 无法找到 _target 属性，请检查环境版本。")
            break

        while not done:
            location = obs[0:2]
            velocity = obs[2:4]

            # 传入获取到的目标
            action, _ = controller.get_action(location, velocity, current_target)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if info.get('TimeLimit.truncated', False):
                done = True
        
        scores.append(total_reward)
        if (i+1) % 10 == 0:
            print(f"  Episode {i+1}: Reward = {total_reward:.4f}")

    avg_score = np.mean(scores)
    std_score = np.std(scores)

    print("\n" + "="*60)
    print(f"🏆 官方基准计算完成 (Official WaypointController)")
    print(f"🏆 Ref Max Score: {avg_score:.6f} (Std: {std_score:.4f})")
    print("="*60)
    print("请将上述数值填入你的 register 代码中的 ref_max_score")

if __name__ == "__main__":
    calculate_official_ref_max_score()