import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import numpy as np
import d4rl

# 1. 【修改点一】导入 LARGE_MAZE
from d4rl.pointmaze.maze_model import LARGE_MAZE 

# 导入官方控制器依赖
from d4rl.pointmaze import q_iteration
from d4rl.pointmaze.gridcraft import grid_env
from d4rl.pointmaze.gridcraft import grid_spec

# =======================================================
# WaypointController (保持不变，为了稳妥把迭代次数改大一点)
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
        
        # ⚠️ 注意：Large 地图更大，保险起见把迭代次数从 50 改为 100
        # 否则 Q-Iteration 可能传导不到起点，导致规划失败
        q_values = q_iteration.q_iteration(env=self.env, num_itrs=100, discount=0.99)
        
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
# 验证主逻辑
# =======================================================

def verify_against_official_large():
    # 2. 【修改点二】环境名称改为 Large
    ENV_NAME = 'maze2d-large-v1'
    
    print(f"🚀 正在加载官方环境: {ENV_NAME}")
    env = gym.make(ENV_NAME)
    
    # Large 的 ref_max_score 应该是 273.99 左右
    official_ref_max = env.ref_max_score
    print(f"📖 官方注册表中的 Ref Max Score: {official_ref_max}")

    # 3. 【修改点三】传入 LARGE_MAZE
    print("🤖 初始化 WaypointController (使用官方 LARGE_MAZE 地图)...")
    controller = WaypointController(LARGE_MAZE)
    
    # Large 地图比较大，随机性强，设置固定种子方便对比
    env.seed(123) 
    np.random.seed(123)

    num_episodes = 500
    scores = []

    print(f"📊 开始运行 {num_episodes} 轮测试...")

    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        try:
            if hasattr(env, 'unwrapped'):
                current_target = np.array(env.unwrapped._target)
            else:
                current_target = np.array(env._target)
        except:
            current_target = np.array(env.goal_locations[0])

        while not done:
            location = obs[0:2]
            velocity = obs[2:4]

            action, _ = controller.get_action(location, velocity, current_target)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if info.get('TimeLimit.truncated', False):
                done = True
        
        scores.append(total_reward)
        if (i+1) % 10 == 0:
            print(f"  Episode {i+1}: Reward = {total_reward:.4f}")

    my_avg_score = np.mean(scores)
    
    print("\n" + "="*60)
    print(f"✅ 验证结果对比:")
    print(f"   官方标准值 (Reference): {official_ref_max:.4f}")
    print(f"   代码复现值 (Calculated): {my_avg_score:.4f}")
    
    diff = abs(official_ref_max - my_avg_score)
    print(f"   差异 (Difference): {diff:.4f}")
    
    # Large 环境本身方差巨大，允许 30-40 分的差异
    if diff < 40.0:
        print("🎉 结论: 代码逻辑正确！")
    else:
        print("⚠️ 结论: 差异较大，建议检查随机种子或迭代次数。")
    print("="*60)

if __name__ == "__main__":
    verify_against_official_large()