"""
Minimal training script for the CBF barrier on SimpleEngine (safety_gym wrapper in this repo).
Usage (from repo root, conda env activated):

PYTHONPATH=$(pwd)/srlnbc-cbf-module/src:/home/lqz27/dyx_ws/Model-FreeNN/srlnbc \
    python -m srlnbc_cbf.train_safetygym --env point_goal --steps 200 --batch-size 64

python -m srlnbc_cbf.train_safetygym --env point_goal --steps 2000 --save-path ./my_cbf_model.pth    
##
PYTHONPATH=$(pwd)/srlnbc-cbf-module/src:/home/lqz27/dyx_ws/Model-FreeNN/srlnbc \
python -m srlnbc_cbf.train_safetygym --env point_goal --steps 2000 --batch-size 64 --save-path ./my_cbf_model.pth

Notes:
- This script imports SimpleEngine and configs from `srlnbc.env` (part of this repo), so make
  sure the repo root is in PYTHONPATH when running.
- The script collects transitions (obs, new_obs, feasible, infeasible) using either random
  actions or an optional policy, then optimizes the barrier network using the 3-term loss.
"""

import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import repo envs; requires repo root in PYTHONPATH
# 修改为本地直接引用
from simple_safety_gym import SimpleEngine
from config import point_goal_config
from diffuser.models.cbf_network import CBFNetwork


def collect_batch(env, batch_size, policy=None, render: bool = False, record_positions: list = None):
    """Collect a batch of transitions from env.
    Returns dict with 'obs','new_obs','feasible','infeasible'.
    policy: a callable obs->action; if None use random actions.
    render: if True, call env.render() each step (may require MuJoCo GUI).
    record_positions: optional list that will be appended with robot (x,y) positions if available.
    """
    obs_list = []
    new_obs_list = []
    feas_list = []
    infeas_list = []

    obs = env.reset()
    while len(obs_list) < batch_size:
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(obs)

        next_obs, reward, done, info = env.step(action)

        # optional rendering
        if render:
            try:
                env.render()
            except Exception:
                pass

        # record robot position if available (SimpleEngine exposes .world.robot_pos())
        if record_positions is not None:
            try:
                wp = getattr(env, 'world', None)
                if wp is not None:
                    rp = wp.robot_pos()
                    record_positions.append(np.array(rp[:2], dtype=np.float32))
            except Exception:
                pass

        obs_list.append(np.array(obs, dtype=np.float32))
        new_obs_list.append(np.array(next_obs, dtype=np.float32))
        feas_list.append(float(info.get('feasible', 0)))
        infeas_list.append(float(info.get('infeasible', 0)))

        obs = next_obs
        if done:
            obs = env.reset()

    batch = {
        'obs': np.stack(obs_list),
        'new_obs': np.stack(new_obs_list),
        'feasible': np.array(feas_list, dtype=np.float32),
        'infeasible': np.array(infeas_list, dtype=np.float32),
    }
    return batch


def train_once(model, optimizer, batch, epsilon=0.01, barrier_lambda=0.1, device='cpu'):
    model.train()
    obs = torch.from_numpy(batch['obs']).to(device)
    next_obs = torch.from_numpy(batch['new_obs']).to(device)
    feasible = torch.from_numpy(batch['feasible']).to(device)
    infeasible = torch.from_numpy(batch['infeasible']).to(device)

    bx = model(obs)
    bx_next = model(next_obs)

    # Ensure shapes: (B,) or (B,1) etc. Convert to (B,) for clamp operations
    if bx.dim() > 1 and bx.size(1) == 1:
        bx = bx.view(-1)
    if bx_next.dim() > 1 and bx_next.size(1) == 1:
        bx_next = bx_next.view(-1)

    # feasible loss: if feasible==1 want B(x) <= -epsilon -> clamp(epsilon + B(x), min=0)
    feasible_loss = feasible * torch.clamp(epsilon + bx, min=0.0)
    if feasible.sum() > 0:
        feasible_loss = feasible_loss.sum() / feasible.sum()
    else:
        feasible_loss = torch.tensor(0.0, device=device)

    # infeasible loss: if infeasible==1 want B(x) >= epsilon -> clamp(epsilon - B(x), min=0)
    infeasible_loss = infeasible * torch.clamp(epsilon - bx, min=0.0)
    if infeasible.sum() > 0:
        infeasible_loss = infeasible_loss.sum() / infeasible.sum()
    else:
        infeasible_loss = torch.tensor(0.0, device=device)

    # invariance loss: B(x_{t+1}) - (1-lambda) B(x_t) <= 0 -> clamp(..., min=0)
    inv_term = bx_next - (1.0 - barrier_lambda) * bx
    invariance_loss = torch.clamp(inv_term, min=0.0).mean()

    loss = feasible_loss + infeasible_loss + invariance_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': float(loss.detach()),
        'feasible_loss': float(feasible_loss.detach()),
        'infeasible_loss': float(infeasible_loss.detach()),
        'invariance_loss': float(invariance_loss.detach()),
    }


def run_training(env_name='point_goal', steps=1000, batch_size=256, lr=1e-3, device='cpu',
                 epsilon=0.01, barrier_lambda=0.1, save_path=None, render: bool = False,
                 plot_file: str = None):
    # choose env config; currently we support point_goal
    if env_name != 'point_goal':
        raise ValueError('This script currently supports env_name="point_goal" only')

    env = SimpleEngine(point_goal_config)
    obs_dim = env.observation_space.shape[0]

    model = CBFNetwork(input_dim=obs_dim, output_dim=1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps_done = 0
    stats = deque(maxlen=50)
    # recording for visualization
    recorded_positions = []  # list of (x,y)
    recorded_certs = []
    t0 = time.time()
    while steps_done < steps:
        batch = collect_batch(env, batch_size, render=render, record_positions=recorded_positions)
        # compute certificate on next_obs for visualization
        try:
            model.eval()
            with torch.no_grad():
                bx_next = model(torch.from_numpy(batch['new_obs']).to(device))
            if bx_next.dim() > 1 and bx_next.size(1) == 1:
                bx_vals = bx_next.view(-1).cpu().numpy().tolist()
            else:
                bx_vals = bx_next.cpu().numpy().tolist()
            recorded_certs.extend(bx_vals)
        except Exception:
            # ignore visualization if model inference fails here
            pass

        info = train_once(model, optimizer, batch, epsilon=epsilon, barrier_lambda=barrier_lambda, device=device)
        steps_done += batch_size
        stats.append(info)
        if steps_done // batch_size % 1 == 0:
            avg = {k: np.mean([s[k] for s in stats]) for k in stats[0].keys()}
            print(f"steps={steps_done:6d} loss={avg['loss']:.6f} f={avg['feasible_loss']:.6f} inf={avg['infeasible_loss']:.6f} inv={avg['invariance_loss']:.6f}")
    print('training done, time', time.time() - t0)
    if save_path:
        torch.save(model.state_dict(), save_path)
        print('saved model to', save_path)
    # plotting / visualization
    if len(recorded_positions) > 0:
        try:
            import matplotlib.pyplot as plt
            pos_arr = np.array(recorded_positions)
            cert_arr = np.array(recorded_certs)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].plot(pos_arr[:, 0], pos_arr[:, 1], '-b', alpha=0.7)
            axs[0].scatter(pos_arr[0, 0], pos_arr[0, 1], c='g', label='start')
            axs[0].scatter(pos_arr[-1, 0], pos_arr[-1, 1], c='r', label='end')
            # hazards if available
            try:
                hazards = getattr(env, 'hazards_pos', None)
                if hazards is not None:
                    h = np.array(hazards)
                    axs[0].scatter(h[:, 0], h[:, 1], c='magenta', marker='x', s=80, label='hazards')
            except Exception:
                pass
            axs[0].set_title('Robot trajectory')
            axs[0].legend()

            axs[1].plot(cert_arr, '-k')
            axs[1].set_title('Certificate B(next) over collected steps')
            axs[1].set_xlabel('step (per sample)')

            plt.tight_layout()
            if plot_file:
                plt.savefig(plot_file)
                print('Saved trajectory plot to', plot_file)
            else:
                plt.show()
        except Exception as e:
            print('Could not plot trajectory:', e)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='point_goal')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--barrier-lambda', type=float, default=0.1)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--render', action='store_true', help='Render env during data collection (requires MuJoCo GUI)')
    parser.add_argument('--plot-file', type=str, default=None, help='If set, save trajectory+certificate plot to this file')
    args = parser.parse_args()

    run_training(env_name=args.env, steps=args.steps, batch_size=args.batch_size,
                 lr=args.lr, device=args.device, epsilon=args.epsilon, barrier_lambda=args.barrier_lambda,
                 save_path=args.save_path, render=args.render, plot_file=args.plot_file)
