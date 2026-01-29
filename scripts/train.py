'''
PYTHONPATH=. python scripts/train.py --config config.maze2d --dataset maze2d-custom-v1
'''

import diffuser.utils as utils
import pdb
import numpy as np

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-515
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# Maze2D 数据
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

# 后续画图
render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

pdb.set_trace()  #debug..................................................
dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

if hasattr(dataset, 'normalizer'):
    print(f"\n[SafeDiffuser] Injecting normalizer limits into Diffusion model...")
    
    # 尝试从管理器中提取 'observations' 的归一化器
    # DatasetNormalizer 通常有一个 .normalizers 字典
    if hasattr(dataset.normalizer, 'normalizers') and 'observations' in dataset.normalizer.normalizers:
        obs_norm = dataset.normalizer.normalizers['observations']
        diffusion.norm_mins = obs_norm.mins
        diffusion.norm_maxs = obs_norm.maxs
    
    # 如果它本身就是 LimitsNormalizer (备用逻辑)
    elif hasattr(dataset.normalizer, 'mins'):
        diffusion.norm_mins = dataset.normalizer.mins
        diffusion.norm_maxs = dataset.normalizer.maxs
        
    else:
        print("⚠️ Warning: Could not find 'mins' in normalizer. Safety Loss may fail!")
        print(f"   Available attributes: {dir(dataset.normalizer)}")

    # 打印确认一下 (如果是 tensor 就打印 shape，如果是 numpy 就直接打印)
    if isinstance(diffusion.norm_mins, (np.ndarray, list)):
        print(f"   mins: {diffusion.norm_mins}")
        print(f"   maxs: {diffusion.norm_maxs}\n")
    else:
        print(f"   mins (tensor): {diffusion.norm_mins}")
        print(f"   maxs (tensor): {diffusion.norm_maxs}\n")

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

# print('Testing forward...', end=' ', flush=True)
# batch = utils.batchify(dataset[0])
# loss, _ = diffusion.loss(*batch)
# loss.backward()
# print('✓')
print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])

# 适应新的3返回值 (diff, barrier, info)
loss_diff, loss_barrier, _ = diffusion.loss(*batch)
loss = loss_diff + loss_barrier  # 简单加和测一下反向传播
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

