import numpy as np
import torch
from .cbf_adapter import NeuralBarrierAdapter
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import einops

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.neural_cbf = NeuralBarrierAdapter(device='cuda' if torch.cuda.is_available() else 'cpu')

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    

    @torch.no_grad()
    def invariance_neural(self, x, xp1):
        """
        基于 TTC 神经网络的避障修正
        """
        if not hasattr(self, 'neural_cbf'):
            return xp1

        config = getattr(self, 'cbf_config', {})
        # 读取 Alpha (力度)
        # alpha = config.get('alpha', 0.05) 
        # 读取 Clip (截断)
        # clip_value = config.get('clip', 0.01)
        # 读取 Threshold (警戒线)
        # threshold = config.get('threshold', 0.05)
        alpha = config.get('alpha', 0.05) 
        clip_value = config.get('clip', 0.01)
        threshold = config.get('threshold', 0)
        # 1. 准备数据
        original_shape = xp1.shape
        xp1_flat = xp1.view(-1, xp1.shape[-1])
        
        # 2. 准备归一化参数
        if isinstance(self.norm_mins, torch.Tensor):
            mins = self.norm_mins.clone().detach().to(xp1.device)
            maxs = self.norm_maxs.clone().detach().to(xp1.device)
        else:
            mins = torch.tensor(self.norm_mins, device=xp1.device, dtype=torch.float32)
            maxs = torch.tensor(self.norm_maxs, device=xp1.device, dtype=torch.float32)
            
        width = maxs - mins
        
        # 3. 反归一化
        norm_pos = xp1_flat[:, 2:4]
        norm_vel = xp1_flat[:, 4:6]
        
        if len(mins) == 4:
            param_pos_idx = slice(0, 2)
            param_vel_idx = slice(2, 4)
        else:
            param_pos_idx = slice(2, 4)
            param_vel_idx = slice(4, 6)
        
        phys_pos = (norm_pos + 1) / 2 * width[param_pos_idx] + mins[param_pos_idx]
        phys_vel = (norm_vel + 1) / 2 * width[param_vel_idx] + mins[param_vel_idx]
        
        phys_state = torch.cat([phys_pos, phys_vel], dim=1)

        # 4. 获取梯度
        h_val, grad_phys = self.neural_cbf.get_correction_gradient(phys_state)
        min_h_val, min_idx = torch.min(h_val, dim=0)
        
        # 只有当机器人靠近任意墙壁 (h < 0.5，大概在墙壁周围 0.65 米范围内) 时才打印
        if min_h_val.item() < 0.5:
            idx = min_idx.item()
            p_x, p_y = phys_pos[idx, 0].item(), phys_pos[idx, 1].item()
            v_x, v_y = phys_vel[idx, 0].item(), phys_vel[idx, 1].item()
            
            # X 现在是 Row，Y 是 Col
            # print(f"靠近障碍-坐标(Row:{p_x:.2f}, Col:{p_y:.2f}) | 速度({v_x:.2f}, {v_y:.2f}) -> h={min_h_val.item():.3f}")


        # 5.  梯度修正逻辑
        
        # 将物理梯度映射回 Normalized 空间
        # grad_norm 代表：为了让 h 增加 1，归一化坐标需要移动多少
        scale = width[param_pos_idx] / 2
        grad_norm = grad_phys * scale
          
        # 计算原始修正量：危险大 -> 梯度大 -> 修正大
        raw_delta = grad_norm * alpha
        
        # 截断 (Clamping) - 防止瞬移
        delta = torch.clamp(raw_delta, -clip_value, clip_value)
        
        # 只在不安全时修正
        is_unsafe = (h_val < threshold).float()
        
        # 最终修正量
        delta = delta * is_unsafe
        # delta = delta

        if is_unsafe.sum() > 0:
            # 计算每个点被推移的欧几里得距离 (Norm)
            delta_norm = torch.linalg.norm(delta, dim=1) 
            
            # 过滤出真正被修改了的点 (delta > 0)
            active_mask = delta_norm > 1e-7 
            num_modified = active_mask.sum().item()
            
            if num_modified > 0:
                active_deltas = delta_norm[active_mask]
                max_delta = active_deltas.max().item()
                mean_delta = active_deltas.mean().item()
                
                # print(f"[CBF]触发危险，修改了 {num_modified} 个轨迹路点.\n")
                # print(f"推力大小 (Delta): 最大 = {max_delta:.5f}, 平均 = {mean_delta:.5f}")

        # 6. 调试打印
        # if is_unsafe.sum() > 0:
        #    idx = torch.where(is_unsafe)[0][0]
        #    print(f"Danger! h={h_val[idx].item():.2f} | Delta={delta[idx].cpu().numpy()}")

        # 7. 应用修正
        xp1_new = xp1_flat.clone()
        xp1_new[:, 2:4] += delta
        
        return xp1_new.view(original_shape)

    
    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Note:  choose any one of the below
        #---------------------------------------start--------------------------------------------------#
        ####################### original diffuser only
        # x = xp1      
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        # self.safe1 = torch.min(b[:,0])
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        # self.safe2 = torch.min(b[:,0])

        ####################### truncate (shield) and GD (classifier-guidance/potential-based)
        # x = self.Shield(x, xp1)
        # x = self.GD(x, xp1)

        ####################### SafeDiffusers 
        x = xp1 # for training only
        # x = self.invariance(x, xp1)    # RoS
        # x = self.invariance_cf(x, xp1)  # RoS closed form

        # x = self.invariance_neural(x, xp1) # 使用新的 TTC 神经避障

        # x = self.invariance_relax(x, xp1, t) # ReS
        # x = self.invariance_relax_cf(x, xp1, t)   #ReS closed form    
        # x = self.invariance_time(x, xp1, t)   # TVS
        # x = self.invariance_time_cf(x, xp1, t)  # TVS closed form
        # x = self.invariance_relax_narrow(x, xp1, t)  # narrow passage case

        ####################### Applying SafeDiffusers to only the last 10 steps
        # if t <= 10:  #10
        #     # x = self.invariance_relax(x, xp1, t)  #done
        #     # x = self.invariance_relax_narrow(x, xp1, t)

        #     x = self.GD(x, xp1)
        # else:
        #     x = xp1
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        #     self.safe1 = torch.min(b[:,0])
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        #     self.safe2 = torch.min(b[:,0])

        
        ###################### umaze case
        # x = self.invariance_umaze(x, xp1)   #umaze
        # x = self.invariance_umaze_relax(x, xp1, t)   #umaze
        #-----------------------------------------end--------------------------------------------------#

        # train模式safe补丁

        # self.safe1 = torch.tensor(0.0, device=device)  # 或者 device='cuda' / x.device
        # self.safe2 = torch.tensor(0.0, device=device)

        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]
        progress = utils.Silent()
        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        # safe1, safe2 = [], []
        for i in reversed(range(0, self.n_timesteps)):  #-50 change here for the number of diffusion steps,
            if i < 0:
                i = 0
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)


            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        

        progress.close()
        # pdb.set_trace()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, return_diffusion = True, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, return_diffusion= return_diffusion, *args, **kwargs)   ## debug

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)


    def _vb_terms_bpd(
        self, x_start, conditions, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # batch_size = x_start.shape(0)
        # conditions = self._format_conditions(conditions, batch_size)

        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        
        mean, _, log_variance = self.p_mean_variance(
             x_t, conditions, t)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, mean, log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)

        # import pdb; pdb.set_trace()
        # decoder_nll = -discretized_gaussian_log_likelihood(
        #     x_start, means=mean, log_scales=0.5 * log_variance
        # )
        
        # assert decoder_nll.shape == x_start.shape
        # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        # output = torch.where((t == 0), decoder_nll, kl)

        return kl


    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

