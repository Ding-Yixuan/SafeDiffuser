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

        self.plan_r1   = 0.6   # Wall 0 半径
        self.plan_r2_x = 0.6   # Wall 1 半径 (X轴/长边)
        self.plan_r2_y = 1.1   # Wall 1 半径 (Y轴/短边)

        self.eval_r1   = 0.6   # 真实 Wall 0 半径
        self.eval_r2_x = 0.6   # 真实 Wall 1 半径 (X轴)
        self.eval_r2_y = 1.1   # 真实 Wall 1 半径 (Y轴)

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
    
    def _update_safety_metrics(self, x):
        """
        使用固定的 Evaluation 参数计算 S-Spec 和 C-Spec。
        该函数只负责更新 self.safe1 和 self.safe2，不修改轨迹。
        x: [Batch, 4] 的轨迹点 (通常是 xp1 或修正后的 rt)
        """
        
        # Wall 1 (S-Spec): 半径 0.6
        yr_e1 = 2 * self.eval_r1 / (self.norm_maxs[0] - self.norm_mins[0])
        xr_e1 = 2 * self.eval_r1 / (self.norm_maxs[1] - self.norm_mins[1])
        
        # Wall 2 (C-Spec): 半径 X=0.6, Y=1.1 (注意长短轴)
        yr_e2 = 2 * self.eval_r2_y / (self.norm_maxs[0] - self.norm_mins[0])
        xr_e2 = 2 * self.eval_r2_x / (self.norm_maxs[1] - self.norm_mins[1])

        # 偏移量 (这是地图固有属性，通常不变)
        off_y_1 = 2 * (2.0 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x_1 = 2 * (2.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y_2 = 2 * (4.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x_2 = 2 * (4.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

        # ============================================================
        # 2. 计算 Barrier 值 (只算分，不求梯度)
        # ============================================================
        
        # S-Spec (Safe 1): Square ^2
        b_eval_1 = ((x[:,2:3] - off_y_1)/yr_e1)**2 + ((x[:,3:4] - off_x_1)/xr_e1)**2 - 1 - 0.01
        
        # C-Spec (Safe 2): Quartic ^4
        b_eval_2 = ((x[:,2:3] - off_y_2)/yr_e2)**4 + ((x[:,3:4] - off_x_2)/xr_e2)**4 - 1 - 0.01

        # ============================================================
        # 3. 更新全局指标
        # ============================================================
        # self.safe1 = torch.min(b_eval_1[:,0] + 0.01)
        # self.safe2 = torch.min(b_eval_2[:,0] + 0.01)
        self.safe1 = torch.min(torch.clamp(b_eval_1[:,0], max=0.0))
        self.safe2 = torch.min(torch.clamp(b_eval_2[:,0], max=0.0))
    
    @torch.no_grad()   #only for sampling
    def invariance_cf(self, x, xp1):  # closed form solution,  RoS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x


        yr = 2 * self.plan_r1 / (self.norm_maxs[0] - self.norm_mins[0])
        xr = 2 * self.plan_r1 / (self.norm_maxs[1] - self.norm_mins[1])
        off_y = 2 * (2.0 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2 * (2.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

        # 3. CBF 计算 (Quadratic: ^2)
        b0 = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01   # robust term increased
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b0

        # self.safe1 = torch.min(b0[:,0] + 0.01)


        yr = 2 * self.plan_r2_y / (self.norm_maxs[0] - self.norm_mins[0])
        xr = 2 * self.plan_r2_x / (self.norm_maxs[1] - self.norm_mins[1])
        off_y = 2 * (4.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2 * (4.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

        # 3. CBF 计算 (Quartic: ^4) - 近似矩形
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        # self.safe2 = torch.min(b[:,0]+ 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h1 = Lfb + k*b
        
        q = -ref[:,2:4].to(b.device)
        
        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out

        self._update_safety_metrics(rt)
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
        
    @torch.no_grad()   #only for sampling
    def invariance_relax_cf(self, x, xp1, t):  # closed-form solution, ReS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        yr = 2 * self.plan_r1 / (self.norm_maxs[0] - self.norm_mins[0])
        xr = 2 * self.plan_r1 / (self.norm_maxs[1] - self.norm_mins[1])
        off_y = 2 * (2.0 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2 * (2.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

        # 3. CBF 计算 (Quadratic: ^2)
        b0 = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 -0.01   # robust term increased
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        # self.safe1 = torch.min(b0[:,0] + 0.01)

        if t >= 10:   # debug  10
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b0.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b0.device)

        G0 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        k = 1
        h0 = Lfb + k*b0

        yr = 2 * self.plan_r2_y / (self.norm_maxs[0] - self.norm_mins[0])
        xr = 2 * self.plan_r2_x / (self.norm_maxs[1] - self.norm_mins[1])
        off_y = 2 * (4.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2 * (4.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

        # 3. CBF 计算 (Quartic: ^4) - 近似矩形
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        # self.safe2 = torch.min(b[:,0]+ 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        k = 1
        h1 = Lfb + k*b
        
   
        q = -ref[:,2:4].to(G0.device)
        q0 = torch.zeros_like(q).to(G0.device)
        q = torch.cat([q, q0], dim = 1)

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()    
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        self._update_safety_metrics(rt)
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
    
    @torch.no_grad()   #only for sampling
    def invariance_time_cf(self, x, xp1, t):  # closed-form solution, TVS-diffuser for maze2d-large-v1
        t_bias = 5  #50 

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        yr = 2 * self.plan_r1 / (self.norm_maxs[0] - self.norm_mins[0])
        xr = 2 * self.plan_r1 / (self.norm_maxs[1] - self.norm_mins[1])
        off_y = 2 * (2.0 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2 * (2.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - nn.Sigmoid()(t_bias - t) -0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        # self.safe1 = torch.min(b[:,0] + 0.01)

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1  #0.3
        h0 = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        yr = 2 * self.plan_r2_y / (self.norm_maxs[0] - self.norm_mins[0])
        xr = 2 * self.plan_r2_x / (self.norm_maxs[1] - self.norm_mins[1])
        off_y = 2 * (4.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        off_x = 2 * (4.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - nn.Sigmoid()(t_bias - t) - 0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        # self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1  #0.4
        h1 = Lfb + k*b
        
   
        q = -ref[:,2:4].to(G0.device)

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()    
        rt[:,2:4] = x[:,2:4] + out
        self._update_safety_metrics(rt)
        # print(out)
        rt = rt.unsqueeze(0)
        return rt        

    @torch.no_grad()
    def invariance_lag_cf(self, x, xp1, t):
        """
        Lagrangian / Gradient-based correction (Revised)
        """
        # 1. 开启梯度计算
        with torch.enable_grad():
            xp1_in = xp1.detach().clone().requires_grad_(True)
            
            # --- 几何参数 (Radius=0.6) ---
            yr_1 = 2 * self.plan_r1 / (self.norm_maxs[0] - self.norm_mins[0])
            xr_1 = 2 * self.plan_r1 / (self.norm_maxs[1] - self.norm_mins[1])
            yr_2 = 2 * self.plan_r2_y / (self.norm_maxs[0] - self.norm_mins[0])
            xr_2 = 2 * self.plan_r2_x / (self.norm_maxs[1] - self.norm_mins[1])

            off_y_1 = 2 * (2.0 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            off_x_1 = 2 * (2.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
            off_y_2 = 2 * (4.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            off_x_2 = 2 * (4.0 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1

            # --- Barrier 计算 ---
            # h(x) < 0 表示不安全
            # 注意：这里取正值部分作为 Loss，即 ReLU(-h)
            
            # Obstacle 1 (Quadratic)
            h1 = ((xp1_in[:,2:3] - off_y_1)/yr_1)**2 + ((xp1_in[:,3:4] - off_x_1)/xr_1)**2 - 1 - 0.01
            
            # Obstacle 2 (Quartic)
            h2 = ((xp1_in[:,2:3] - off_y_2)/yr_2)**4 + ((xp1_in[:,3:4] - off_x_2)/xr_2)**4 - 1 - 0.01

            # --- Loss 计算 (Sum 模式) ---
            # 使用 sum() 保证每个样本的梯度力大小独立于 batch size
            lag_lambda = 5.0  # 稍微调大 Lambda，代表"斥力场"的强度
            loss = lag_lambda * (torch.relu(-h1).sum() + torch.relu(-h2).sum())
            
            # 初始化修正量 (全0)
            correction = torch.zeros_like(xp1)

            # --- 梯度反传 ---
            # 修正 1: 使用 .item() 判断
            if loss.item() > 1e-6:
                grads = torch.autograd.grad(loss, xp1_in)[0]
                
                # 修正 3: 限制单步最大位移 (Clamp)
                # 防止梯度爆炸导致小车瞬移飞出地图
                # step_size 可以理解为"时间步长"或"学习率"
                step_size = 0.5 
                
                # 计算原始推力 (负梯度方向)
                raw_push = - step_size * grads
                
                # 截断推力：每次最多修正 0.05 (归一化坐标系下)
                # 0.05 在 maze2d 里大概对应 5cm-10cm，足够了
                clamped_push = torch.clamp(raw_push, -0.05, 0.05)
                
                # 修正 2: 只更新位置维度 (2:4)
                correction[:, 2:4] = clamped_push[:, 2:4]
            
            # 日志 (保持你的逻辑)
            # self.safe1 = torch.min(h1.detach() + 0.01)
            # self.safe2 = torch.min(h2.detach() + 0.01)

        # 应用修正
        xp1_out = xp1 + correction
        self._update_safety_metrics(xp1_out)
        
        return xp1_out

    
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
        x = xp1  
        self._update_safety_metrics(x.squeeze(0) if x.ndim > 2 else x)
        # 
        #     
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
        # x = xp1 # for training only
        # x = self.invariance(x, xp1)    # RoS
        # x = self.invariance_cf(x, xp1)  # RoS closed form

        # x = self.invariance_neural(x, xp1) # 使用新的 TTC 神经避障
        # x = self.invariance_lag_cf(x, xp1, t) # 手动构造的cbf的lagrangian修正

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
