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
    
    @torch.no_grad()   #only for sampling
    def Shield(self, x0, xp10):  #Truncate method

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        for k in range(nBatch):
            if b[k, 0] < 0: 
                theta = torch.atan2((xp1[k,2:3] - off_y)/yr, (xp1[k,3:4] - off_x)/xr)
                xp1[k,2] = yr*torch.sin(theta) + off_y
                xp1[k,3] = xr*torch.cos(theta) + off_x

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

         #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1
    
    @torch.no_grad()   #only for sampling
    def GD(self, x0, xp10):    #classifier guidance or potential-based method

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        for k in range(nBatch):
            if b[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(2*((xp1[k,2:3] - off_y)/yr)/yr)
                u2 = 0.2/(2*((xp1[k,3:4] - off_x)/xr)/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001  #note no 0.1/0.01 for GD, but has for potential
                xp1[k,3] = xp1[k,3] + u2*0.001
            elif b2[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(4*((xp1[k,2:3] - off_y)/yr)**3/yr)
                u2 = 0.2/(4*((xp1[k,3:4] - off_x)/xr)**3/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001
                xp1[k,3] = xp1[k,3] + u2*0.001
            # else:
            #     x[k,2] = xp1[k,2]
            #     x[k,3] = xp1[k,3]

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1

    @torch.no_grad()   #only for sampling
    def invariance_umaze(self, x, xp1):   #  RoS-diffuser for umaze

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1.52/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1.52/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

       #CBF
        b = 1 - ((x[:,2:3] - off_y)/yr)**4 - ((x[:,3:4] - off_x)/xr)**4
        Lfb = 0
        Lgbu1 = -4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = -4*((x[:,3:4] - off_x)/xr)**3/xr

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        self.safe1 = torch.min(b[:,0])

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1.2/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*0.6/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0])

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        rt = rt.unsqueeze(0)
        return rt
    
    @torch.no_grad()   #only for sampling
    def invariance_umaze_relax(self, x, xp1, t):  #  ReS-diffuser for umaze

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1.52/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1.52/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = 1 - ((x[:,2:3] - off_y)/yr)**4 - ((x[:,3:4] - off_x)/xr)**4
        Lfb = 0
        Lgbu1 = -4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = -4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe1 = torch.min(b[:,0])

        if t >= 10:
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1.2/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*0.6/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0])

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0], dim = 1)
        Q = Variable(torch.eye(4))
        Q = Q.unsqueeze(0).expand(nBatch, 4, 4).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()   #only for sampling
    def invariance(self, x, xp1):    #  RoS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01  # robust term 09/25
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        self.safe1 = torch.min(b[:,0] + 0.01)  # robust term 09/25

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term 09/25
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0]+ 0.01) # robust term 09/25

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        # print(out)
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()   #only for sampling
    def invariance_cf(self, x, xp1):  # closed form solution,  RoS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b0 = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01  # robust term 09/25
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b0

        self.safe1 = torch.min(b0[:,0] + 0.01)  # robust term 09/25

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term 09/25
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0]+ 0.01) # robust term 09/25

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
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
        



    @torch.no_grad()   #only for sampling
    def invariance_relax(self, x, xp1, t):  #  ReS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        if t >= 10:   # debug  10
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0], dim = 1)
        Q = Variable(torch.eye(4))
        Q = Q.unsqueeze(0).expand(nBatch, 4, 4).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt
    
    @torch.no_grad()   #only for sampling
    def invariance_relax_cf(self, x, xp1, t):  # closed-form solution, ReS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        if t >= 10:   # debug  10
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G0 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        k = 1
        h0 = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

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
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
    

    @torch.no_grad()   #only for sampling
    def invariance_relax_narrow(self, x, xp1, t):  #  ReS-diffuser for maze2d-large-v1,  narrow passage case

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        if t >= 10:   # debug  10
            sign = 1   #relax
        else:
            sign = 0   #non-relax

        #normalize obstacle 1,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])

        off_x = 2*(5.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4  # 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        self.safe1 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0, rx0, rx0, rx0, rx0], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        ########################################### obs 2
        off_x = 2*(5.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.6 #0.01
        Lfb = 0
        Lgbu12 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu22 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b2[:,0] + 0.01)

        G2 = torch.cat([-Lgbu12, -Lgbu22, rx0, rx1, rx0, rx0, rx0, rx0], dim = 1)
        G2 = G2.unsqueeze(1)
        k = 1
        h2 = Lfb + k*b2

        ########################################### obs 3
        off_x = 2*(3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b3 = ((x[:,2:3] - off_y)/yr/0.5)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu13 = 4*((x[:,2:3] - off_y)/yr/0.5)**3/yr
        Lgbu23 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G3 = torch.cat([-Lgbu13, -Lgbu23, rx0, rx0, rx1, rx0, rx0, rx0], dim = 1)
        G3 = G3.unsqueeze(1)
        k = 1
        h3 = Lfb + k*b3

        ########################################### obs 4
        off_x = 2*(8.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(3.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b4 = ((x[:,2:3] - off_y)/yr/1.8)**4 + ((x[:,3:4] - off_x)/xr/1.8)**4 - 1 - 0.01
        Lfb = 0
        Lgbu14 = 4*((x[:,2:3] - off_y)/yr/1.8)**3/yr
        Lgbu24 = 4*((x[:,3:4] - off_x)/xr/1.8)**3/xr

        G4 = torch.cat([-Lgbu14, -Lgbu24, rx0, rx0, rx0, rx1, rx0, rx0], dim = 1)
        G4 = G4.unsqueeze(1)
        k = 1
        h4 = Lfb + k*b4

        ########################################### obs 5
        off_x = 2*(7.6-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(7-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b5 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4 #0.01
        Lfb = 0
        Lgbu15 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu25 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G5 = torch.cat([-Lgbu15, -Lgbu25, rx0, rx0, rx0, rx0, rx1, rx0], dim = 1)
        G5 = G5.unsqueeze(1)
        k = 1
        h5 = Lfb + k*b5

        ########################################### obs 6
        off_x = 2*(10-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(6.3-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b6 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu16 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu26 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G6 = torch.cat([-Lgbu16, -Lgbu26, rx0, rx0, rx0, rx0, rx0, rx1], dim = 1)
        G6 = G6.unsqueeze(1)
        k = 1
        h6 = Lfb + k*b6

        b0 = torch.cat([b, b2, b3, b4, b5, b6], dim = 1)
        idx = torch.argmin(b0, dim = 1).cpu().numpy()
        G0 = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        h0 = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)
        rows = len(G0[:,0,0])
        G = []
        h = []
        for i in range(rows):
            G.append(G0[i:i+1,idx[i]:idx[i]+1])
            h.append(h0[i:i+1,idx[i]:idx[i]+1])
        G = torch.cat(G, dim = 0)
        h = torch.cat(h, dim = 0)


        

        # G = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        # h = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)

        # G = torch.cat([G1, G2, G3, G5], dim = 1)
        # h = torch.cat([h1, h2, h3, h5], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0, q0, q0], dim = 1)
        Q = Variable(torch.eye(8))
        Q = Q.unsqueeze(0).expand(nBatch, 8, 8).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt


    @torch.no_grad()   #only for sampling
    def invariance_time(self, x, xp1, t):  #  TVS-diffuser for maze2d-large-v1
        t_bias = 5  #50

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - nn.Sigmoid()(t_bias - t) -0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1  #0.3
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - nn.Sigmoid()(t_bias - t) - 0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1  #0.4
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
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
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - nn.Sigmoid()(t_bias - t) -0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1  #0.3
        h0 = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - nn.Sigmoid()(t_bias - t) - 0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

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
        # print(out)
        rt = rt.unsqueeze(0)
        return rt        


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
        clip_value = config.get('clip', 0.5)
        threshold = config.get('threshold', 0.05)
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
        # ==================== [新增的照妖镜：排查下方的墙] ====================
        # 假设下方墙壁的中心大约在 Y=2.5 左右
        # 抓取所有 Y 在 1.5 到 3.5 之间，X 在 1.5 到 3.5 之间的机器人状态
        lower_wall_mask = (phys_pos[:, 1] > 1.5) & (phys_pos[:, 1] < 3.5) & \
                          (phys_pos[:, 0] > 1.5) & (phys_pos[:, 0] < 3.5)
        
        if lower_wall_mask.sum() > 0:
            # 随机抓取一个在下方墙壁附近的数据点
            idx = torch.where(lower_wall_mask)[0][0]
            p_x, p_y = phys_pos[idx, 0].item(), phys_pos[idx, 1].item()
            v_x, v_y = phys_vel[idx, 0].item(), phys_vel[idx, 1].item()
            
            # 临时算一下它的 h 值
            with torch.no_grad():
                test_h = self.neural_cbf.model(phys_state[idx:idx+1]).item()
            
            print(f"👀 [下墙监控] 位置:({p_x:.2f}, {p_y:.2f}) | 速度:({v_x:.2f}, {v_y:.2f}) -> h={test_h:.3f}")
        # ====================================================================
        # 4. 获取梯度
        h_val, grad_phys = self.neural_cbf.get_correction_gradient(phys_state)
        
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
        # is_unsafe = torch.ones_like(h_val) # 全部标记为不安全，全图开启护盾
        # delta = delta * 1.0 # 直接应用所有推力
        # 更优雅的写法，只在真正危险 (h < 0) 时发力
        # is_unsafe = (h_val < 0.0).float() # 小于0就亮红灯
        # delta = delta * is_unsafe
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
                
                print(f"[CBF] ⚠️ 触发危险! 强行修改了 {num_modified} 个轨迹路点.")
                print(f"      推力大小 (Delta): 最大 = {max_delta:.5f}, 平均 = {mean_delta:.5f}")

        # --- 新逻辑 End ---

        # 6. 调试打印 (可选)
        # if is_unsafe.sum() > 0:
        #    idx = torch.where(is_unsafe)[0][0]
        #    print(f"Danger! h={h_val[idx].item():.2f} | Delta={delta[idx].cpu().numpy()}")

        # 7. 应用修正
        xp1_new = xp1_flat.clone()
        xp1_new[:, 2:4] += delta
        
        return xp1_new.view(original_shape)


    # @torch.no_grad()
    # def invariance_neural(self, x, xp1):
    #     # 1. 检查模块是否存在
    #     has_cbf = hasattr(self, 'neural_cbf')
        
    #     # 2. 读取配置
    #     config = getattr(self, 'cbf_config', {})
        
    #     # === [强制体检] 只在前 3 个时间步打印，避免刷屏 ===
    #     # 我们利用 x 的形状或其他特征来判断是不是第一次调用，或者简单点，直接打印
    #     # 为了防止刷屏，我们只在 config 为空或者 neural_cbf 缺失时疯狂打印
        
    #     print_debug = False
    #     if not hasattr(self, '_debug_counter'):
    #         self._debug_counter = 0
        
    #     if self._debug_counter < 3: # 只打前 3 次
    #         print_debug = True
    #         self._debug_counter += 1
            
    #     if print_debug:
    #         print(f"\n[CBF DIAGNOSIS] Step {self._debug_counter}")
    #         print(f"  > 模块挂载状态: {'✅ 正常' if has_cbf else '❌ 缺失 (致命错误!)'}")
    #         print(f"  > 接收到的参数: {config}")
            
    #         if not has_cbf:
    #             print("  > ⚠️ 警告：因为模块缺失，避障直接跳过！请检查 plan_cbf.py 的挂载逻辑。")

    #     if not has_cbf:
    #         return xp1

    #     # 3. 正常逻辑
    #     alpha = config.get('alpha', 0.11) 
    #     clip_value = config.get('clip', 0.01)
    #     threshold = config.get('threshold', 0.03)

    #     # ... (数据准备代码) ...
    #     original_shape = xp1.shape
    #     xp1_flat = xp1.view(-1, xp1.shape[-1])
        
    #     if isinstance(self.norm_mins, torch.Tensor):
    #         mins = self.norm_mins.clone().detach().to(xp1.device)
    #         maxs = self.norm_maxs.clone().detach().to(xp1.device)
    #     else:
    #         mins = torch.tensor(self.norm_mins, device=xp1.device, dtype=torch.float32)
    #         maxs = torch.tensor(self.norm_maxs, device=xp1.device, dtype=torch.float32)
            
    #     width = maxs - mins
        
    #     # ... (反归一化代码) ...
    #     norm_pos = xp1_flat[:, 2:4]
    #     norm_vel = xp1_flat[:, 4:6]
        
    #     if len(mins) == 4:
    #         param_pos_idx = slice(0, 2)
    #         param_vel_idx = slice(2, 4)
    #     else:
    #         param_pos_idx = slice(2, 4)
    #         param_vel_idx = slice(4, 6)
        
    #     phys_pos = (norm_pos + 1) / 2 * width[param_pos_idx] + mins[param_pos_idx]
    #     phys_vel = (norm_vel + 1) / 2 * width[param_vel_idx] + mins[param_vel_idx]
    #     phys_state = torch.cat([phys_pos, phys_vel], dim=1)
        
    #     # 4. 获取梯度
    #     h_val, grad_phys = self.neural_cbf.get_correction_gradient(phys_state)
        
    #     if print_debug:
    #         print(f"  > h_val (安全值): min={h_val.min().item():.4f}, max={h_val.max().item():.4f}")
    #         unsafe_cnt = (h_val < threshold).sum().item()
    #         print(f"  > 危险点数量: {unsafe_cnt}")

    #     # 5. 计算修正
    #     scale = width[param_pos_idx] / 2
    #     grad_norm = grad_phys * scale
        
    #     raw_delta = grad_norm * alpha
    #     delta = torch.clamp(raw_delta, -clip_value, clip_value)
    #     is_unsafe = (h_val < threshold).float()
    #     delta = delta * is_unsafe
        
    #     xp1_new = xp1_flat.clone()
    #     xp1_new[:, 2:4] += delta
        
    #     return xp1_new.view(original_shape)
    
    
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
        # x = xp1 # for training only
        # x = self.invariance(x, xp1)    # RoS
        # x = self.invariance_cf(x, xp1)  # RoS closed form

        x = self.invariance_neural(x, xp1) # 使用新的 TTC 神经避障

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

            
            # safe1.append(self.safe1.unsqueeze(0))
            # safe2.append(self.safe2.unsqueeze(0))
            # ================= [修复开始] =================
        # 兼容性修复：防止 safe1/safe2 是 int 类型时报错
        
        # # 处理 safe1
        #     if isinstance(self.safe1, torch.Tensor):
        #         safe1.append(self.safe1.unsqueeze(0))
        #     else:
        #     # 如果是 int/float，先转成 Tensor 再存
        #         safe1.append(torch.tensor([self.safe1], device=x.device))

        # # 处理 safe2
        #     if isinstance(self.safe2, torch.Tensor):
        #         safe2.append(self.safe2.unsqueeze(0))
        #     else:
        #         safe2.append(torch.tensor([self.safe2], device=x.device))
            
        # ================= [修复结束] =================

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        
        # self.safe1 = torch.cat(safe1, dim=0)
        # self.safe2 = torch.cat(safe2, dim=0)

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

