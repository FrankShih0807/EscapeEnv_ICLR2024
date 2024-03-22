import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer
import numpy as np


class LKTD(Optimizer):
    def __init__(self, 
                params,
                lr=1e-4, 
                pseudo_population=50,
                obs_sd=1,
                prior_sd=1,
                sparse_sd=0.1,
                sparse_ratio=1,
                alpha=0.9
                ):

        defaults = dict(lr=lr, pseudo_population=pseudo_population, obs_sd=obs_sd, prior_sd=prior_sd, sparse_sd=sparse_sd, sparse_ratio=sparse_ratio, alpha=alpha)
        super().__init__(params, defaults)
    
    def update_parameters(self, parameter_name, parameter_value):
        for group in self.param_groups:
            group[parameter_name] = parameter_value
        
    def learning_rate(self, lr_init, k):
        # return lr_init / pow(k+1, 1)
        n = 1
        return lr_init * (n / (k + n)) ** 0.9

    @torch.no_grad()
    def step(self, aug_param:Variable, observation:Tensor, current_step:int, closure=None):
        ''' One sigle step of LKTD algorithm
            observation (Tensor):  
            measurement (Tensor):
        '''
        if closure is not None:
            loss = closure()
        
        batch_size = aug_param.numel()
        batch_ratio = batch_size/self.defaults['pseudo_population']
        for group in self.param_groups:
            lr = self.learning_rate(group['lr'], current_step)
            B_t = lr
            w_sd = np.sqrt(B_t * batch_ratio)
            alpha_var_inv = 1/group['alpha']/group['obs_sd']**2
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                prior_grad = self._prior_gradient(p, group['prior_sd'], group['sparse_sd'], group['sparse_ratio'])
                p.sub_(lr/2 * (alpha_var_inv * p.grad + prior_grad * batch_ratio) + w_sd * torch.randn_like(p, device=p.device))
                
                
            
        
        ### Update augmented variables
        if aug_param.grad is not None:
            R_t = 2 * (1-self.defaults['alpha']) * self.defaults['obs_sd']**2
            v_sd = np.sqrt(batch_ratio * R_t) 
            k_var = B_t / (R_t + B_t)
            aug_param.sub_(lr/2 * batch_ratio * alpha_var_inv * aug_param.grad  + w_sd * torch.randn_like(aug_param, device=p.device))
            aug_param.add_(k_var * (observation-aug_param.detach() + v_sd * torch.randn_like(aug_param, device=p.device)) )


    def _prior_gradient(self, param, prior_sd, sparse_sd, sparse_ratio):
        # param_trunc = param.clamp(min=-prior_sd, max=prior_sd)
        if sparse_ratio == 1:
            return param/prior_sd**2
        elif sparse_ratio < 1:
            A = sparse_ratio/(1-sparse_ratio) * sparse_sd/prior_sd * torch.exp(-(1/prior_sd**2 - 1/sparse_sd**2)* torch.square(param)/2) + 1
            coef = 1/prior_sd**2 + torch.div((1/sparse_sd**2 - 1/prior_sd**2),A)
            return coef * param

if __name__ == "__main__":

    pass