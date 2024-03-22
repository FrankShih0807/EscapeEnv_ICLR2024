import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer
import numpy as np


class SGHMC(Optimizer):
    def __init__(self, 
                params,
                lr=1e-5, 
                pseudo_population=50,
                alpha=0.9,
                obs_sd=1,
                prior_sd=1,
                sparse_sd=0.1,
                sparse_ratio=1,
                ):

        defaults = dict(lr=lr, pseudo_population=pseudo_population, alpha=alpha, obs_sd=obs_sd, prior_sd=prior_sd, sparse_sd=sparse_sd, sparse_ratio=sparse_ratio)
        super().__init__(params, defaults)

    def update_parameters(self, parameter_name, parameter_value):
        for group in self.param_groups:
            group[parameter_name] = parameter_value
            
    def learning_rate(self, lr_init, k):
        # return lr_init / pow(k+1, 1)
        n = 1
        return lr_init * (n / (k + n)) ** 0.9

    @torch.no_grad()
    def step(self, batch_size, current_step=0, closure=None):
        ''' One sigle step of LKTD algorithm
            observation (Tensor):  
            measurement (Tensor):
        '''
        if closure is not None:
            loss = closure()
            
        batch_ratio = batch_size/self.defaults['pseudo_population']
        for group in self.param_groups:
            lr = self.learning_rate(group['lr'], current_step)
            alpha = group['alpha']
            w_sd = np.sqrt(lr * batch_ratio)
            var_inv = 1/group['obs_sd']**2
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                # if current_step == 0:
                #     state['momentum'] = torch.zeros_like(p)
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                v = state['momentum']
                prior_grad = self._prior_gradient(p, group['prior_sd'], group['sparse_sd'], group['sparse_ratio'])
                v = (1 - alpha) * v + lr * (var_inv * p.grad + prior_grad * batch_ratio) + np.sqrt(2*alpha) * w_sd * torch.randn_like(p, device=p.device)
                p.sub_(v)
            # print(1-alpha)

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