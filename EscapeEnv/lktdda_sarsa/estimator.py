from EscapeEnv.common.base_estimator import BaseEstimator
from EscapeEnv.common.scheduler import CyclicalParamScheduler, LinearParamScheduler, ConstantParamScheduler, PolynomialParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from torch.autograd import Variable
from my_package.optimizers import LKTDDA
import numpy as np
from copy import deepcopy
import random


class LKTDDAEstimator(BaseEstimator):
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device)
        

        self.optimizer = LKTDDA(self.qnet.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        # schedulers
        if self.estimator_kwargs['power'] > 0:
            self.lr_scheduler = PolynomialParamScheduler(self.optimizer, 'lr', self.learning_rate, self.estimator_kwargs['power'])
        else:
            self.lr_scheduler = ConstantParamScheduler(self.optimizer, 'lr', self.learning_rate)
        
        self.schedulers = [self.lr_scheduler]
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.resample_buffer = ResampleBuffer(state_sd=self.optimizer_kwargs['state_sd'])
        
        # self.batch_ratio = self.batch_size / self.optimizer_kwargs['pseudo_population']
        
    def evaluate_q_value(self, s):
        q_value_list = []
        for net in self.resample_buffer.old_pool:
            q_value_list.append(net(s).detach())
        return torch.stack(q_value_list, dim=0)
            
        
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()
            
        batch = buffer.sample()
        states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal = self.batch_extract(batch)
        
        r = rewards.clone()
        xi = Variable(r.clone().detach(), requires_grad=True)
        
        for k in range(self.loops_per_train):
            if xi.grad is not None:
                xi.grad.zero_()
            self.optimizer.zero_grad()
            qa = self.qnet(states).gather(-1, actions).squeeze(dim=-1)
                
            predict_r = qa
            predict_r[non_final_mask] -= discount_factor *  self.qnet(non_final_next_states).gather(-1, non_final_next_actions).squeeze(dim=-1).detach()

            loss = self.mse_loss(xi, predict_r)
            loss.backward()
            
            if self.n_updates == 0:
                self.optimizer.step(aug_param=xi, observation=r, current_step=k)
            else:
                if k==0:
                    with torch.no_grad():
                        for p in self.qnet.parameters():
                            p.add_(self.optimizer_kwargs['state_sd'] * torch.randn_like(p))
                resample_net = self.resample_buffer.resampling(self.qnet)
                self.optimizer.step(aug_param=xi, observation=r, resample_net_params=resample_net.parameters(), current_step=k)
                
            if k>=10:
                self.resample_buffer.add(self.qnet)

        self.resample_buffer.reset()
        self.n_updates += 1
        return loss.item()/self.batch_size
    
        
class ResampleBuffer(object):
    def __init__(self, state_sd=1) -> None:
        self.new_pool = []
        self.old_pool = []
        self.state_sd = state_sd
        
        self.normal_dist = torch.distributions.Normal(0, self.state_sd)
    
    def add(self, net):
        self.new_pool.append(deepcopy(net))
        
    def resampling(self, cur_net):
        cur_param_vec = utils.parameters_to_vector(cur_net.parameters()).detach()
        param_diff = self.old_param_vec-cur_param_vec

        log_weight = self.normal_dist.log_prob(param_diff).sum(dim=-1)
        log_weight -= log_weight.max()
        log_weight.clamp_(min=-6).numpy()
        weight_vec = np.exp(log_weight)
        
        # weight_vec = self.calculate_weight(param_diff)
        
        return random.choices(self.old_pool, weight_vec)[0]
    
    def calculate_weight(self, param_vec):
        mse = (param_vec ** 2).sum(dim=-1)
        mse = (mse - mse.max()) / self.state_sd ** 2
        mse.clamp_(min=-6)
        return mse.exp().numpy()
        
    def last_net(self):
        return self.old_pool[-1]
    
    def reset(self):
        self.old_pool = self.new_pool
        self.new_pool = []
        old_param_vec_list = []
        for net in self.old_pool:
            old_param_vec_list.append(utils.parameters_to_vector(net.parameters()).detach())
        self.old_param_vec = torch.stack(old_param_vec_list)
        
if __name__ == '__main__':
    a = [1,2,3,4]
    b = [2,3,4,5]
    
    a = b
    b = []
    print(a)
    print(b)
    
    a = 10 * torch.randn(4)
    a.clamp_(min=-5)
    print(a)