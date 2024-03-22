from EscapeEnv.common.base_estimator import BaseEstimator
from EscapeEnv.common.scheduler import CyclicalParamScheduler, LinearParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from my_package.optimizers import LKTD
from copy import deepcopy


class LKTDEstimator(BaseEstimator):
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device)
        

        self.optimizer = LKTD(self.qnet.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        # schedulers
        self.lr_scheduler = CyclicalParamScheduler(self.optimizer, 'lr', self.learning_rate, self.estimator_kwargs['cycle_len'])
        self.sr_scheduler = LinearParamScheduler(self.optimizer, 'sparse_ratio', 1.0, self.optimizer_kwargs['sparse_ratio'], self.estimator_kwargs['sr_decay'])
        
        self.schedulers = [self.lr_scheduler, self.sr_scheduler]
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.net_buffer = NetBuffer()
        
    def evaluate_q_value(self, s):
        if self.net_buffer.active:
            q_value_list = []
            for net in self.net_buffer.net_pool:
                q_value_list.append(net(s).detach())
            return torch.stack(q_value_list, dim=0)
        else:
            return super().evaluate_q_value(s)
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()
        batch = buffer.sample()
        # batch data
        states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal = self.batch_extract(batch)

        self.net_buffer.reset()
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

            self.optimizer.step(xi, r, k)
            
            if k >= self.loops_per_train:
                self.net_buffer.add(self.qnet)
                
        self.n_updates += 1
        return loss.item()/self.batch_size
    
class NetBuffer(object):
    def __init__(self) -> None:
        self.net_pool = []
        self.active = False
        
    def add(self, net):
        self.active = True
        self.net_pool.append(deepcopy(net))
        
    def reset(self):
        self.net_pool = []