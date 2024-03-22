from EscapeEnv.common.base_estimator import ActorCriticEstimator
from EscapeEnv.common.scheduler import ConstantParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from my_package.optimizers import LKTDDA
from torch.autograd import Variable
import torch.nn.utils as utils
import random
import numpy as np
from copy import deepcopy



class FT_A2CEstimator(ActorCriticEstimator):
    def __init__(self, network, learning_rate, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, learning_rate, optimizer_kwargs, estimator_kwargs, device)
        
        self.loops_per_train = self.estimator_kwargs['loops_per_train']
        self.vf_coef = self.estimator_kwargs['vf_coef']
        self.ent_coef = self.estimator_kwargs['ent_coef']
        self.pseudo_population = self.optimizer_kwargs['pseudo_population']
        self.max_grad_norm = self.estimator_kwargs['max_grad_norm']

        self.actor_optimizer = optim.Adam(self.ac_net.actor_parameters(), lr=0.5e-3)
        # self.actor_optimizer = optim.RMSprop(self.ac_net.actor_parameters(), alpha=0.99, eps=1e-5, weight_decay=0)

        self.critic_optimizer = LKTDDA(self.ac_net.critic_parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
                
        # self.actor_lr_scheduler = ConstantParamScheduler(self.actor_optimizer, 'lr', self.learning_rate)
        # self.critic_lr_scheduler = ConstantParamScheduler(self.critic_optimizer, 'lr', self.vf_coef * self.learning_rate)
        
        self.schedulers = []
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.resample_buffer = ResampleBuffer(state_sd=self.optimizer_kwargs['state_sd'])
    
    # def evaluate_values(self, s):
    #     value_list = []
    #     for net in self.resample_buffer.old_pool:
    #         value_list.append(net(s).detach())
    #     return torch.stack(value_list, dim=0)
        
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()

        # optimize actor network
        rollout_data = buffer.get()
        actions = buffer.actions
        

        values, log_prob, entropy = self.ac_net.evaluate_actions(rollout_data.observations, actions)
        advantages = rollout_data.advantages
        
        policy_loss = - (advantages * log_prob).mean()


        entropy_loss = - torch.mean(entropy)
        # actor_loss = policy_loss + self.ent_coef * entropy_loss 
        value_loss = F.mse_loss(rollout_data.returns, values.flatten())
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        
        self.actor_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.ac_net.actor_parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Sample critic network
        
        returns = rollout_data.returns.clone()
        xi = Variable(rollout_data.returns.clone(), requires_grad=True)
        
        for k in range(self.loops_per_train):
            if xi.grad is not None:
                xi.grad.zero_()
            self.critic_optimizer.zero_grad()
            values, _, _ = self.ac_net.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()


            loss = self.mse_loss(xi, values)
            loss.backward()
            
            if self.n_updates == 0:
                self.critic_optimizer.step(aug_param=xi, observation=returns, current_step=k)
            else:
                resample_net = self.resample_buffer.resampling(self.ac_net)
                self.critic_optimizer.step(aug_param=xi, observation=returns.clone(), resample_net_params=resample_net.critic_parameters(), current_step=k)
                
            if k>=5:
                self.resample_buffer.add(self.ac_net)

        self.resample_buffer.reset()
        self.n_updates += 1
        return loss.item()


class ResampleBuffer(object):
    def __init__(self, state_sd=1) -> None:
        self.new_pool = []
        self.old_pool = []
        self.state_sd = state_sd
        
        self.normal_dist = torch.distributions.Normal(0, self.state_sd)
    
    def add(self, net):
        self.new_pool.append(deepcopy(net))
        
    def resampling(self, cur_net):
        cur_param_vec = utils.parameters_to_vector(cur_net.critic_parameters()).detach()
        param_diff = self.old_param_vec-cur_param_vec

        log_weight = self.normal_dist.log_prob(param_diff).sum(dim=-1)
        log_weight -= log_weight.max()

        log_weight.clamp_(min=-6).numpy()
        weight_vec = np.exp(log_weight)
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
            old_param_vec_list.append(utils.parameters_to_vector(net.critic_parameters()).detach())
        self.old_param_vec = torch.stack(old_param_vec_list)



if __name__ == '__main__':
    x = torch.tensor([1,2,3,4,5])
    legal = torch.tensor([0,1,0,1,0])
    
    y = (x.exp() * legal).max(dim=-1)[0].log()
    print(y)
    
