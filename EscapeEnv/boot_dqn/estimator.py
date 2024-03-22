from EscapeEnv.common.base_estimator import BaseEstimator
from EscapeEnv.common.scheduler import ConstantParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random



class BootDQNEstimator(BaseEstimator):
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device)
        

        self.use_ddqn = self.estimator_kwargs['use_ddqn']
        self.use_legal = self.estimator_kwargs['use_legal']
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        self.lr_scheduler = ConstantParamScheduler(self.optimizer, 'lr', self.learning_rate)
        self.schedulers = [self.lr_scheduler]
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.random_state = np.random.RandomState()
        self.n_heads = self.estimator_kwargs['n_heads']
        self.bernoulli_p = self.estimator_kwargs['bernoulli_p']
        
        self.heads = list(range(self.n_heads))
        self.active_head = random.choice(self.heads)
        
    
    def predict_nograd(self, s):
        with torch.no_grad():
            q_as = self.qnet(s, self.active_head)
        return q_as
    
    def evaluate_q_value(self, s):
        with torch.no_grad():
            q_as = self.qnet(s)
        return q_as
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()

        # batch data
        batch = buffer.sample()
        states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal = self.batch_extract(batch)

        
        for _ in range(self.loops_per_train):
            all_target_next_Qs = self.qnet_target(non_final_next_states).detach()  # n_heads * batch_size (non final) * num_actions
            all_Qs = self.qnet(states)  # n_heads * batch_size * num_actions
            if self.use_ddqn:
                all_policy_next_Qs = self.qnet(non_final_next_states).detach()  # n_heads * batch_size (non final) * num_actions
            
            self.optimizer.zero_grad()
            exp_mask = np.random.binomial(1, self.bernoulli_p, [self.n_heads, self.batch_size])   # n_heads * batch_size
            exp_mask = torch.tensor(exp_mask)
            
            total_used = exp_mask.sum(dim=0)
            if self.use_legal == True:
                if self.use_ddqn:
                    policy_next_actions = (all_policy_next_Qs.exp() * non_final_next_legal.tile(self.n_heads, 1, 1)).max(-1, keepdim=True)[1]  # n_heads * batch_size (non final) * 1
                else:
                    policy_next_actions = (all_target_next_Qs.exp() * non_final_next_legal.tile(self.n_heads, 1, 1)).max(-1, keepdim=True)[1]
            else:
                if self.use_ddqn:
                    policy_next_actions = all_policy_next_Qs.max(-1, keepdim=True)[1]  # n_heads * batch_size (non final) * 1
                else:
                    policy_next_actions = all_target_next_Qs.max(-1, keepdim=True)[1]
                    
            
            target_qa = rewards.clone().unsqueeze(0).repeat(self.n_heads, 1)
            target_qa[:, non_final_mask] += discount_factor * all_target_next_Qs.gather(-1, policy_next_actions).squeeze(dim=-1)
            qa = all_Qs.gather(-1, actions.unsqueeze(0).repeat(self.n_heads, 1, 1)).squeeze(dim=-1)
            
            full_loss = (qa - target_qa) ** 2
            mean_loss = torch.sum(full_loss * exp_mask, dim=-1) / exp_mask.sum(dim=-1)
            loss = mean_loss.mean()
            loss.backward()
            
            for param in self.qnet.core_net.parameters():
                if param.grad is not None:
                    param.grad.data *= 1. / self.n_heads
                    
            nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
            self.optimizer.step()
            
        self.n_updates += 1
        if self.n_updates % 10 == 0:
            self.active_head = random.choice(self.heads)
        return loss.item()
        

if __name__ == '__main__':
    x = torch.tensor([1,2,3,4,5])
    legal = torch.tensor([0,1,0,1,0])
    
    y = (x.exp() * legal).max(dim=-1)[0].log()
    print(y)