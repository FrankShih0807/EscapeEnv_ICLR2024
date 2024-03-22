from EscapeEnv.common.base_estimator import ActorCriticEstimator
from EscapeEnv.common.scheduler import ConstantParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class A2CEstimator(ActorCriticEstimator):
    def __init__(self, network, learning_rate, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, learning_rate, optimizer_kwargs, estimator_kwargs, device)
        
        self.vf_coef = self.estimator_kwargs['vf_coef']
        self.ent_coef = self.estimator_kwargs['ent_coef']
        self.max_grad_norm = self.estimator_kwargs['max_grad_norm']
        # self.rms_prop_eps = self.estimator_kwargs['rms_prop_eps']
        self.use_rms_prop = self.estimator_kwargs['use_rms_prop']

        if self.use_rms_prop:
            self.optimizer = optim.RMSprop(self.ac_net.parameters(), **self.optimizer_kwargs)
        else:
            self.optimizer = optim.Adam(self.ac_net.parameters(), lr=self.learning_rate)
                
        self.lr_scheduler = ConstantParamScheduler(self.optimizer, 'lr', self.learning_rate)
        self.schedulers = [self.lr_scheduler]
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()

        # batch data
        rollout_data = buffer.get()
        actions = buffer.actions
        values, log_prob, entropy = self.ac_net.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        
        advantages = rollout_data.advantages
        
        policy_loss = - (advantages * log_prob).mean()
        value_loss = F.mse_loss(rollout_data.returns, values)
        
        
        entropy_loss = - torch.mean(entropy)
        
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.n_updates += 1
        return loss.item()

if __name__ == '__main__':
    x = torch.tensor([1,2,3,4,5])
    legal = torch.tensor([0,1,0,1,0])
    
    y = (x.exp() * legal).max(dim=-1)[0].log()
    print(y)
    
