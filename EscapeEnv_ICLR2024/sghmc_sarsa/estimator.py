from EscapeEnv_ICLR2024.common.base_estimator import BaseEstimator
from EscapeEnv_ICLR2024.common.scheduler import CyclicalParamScheduler, LinearParamScheduler
import torch.nn as nn
from .SGHMC import SGHMC



class SGHMCEstimator(BaseEstimator):
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device)
        

        self.optimizer = SGHMC(self.qnet.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        self.lr_scheduler = CyclicalParamScheduler(self.optimizer, 'lr', self.learning_rate, self.estimator_kwargs['cycle_len'])
        self.sr_scheduler = LinearParamScheduler(self.optimizer, 'sparse_ratio', 1.0, self.optimizer_kwargs['sparse_ratio'], self.estimator_kwargs['sr_decay'])
        self.schedulers = [self.lr_scheduler, self.sr_scheduler]
        self.mse_loss = nn.MSELoss(reduction='sum')
        
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()
        batch = buffer.sample()
        # batch data
        states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal = self.batch_extract(batch)

        r = rewards.clone()
        
        for k in range(self.loops_per_train):

            self.optimizer.zero_grad()
            qa = self.qnet(states).gather(-1, actions).squeeze(dim=-1)
            predict_r = qa
            predict_r[non_final_mask] -= discount_factor *  self.qnet(non_final_next_states).gather(-1, non_final_next_actions).squeeze(dim=-1).detach()
            
            loss = self.mse_loss(r, predict_r)
            loss.backward()

            self.optimizer.step(self.batch_size, k)
                
        self.n_updates += 1
        return loss.item()/self.batch_size
        