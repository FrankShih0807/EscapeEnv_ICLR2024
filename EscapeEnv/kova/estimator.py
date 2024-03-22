from EscapeEnv.common.base_estimator import BaseEstimator
from EscapeEnv.common.scheduler import ConstantParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from my_package.optimizers import ExtendedKalmanFilter


class KOVAEstimator(BaseEstimator):
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device)
        
        self.param_vec = utils.parameters_to_vector(self.qnet.parameters())
        self.use_ddqn = self.estimator_kwargs['use_ddqn']
        self.use_legal = self.estimator_kwargs['use_legal']
        self.optimizer = ExtendedKalmanFilter(net_params=self.qnet.parameters(), params=self.param_vec, batch_size=batch_size, lr=self.learning_rate, **self.optimizer_kwargs)
        self.lr_scheduler = ConstantParamScheduler(self.optimizer, 'lr', self.learning_rate)
        self.schedulers = [self.lr_scheduler]
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        matrix = torch.diag(torch.ones(self.batch_size))
        matrix -= torch.tril(torch.ones(self.batch_size, self.batch_size), diagonal=-1)
        matrix += torch.tril(torch.ones(self.batch_size, self.batch_size), diagonal=-2)    
        self.matrix = matrix
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()

        # batch data
        batch = buffer.sample()
        states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal = self.batch_extract(batch)

        
        for _ in range(self.loops_per_train):
            # self.optimizer.zero_grad()
            # qa = self.qnet(states).gather(-1, actions).squeeze(dim=-1)
            target_qa = rewards.clone()
            if self.use_legal == True:
                if self.use_ddqn == True:
                    policy_next_action = (self.qnet(non_final_next_states).detach().exp() * non_final_next_legal).max(dim=-1, keepdim=True)[1]
                    target_qa[non_final_mask] += discount_factor * self.qnet_target(non_final_next_states).gather(-1, policy_next_action).squeeze(dim=-1).detach()
                else:
                    target_next_action = (self.qnet_target(non_final_next_states).detach().exp() * non_final_next_legal).max(dim=-1, keepdim=True)[1]
                    target_qa[non_final_mask] += discount_factor *  (self.qnet_target(non_final_next_states).gather(-1, target_next_action).squeeze(dim=-1).detach())
            else:
                if self.use_ddqn == True:
                    policy_next_action = self.qnet(non_final_next_states).detach().max(dim=-1, keepdim=True)[1]
                    target_qa[non_final_mask] += discount_factor * self.qnet_target(non_final_next_states).gather(-1, policy_next_action).squeeze(dim=-1).detach()
                else:
                    target_qa[non_final_mask] += discount_factor *  self.qnet_target(non_final_next_states).max(dim=-1, keepdim=True)[0].squeeze(dim=-1).detach()
            
            grad_list = []
            self.optimizer.zero_grad()
            qa = self.qnet(states).gather(-1, actions).squeeze(dim=-1)
            for i in range(self.batch_size):
                qa[i].backward(retain_graph=True)
                grad_list.append(self.qnet.gradient_to_vector())
            grad_mat = self.matrix @ torch.stack(grad_list, 0)
            del grad_list
            qa_diff = (target_qa-qa).detach()
            self.optimizer.step(grad_mat, qa_diff)
            utils.vector_to_parameters(self.param_vec, self.qnet.parameters())
        
        loss = self.mse_loss(qa, target_qa).detach()
        self.n_updates += 1
        return loss.item()

if __name__ == '__main__':
    x = torch.tensor([1,2,3,4,5])
    legal = torch.tensor([0,1,0,1,0])
    
    y = (x.exp() * legal).max(dim=-1)[0].log()
    print(y)
    
    n = 5  # Example size

    # Create a matrix with 1s on the diagonal
    matrix = torch.diag(torch.ones(n))
    matrix -= torch.tril(torch.ones(n, n), diagonal=-1)
    matrix += torch.tril(torch.ones(n, n), diagonal=-2)

    print(matrix)