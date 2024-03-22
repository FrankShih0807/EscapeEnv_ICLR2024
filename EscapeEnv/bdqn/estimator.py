from EscapeEnv.common.base_estimator import BaseEstimator
from EscapeEnv.common.scheduler import ConstantParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy



class BayesianDQNEstimator(BaseEstimator):
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device)
        

        self.use_ddqn = self.estimator_kwargs['use_ddqn']
        self.use_legal = self.estimator_kwargs['use_legal']
        # Bayesian DQN 
        self.bdqn_learn_frequency = self.estimator_kwargs['bdqn_learn_frequency']
        self.thompson_sampling_frequency = self.estimator_kwargs['thompson_sampling_frequency']
        
        
        # self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        self.optimizer = optim.RMSprop(self.qnet.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01, centered=True)
        # self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.learning_rate)
        
        self.lr_scheduler = ConstantParamScheduler(self.optimizer, 'lr', self.learning_rate)
        self.schedulers = [self.lr_scheduler]
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.eps = 1e-5
        
        self.target_batch_size = self.estimator_kwargs['target_batch_size']
        self.prior_var = self.estimator_kwargs['prior_var']
        self.noise_var = self.estimator_kwargs['noise_var']
        self.var_k = self.estimator_kwargs['var_k']

        # self.num_actions = self.estimator_kwargs.action_dim
        self.feature_size = self.qnet.net_arch[-1]
        self.num_actions = self.qnet.output_size
        
        
        self.sampled_mean = self.qnet.sampled_mean
        self.policy_mean = self.qnet.policy_mean

        self.policy_cov = torch.normal(0, 1, size=(self.num_actions, self.feature_size, self.feature_size), device=self.device)  # size = [4, 32, 32]
        self.cov_decom = self.policy_cov
        for idx in range(self.num_actions):
            # self.policy_cov[idx] = self.var_k * torch.eye(self.feature_size)
            self.policy_cov[idx] = torch.eye(self.feature_size)
            self.cov_decom[idx] = torch.linalg.cholesky((self.policy_cov[idx] + self.policy_cov[idx].T)/2.0)
        self.ppt = torch.zeros(self.num_actions, self.feature_size, self.feature_size, device=self.device)
        self.py = torch.zeros(self.num_actions, self.feature_size, device=self.device)
        
        self.update_target()
    
    def update_target(self):
        self.qnet_target = deepcopy(self.qnet)
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()
            
        if self.n_updates % self.bdqn_learn_frequency == 0 and self.n_updates > 0:
            self.update_posterior(buffer, discount_factor)
        if self.n_updates % self.thompson_sampling_frequency == 0 and self.n_updates > 0:
            self.thompson_sample()
        
        for _ in range(self.loops_per_train):
            loss = self.update_weight(buffer, discount_factor)
        
        self.n_updates += 1
        return loss

        
    def update_weight(self, buffer, discount_factor):
        batch = buffer.sample()
        states, actions, rewards, non_final_mask, non_final_next_states, _, non_final_next_legal = self.batch_extract(batch)
        
        self.optimizer.zero_grad()
        qa, target_qa  = self.find_qvals(states, non_final_next_states, non_final_mask, rewards, actions, non_final_next_legal, discount_factor)
        
        loss = self.mse_loss(qa, target_qa)
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
            
    def update_posterior(self, buffer, discount_factor):
        self.ppt *= 0
        self.py *= 0
        num_iters = int(self.target_batch_size/self.batch_size)
        for _ in range(num_iters):
            
            # batch data
            batch = buffer.sample()
            states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal = self.batch_extract(batch)
            # self.optimizer.zero_grad()

            q_feature, target_qa  = self.find_state_rep(states, non_final_next_states, non_final_mask, rewards, actions, non_final_next_legal, discount_factor)
            q_feature = q_feature.detach()

            for idx in range(self.batch_size):
                self.ppt[actions[idx].item(), :, :] += torch.matmul(q_feature[idx].unsqueeze(0).mT, q_feature[idx].unsqueeze(0))
                self.py[actions[idx].item(), :] += q_feature[idx] * target_qa[idx].item()


        for idx in range(self.num_actions):
            inv = torch.inverse(self.ppt[idx]/self.noise_var + 1/self.prior_var*torch.eye(self.feature_size))
            self.policy_mean[idx] = torch.matmul(inv, self.py[idx])/self.noise_var
            self.policy_cov[idx] = self.var_k * inv
            try:
                # self.cov_decom[idx] = torch.linalg.cholesky((self.policy_cov[idx]+self.policy_cov[idx].T)/2 + self.eps * torch.eye(self.feature_size))
                self.cov_decom[idx] = torch.linalg.cholesky((self.policy_cov[idx]+self.policy_cov[idx].T)/2)
            except RuntimeError:
                pass
            # print(self.cov_decom)
        self.qnet.update_policy_mean(self.policy_mean)


    def thompson_sample(self):
        for idx in range(self.num_actions):
            sample = torch.normal(0, 1, size=(self.feature_size, 1), device=self.device)
            self.sampled_mean[idx] = self.policy_mean[idx] + torch.matmul(self.cov_decom[idx], sample)[:,0] 
        self.qnet.update_sampled_mean(self.sampled_mean)

    def find_qvals(self, states, non_final_next_states, non_final_mask, rewards, actions, non_final_next_legal, discount_factor):
        qa = self.qnet(states).gather(-1, actions).squeeze(dim=-1)
        target_qa = rewards.clone()
        if self.use_legal == True:
            if self.use_ddqn == True:
                policy_next_action = (self.qnet.sampled_forward(non_final_next_states).detach().exp() * non_final_next_legal).max(dim=-1, keepdim=True)[1]
                target_qa[non_final_mask] += discount_factor * self.qnet_target(non_final_next_states).gather(-1, policy_next_action).squeeze(dim=-1).detach()
            else:
                target_next_action = (self.qnet_target(non_final_next_states).detach().exp() * non_final_next_legal).max(dim=-1, keepdim=True)[1]
                target_qa[non_final_mask] += discount_factor *  self.qnet_target(non_final_next_states).gather(-1, target_next_action).squeeze(dim=-1).detach()
        else:
            if self.use_ddqn == True:
                policy_next_action = self.qnet.sampled_forward(non_final_next_states).detach().max(dim=-1, keepdim=True)[1]
                target_qa[non_final_mask] += discount_factor * self.qnet_target(non_final_next_states).gather(-1, policy_next_action).squeeze(dim=-1).detach()
            else:
                target_qa[non_final_mask] += discount_factor *  self.qnet_target(non_final_next_states).max(dim=-1, keepdim=True)[0].squeeze(dim=-1).detach()
        return qa, target_qa
    
    def find_state_rep(self, states, non_final_next_states, non_final_mask, rewards, actions, non_final_next_legal, discount_factor):
        policy_state_rep = self.qnet.state_feature(states).detach()
        target_qa = rewards.clone()
        target_qa[non_final_mask] += discount_factor *  self.qnet_target(non_final_next_states).max(dim=-1, keepdim=True)[0].squeeze(dim=-1).detach()
                
        return policy_state_rep, target_qa
    
if __name__ == '__main__':
    x = torch.tensor([1,2,3,4,5])
    legal = torch.tensor([0,1,0,1,0])
    
    y = (x.exp() * legal).max(dim=-1)[0].log()
    print(y)
    
