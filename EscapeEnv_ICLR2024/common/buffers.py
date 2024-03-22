
from abc import ABC, abstractmethod
import collections
import random
import torch
from collections import namedtuple
from copy import deepcopy
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'next_action', 'next_legal'))

class BaseBuffer(object):
    def __init__(self, size, batch_size):
        self.size = size
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=size)

    def add(self, state, action, reward, next_state, done, next_action=None, next_legal=None):
        '''store transition tuple to buffer'''
        self.buffer.append(Transition(state, action, reward, next_state, done, next_action, next_legal))

    def sample(self, batch_size=None): 
        if batch_size is None:
            samples = random.sample(self.buffer, self.batch_size)
        else:
            assert batch_size > 0
            samples = random.sample(self.buffer, batch_size)
        return Transition(*zip(*samples))

    def __len__(self):
        return len(self.buffer)

       
    
class QvalueBuffer(object):
    def __init__(self, size) -> None:
        self.size = size
        self.ensemble_tensor = None
        
    def add(self, q_value_tensor):
        while q_value_tensor.dim() < 4:
            q_value_tensor = q_value_tensor.unsqueeze(dim = 0)
            
            
        if self.ensemble_tensor is None:
            self.ensemble_tensor = q_value_tensor.clone()
        else:
            self.ensemble_tensor = torch.cat([self.ensemble_tensor, q_value_tensor])
        if self.ensemble_tensor.shape[0] > self.size:
            self.ensemble_tensor = self.ensemble_tensor[-self.size::]
        
    def mean(self):
        return torch.mean(self.ensemble_tensor, dim=0)
    
    def last(self):
        return self.ensemble_tensor[-1,:,:,:]
    
    def quantile(self, p=0.05):
        if self.ensemble_tensor is not None:
            q = torch.tensor([p/2, 0.5, 1-p/2])
            quantiles = self.ensemble_tensor.quantile(q=q, dim=0)
            # hi = self.ensemble_tensor.quantile(q=1-p/2, dim=0)
            return quantiles[0], quantiles[1], quantiles[2]
    
    def prediction(self):
        lo, med, hi = self.quantile()
        center = self.last()
        upper = center + (hi-lo)/2
        lower = center - (hi-lo)/2
        return lower, upper
        
    def __len__(self):
        return self.ensemble_tensor.shape[0]
    
class ValueBuffer(object):
    def __init__(self, size) -> None:
        self.size = size
        self.ensemble_tensor = None
        
    def add(self, value_tensor):
        # while value_tensor.dim() < 3:
        value_tensor = value_tensor.unsqueeze(dim = 0).squeeze(dim=-1)
            
            
        if self.ensemble_tensor is None:
            self.ensemble_tensor = value_tensor.clone()
        else:
            self.ensemble_tensor = torch.cat([self.ensemble_tensor, value_tensor])
        if self.ensemble_tensor.shape[0] > self.size:
            self.ensemble_tensor = self.ensemble_tensor[-self.size::]

    def mean(self):
        return torch.mean(self.ensemble_tensor, dim=0)
    
    def last(self):
        return self.ensemble_tensor[-1,:,:]
    
    def quantile(self, p=0.05):
        if self.ensemble_tensor is not None:
            q = torch.tensor([p/2, 0.5, 1-p/2])
            quantiles = self.ensemble_tensor.quantile(q=q, dim=0)
            return quantiles[0], quantiles[1], quantiles[2]
    
    def prediction(self):
        lo, med, hi = self.quantile()
        center = self.last()
        upper = center + (hi-lo)/2
        lower = center - (hi-lo)/2
        return lower, upper
        
    def __len__(self):
        return self.ensemble_tensor.shape[0]
        
        
RolloutData = namedtuple('RolloutData', ('observations', 'actions', 'values', 'log_probs', 'advantages', 'returns'))

class RolloutBuffer(object):
    def __init__(self, 
                 buffer_size, 
                 state_dim,
                 num_actions,
                 gae_lambda = 1,
                 gamma = 0.99,
                 ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()
        
    def reset(self):
        self.observations = torch.zeros((self.buffer_size, self.state_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.buffer_size, 1), dtype=torch.long)
        self.rewards = torch.zeros((self.buffer_size), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size), dtype=torch.float32)
        self.episode_starts = torch.zeros((self.buffer_size), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size), dtype=torch.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, episode_start, value, log_prob, new_obs):
        '''store transition tuple to buffer'''
        # action = action.reshape((1, self.action_dim))

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.clone().cpu().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    
    def compute_returns_and_advantages(self, last_values, dones):
        last_values = last_values.clone().cpu().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values
        
    def get(self):
        data = RolloutData(
            self.observations,
            self.actions,
            self.values.flatten(),
            self.log_probs.flatten(),
            self.advantages.flatten(),
            self.returns.flatten(),
        )
        return data


    def __len__(self):
        return len(self.buffer)

if __name__ == '__main__':
    

    # buffer = QvalueBuffer(20)
    
    # for i in range(20):
    #     buffer.add(torch.randn([2,5,5,4]))
    
    # print(buffer.ensemble_tensor.shape)
    
    # a = torch.randn([2,3,4])
    # print(a[0])
    
    for i in reversed(range(20)):
        print(i)