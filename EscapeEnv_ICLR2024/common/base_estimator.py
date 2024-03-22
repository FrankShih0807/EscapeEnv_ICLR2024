from EscapeEnv_ICLR2024.common.torch_layers import QNetwork, EnsembleNet, BayesianNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import random
from my_package.utils.scheduler import LinearParamScheduler, PolynomialParamScheduler
from copy import deepcopy

from abc import ABC, abstractmethod



class BaseEstimator(object):
    
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        
        self.qnet = network
        # initialize the weights using Xavier init
        # for p in self.qnet.parameters():
        #     if len(p.data.shape) > 1:
        #         nn.init.normal_(p.data, std=0.01)
        self.qnet_target = deepcopy(self.qnet)
        self.qnet_target.eval()
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loops_per_train = loops_per_train
        self.optimizer_kwargs = optimizer_kwargs
        self.estimator_kwargs = estimator_kwargs
        
        self.parameter_size = sum(p.numel() for p in self.qnet.parameters())
        self.device = device

        self.n_updates = 0
    
        
    def update_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        
    def predict_nograd(self, s):
        with torch.no_grad():
            q_as = self.qnet(s)
        return q_as
    
    def evaluate_q_value(self, s):
        return self.predict_nograd(s)
    
    @abstractmethod
    def update(self, batch, discount_factor):
        '''_summary_
        Args:
            batch (_type_): _description_
            discount_factor (_type_): _description_
        Raises:
            NotImplementedError: _description_
        '''
        raise NotImplementedError()
        
    
    def batch_extract(self, batch):
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward) 
        non_final_mask = torch.tensor(tuple(map(lambda s: s is False, batch.done)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_next_actions = torch.cat([s for s in batch.next_action if s is not None])
        if len([s for s in batch.next_legal if s is not None]) > 0:
            non_final_next_legal = torch.tensor(np.stack([s for s in batch.next_legal if s is not None]), device=self.device)
        else:
            non_final_next_legal = 0

        return states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal

    
        
    
    
if __name__ == '__main__':
    a = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    b = (1-a).clone().type(torch.bool)
    print(b)
    a[0] = 1
    print(b)
    print(a)
    
    a = [True, False, True, True]
    print(tuple(map(lambda s: s is False, a)))