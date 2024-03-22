import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer
import numpy as np


class ExtendedKalmanFilter(Optimizer):
    def __init__(self, 
                net_params,
                params,
                batch_size,
                lr=1.0, 
                onv_coef=1.0,
                eta=0.01
                ):
        defaults = dict(lr=lr, batch_size=batch_size, onv_coef=onv_coef, eta=eta)
        super().__init__(net_params, defaults)
        self.params = params
        self.param_size = params.numel()
        self.batch_size = batch_size
        self.lr = lr
        self.eta = eta
        self.eps = 1e-5
        
        self.onv_coef = onv_coef
        self.observation_noise_var = self.onv_coef * self.batch_size
        self.p_hat_predicted = None
        self.covariance = torch.eye(self.param_size)
        
        self.P_nt = batch_size * torch.eye(batch_size)
        self.P_tt = torch.eye(self.param_size)
        # print(self.P_nt)
    def step(self, q_function_gradient:Tensor, obs_diff:Tensor):
        ''' One sigle step of LKTD algorithm
            observation (Tensor):  
            measurement (Tensor):
        '''

        self.p_hat_predicted = self.covariance / (1 - self.eta)
        P_theta_r = torch.matmul(self.p_hat_predicted, q_function_gradient.T)

        self.gradq_P_gradq = torch.matmul(q_function_gradient, P_theta_r)
        self.P_r = self.gradq_P_gradq + self.observation_noise_var * torch.eye(self.batch_size)
        try:
            self.K = torch.matmul(P_theta_r, torch.linalg.inv(self.P_r))
        except:
            print(self.P_r)
            pass

        weight_update = P_theta_r @ torch.linalg.solve(self.P_r, obs_diff)
        self.params.add_(self.lr * weight_update)
        # self.obs_diff = obs_diff
        # self.KP = torch.matmul(self.K, self.P_r)
        # self.KPK = self.lr * torch.matmul(self.KP, self.K.T)
        self.KPK = self.lr * self.K @ self.P_r @ self.K.T
        self.covariance = self.p_hat_predicted - self.KPK

        return self.params
        


if __name__ == "__main__":

    pass