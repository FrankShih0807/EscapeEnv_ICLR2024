import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from copy import deepcopy
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
# from matplotlib import cm
import time
import random

from abc import ABC, abstractmethod
from EscapeEnv.common.logger import Logger, configure

class BaseAgent(ABC):
    def __init__(self,
                 env,
                 estimator_class,
                 buffer_class,
                 network_class,
                 net_arch=[32, 32],
                 activation_fn='nn.ReLU',
                 buffer_size=20000,
                 train_start=1000,
                 update_target_every=1,
                 discount_factor=0.9,
                 learning_rate=1e-4,
                 estimator_kwargs=dict(),
                 optimizer_kwargs=dict(),
                 exploration_fraction=0.1,
                 exploration_final_eps = 0.01,
                 batch_size=20,
                 train_every=5,
                 loops_per_train=5,
                 device=None,
                 dtype=torch.float32,
                 save_path=None,
                 verbose=0
                 ):

        # Environment setting
        self.env = env
        self.is_legal_action = self.env.is_legal_action
        
        self.num_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.action2str = env.action2str
        
        # 
        self.buffer_class = buffer_class
        self.network_class = network_class
        self.net_arch = net_arch
        self.activation_fn = eval(activation_fn)
        
        # Agent Parameters
        self.buffer_size = buffer_size
        self.train_start = train_start
        self.update_target_every = update_target_every
        self.discount_factor = discount_factor
        self.train_every = train_every

        self.estimator_class = estimator_class
        # Estimator Parameters
        self.exploration_final_eps = exploration_final_eps
        self.batch_size = batch_size
        self.loops_per_train = loops_per_train
        self.exploration_fraction = exploration_fraction
        
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.estimator_kwargs = estimator_kwargs

        self.device = device
        self.dtype = dtype
        
        self.save_path = save_path
        
        self.verbose = verbose
        if self.verbose == 1:
            format_strings = ["stdout", "csv", "tensorboard"]
        else:
            format_strings = ["csv", "tensorboard"]
        
        self.logger = configure(self.save_path, format_strings)


        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.num_timesteps = 0
        # Total training step
        self.num_trainsteps = 0
        # Total computation_time
        self.computation_time = 0

        # self.avg_time_spent = []

        
        # build
        self._build_network()
        self._build_estimator()
        self._build_buffer()
        # self._build_callback()


    @abstractmethod
    def _build_network(self):
        '''define self.network'''
        raise NotImplementedError()
    
    @abstractmethod
    def _build_estimator(self):
        '''define self.q_estimator'''
        raise NotImplementedError()
    
    @abstractmethod
    def _build_buffer(self):
        '''define self.buffer'''
        raise NotImplementedError()
    
    # def _build_callback(self):
    #     '''define self.callback'''
    #     self.callback = self.callback_class()
    
    def get_env(self):
        return self.env
        
    def feed(self, transition):        
        self.buffer.add(*transition)
        self.num_timesteps += 1
        
        self.logger.record("rollout/exploration_rate", self.eps)
        self.logger.record("rollout/timesteps", self.num_timesteps)
        self.logger.record("rollout/n_train", self.num_trainsteps)
        self.logger.record("rollout/progress", self.num_timesteps/self.total_timesteps)
        
        self.callback.on_step()
        
        if self.num_timesteps >= self.train_start and self.num_timesteps % self.train_every == 0:
            self.train()
            self.num_trainsteps += 1
            self.logger.record("train/computation_time", self.computation_time/self.num_trainsteps, exclude=['csv', 'tensorboard'])
        
        
    def train(self):
        # batch = self.buffer.sample()
        tic = time.time()
        loss = self.q_estimator.update(self.buffer, self.discount_factor)
        toc = time.time()
        self.computation_time += toc-tic
        
        for scheduler in self.q_estimator.schedulers:
            self.logger.record("parameters/"+scheduler.param_name, scheduler.param_value)
        self.logger.record("train/loss", loss, exclude='csv')
        # self.logger.record_mean("train/computation_time", toc-tic, exclude=['csv', 'tensorboard'])
        self.callback.on_training_end()

        # self.logger.dump(step=self.num_timesteps)
        
        
        if self.num_trainsteps % self.update_target_every == 0:
            self.q_estimator.update_target()
            
            
    
    def epsilon_greedy_action(self, state, legal_action=None):
        ''' Select action with epsilon Q-greedy
        Args:
            state(torch.tensor):
            action_space(list): a list of available actions
        '''
        self.eps = self.exploration_scheduler[min(self.num_timesteps, int(self.exploration_fraction*self.total_timesteps-1))]
        # state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1,-1)
        sample = random.random()
        if legal_action is None:
            legal_action = list(range(self.num_actions))
        else:
            legal_action = legal_action.nonzero()[0]
        
        if sample < self.eps:
            action = random.choice(legal_action)
            return torch.tensor(action, dtype=torch.long).view(1,1)
            # return action
        else:
            q_value = self.q_estimator.predict_nograd(state)
            action_index = q_value[:,legal_action].argmax()
            action = legal_action[action_index]
            return torch.tensor(action, dtype=torch.long).view(1,1)   
            # return action       
        
    

    def learn(self, total_timesteps, callback=None):
        self.total_timesteps = total_timesteps
        self.exploration_scheduler = np.linspace(1, self.exploration_final_eps, int(self.exploration_fraction*self.total_timesteps))
        i_episode = 0
        
        self.callback = callback
        self.callback.init_callback(self)
        # Exploration scheduler

        
        while self.num_timesteps<total_timesteps:
            done = False
            state = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=self.dtype).reshape(1,-1)

            legal_action = self.env.legal_action()
            action = self.epsilon_greedy_action(state, legal_action) 
            next_state, reward, done, info = self.env.step(action.item())
            next_legal = info['next_legal']
            
            reward = torch.tensor(reward, device=self.device, dtype=self.dtype).reshape(1)
            next_state = torch.tensor(next_state, device=self.device, dtype=self.dtype).reshape(1,-1)
            # next_legal = torch.tensor(next_legal, device=self.device, dtype=torch.long).reshape(1,-1)

            # Check for early Finish
            if done:
                next_action = None
                next_legal = None
                transition_tuple = [state, action, reward, next_state, done, next_action, next_legal]
                self.feed(transition_tuple)
            
            else:
                for t in range(200):
                    # action type: tensor
                    legal_action = info['next_legal']
                    next_action = self.epsilon_greedy_action(next_state, legal_action) 
                    state_2, reward_1, done_1, info= self.env.step(next_action.item())
                    next_legal = info['next_legal']

                    reward_1 = torch.tensor(reward_1, device=self.device, dtype=self.dtype).reshape(1)
                    state_2 = torch.tensor(state_2, device=self.device, dtype=self.dtype).reshape(1,-1)
                    # next_legal = torch.tensor(next_legal, device=self.device, dtype=torch.long).reshape(1,-1)

                    transition_tuple = [state, action, reward, next_state, done, next_action, next_legal]              
                    self.feed(transition_tuple)

                    state = next_state.clone()
                    next_state = state_2.clone()
                    action = next_action.clone()
                    reward = reward_1.clone()
                    done = done_1
                    
                    if done:
                        next_action = None
                        next_state = None
                        next_legal = None
                        transition_tuple = [state, action, reward, next_state, done, next_action, next_legal]
                        self.feed(transition_tuple)  
                        break

            # record time spend                    
            self.logger.record_mean("rollout/episode_len", self.env.current_step)
            i_episode += 1
            

class ActorCriticAgent(BaseAgent):
    def __init__(self,
                 env,
                 net_arch=[32, 32],
                 activation_fn='F.relu',
                 share_feature=False,
                 n_steps=-1,
                 gamma=0.99,
                 gae_lambda=1.0,
                 learning_rate=1e-4,
                 estimator_kwargs=dict(),
                 optimizer_kwargs=dict(),
                 device=None,
                 dtype=torch.float32,
                 save_path=None,
                 verbose=0
                 ):

        # Environment setting
        self.env = env
        self.is_legal_action = self.env.is_legal_action
        
        self.num_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.action2str = env.action2str
        
        self.net_arch = net_arch
        self.activation_fn = eval(activation_fn)
        self.share_feature = share_feature
        
        # Agent Parameters
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Estimator Parameters
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.estimator_kwargs = estimator_kwargs

        self.device = device
        self.dtype = dtype
        
        self.save_path = save_path
        
        self.verbose = verbose
        if self.verbose == 1:
            format_strings = ["stdout", "csv", "tensorboard"]
        else:
            format_strings = ["csv", "tensorboard"]
        
        self.logger = configure(self.save_path, format_strings)


        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.num_timesteps = 0
        # Total training step
        self.num_trainsteps = 0
        # Total computation_time
        self.computation_time = 0

        
        # build
        # self._build_network()
        # self._build_estimator()
        # self._build_buffer()

        
    def _build(self):
        self._build_network()
        self._build_estimator()
        self._build_buffer()
        
    def feed(self, transition):        
        self.buffer.add(*transition)
        self.num_timesteps += 1
        
        self.logger.record("rollout/timesteps", self.num_timesteps)
        self.logger.record("rollout/progress", self.num_timesteps/self.total_timesteps)
        self.callback.on_step()
        
        # if self.num_timesteps >= self.train_start and self.num_timesteps % self.train_every == 0:
        #     self.train()
        #     self.logger.record("train/computation_time", self.computation_time/self.num_trainsteps, exclude=['csv', 'tensorboard'])
            
    def train(self):
        tic = time.time()
        loss = self.ac_estimator.update(self.buffer, self.gamma)
        toc = time.time()
        self.computation_time += toc-tic
        
        for scheduler in self.ac_estimator.schedulers:
            self.logger.record("parameters/"+scheduler.param_name, scheduler.param_value)
        self.logger.record("train/loss", loss, exclude='csv')
        # self.logger.record_mean("train/computation_time", toc-tic, exclude=['csv', 'tensorboard'])
        
        self.num_trainsteps += 1
        self.logger.record("train/computation_time", self.computation_time/self.num_trainsteps, exclude=['csv', 'tensorboard'])
        self.logger.record("rollout/n_train", self.num_trainsteps)
        self.callback.on_training_end()
        
            
    def policy_action(self, state, legal_action=None):
        # state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1,-1)

        if legal_action is None:
            legal_action = list(range(self.num_actions))
        else:
            legal_action = legal_action.nonzero()[0]
        
        probs = self.ac_estimator.predict_probs(state).flatten()
        restrict_probs = torch.zeros_like(probs)
        restrict_probs[legal_action] = probs[legal_action]
        actions = torch.multinomial(restrict_probs, num_samples=1, replacement=False)
        return actions

        
    def learn(self, total_timesteps, callback=None, ep_max_len=200):
        self.total_timesteps = total_timesteps
        if self.n_steps == -1:
            self.n_steps = ep_max_len
            
        self._build()    
        self.callback = callback
        self.callback.init_callback(self)
        
        # Exploration scheduler

        obs_tensor = self.env.reset()
        legal_action = self.env.legal_action()
        obs_tensor = torch.tensor(obs_tensor, device=self.device, dtype=self.dtype).reshape(1,-1)
        self._last_episode_starts = True
        
        while self.num_timesteps<total_timesteps:
            # done = False

            n_steps = 0
            self.buffer.reset()
            while n_steps < self.n_steps:
                # print(type(obs_tensor))
                # obs_tensor = torch.from_numpy(obs_tensor, device=self.device, dtype=self.dtype).reshape(1,-1)

                action, value, log_prob = self.ac_estimator.predict_nograd(obs_tensor)
                action = self.policy_action(obs_tensor, legal_action)
                new_obs, reward, done, info = self.env.step(action.item())
                legal_action = info['next_legal']
                new_obs = torch.tensor(new_obs, device=self.device, dtype=self.dtype).reshape(1,-1)
                
                self.num_timesteps += 1
                n_steps += 1

                transitions = [obs_tensor, action, reward, self._last_episode_starts, value, log_prob, new_obs]
                self.feed(transitions)
                # self.buffer.add(obs_tensor, action, reward, self._last_episode_starts, value, log_prob)
                obs_tensor = new_obs 
                self._last_episode_starts = done
                
                if done:
                    self.logger.record_mean("rollout/episode_len", self.env.current_step)
                    obs_tensor = self.env.reset()
                    legal_action = self.env.legal_action()
                    obs_tensor = torch.tensor(obs_tensor, device=self.device, dtype=self.dtype).reshape(1,-1)
                    
                    # print('Hahaha')
                    
                elif self.env.current_step == ep_max_len:
                    self.logger.record_mean("rollout/episode_len", self.env.current_step)
                    obs_tensor = self.env.reset()
                    legal_action = self.env.legal_action()
                    obs_tensor = torch.tensor(obs_tensor, device=self.device, dtype=self.dtype).reshape(1,-1)
                    self._last_episode_starts = True
                    

            # if done:
            #     value = torch.zeros([1,1])
            # else:
            value = self.ac_estimator.ac_net.predict_values(new_obs)

            self.buffer.compute_returns_and_advantages(last_values=value, dones=done)

            self.train()
            # record time spend                    
            
            
if __name__ == '__main__':
    a = [1,2,3,4]
    print(random.choice(a))