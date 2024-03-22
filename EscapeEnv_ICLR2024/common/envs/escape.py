from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union
import os
import gym
import numpy as np
from gym import spaces
from itertools import product
import random
from tqdm import tqdm

class BaseEscapeEnv(gym.Env):
    def __init__(self, dim=2, penalty=-200, obs_fn='id') -> None:
        super().__init__()
        self.dim = dim
        self.n_grid = 10
        self.shape = tuple([self.n_grid+2]*dim)
        self.shape_inner = tuple([self.n_grid]*dim)
        self.goal = tuple([self.n_grid]*dim)
        
        self.action2str = ['dim'+str(s)+'+' for s in range(dim)] + ['dim'+str(s)+'-' for s in range(dim)]
        self.action_map = [np.int_(row) for row in np.eye(dim)] + [np.int_(row) for row in -np.eye(dim)]
        
        if obs_fn == 'id':
            self.ticks = np.arange(self.n_grid+1)
            self.coordinate_map = np.arange(self.n_grid+2)-0.5
            
        elif obs_fn == 'grid':
            self.ticks = np.linspace(0.0, 1.0, self.n_grid+1)
            self.coordinate_map = np.linspace(0.0, 1.0 + 1/self.n_grid, self.n_grid+2) - 1/(2*self.n_grid)
        
        self.reward_mean = -1
        self.reward_sd = 0.1
        self.penalty = penalty 
        
        self.obs_fn = obs_fn
    
        self.action_space = spaces.Discrete(2*dim)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(dim,))


    
    def step(self, a):
        self._take_action(a)
        terminated = self._check_status()
        r = self._get_reward()
        obs = self._get_observation(self.s)
        return obs, r, terminated, {}
    
    
    def _take_action(self, a):
        self.s += self.action_map[a]
        
    def _check_status(self):
        
        for s in self.s:
            if s <= 0 or s > self.n_grid:
                return True
        if (self.s - self.goal).sum() == 0:
            return True
        else: 
            return False
        
    def _get_reward(self):
        for s in self.s:
            if s <= 0 or s > self.n_grid:
                return np.random.normal(loc=self.penalty, scale=self.reward_sd)
        return np.random.normal(loc=self.reward_mean, scale=self.reward_sd)
            
    def _get_observation(self, state):
            coordinate = [self.coordinate_map[s] for s in state]
            return np.array(coordinate)
    
    def reset(self):
        self.s = np.int_([1]*self.dim)
        self.current_step = 0
        
        return self._get_observation(self.s)
    
    def print_attributes(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
            
    def optimal_action(self):
        return self.s.argmin()
    
    def random_action(self):
        return random.randint(0, len(self.action_map)-1)
    
    def geometric_series(self, a, r, n):
        """Generate the first n terms of a geometric series with initial value a and ratio r."""
        return np.array([a * (1 - r**i) / (1 - r) for i in range(n)])
    
    def simulate_state_value(self, gamma):
        reward_table = self.geometric_series(self.reward_mean, gamma, self.dim * self.n_grid)
        state_value = np.zeros(self.shape)
        
        for init_coordinate in product(range(1,self.n_grid+1), repeat=self.dim):
            n_step = self.dim * self.n_grid - sum(init_coordinate)
            state_value[init_coordinate] = reward_table[n_step]
        return state_value[(slice(1, -1),) * self.dim]
    
    def state_space(self):
        obs_list= []
        for state in product(range(1,self.n_grid+1), repeat=self.dim):
            obs = self._get_observation(state)
            obs_list.append(obs)
        return np.stack(obs_list, axis=0)




class EscapeEnv(gym.Env):
    def __init__(self, n_grid=10, reward=-1, goal_reward=0, noise_sd=0.1, is_legal_action=False, random_init=False, boundary_penalty=-100):
        super().__init__()

        self.goal = np.int_([n_grid-1, n_grid-1])
        self.reward = reward
        self.goal_reward = goal_reward
        self.noise_sd = noise_sd
        self.n_grid = n_grid
        self.is_legal_action = is_legal_action
        self.random_init = random_init
        self.boundary_penalty = boundary_penalty
        
        self.action2str = [ "E", "N", "W", "S"]
        self.action_map = [np.int_([1, 0]), np.int_([0, 1]), np.int_([-1,0]), np.int_([0,-1])]
        
        self.coordinate_map = np.linspace(0.05, 0.95, 10)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,))
        
        self.legal_action_map = np.ones([n_grid, n_grid, self.action_space.n])
        self.legal_action_map[-1,-1,:] = 0
        if is_legal_action:
            self.legal_action_map[-1,:,0] = 0
            self.legal_action_map[:,-1,1] = 0
            self.legal_action_map[0,:,2] = 0
            self.legal_action_map[:,0,3] = 0
        

    def step(self, action):
        if self.is_legal_action:
            if action not in self.legal_action().nonzero()[0]:
                print(self.legal_action())
                raise ValueError("Not a legal action at state {} and action {}".format(self.state, action)) 
        self._take_action(action)
        self.current_step += 1
        done = self._is_done()
        out_bound = self._out_bound()
        
        reward = self.reward
        
        if out_bound:
            reward += self.boundary_penalty
        elif done:
            reward += self.goal_reward
        
        if self.noise_sd > 0:
            reward += self.noise_sd * np.random.normal()
        state_coordinate = self._position_map(self.state)
        
        self.log = f"Went {self.action2str[action]} in state {np.round(self.prev_state,2)}, got to state {np.round(self.state,2)}"
        return state_coordinate, reward, done, {'next_legal': self.legal_action()}
    

    def reset(self, init_state=None):
        done = True
        
        if init_state is not None:
            self.state = init_state
            done = self._is_done()
            if done:
                raise ValueError("The initial state is in the goal region") 
        elif self.random_init:
            done = True
            while done:
                self.state = np.random.randint(0, self.n_grid, 2)
                done = self._is_done()
        else:
            self.state = np.int_([0, 0])
            done = False
                
        self.current_step = 0
        return self._position_map(self.state)

    def _position_map(self, state):
        x = 0.05 + 0.1 * state[0]
        y = 0.05 + 0.1 * state[1]
        return np.array([x,y])
    
    def legal_action(self):
        if self.is_legal_action:
            return self.legal_action_map[self.state[0], self.state[1], :]
        else:
            return np.ones(self.action_space.n)
        
    def _take_action(self, action):
        self.prev_state = self.state
        self.state += self.action_map[action]
        self.state =  np.clip(self.state, 0, 9)

    def _is_done(self):
        if self.state[0]==self.goal[0] and self.state[1]==self.goal[1]:
            return True
        elif self.state[0]<0 or self.state[1]<0 or self.state[0]>=self.n_grid or self.state[1]>=self.n_grid :
            return True
        else:
            return False
        
    def _out_bound(self):
        if self.state[0]<0 or self.state[1]<0 or self.state[0]>=self.n_grid or self.state[1]>=self.n_grid :
            return True
        else:
            return False
        
    def optimal_action(self):
        return self.state.argmin()

    
    def render(self, mode='human'):
        
        # print(self.log)
        if self._is_done():
            print('Escape the room in {} steps.'.format(self.current_step))
        elif self.current_step>=200:
            print('Failed to escape the room.')
            
    def geometric_series(self, a, r, n):
        """Generate the first n terms of a geometric series with initial value a and ratio r."""
        return np.array([a * (1 - r**i) / (1 - r) for i in range(n)])

    def load_q_value(self, gamma=0.9, n_trials=1000, eps=0):
        path = os.path.join(os.getcwd(), 'q_values')
        os.makedirs(path, exist_ok=True)
        if self.is_legal_action:
            name = 'escape_legal'
            name += str(eps).replace('.','_')
            name += '.npy'
        else:
            name = 'escape'
            name += str(eps).replace('.','_')     
            name += '.npy'
        file_name = os.path.join(path, name)
        
        if os.path.isfile(file_name):
            # print(os.listdir(path))
            q_value = np.load(file_name)
            print("load q value from {}".format(file_name))

        else:
            print("start simulation")
            q_value = self.simulate_q_value(gamma, n_trials, eps, file_name)
            print("finish simulation")
        # for a in range(4):
        #     print(q_value[:,:,a].round(3))
        # raise
        return q_value
        
    
    def simulate_q_value(self, gamma, n_trials, eps, file_name):
        print(file_name)
        
        reward_table = self.geometric_series(self.reward, gamma, 3 * self.n_grid)
        q_value = np.zeros([self.n_grid, self.n_grid, self.action_space.n])
            
        for init_grid in tqdm(product(range(self.n_grid), repeat=2)):
            for a in range(self.action_space.n):
                rewards = []
                for n in range(n_trials):
                    try:
                        self.reset(init_state=init_grid)
                    except:
                        continue
                    if self.is_legal_action:
                        if self.legal_action()[a] == 0:
                            continue
                    self.step(a)
                    
                    while self._is_done() is False:
                        
                        sample = random.random()
                        legal_action = self.legal_action().nonzero()[0]
                        # print('\n')
                        # print(legal_action)
                        if sample < eps:
                            action = random.choice(legal_action)
                            # print("random")
                        else:
                            action = self.optimal_action()
                            # print("optim")
                        # print(action)
                        self.step(action)
                    
                    time_spent = self.current_step
                    rewards.append(reward_table[time_spent])
                if len(rewards) > 0:
                    q_value[init_grid[0], init_grid[1], a] = np.mean(rewards)

                # print("state {}: {}".format(init_grid, q_value[init_grid[0], init_grid[1], a]))
        np.save(file_name, q_value)
        return q_value
        
        
        
    def simulate_state_value(self, gamma, file_name):

        
        reward_table = self.geometric_series(self.reward, gamma, 3 * self.n_grid)
        values = np.zeros([self.n_grid, self.n_grid])
            
        for init_grid in product(range(self.n_grid), repeat=2):
            
            time_spent = 0
            for x in init_grid:
                time_spent += self.n_grid - x - 1
                
            values[init_grid[0], init_grid[1]] = reward_table[time_spent]

        np.save(file_name, values)
        return values
    
    def load_state_value(self, gamma=0.9):
        path = os.path.join(os.getcwd(), 'state_values')
        os.makedirs(path, exist_ok=True)
        name = 'escape'
        name += str(gamma).replace('.','_')     
        name += '.npy'
        file_name = os.path.join(path, name)
        
        if os.path.isfile(file_name):
            # print(os.listdir(path))
            state_value = np.load(file_name)
            print("load state value from {}".format(file_name))

        else:
            state_value = self.simulate_state_value(gamma, file_name)
        return state_value
    
            
if __name__ == '__main__':
    
    env = EscapeEnv(is_legal_action=False)
    
    env.load_state_value(0.99)
