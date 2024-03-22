# modified from stable_baselines3

from abc import ABC, abstractmethod
from EscapeEnv.common.logger import Logger
from typing import Any, Callable, Dict, List, Optional, Union
from EscapeEnv.common.buffers import QvalueBuffer, ValueBuffer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # The RL model
    # Type hint as string to avoid circular import

    logger: Logger

    def __init__(self, verbose: int = 0):
        super().__init__()
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None 
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    # Type hint as string to avoid circular import
    def init_callback(self, model) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.get_env()
        self.logger = model.logger
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        self.num_trainsteps = self.model.num_trainsteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass
    
    
class EscapeCallback(BaseCallback):
    def __init__(self, callback_kwargs=dict(), eval_freq=10000, verbose: int = 1):
        super().__init__(verbose)
        
        self.eval_freq = eval_freq
        self.callback_kwargs = callback_kwargs
        # print(callback_kwargs)
        for key, value in callback_kwargs.items():
            setattr(self, key, value)
            print(key, value)
        
    def _init_callback(self) -> None:
        print("initiate callback")
        self.true_q_value = torch.tensor(self.training_env.load_q_value(gamma=self.model.discount_factor, eps=self.model.exploration_final_eps), dtype=torch.float32)
        # self.true_q_value = torch.tensor(self.training_env.load_q_value(gamma=self.model.discount_factor, eps=0.0), dtype=torch.float32)
        self.q_domain = torch.tensor(self.training_env.legal_action_map, dtype=torch.float32)
            
        self.q_value_buffer = QvalueBuffer(size=self.ensemble_size)
        self.total_timesteps = self.model.total_timesteps
        # return super()._init_callback()
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.verbose:
            self.logger.dump(step=self.num_timesteps)
            if self.num_timesteps > self.burn_in * self.total_timesteps:
                self.plot_action_prob()
                self.plot_cr()
            # print(self.q_value_buffer.quantile())


    def _on_training_end(self) -> None:
        if self.num_timesteps < self.burn_in * self.total_timesteps:
            self.logger.record("metric/burn-in", self.burn_in, exclude=['csv', 'tensorboard'])
        else:
            if "sampling_threshold" not in self.callback_kwargs:        
                self.sampling_threshold = 1.0
                
            if self.logger.name_to_value['parameters/lr'] <= self.sampling_threshold * self.model.learning_rate:          
                self.q_value_buffer.add(self.grid_q_value())
                
                metric_dict = self.metric()
                for a in range(self.model.num_actions):
                    action_str = self.model.action2str[a]
                    self.logger.record_mean("metric/mse_"+action_str, metric_dict['mse'][a].item())
                    self.logger.record_mean("metric/start_mse_"+action_str, metric_dict['start_mse'][a].item())
                    self.logger.record_mean("metric/cr_"+action_str, metric_dict['cr'][a].item())
                    self.logger.record_mean("metric/start_cr_"+action_str, metric_dict['start_cr'][a].item())
                    self.logger.record_mean("metric/range_"+action_str, metric_dict['ci_range'][a].item())
                self.logger.record("metric/ens_size", len(self.q_value_buffer), exclude=['csv', 'tensorboard'])

    def grid_q_value(self):
        x = y = self.training_env.coordinate_map
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        states = torch.stack([X,Y], dim=-1)
        q_values= self.model.q_estimator.evaluate_q_value(states)
    
        return q_values
    
    def q_value_to_vote(self, q_value_tensor):
        num_ens = q_value_tensor.shape[0]
        max_index = (q_value_tensor.exp() * self.q_domain.tile(num_ens, 1, 1, 1)).max(dim=-1)[1]
        # max_index = q_value_tensor.exp().max(dim=-1)[1]

        votes = F.one_hot(max_index, num_classes=self.model.num_actions).type(torch.float32).mean(dim=0)
        votes[-1,-1,:] = 0
        return votes
    
    def metric(self):
        ensemble_mean = self.q_value_buffer.mean()
        q_lo, q_hi = self.q_value_buffer.prediction()
        
        q_val_diff = self.q_domain * (ensemble_mean - self.true_q_value)
        
        start_mse = q_val_diff[0,0,:] ** 2
        
        mse =  torch.sum( q_val_diff ** 2, dim=[0,1]) / torch.sum(self.q_domain, dim=[0,1])
        is_cover = (q_hi > self.true_q_value) * (q_lo < self.true_q_value)
        
        start_cr = is_cover[0,0,:]

        cr = torch.sum(is_cover * self.q_domain, dim=[0,1]) / torch.sum(self.q_domain, dim=[0,1])
        ci_range =  torch.sum(q_hi - q_lo, dim=[0,1])/torch.sum(self.q_domain, dim=[0,1])
        metric_dict = dict(mse=mse, cr=cr, ci_range=ci_range, start_mse=start_mse, start_cr=start_cr)
        return metric_dict
    
    def plot_action_prob(self):
        action2str = self.training_env.action2str
        action_map = self.training_env.action_map
        n_grid = 10
        fig = plt.figure(figsize=(6,6))
        plt.subplot(1,1,1, aspect=1)
        plt.title('Action Probability')

        # ax = fig.add_subplot(111)
        x = y = self.training_env.coordinate_map
        X, Y = np.meshgrid(x, y, indexing='ij')
        qa = self.q_value_buffer.ensemble_tensor
        zs = self.q_value_to_vote(qa)
        colors = ["blue", "red","green", "orange"]
        for a in range(self.model.num_actions):
            Z = zs[:,:,a]
            U = action_map[a][0] * Z
            V = action_map[a][1] * Z
            plt.quiver(X,Y,U,V, scale_units="inches", scale=2.5, label='Action {}'.format(action2str[a]), color=colors[a], alpha=0.7, headlength=8)

        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.75))
        plt.grid(color="grey", linestyle="--", linewidth=1)
        ticks = np.arange(0, 1+1/n_grid, 1/n_grid)
        plt.xticks(ticks)
        plt.yticks(ticks)

        plt.xlim(0,1)
        plt.ylim(0,1)

        path_figs = os.path.join(self.model.save_path, "action_prob_plot.png")
        plt.savefig(path_figs)
        plt.close()
        
    def plot_cr(self):
        exp_dir = self.model.save_path

        results_path = os.path.join(exp_dir, 'progress.csv')
        if os.path.exists(results_path):
            df_temp = pd.read_csv(results_path)
            df = df_temp[[col for col in df_temp.columns if col.startswith('metric')]+['rollout/timesteps']]
            df.columns = [col.split('/')[-1] for col in df.columns]
            df = df.dropna()
            df_melted = df.melt(id_vars=['timesteps'], value_vars=['cr_E', 'cr_N'], 
                    var_name='Variable', value_name='Value')
            sns.lineplot(x='timesteps', y='Value', hue='Variable', data=df_melted)
            plt.axhline(y=0.95, color='red', linestyle='--', label='95%')
            plt.ylim(0.0, 1.0)
            
            plt.title('Coverage rate')
            path_figs = os.path.join(self.model.save_path, "cr_plot.png")
            plt.savefig(path_figs)
            plt.close()



class ActorCriticCallback(BaseCallback):
    def __init__(self, callback_kwargs=dict(), eval_freq=10000, verbose: int = 1):
        super().__init__(verbose)
        
        self.eval_freq = eval_freq
        self.callback_kwargs = callback_kwargs
        for key, value in callback_kwargs.items():
            setattr(self, key, value)
            print(key, value)
        
    def _init_callback(self) -> None:
        print("initiate callback")
        self.true_values = torch.tensor(self.training_env.load_state_value(gamma=self.model.gamma), dtype=torch.float32)
        self.value_buffer = ValueBuffer(size=self.ensemble_size)
        
        self.total_timesteps = self.model.total_timesteps
        
    def _on_step(self) -> bool:
        pass

    def _on_training_end(self) -> None:
        if self.num_timesteps < self.burn_in * self.total_timesteps:
            self.logger.record("metric/burn-in", self.burn_in, exclude=['csv', 'tensorboard'])
        else:
            self.value_buffer.add(self.grid_values())
            
            metric_dict = self.metric()
            # for a in range(self.model.num_actions):
                # action_str = self.model.action2str[a]
            self.logger.record_mean("metric/mse", metric_dict['mse'].item())
            self.logger.record_mean("metric/cr", metric_dict['cr'].item())
            self.logger.record_mean("metric/range", metric_dict['ci_range'].item())
            
            self.logger.record("metric/ens_size", len(self.value_buffer), exclude=['csv', 'tensorboard'])
            # self.logger.dump(step=self.num_timesteps)
            
        if self.num_timesteps % self.eval_freq == 0 and self.verbose:
            self.logger.dump(step=self.num_timesteps)
            if self.num_timesteps > self.burn_in * self.total_timesteps:
                self.plot_action_prob()
                self.plot_cr()
                
                
    def grid_values(self):
        x = y = self.training_env.coordinate_map
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        states = torch.stack([X,Y], dim=-1)
        values= self.model.ac_estimator.evaluate_values(states)
        return values
    
    
    def metric(self):
        ensemble_mean = self.value_buffer.mean()
        lo, hi = self.value_buffer.prediction()
        
        val_diff = ensemble_mean - self.true_values
        
        mse =  torch.mean( val_diff ** 2, dim=[0,1]) 
        is_cover = (hi > self.true_values) * (lo < self.true_values)

        cr = torch.mean(is_cover.to(dtype=torch.float32))
        ci_range =  torch.mean(hi - lo)
        metric_dict = dict(mse=mse, cr=cr, ci_range=ci_range)
        return metric_dict
    
    def plot_action_prob(self):
        action2str = self.training_env.action2str
        action_map = self.training_env.action_map
        n_grid = 10
        fig = plt.figure(figsize=(6,6))
        plt.subplot(1,1,1, aspect=1)
        plt.title('Action Probability')

        x = y = self.training_env.coordinate_map
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        # qa = self.value_buffer.ensemble_tensor
        states = torch.stack([X,Y], dim=-1)
        zs = self.model.ac_estimator.predict_probs(states)
        colors = ["blue", "red","green", "orange"]
        for a in range(self.model.num_actions):
            Z = zs[:,:,a]
            U = action_map[a][0] * Z
            V = action_map[a][1] * Z
            plt.quiver(X,Y,U,V, scale_units="inches", scale=2.5, label='Action {}'.format(action2str[a]), color=colors[a], alpha=0.7, headlength=8)

        plt.grid(color="grey", linestyle="--", linewidth=1)
        ticks = np.arange(0, 1+1/n_grid, 1/n_grid)
        plt.xticks(ticks)
        plt.yticks(ticks)

        plt.xlim(0,1)
        plt.ylim(0,1)

        path_figs = os.path.join(self.model.save_path, "action_prob_plot.png")
        plt.savefig(path_figs)
        plt.close()
        
    def plot_cr(self):
        exp_dir = self.model.save_path

        results_path = os.path.join(exp_dir, 'progress.csv')
        if os.path.exists(results_path):
            df_temp = pd.read_csv(results_path)
            df = df_temp[[col for col in df_temp.columns if col.startswith('metric')]+['rollout/timesteps']]
            df.columns = [col.split('/')[-1] for col in df.columns]
            df = df.dropna()
            # df_melted = df.melt(id_vars=['timesteps'], value_vars=['cr_E', 'cr_N'], 
            #         var_name='Variable', value_name='Value')
            sns.lineplot(x='timesteps', y='cr', data=df)
            plt.axhline(y=0.95, color='red', linestyle='--', label='95%')
            plt.ylim(0.0, 1.0)
            
            plt.title('Coverage rate')
            path_figs = os.path.join(self.model.save_path, "cr_plot.png")
            plt.savefig(path_figs)
            plt.close()
        
if __name__ == '__main__':
    x = y = np.arange(0.05, 0.85, 0.1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x_length = X.shape[0]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    Z = torch.stack([X,Y], dim=-1)
    print(X)
    print(Y)
    
