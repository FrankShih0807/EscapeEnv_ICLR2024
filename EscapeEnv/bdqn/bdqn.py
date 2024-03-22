from EscapeEnv.common.base_agent import BaseAgent
from EscapeEnv.bdqn.estimator import BayesianDQNEstimator
from EscapeEnv.common.torch_layers import BayesianNet
from EscapeEnv.common.buffers import BaseBuffer
import time

class BayesianDQN(BaseAgent):
    def __init__(self, 
                 env,
                 estimator_class=BayesianDQNEstimator, 
                 buffer_class=BaseBuffer,
                 network_class=BayesianNet, 
                 **kwargs):
        super().__init__(env, estimator_class, buffer_class, network_class, **kwargs)
        
    def _build_network(self):
        self.network = self.network_class(input_size=self.state_dim, output_size=self.num_actions, net_arch=self.net_arch, activation_fn=self.activation_fn)
    
    def _build_estimator(self):
        self.q_estimator = self.estimator_class(self.network, self.batch_size, self.learning_rate, self.loops_per_train, optimizer_kwargs=self.optimizer_kwargs, estimator_kwargs=self.estimator_kwargs, device=self.device)
        
    def _build_buffer(self):
        self.buffer = self.buffer_class(size=self.buffer_size, batch_size=self.batch_size)
        
        
    def train(self):
        # batch = self.buffer.sample()
        tic = time.time()
        loss = self.q_estimator.update(self.buffer, self.discount_factor)
        toc = time.time()
        
        for scheduler in self.q_estimator.schedulers:
            self.logger.record("parameters/"+scheduler.param_name, scheduler.param_value)
        self.logger.record("train/loss", loss, exclude='csv')
        self.logger.record_mean("train/computation_time", toc-tic, exclude=['csv', 'tensorboard'])
        self.callback.on_training_end()

        # self.logger.dump(step=self.num_timesteps)
        
        if self.num_trainsteps % self.update_target_every == 0:
            self.q_estimator.update_target()
        
        
if __name__ == '__main__':
    pass