from EscapeEnv_ICLR2024.common.base_agent import BaseAgent
from EscapeEnv_ICLR2024.dqn.estimator import DQNEstimator
from EscapeEnv_ICLR2024.common.torch_layers import QNetwork
from EscapeEnv_ICLR2024.common.buffers import BaseBuffer

class DQN(BaseAgent):
    def __init__(self, 
                 env,
                 estimator_class=DQNEstimator, 
                 buffer_class=BaseBuffer,
                 network_class=QNetwork, 
                 **kwargs):
        super().__init__(env, estimator_class, buffer_class, network_class, **kwargs)
    def _build_network(self):
        self.network = self.network_class(input_size=self.state_dim, output_size=self.num_actions, net_arch=self.net_arch, activation_fn=self.activation_fn)
    
    def _build_estimator(self):
        self.q_estimator = self.estimator_class(self.network, self.batch_size, self.learning_rate, self.loops_per_train, optimizer_kwargs=self.optimizer_kwargs, estimator_kwargs=self.estimator_kwargs, device=self.device)
        
    def _build_buffer(self):
        self.buffer = self.buffer_class(size=self.buffer_size, batch_size=self.batch_size)
        
        
if __name__ == '__main__':
    pass