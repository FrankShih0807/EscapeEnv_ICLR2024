from EscapeEnv.common.base_agent import ActorCriticAgent
from EscapeEnv.a2c.estimator import A2CEstimator
from EscapeEnv.common.torch_layers import ActorCriticNetwork
from EscapeEnv.common.buffers import RolloutBuffer

class A2C(ActorCriticAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _build_network(self):
        self.network = ActorCriticNetwork(input_size=self.state_dim, actor_output_size=self.num_actions, net_arch=self.net_arch, activation_fn=self.activation_fn, share_features=self.share_feature)
    
    def _build_estimator(self):
        self.ac_estimator = A2CEstimator(network=self.network, learning_rate=self.learning_rate, optimizer_kwargs=self.optimizer_kwargs, estimator_kwargs=self.estimator_kwargs, device=self.device)
        
    def _build_buffer(self):
        self.buffer = RolloutBuffer(buffer_size=self.n_steps, state_dim=self.state_dim, num_actions=self.num_actions, gae_lambda=self.gae_lambda, gamma=self.gamma)
        
        
if __name__ == '__main__':
    pass