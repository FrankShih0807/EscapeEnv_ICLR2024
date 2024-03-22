import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ReLU
import torch.nn.utils as utils
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


class BaseNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[32, 32], activation_fn=nn.ReLU):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        self.activation_fn= activation_fn
        
        self.layers = nn.ModuleList()

        # Add the first layer (input layer)
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(self.activation_fn())

        # Add hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(self.activation_fn())

        # Add the output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

        self._weight_init()

    def _weight_init(self):
        # print('weight initialize')
        with torch.no_grad():
            for p in self.parameters():
                p.data = 0.1 * torch.randn_like(p)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
class QNetwork(BaseNetwork):
    def __init__(self, input_size, output_size, net_arch=[32, 32], activation_fn=nn.ReLU):
        super().__init__(input_size, output_size, net_arch, activation_fn)
    
    def gradient_to_vector(self):
        grad_list = []
        for p in self.parameters():
            grad_list.append(p.grad.flatten())
        grad_vec = torch.cat(grad_list)
        del grad_list
        return grad_vec
    



''' Bootstrap DQN '''

class CoreNet(nn.Module):
    def __init__(self, input_size, net_arch, activation_fn):
        super(CoreNet, self).__init__()
        
        self.net_arch = net_arch
        self.activation_fn= activation_fn
        
        self.layers = nn.ModuleList()

        # Add the first layer (input layer)
        self.layers.append(nn.Linear(input_size, net_arch[0]))
        self.layers.append(self.activation_fn())

        # Add hidden layers
        for i in range(1, len(net_arch)):
            self.layers.append(nn.Linear(net_arch[i-1], net_arch[i]))
            self.layers.append(self.activation_fn())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HeadNet(nn.Module):
    def __init__(self, last_layer, output_size, n_heads):
        super(HeadNet, self).__init__()
        self.output_size = output_size
        self.n_heads = n_heads
        self.fc1 = nn.Linear(last_layer, output_size * n_heads)

    def forward(self, x):
        x = self.fc1(x)
        new_shape = x.shape[:-1] + (self.output_size, self.n_heads) 
        return x.view(new_shape)


class EnsembleNet(nn.Module):
    def __init__(self, n_heads, input_size, output_size, net_arch=[32, 32], activation_fn=nn.ReLU):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(input_size, net_arch, activation_fn)
        self.head_net = HeadNet(net_arch[-1], output_size, n_heads)
        
    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return self.head_net(x)

    def forward(self, x, k=None):
        core_cache = self._core(x)
        net_heads = self._heads(core_cache)
        
        num_dims = len(net_heads.shape)
        permute_order = (num_dims - 1,) + tuple(range(num_dims - 1))
        net_heads = net_heads.permute(permute_order)
        # net_heads = torch.transpose(net_heads, dim0=0, dim1=-1)
        if k is None:
            return net_heads
        else:
            return net_heads[k]
    
''' Bayesian DQN '''
def layer_init(layer, w_scale=1):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class BayesianNet(nn.Module):
    def __init__(self, input_size, output_size ,net_arch=[32,32], activation_fn=nn.ReLU, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net_arch = net_arch
        self.device = device
        
        
        q_net = []
        last_layer_dim = input_size
        
        for curr_layer_dim in net_arch:
            q_net.append(layer_init(nn.Linear(last_layer_dim, curr_layer_dim)))
            q_net.append(activation_fn())
            last_layer_dim = curr_layer_dim
         
        self.q_net = nn.Sequential(*q_net)
        self.sampled_mean = torch.normal(0, 0.01, size=(self.output_size, last_layer_dim), device=self.device)
        self.policy_mean = torch.normal(0, 0.01, size=(self.output_size, last_layer_dim), device=self.device)
        # self.policy_mean = self.sampled_mean.clone()
        
        
    def sampled_forward(self, x):
        feature_predict = self.q_net(x)
        return torch.matmul(feature_predict, self.sampled_mean.T)
    
    def forward(self, x):
        feature_predict = self.q_net(x)
        return torch.matmul(feature_predict, self.policy_mean.T)
    
    def state_feature(self, x):
        feature_predict = self.q_net(x)
        return feature_predict
    
    def update_sampled_mean(self, sampled_mean):
        self.sampled_mean = sampled_mean.clone()
    
    def update_policy_mean(self, policy_mean):
        self.policy_mean = policy_mean.clone()
        
        
''' Distributional DQN '''
class QuantileNetwork(BaseNetwork):
    def __init__(self, input_size, num_actions, n_quantiles, net_arch=[32, 32], activation_fn=nn.ReLU):
        super().__init__(input_size, num_actions * n_quantiles, net_arch, activation_fn)
        self.n_quantiles = n_quantiles
        self.num_actions = num_actions
    
    def forward(self, x):
        x = super().forward(x)
        return x.reshape(*x.shape[:-1], self.n_quantiles, self.num_actions)

    def _predict(self, x):
        q_values = self(x).mean(dim=-2)
        return q_values
    
    def eval_q_values(self, x):
        q_values = self._predict(x)
        num_dims = len(q_values.shape)
        permute_order = (num_dims - 2,) + tuple(range(num_dims - 2)) + (num_dims - 1,)
        return q_values.permute(permute_order)
        

class ActorNetwork(BaseNetwork):
    def __init__(self, state_size, num_actions, net_arch=[32, 32], activation_fn=nn.ReLU):
        super().__init__(input_size=state_size, output_size=num_actions, net_arch=net_arch, activation_fn=activation_fn)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.q_net(x) 
        y = self.softmax(x)
        return y

class CriticNetwork(BaseNetwork):
    def __init__(self, state_dim, net_arch=[32, 32], activation_fn=nn.ReLU):
        super().__init__(input_size=state_dim, output_size=1, net_arch=net_arch, activation_fn=activation_fn)
        
        
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, actor_output_size, net_arch=[32, 32], critic_output_size=1, share_features=False, activation_fn=nn.ReLU):
        super(ActorCriticNetwork, self).__init__()
        self.activation_fn = activation_fn()
        self.share_features = share_features
        self.shared_layers = nn.ModuleList()
        self.actor_output_size = actor_output_size
        
        if share_features:
            # Define shared hidden layers
            self.shared_layers = nn.ModuleList([nn.Linear(input_size, net_arch[0])])
            self.shared_layers.extend([
                nn.Linear(net_arch[i], net_arch[i+1])
                for i in range(len(net_arch) - 1)
            ])
            final_dim = net_arch[-1]
        else:
            # Define separate hidden layers for actor and critic
            self.actor_layers = nn.ModuleList([nn.Linear(input_size, net_arch[0])])
            self.actor_layers.extend([
                nn.Linear(net_arch[i], net_arch[i+1])
                for i in range(len(net_arch) - 1)
            ])
            self.critic_layers = nn.ModuleList([nn.Linear(input_size, net_arch[0])])
            self.critic_layers.extend([
                nn.Linear(net_arch[i], net_arch[i+1])
                for i in range(len(net_arch) - 1)
            ])
            final_dim = net_arch[-1]

        # Define actor and critic heads
        self.actor_head = nn.Linear(final_dim, actor_output_size)
        self.critic_head = nn.Linear(final_dim, critic_output_size)
        

        self._weight_init()

    def _weight_init(self):
        # print('weight initialize')
        with torch.no_grad():
            for p in self.parameters():
                p.data = 0.01 * torch.randn_like(p)
    


    def forward(self, x):
        if self.share_features:
            # Pass input through shared layers
            for layer in self.shared_layers:
                x = self.activation_fn(layer(x))
        else:
            # Pass input through separate actor layers
            actor_x = x
            for layer in self.actor_layers:
                actor_x = self.activation_fn(layer(actor_x))
            
            # Pass input through separate critic layers
            critic_x = x
            for layer in self.critic_layers:
                critic_x = self.activation_fn(layer(critic_x))
            x = actor_x  # For actor output
            critic_x = critic_x  # For critic output, reassigning for clarity

        # Pass through actor and critic heads
        policy_dist = F.softmax(self.actor_head(x), dim=-1)
        values = self.critic_head(critic_x if not self.share_features else x)
        
        n = len(policy_dist.shape)
        actions = torch.multinomial(policy_dist.view(-1, self.actor_output_size), num_samples=1, replacement=False).view(policy_dist.shape[0:n-1]).unsqueeze(dim=-1)
        log_prob = policy_dist.gather(-1, actions).log()
        
        return actions, values, log_prob
    


    def action_probs(self, x):
        if self.share_features:
            # Pass input through shared layers
            for layer in self.shared_layers:
                x = self.activation_fn(layer(x))
        else:
            # Pass input through separate actor layers
            actor_x = x
            for layer in self.actor_layers:
                actor_x = self.activation_fn(layer(actor_x))
            
            # Pass input through separate critic layers
            critic_x = x
            for layer in self.critic_layers:
                critic_x = self.activation_fn(layer(critic_x))
            x = actor_x  # For actor output
            critic_x = critic_x  # For critic output, reassigning for clarity

        # Pass through actor and critic heads
        policy_dist = F.softmax(self.actor_head(x), dim=-1)
        # values = self.critic_head(critic_x if not self.share_features else x)
        
        return policy_dist
    
    def evaluate_actions(self, x, actions):
        
        if self.share_features:
            # Pass input through shared layers
            for layer in self.shared_layers:
                x = self.activation_fn(layer(x))
        else:
            # Pass input through separate actor layers
            actor_x = x
            for layer in self.actor_layers:
                actor_x = self.activation_fn(layer(actor_x))
            
            # Pass input through separate critic layers
            critic_x = x
            for layer in self.critic_layers:
                critic_x = self.activation_fn(layer(critic_x))
            x = actor_x  # For actor output
            critic_x = critic_x  # For critic output, reassigning for clarity
            
        policy_dist = F.softmax(self.actor_head(x), dim=-1)
        values = self.critic_head(critic_x if not self.share_features else x)
        # actions = torch.multinomial(policy_dist, num_samples=1, replacement=False)
        log_prob = policy_dist.gather(-1, actions).log()
        
        # Ensure probabilities are greater than 0 to avoid log(0)
        prob_vector = torch.clamp(policy_dist, min=1e-9)
        entropy = -(prob_vector * torch.log(prob_vector)).sum(dim=-1, keepdim=True)
        
        return values, log_prob, entropy
    
    def predict_values(self, x):
        _, values, _ = self(x)
        return values
    
    def actor_parameters(self):
        if self.share_features:
            return list(self.shared_layers.parameters()) + list(self.actor_head.parameters())
        else:
            return list(self.actor_layers.parameters()) + list(self.actor_head.parameters())

    def critic_parameters(self):
        if self.share_features:
            return list(self.critic_head.parameters())
        else:
            return list(self.critic_layers.parameters()) + list(self.critic_head.parameters())

    # def group_parameters(self):
    #     if self.share_features:
    #         # All parameters in shared layers are considered shared
    #         shared_params = list(self.shared_layers.parameters())
    #         actor_only_params = list(self.actor_head.parameters())
    #         critic_only_params = list(self.critic_head.parameters())
    #     else:
    #         # No shared parameters, separate actor and critic parameters
    #         shared_params = []
    #         actor_only_params = list(self.actor_layers.parameters()) + list(self.actor_head.parameters())
    #         critic_only_params = list(self.critic_layers.parameters()) + list(self.critic_head.parameters())

    #     return {
    #         "shared": shared_params,
    #         "actor_only": actor_only_params,
    #         "critic_only": critic_only_params
    #     }
if __name__ == "__main__":
    
    # Instantiate the ActorCritic network with shared features
    ac_network_shared = ActorCriticNetwork(input_size=2, net_arch=[32, 32], actor_output_size=4, share_features=True)

    # Instantiate the ActorCriticNetwork network without shared features
    ac_network_separate = ActorCriticNetwork(input_size=2, net_arch=[32, 32], actor_output_size=4, share_features=False)

    # print("Shared Network Parameters:", param_groups_shared)
    # print("Separate Network Parameters:", param_groups_separate)
    
    # for p in ac_network_shared.parameter_group['actor_only']:
    #     print(p)
    
    x = torch.randn([2, 2])
    actions, values, log_prob = ac_network_separate(x)

    for p in ac_network_separate.actor_parameters():
        print(p.shape)
        
    print('=========')
    
    for p in ac_network_separate.critic_parameters():
        print(p.shape)
    
    # values, log_prob, entropy = ac_network_shared.evaluate_actions(x, actions)
    
    # print(log_prob)
    # print(entropy)
    
    # print('--------')
    # print(ac_network_shared.predict_values(x))