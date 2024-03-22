from EscapeEnv.common.base_estimator import BaseEstimator
from EscapeEnv.common.scheduler import ConstantParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Optional, Sequence


class QRDQNEstimator(BaseEstimator):
    def __init__(self, network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device) -> None:
        super().__init__(network, batch_size, learning_rate, loops_per_train, optimizer_kwargs, estimator_kwargs, device)
        

        self.use_ddqn = self.estimator_kwargs['use_ddqn']
        self.use_legal = self.estimator_kwargs['use_legal']
        self.n_quantiles = self.estimator_kwargs['n_quantiles']
        self.max_grad_norm = self.estimator_kwargs['max_grad_norm']
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        self.lr_scheduler = ConstantParamScheduler(self.optimizer, 'lr', self.learning_rate)
        self.schedulers = [self.lr_scheduler]
        # self.mse_loss = nn.HuberLoss()
    
    def predict_nograd(self, s):
        with torch.no_grad():
            q_as = self.qnet._predict(s)
        return q_as
    
    def evaluate_q_value(self, s):
        return self.qnet.eval_q_values(s)
    
    
    def update(self, buffer, discount_factor):
        for schedule in self.schedulers:
            schedule.step()

    
        for _ in range(self.loops_per_train):
            # batch data
            batch = buffer.sample()
            states, actions, rewards, non_final_mask, non_final_next_states, non_final_next_actions, non_final_next_legal = self.batch_extract(batch)
            
            self.optimizer.zero_grad()
            
            
            target_quantiles = rewards.unsqueeze(dim=1).expand(self.batch_size, self.n_quantiles).clone()

            if self.use_legal == True:
                if self.use_ddqn == True:
                    policy_next_action = (self.qnet._predict(non_final_next_states).detach().exp() * non_final_next_legal).max(dim=-1, keepdim=True)[1]
                    policy_next_action = policy_next_action.unsqueeze(dim=1).tile(1, self.n_quantiles, 1)
                    # target_quantiles = target_quantiles.clone()
                    target_quantiles[non_final_mask] += discount_factor * self.qnet_target(non_final_next_states).gather(-1, policy_next_action).squeeze(dim=-1).detach()
                else:
                    target_next_action = (self.qnet_target(non_final_next_states).detach().exp() * non_final_next_legal).max(dim=-1, keepdim=True)[1]
                    target_quantiles[non_final_mask] += discount_factor *  (self.qnet_target(non_final_next_states).gather(-1, target_next_action).squeeze(dim=-1).detach())
            else:
                if self.use_ddqn == True:
                    policy_next_action = self.qnet_predict(non_final_next_states).detach().max(dim=-1, keepdim=True)[1]
                    target_quantiles[non_final_mask] += discount_factor * self.qnet_target(non_final_next_states).gather(-1, policy_next_action).squeeze(dim=-1).detach()
                else:
                    target_quantiles[non_final_mask] += discount_factor *  self.qnet_target(non_final_next_states).max(dim=-1, keepdim=True)[0].squeeze(dim=-1).detach()
                    
            # sarsa
            # next_actions = non_final_next_actions.unsqueeze(dim=1).tile(1, self.n_quantiles, 1)
            # target_quantiles[non_final_mask] += discount_factor * self.qnet(non_final_next_states).gather(-1, next_actions).squeeze(dim=-1)
                    
            current_quantiles = self.qnet(states)
            current_actions = actions.unsqueeze(dim=1).tile(1, self.n_quantiles, 1)
            current_quantiles = current_quantiles.gather(-1, current_actions).squeeze(dim=-1)            
            
            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.n_updates += 1
        return loss.item()

def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    cum_prob: Optional[torch.Tensor] = None,
    sum_over_quantiles: bool = True,
) -> torch.Tensor:
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.

    :param current_quantiles: current estimate of quantiles, must be either
        (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
        (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
        (if None, calculating unit quantiles)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if current_quantiles.ndim != target_quantiles.ndim:
        raise ValueError(
            f"Error: The dimension of curremt_quantile ({current_quantiles.ndim}) needs to match "
            f"the dimension of target_quantiles ({target_quantiles.ndim})."
        )
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError(
            f"Error: The batch size of curremt_quantile ({current_quantiles.shape[0]}) needs to match "
            f"the batch size of target_quantiles ({target_quantiles.shape[0]})."
        )
    if current_quantiles.ndim not in (2, 3):
        raise ValueError(f"Error: The dimension of current_quantiles ({current_quantiles.ndim}) needs to be either 2 or 3.")

    if cum_prob is None:
        n_quantiles = current_quantiles.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, -1, 1)
        elif current_quantiles.ndim == 3:
            # For TQC, current_quantiles have a shape (batch_size, n_critics, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_critics, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, 1, -1, 1)

    # QR-DQN
    # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
    # TQC
    # target_quantiles: (batch_size, 1, n_target_quantiles) -> (batch_size, 1, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_critics, n_quantiles) -> (batch_size, n_critics, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_critics, n_quantiles, n_target_quantiles)
    # Note: in both cases, the loss has the same shape as pairwise_delta
    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()
    return loss
if __name__ == '__main__':
    x = torch.tensor([1,2,3,4,5])
    legal = torch.tensor([0,1,0,1,0])
    
    y = (x.exp() * legal).max(dim=-1)[0].log()
    print(y)
    
