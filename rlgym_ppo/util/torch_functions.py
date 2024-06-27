"""
File: torch_functions.py
Author: Matthew Allen

Description:
    A helper file for misc. PyTorch functions.

"""

from typing import List

import torch
import torch.nn as nn
from rlgym.api import ActionType, ObsType, RewardType

from rlgym_ppo.api import ActionType, AgentID, ObsType, RewardType, RewardTypeWrapper
from rlgym_ppo.batched_agents import Trajectory


class MapContinuousToAction(nn.Module):
    """
    A class for policies using the continuous action space. Continuous policies output N*2 values for N actions where
    each value is in the range [-1, 1]. Half of these values will be used as the mean of a multi-variate normal distribution
    and the other half will be used as the diagonal of the covariance matrix for that distribution. Since variance must
    be positive, this class will map the range [-1, 1] for those values to the desired range (defaults to [0.1, 1]) using
    a simple linear transform.
    """

    def __init__(self, range_min=0.1, range_max=1):
        super().__init__()

        tanh_range = [-1, 1]
        self.m = (range_max - range_min) / (tanh_range[1] - tanh_range[0])
        self.b = range_min - tanh_range[0] * self.m

    def forward(self, x):
        n = x.shape[-1] // 2
        # map the right half of x from [-1, 1] to [range_min, range_max].
        return x[..., :n], x[..., n:] * self.m + self.b


def compute_gae(
    trajectories: List[
        Trajectory[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]]
    ],
    gamma=0.99,
    lmbda=0.95,
    return_std=1,
):
    """
    Function to estimate the advantage function for a series of states and actions using the
    general advantage estimator (GAE).

    :param trajectories: List of trajectories with value predictions.
    :param gamma: Gamma hyper-parameter.
    :param lmbda: Lambda hyper-parameter.
    :param return_std: Standard deviation of the returns (used for reward normalization).
    :return: Bootstrapped value function estimates, GAE results, returns.
    """
    observations: List[ObsType] = []
    actions: List[ActionType] = []
    log_probs: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []
    terminateds: List[bool] = []
    truncateds: List[bool] = []
    values: List[float] = []
    advantages: List[float] = []
    returns: List[float] = []
    for trajectory in trajectories:
        cur_return = trajectory.final_val_pred
        next_val_pred = trajectory.final_val_pred
        cur_advantages = torch.Tensor(0)
        for timestep in reversed(trajectory.complete_timesteps):
            (obs, action, log_prob, reward, terminated, truncated, val_pred) = timestep
            reward_tensor = reward.as_tensor()
            delta = reward_tensor + gamma * next_val_pred - val_pred
            next_val_pred = val_pred
            cur_advantages = delta + gamma * lmbda * cur_advantages
            cur_return = reward_tensor + gamma * cur_return
            returns.append(cur_return.detach().item())
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward_tensor)
            terminateds.append(terminated)
            truncateds.append(truncated)
            values.append(val_pred)
            advantages.append(cur_advantages)

    return (
        observations,
        actions,
        log_probs,
        rewards,
        terminateds,
        truncateds,
        values,
        advantages,
        returns,
    )


class MultiDiscreteRolv(nn.Module):
    """
    A class to handle the multi-discrete action space in Rocket League. There are 8 potential actions, 5 of which can be
    any of {-1, 0, 1} and 3 of which can be either of {0, 1}. This class takes 21 logits, appends -inf to the final 3
    such that each of the 8 actions has 3 options (to avoid a ragged list), then builds a categorical distribution over
    each class for each action. Credit to Rolv Arild for coming up with this method.
    """

    def __init__(self, bins):
        super().__init__()
        self.distribution = None
        self.bins = bins

    def make_distribution(self, logits):
        """
        Function to make the multi-discrete categorical distribution for a group of logits.
        :param logits: Logits which parameterize the distribution.
        :return: None.
        """

        # Split the 21 logits into the expected bins.
        logits = torch.split(logits, self.bins, dim=-1)

        # Separate triplets from the split logits.
        triplets = torch.stack(logits[:5], dim=-1)

        # Separate duets and pad the final dimension with -inf to create triplets.
        duets = torch.nn.functional.pad(
            torch.stack(logits[5:], dim=-1), pad=(0, 0, 0, 1), value=float("-inf")
        )

        # Un-split the logits now that the duets have been converted into triplets and reshape them into the correct shape.
        logits = torch.cat((triplets, duets), dim=-1).swapdims(-1, -2)

        # Construct a distribution with our fixed logits.
        self.distribution = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        return self.distribution.log_prob(action).sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy().sum(
            dim=-1
        )  # Unsure about this sum operation.
