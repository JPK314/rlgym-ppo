from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType

from rlgym_ppo.api import ObsStandardizer
from rlgym_ppo.ppo import TrajectoryProcessor
from rlgym_ppo.util import WelfordRunningStat


class NumpyObsStandardizer(ObsStandardizer[AgentID, np.ndarray]):
    def __init__(
        self, steps_per_obs_stats_update: int, steps_until_fixed: int = np.inf
    ):
        self.obs_stats = None
        self.steps_per_obs_stats_update = steps_per_obs_stats_update
        self.obs_stats_start_index = 0
        self.steps_until_fixed = steps_until_fixed
        self.steps = 0

    def standardize(self, obs_list) -> List[Tuple[AgentID, np.ndarray]]:
        if self.obs_stats == None:
            (_, obs) = obs_list[0]
            self.obs_stats = WelfordRunningStat(obs.shape)
        if self.steps < self.steps_until_fixed:
            stats_update_batch = [
                o[1]
                for o in obs_list[
                    self.obs_stats_start_index :: self.steps_per_obs_stats_update
                ]
            ]
            self.obs_stats_start_index = (
                self.steps_per_obs_stats_update
                - 1
                - (
                    (len(obs_list) - self.obs_stats_start_index - 1)
                    % self.steps_per_obs_stats_update
                )
            )
            for sample in stats_update_batch:
                self.obs_stats.update(sample)
        return [
            (agent_id, (obs - self.obs_stats.mean) / self.obs_stats.std)
            for (agent_id, obs) in obs_list
        ]


# TODO: move to ppo
@dataclass
class GAETrajectoryProcessorData:
    average_undiscounted_episodic_return: float
    average_return: float
    return_standard_deviation: float


class GAETrajectoryProcessor(
    TrajectoryProcessor[
        AgentID, ObsType, ActionType, RewardType, GAETrajectoryProcessorData
    ]
):
    def __init__(
        self,
        gamma=0.99,
        lmbda=0.95,
        standardize_returns=True,
        max_returns_per_stats_increment=150,
    ):
        """
        :param gamma: Gamma hyper-parameter.
        :param lmbda: Lambda hyper-parameter.
        :param return_std: Standard deviation of the returns (used for reward normalization).
        """
        self.gamma = gamma
        self.lmbda = lmbda
        self.return_stats = WelfordRunningStat(1)
        self.standardize_returns = standardize_returns
        self.max_returns_per_stats_increment = max_returns_per_stats_increment

    def process_trajectories(self, trajectories, dtype, device):
        return_std = self.return_stats.std[0] if self.standardize_returns else None
        gamma = self.gamma
        lmbda = self.lmbda
        observations: List[Tuple[AgentID, ObsType]] = []
        actions: List[ActionType] = []
        log_probs: List[torch.Tensor] = []
        values: List[float] = []
        advantages: List[float] = []
        returns: List[float] = []
        reward_sum = torch.as_tensor(0, dtype=dtype, device=device)
        for trajectory in trajectories:
            cur_return = torch.as_tensor(0, dtype=dtype, device=device)
            next_val_pred = (
                trajectory.final_val_pred
                if trajectory.truncated
                else torch.as_tensor(0, dtype=dtype, device=device)
            )
            cur_advantages = torch.as_tensor(0, dtype=dtype, device=device)
            for timestep in reversed(trajectory.complete_timesteps):
                (obs, action, log_prob, reward, val_pred) = timestep
                reward_tensor = reward.as_tensor(dtype=dtype, device=device)
                reward_sum += reward_tensor
                if return_std is not None:
                    norm_reward_tensor = torch.clamp(
                        reward_tensor / return_std, min=-10, max=10
                    )
                else:
                    norm_reward_tensor = reward_tensor
                delta = norm_reward_tensor + gamma * next_val_pred - val_pred
                next_val_pred = val_pred
                cur_advantages = delta + gamma * lmbda * cur_advantages
                cur_return = reward_tensor + gamma * cur_return
                returns.append(cur_return.detach().item())
                observations.append((trajectory.agent_id, obs))
                actions.append(action)
                log_probs.append(log_prob)
                values.append(val_pred)
                advantages.append(cur_advantages)

        if self.standardize_returns:
            # Update the running statistics about the returns.
            n_to_increment = min(self.max_returns_per_stats_increment, len(returns))
            for sample in returns[:n_to_increment]:
                self.return_stats.update(sample)
            avg_return = self.return_stats.mean
            return_std = self.return_stats.std
        else:
            avg_return = np.nan
            return_std = np.nan
        avg_reward = (reward_sum / len(observations)).cpu().item()
        trajectory_processor_data = GAETrajectoryProcessorData(
            average_undiscounted_episodic_return=avg_reward,
            average_return=avg_return,
            return_standard_deviation=return_std,
        )
        return (
            (
                observations,
                actions,
                torch.cat(log_probs).to(device=device),
                torch.stack(values).to(device=device),
                torch.stack(advantages),
            ),
            trajectory_processor_data,
        )

    def save(self) -> dict:
        return {
            "gamma": self.gamma,
            "lambda": self.lmbda,
            "standardize_returns": self.standardize_returns,
            "max_returns_per_stats_increment": self.max_returns_per_stats_increment,
            "return_running_stats": self.return_stats.to_json(),
        }

    def load(self, state):
        self.gamma = state["gamma"]
        self.lmbda = state["lambda"]
        self.standardize_returns = state["standardize_returns"]
        self.max_returns_per_stats_increment = state["max_returns_per_stats_increment"]
        self.return_stats = self.return_stats.from_json(state["return_running_stats"])
