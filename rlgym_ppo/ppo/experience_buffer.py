"""
File: experience_buffer.py
Author: Matthew Allen

Description:
    A buffer containing the experience to be learned from on this iteration. The buffer may be added to, removed from,
    and shuffled. When the maximum specified size of the buffer is exceeded, the least recent entries will be removed in
    a FIFO fashion.
"""

import time
from typing import Generic, List, Tuple

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType


class ExperienceBuffer(Generic[AgentID, ObsType, ActionType]):
    @staticmethod
    def _cat(t1, t2, size):
        if len(t2) > size:
            # t2 alone is larger than we want; copy the end
            t = t2[-size:].clone()

        elif len(t2) == size:
            # t2 is a perfect match; just use it directly
            t = t2

        elif len(t1) + len(t2) > size:
            # t1+t2 is larger than we want; use t2 wholly with the end of t1 before it
            t = torch.cat((t1[len(t2) - size :], t2), 0)

        else:
            # t1+t2 does not exceed what we want; concatenate directly
            t = torch.cat((t1, t2), 0)

        del t1
        del t2
        return t

    def __init__(self, max_size, seed, device):
        self.device = device
        self.seed = seed
        self.observations: List[Tuple[AgentID, ObsType]] = []
        self.actions: List[ActionType] = []
        self.log_probs = torch.FloatTensor().to(self.device)
        self.values = torch.FloatTensor().to(self.device)
        self.advantages = torch.FloatTensor().to(self.device)
        self.max_size = max_size
        self.rng = np.random.RandomState(seed)

    def submit_experience(
        self,
        observations: List[Tuple[AgentID, ObsType]],
        actions: List[ActionType],
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
        advantages: List[torch.Tensor],
    ):
        """
        Function to add experience to the buffer.

        :param observations: An ordered sequence of observations from the environment.
        :param actions: The corresponding actions that were taken at each state in the `states` sequence.
        :param log_probs: The log probability for each action in `actions`
        :param rewards: A list of rewards such that rewards[i] is the reward for taking action actions[i] from observation observations[i]
        :param terminateds: An ordered sequence of the terminated flags from the environment.
        :param truncateds: An ordered sequence of the truncated flag from the environment.
        :param values: The output of the value function estimator evaluated on the observations.
        :param advantages: The advantage of each action at each state in `states` and `actions`

        :return: None
        """

        _cat = ExperienceBuffer._cat
        self.observations += observations
        self.actions += actions
        self.log_probs = _cat(
            self.log_probs,
            torch.as_tensor(log_probs, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.values = _cat(
            self.values,
            torch.as_tensor(values, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.advantages = _cat(
            self.advantages,
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            self.max_size,
        )

    def _get_samples(self, indices) -> Tuple[
        List[ActionType],
        torch.Tensor,
        List[Tuple[AgentID, ObsType]],
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            [self.observations[index] for index in indices],
            [self.actions[index] for index in indices],
            self.log_probs[indices],
            self.values[indices],
            self.advantages[indices],
        )

    def get_all_batches_shuffled(self, batch_size):
        """
        Function to return the experience buffer in shuffled batches. Code taken from the stable-baeselines3 buffer:
        https://github.com/DLR-RM/stable-baselines3/blob/2ddf015cd9840a2a1675f5208be6eb2e86e4d045/stable_baselines3/common/buffers.py#L482
        :param batch_size: size of each batch yielded by the generator.
        :return:
        """

        total_samples = self.rewards.shape[0]
        indices = self.rng.permutation(total_samples)
        start_idx = 0
        while start_idx + batch_size <= total_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def clear(self):
        """
        Function to clear the experience buffer.
        :return: None.
        """
        del self.observations
        del self.actions
        del self.log_probs
        del self.values
        del self.advantages
        self.__init__(self.max_size, self.seed, self.device)
