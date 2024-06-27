from abc import abstractmethod
from typing import Generic, List, Tuple

import torch.nn as nn
from rlgym.api import ActionType, AgentID, ObsType
from torch import Tensor


# TODO: maybe add ActionParser to constructor?
class PPOPolicy(nn.Module, Generic[AgentID, ObsType, ActionType]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_action(
        self, obs: List[Tuple[AgentID, ObsType]], deterministic=False
    ) -> Tuple[List[ActionType], List[Tensor]]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs_list: list of tuples of agent IDs and observations parallel with returned list. Agent IDs may not be unique here.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Tuple of lists of chosen action and its logprob, each parallel with obs_list.
        """
        raise NotImplementedError

    @abstractmethod
    def get_backprop_data(self, obs: List[ObsType], acts: List[ActionType]):
        """
        Function to compute the data necessary for backpropagation.
        :param obs: Observations to pass through the policy.
        :param acts: Actions taken by the policy.
        :return: Action log probs & entropy.
        """
        raise NotImplementedError
