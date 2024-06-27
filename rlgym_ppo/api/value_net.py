from abc import abstractmethod
from typing import Generic, List, Tuple

import torch.nn as nn
from rlgym.api import AgentID, ObsType
from torch import Tensor


class ValueNet(nn.Module, Generic[AgentID, ObsType]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs_list: List[Tuple[AgentID, ObsType]]) -> Tuple[Tensor, Tensor]:
        """
        :obs_list: list of agent_id and obs pairs to potentially compute values for.
        """
        raise NotImplementedError
