from abc import abstractmethod
from typing import Generic, List, Tuple

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor, device, dtype

from rlgym_ppo.api import RewardTypeWrapper
from rlgym_ppo.experience import Trajectory


class ObsStandardizer(Generic[AgentID, ObsType]):
    @abstractmethod
    def standardize(
        self, obs_list: List[Tuple[AgentID, ObsType]]
    ) -> List[Tuple[AgentID, ObsType]]:
        """
        :param obs_list: list of tuples of agent IDs and observations parallel with returned list. Agent IDs may not be unique here.
        :return: List of tuples of agent IDs and observations, with observations standardized.
        """
        raise NotImplementedError


class TrajectoryProcessor(Generic[AgentID, ObsType, ActionType, RewardType]):
    @abstractmethod
    def process_trajectories(
        self,
        trajectories: List[
            Trajectory[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]]
        ],
        dtype: dtype,
        device: device,
    ) -> Tuple[
        List[Tuple[AgentID, ObsType]],
        List[ActionType],
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        :param trajectories: List of Trajectory instances from which to generate experience.
        :return: Tuple of parallel lists (considering tensors as a list in their first dimension)
            with (AgentID, ObsType), ActionType, log prob, value, and advantage respectively.
            log prob, value, and advantage tensors should be with dtype=dtype and device=device.
        """
        raise NotImplementedError
