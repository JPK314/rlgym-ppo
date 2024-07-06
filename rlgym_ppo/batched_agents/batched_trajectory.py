"""
File: batched_trajectory.py
Author: Matthew Allen and Jonathan Keegan

Description:
    A class to maintain timesteps from batched agents in synchronized sequences.
"""

from typing import Dict, Generic, Iterable, List

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor

from rlgym_ppo.experience import Trajectory


class BatchedTrajectory(Generic[AgentID, ActionType, ObsType, RewardType]):
    def __init__(self, agent_ids: Iterable[AgentID]):
        self.agent_trajectories: Dict[AgentID, Trajectory] = {
            agent_id: Trajectory(agent_id) for agent_id in agent_ids
        }

    def add_last_obs(self, obs_dict: Dict[AgentID, ObsType]):
        """
        Function to update batched trajectories with new timestep data
        :return: None.
        """
        for agent_id, trajectory in self.agent_trajectories.items():
            trajectory.final_obs = obs_dict[agent_id]
            if trajectory.truncated == None:
                trajectory.truncated = True

    def add_timesteps(
        self,
        obs_dict: Dict[AgentID, ObsType],
        action_dict: Dict[AgentID, ActionType],
        log_prob_dict: Dict[AgentID, Tensor],
        rew_dict: Dict[AgentID, RewardType],
        terminated_dict: Dict[AgentID, bool],
        truncated_dict: Dict[AgentID, bool],
    ):
        """
        Function to update batched trajectories with new timestep data
        :return: None.
        """
        for agent_id, trajectory in self.agent_trajectories.items():
            trajectory.add_timestep(
                obs_dict[agent_id],
                action_dict[agent_id],
                log_prob_dict[agent_id],
                rew_dict[agent_id],
                terminated_dict[agent_id],
                truncated_dict[agent_id],
            )

    def get_all(self) -> List[Trajectory]:
        """
        Function to retrieve all timestep sequences tracked by this object.
        :return: List of trajectories.
        """
        return list(self.agent_trajectories.values())
