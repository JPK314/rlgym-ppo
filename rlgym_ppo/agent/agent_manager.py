from typing import Generic, Iterable, List, Tuple, cast

import numpy as np
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from torch import Tensor, as_tensor, int64, stack

from rlgym_ppo.api import Agent, AgentData, StateMetrics
from rlgym_ppo.experience import Timestep


class AgentManager(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
        AgentData,
    ]
):
    def __init__(
        self,
        agents: List[
            Agent[
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                ObsSpaceType,
                ActionSpaceType,
                StateMetrics,
                AgentData,
            ]
        ],
    ) -> None:
        self.agents = agents
        self.n_agents = len(agents)
        assert self.n_agents > 0, "There must be at least one agent!"

    def get_actions(
        self, obs_list: List[Tuple[AgentID, ObsType]]
    ) -> Tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs_list: list of tuples of agent IDs and observations parallel with returned list. Agent IDs may not be unique here.
        :return: Tuple of lists of chosen action and Tensor, with the action list and the first dimension of the tensor parallel with obs_list.
        """
        obs_list_len = len(obs_list)
        agents_actions = [agent.get_actions(obs_list) for agent in self.agents]
        # agents earlier in the list have higher priority
        action_idx_agent_idx_map = np.array([-1] * obs_list_len)
        for agent_idx, agent_actions in enumerate(agents_actions):
            action_idx_mask1 = action_idx_agent_idx_map == -1
            action_idx_mask2 = np.array(
                [action is not None for action in agent_actions[0]]
            )
            action_idx_mask = np.logical_and(action_idx_mask1, action_idx_mask2)
            action_idx_agent_idx_map[action_idx_mask] = agent_idx
        assert not (
            action_idx_agent_idx_map == -1
        ).any(), "Agents didn't provide actions for all observations!"
        agents_log_probs = stack(
            [agent_action[1] for agent_action in agents_actions]
        ).to(device="cpu")
        actions: List[ActionType] = [None] * obs_list_len
        for action_idx, agent_idx in enumerate(action_idx_agent_idx_map):
            actions[action_idx] = agents_actions[agent_idx][0][action_idx]
        # TODO: this looks insane but probably works? Check the output with multiple agents
        log_prob_gather_index = (
            as_tensor(action_idx_agent_idx_map, dtype=int64)
            .unsqueeze(dim=1)
            .repeat(obs_list_len, 1, 1)
        )
        log_probs = agents_log_probs.gather(dim=0, index=log_prob_gather_index)[0].to(
            device="cpu"
        )
        return actions, log_probs

    def process_timestep_data(
        self, timesteps: List[Timestep], state_metrics: List[StateMetrics]
    ):
        for agent in self.agents:
            agent.process_timestep_data(timesteps, state_metrics)

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        for agent in self.agents:
            agent.set_space_types(obs_space, action_space)

    def set_device(self, device: str):
        for agent in self.agents:
            agent.set_device(device)

    def load_agents(self):
        for agent in self.agents:
            agent.load()

    def save_agents(self):
        for agent in self.agents:
            agent.save()

    def cleanup(self):
        for agent in self.agents:
            agent.cleanup()

    def is_learning(self):
        return any([agent.is_learning() for agent in self.agents])
