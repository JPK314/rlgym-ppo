from typing import Generic, List, Optional, Tuple

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


class Trajectory(Generic[AgentID, ActionType, ObsType, RewardType]):
    def __init__(self, agent_id: AgentID):
        """
        agent_id: the AgentID for the agent which is producing this trajectory.
        """
        self.agent_id = agent_id
        self.done = False
        self.complete_timesteps: List[
            Tuple[
                ObsType,
                ActionType,
                Tensor,
                RewardType,
                bool,
                bool,
                Optional[Tensor],
            ]
        ] = []
        self.final_obs: Optional[ObsType] = None
        self.final_val_pred: Tensor = torch.tensor(0, dtype=torch.float32)
        self.truncated: Optional[bool] = None

    def add_timestep(
        self,
        obs: ObsType,
        action: ActionType,
        log_prob: Tensor,
        reward: RewardType,
        terminated: bool,
        truncated: bool,
    ):
        """
        :obs: the observation prior to the action taken
        :action: the action taken as a result of this obs
        :log_prob: the log prob of this action
        :reward: the reward resulting from stepping the env with this action from the state used to construct the obs
        :terminated: whether or not the episode has been terminated
        :truncated: whether or not the episode has been truncated
        """
        if not self.done:
            self.complete_timesteps.append(
                (obs, action, log_prob, reward, terminated, truncated, None)
            )
            self.done = terminated or truncated
            if self.done:
                self.truncated = truncated

    def update_val_preds(
        self, val_preds: List[Tensor], final_val_pred: Optional[Tensor]
    ):
        """
        :val_preds: list of torch tensors for value prediction, parallel with self.complete_timesteps
        :final_val_pred: value prediction for self.final_obs
        """
        for idx, timestep in enumerate(self.complete_timesteps):
            (obs, action, log_prob, reward, terminated, truncated, _) = timestep
            self.complete_timesteps[idx] = (
                obs,
                action,
                log_prob,
                reward,
                terminated,
                truncated,
                val_preds[idx],
            )
        self.final_val_pred = final_val_pred
