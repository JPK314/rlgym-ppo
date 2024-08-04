"""
File: learner.py
Author: Matthew Allen and Jonathan Keegan

Description:
The primary algorithm file. The Learner object coordinates timesteps from the workers 
and sends them to PPO, keeps track of the misc. variables and statistics for logging,
reports to wandb and the console, and handles checkpointing.
"""

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional

import numpy as np
import torch
import torch.nn as nn
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    EngineActionType,
    ObsSpaceType,
    ObsType,
    RewardType,
    RLGym,
    StateType,
)
from wandb.wandb_run import Run

import wandb
from rlgym_ppo.agent import AgentManager
from rlgym_ppo.api import (
    Agent,
    ObsStandardizer,
    RewardTypeWrapper,
    StateMetrics,
    TypeSerde,
)
from rlgym_ppo.env_processing import EnvProcessInterface
from rlgym_ppo.util import KBHit
from rlgym_ppo.util.torch_functions import get_device


@dataclass
class LearnerConfig:
    n_proc: int = 8
    min_inference_size: int = 80
    render: bool = False
    render_delay: float = 0
    timestep_limit: int = 5_000_000_000
    instance_launch_delay: Optional[float] = None
    random_seed: int = 123
    shm_buffer_size: int = 8192
    device: str = "auto"
    recalculate_agent_id_every_step: bool = False


class Learner(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
    ]
):
    def __init__(
        self,
        env_create_function: Callable[
            [],
            RLGym[
                AgentID,
                ObsType,
                ActionType,
                EngineActionType,
                RewardTypeWrapper[RewardType],
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ],
        ],
        agents: List[
            Agent[
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                ObsSpaceType,
                ActionSpaceType,
                StateMetrics,
                Any,
            ]
        ],
        agent_id_serde: TypeSerde[AgentID],
        action_type_serde: TypeSerde[ActionType],
        obs_type_serde: TypeSerde[ObsType],
        reward_type_serde: TypeSerde[RewardTypeWrapper[RewardType]],
        obs_space_type_serde: TypeSerde[ObsSpaceType],
        action_space_type_serde: TypeSerde[ActionSpaceType],
        state_metrics_type_serde: Optional[TypeSerde[StateMetrics]] = None,
        # TODO: add List[Tuple[AgentID, RewardType]] to collect_state_metrics_fn? Or can this be done in trajectory processor impl?
        collect_state_metrics_fn: Optional[Callable[[StateType], StateMetrics]] = None,
        obs_standardizer: Optional[ObsStandardizer] = None,
        config=LearnerConfig(),
    ):
        self.config = config
        assert (
            env_create_function is not None
        ), "MUST PROVIDE A FUNCTION TO CREATE RLGYM FUNCTIONS TO INITIALIZE RLGYM-PPO"

        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

        self.device = get_device(self.config.device)
        print(f"Using device {self.device}")

        print("Initializing processes...")

        self.agent_manager = AgentManager(agents)

        self.cumulative_timesteps = 0
        self.env_process_interface = EnvProcessInterface(
            env_create_function,
            agent_id_serde,
            action_type_serde,
            obs_type_serde,
            reward_type_serde,
            obs_space_type_serde,
            action_space_type_serde,
            state_metrics_type_serde=state_metrics_type_serde,
            collect_state_metrics_fn=collect_state_metrics_fn,
            obs_standardizer=obs_standardizer,
            min_inference_size=self.config.min_inference_size,
            seed=self.config.random_seed,
            recalculate_agent_id_every_step=self.config.recalculate_agent_id_every_step,
        )
        obs_space, action_space, self.initial_obs_list = (
            self.env_process_interface.init_processes(
                n_processes=self.config.n_proc,
                spawn_delay=self.config.instance_launch_delay,
                render=self.config.render,
                render_delay=self.config.render_delay,
                shm_buffer_size=self.config.shm_buffer_size,
            )
        )
        print("Loading agents...")
        self.agent_manager.set_space_types(obs_space, action_space)
        self.agent_manager.set_device(self.device)
        self.agent_manager.load_agents()
        print("Learner successfully initialized!")

    def learn(self):
        """
        Function to wrap the _learn function in a try/catch/finally
        block to ensure safe execution and error handling.
        :return: None
        """
        try:
            self._learn()
        except Exception:
            import traceback

            print("\n\nLEARNING LOOP ENCOUNTERED AN ERROR\n")
            traceback.print_exc()

            try:
                self.save(self.cumulative_timesteps)
            except:
                print("FAILED TO SAVE ON EXIT")

        finally:
            self.cleanup()

    def _learn(self):
        """
        Learning function. This is where the magic happens.
        :return: None
        """

        # Class to watch for keyboard hits
        # TODO: add keys to increase / decrease number of env processes
        kb = KBHit()
        print(
            "Press (p) to pause (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\n"
        )

        # Handle actions for observations created on process init
        actions, log_probs = self.agent_manager.get_actions(self.initial_obs_list)
        self.env_process_interface.send_actions(actions, log_probs)

        # While the number of timesteps we have collected so far is less than the
        # amount we are allowed to collect.
        while self.cumulative_timesteps < self.config.timestep_limit:
            # Collect the desired number of timesteps from our agent.
            obs_list, timesteps, state_metrics = (
                self.env_process_interface.collect_step_data()
            )
            self.cumulative_timesteps += len(timesteps)
            self.agent_manager.process_timestep_data(timesteps, state_metrics)
            actions, log_probs = self.agent_manager.get_actions(obs_list)

            self.env_process_interface.send_actions(actions, log_probs)

            # Check if keyboard press
            # p: pause, any key to resume
            # c: checkpoint
            # q: checkpoint and quit

            if kb.kbhit():
                c = kb.getch()
                if c == "p":  # pause
                    print("Paused, press any key to resume")
                    while True:
                        if kb.kbhit():
                            break
                if c in ("c", "q"):
                    self.agent_manager.save_agents()
                if c == "q":
                    return
                if c in ("c", "p"):
                    print("Resuming...\n")

    def save(self, cumulative_timesteps):
        raise NotImplementedError

    def load(self, folder_path, load_wandb, new_policy_lr=None, new_critic_lr=None):
        raise NotImplementedError

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """
        self.env_process_interface.cleanup()
        self.agent_manager.cleanup()
