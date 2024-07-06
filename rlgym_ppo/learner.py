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
from typing import Callable, Generic, List, Optional, Tuple, Type, TypedDict, Union

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
from rlgym_ppo import LearnerConfig, LearnerMetrics
from rlgym_ppo.api import (
    ActorCriticManager,
    MetricsLogger,
    ObsStandardizer,
    PPOPolicy,
    RewardTypeWrapper,
    TrajectoryProcessor,
    TypeSerde,
    ValueNet,
)
from rlgym_ppo.batched_agents import BatchedAgentManager
from rlgym_ppo.experience import ExperienceBuffer, Trajectory
from rlgym_ppo.ppo import PPOLearner
from rlgym_ppo.util import KBHit, WelfordRunningStat


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
        actor_critic_manager: ActorCriticManager,
        agent_id_serde: TypeSerde[AgentID],
        action_type_serde: TypeSerde[ActionType],
        obs_type_serde: TypeSerde[ObsType],
        reward_type_serde: TypeSerde[RewardTypeWrapper[RewardType]],
        obs_space_type_serde: TypeSerde[ObsSpaceType],
        action_space_type_serde: TypeSerde[ActionSpaceType],
        policy_factory: Callable[
            [ObsSpaceType, ActionSpaceType, str],
            PPOPolicy[AgentID, ObsType, ActionType],
        ],
        value_net_factory: Callable[[ObsSpaceType, str], ValueNet[AgentID, ObsType]],
        trajectory_processor_factory: Callable[
            ..., TrajectoryProcessor[AgentID, ObsType, ActionType, RewardType]
        ],
        obs_standardizer: Optional[ObsStandardizer] = None,
        metrics_logger: Optional[
            MetricsLogger[StateType, AgentID, ActionType, ObsType, RewardType]
        ] = None,
        wandb_run: Optional[Run] = None,
        config=LearnerConfig(),
    ):
        self.config = config
        assert (
            env_create_function is not None
        ), "MUST PROVIDE A FUNCTION TO CREATE RLGYM FUNCTIONS TO INITIALIZE RLGYM-PPO"

        checkpoints_save_folder = self.config.checkpoints_save_folder
        if checkpoints_save_folder is None:
            checkpoints_save_folder = os.path.join(
                "data", "checkpoints", "rlgym-ppo-run"
            )

        # Add the option for the user to turn off the addition of Unix Timestamps to
        # the ``checkpoints_save_folder`` path
        if self.config.add_unix_timestamp:
            checkpoints_save_folder = f"{checkpoints_save_folder}-{time.time_ns()}"

        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

        self.checkpoints_save_folder = checkpoints_save_folder
        self.metrics_logger = metrics_logger
        self.ts_since_last_save = 0

        if self.config.device in {"auto", "gpu"} and torch.cuda.is_available():
            self.device = "cuda:0"
            torch.backends.cudnn.benchmark = True
        elif self.config.device == "auto" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = self.config.device

        print(f"Using device {self.device}")
        self.return_stats = WelfordRunningStat(1)
        self.average_reward = None
        self.iteration = 0

        self.experience_buffer: ExperienceBuffer[AgentID, ObsType, ActionType] = (
            ExperienceBuffer(
                trajectory_processor_factory(**self.config.trajectory_processor_args),
                self.config.exp_buffer_size,
                seed=self.config.random_seed,
                device="cpu",
            )
        )

        print("Initializing processes...")
        self.encode_state_metrics_fn = (
            None if metrics_logger is None else self.metrics_logger.encode_state_metrics
        )
        self.decode_state_metrics_fn = (
            None if metrics_logger is None else self.metrics_logger.decode_state_metrics
        )
        self.collect_ppo_metrics_fn = (
            None if metrics_logger is None else self.metrics_logger.collect_ppo_metrics
        )
        self.agent = BatchedAgentManager(
            env_create_function,
            actor_critic_manager,
            agent_id_serde,
            action_type_serde,
            obs_type_serde,
            reward_type_serde,
            obs_space_type_serde,
            action_space_type_serde,
            obs_standardizer=obs_standardizer,
            min_inference_size=self.config.min_inference_size,
            seed=self.config.random_seed,
            recalculate_agent_id_every_step=self.config.recalculate_agent_id_every_step,
        )
        (policy, value_net) = self.agent.init_processes(
            policy_factory=policy_factory,
            value_net_factory=value_net_factory,
            device=self.device,
            n_processes=self.config.n_proc,
            encode_metrics_fn=self.encode_state_metrics_fn,
            decode_metrics_fn=self.decode_state_metrics_fn,
            spawn_delay=self.config.instance_launch_delay,
            render=self.config.render,
            render_delay=self.config.render_delay,
            shm_buffer_size=self.config.shm_buffer_size,
        )
        print("Initializing PPO...")
        if self.config.ppo_minibatch_size is None:
            ppo_minibatch_size = self.config.ppo_batch_size
        else:
            ppo_minibatch_size = self.config.ppo_minibatch_size

        self.ppo_learner = PPOLearner(
            policy=policy,
            value_net=value_net,
            device=self.device,
            batch_size=self.config.ppo_batch_size,
            mini_batch_size=ppo_minibatch_size,
            n_epochs=self.config.ppo_epochs,
            policy_lr=self.config.policy_lr,
            critic_lr=self.config.critic_lr,
            clip_range=self.config.ppo_clip_range,
            ent_coef=self.config.ppo_ent_coef,
        )
        wandb_config = {
            key: value
            for (key, value) in self.config.__dict__.items()
            if key
            in [
                "n_proc",
                "min_inference_size",
                "timestep_limit",
                "exp_buffer_size",
                "ts_per_iteration",
                "standardize_obs",
                "policy_layer_sizes",
                "critic_layer_sizes",
                "ppo_epochs",
                "ppo_batch_size",
                "ppo_minibatch_size",
                "ppo_ent_coef",
                "ppo_clip_range",
                "policy_lr",
                "critic_lr",
            ]
        }
        wandb_config = {**wandb_config, **self.config.trajectory_processor_args}
        self.wandb_run = wandb_run
        wandb_loaded = self.config.checkpoint_load_folder is not None and self.load(
            self.config.checkpoint_load_folder,
            self.config.load_wandb,
            self.config.policy_lr,
            self.config.critic_lr,
        )

        if self.config.log_to_wandb and self.wandb_run is None and not wandb_loaded:
            project = (
                "rlgym-ppo"
                if self.config.wandb_project_name is None
                else self.config.wandb_project_name
            )
            group = (
                "unnamed-runs"
                if self.config.wandb_group_name is None
                else self.config.wandb_group_name
            )
            run_name = (
                "rlgym-ppo-run"
                if self.config.wandb_run_name is None
                else self.config.wandb_run_name
            )
            print("Attempting to create new wandb run...")
            self.wandb_run = wandb.init(
                project=project,
                group=group,
                config=wandb_config,
                name=run_name,
                reinit=True,
            )
            print("Created new wandb run!", self.wandb_run.id)
        print("Learner successfully initialized!")

    def update_learning_rate(self, new_policy_lr=None, new_critic_lr=None):
        if new_policy_lr is not None:
            self.policy_lr = new_policy_lr
            for param_group in self.ppo_learner.policy_optimizer.param_groups:
                param_group["lr"] = new_policy_lr
            print(f"New policy learning rate: {new_policy_lr}")

        if new_critic_lr is not None:
            self.critic_lr = new_critic_lr
            for param_group in self.ppo_learner.value_optimizer.param_groups:
                param_group["lr"] = new_critic_lr
            print(f"New policy learning rate: {new_policy_lr}")

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
                self.save(self.agent.cumulative_timesteps)
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
        kb = KBHit()
        print(
            "Press (p) to pause (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\n"
        )

        # While the number of timesteps we have collected so far is less than the
        # amount we are allowed to collect.
        while self.agent.cumulative_timesteps < self.config.timestep_limit:
            iteration_start = time.perf_counter()
            report = {}

            # Collect the desired number of timesteps from our agent.
            trajectories, collected_metrics, steps_collected, collection_time = (
                self.agent.collect_timesteps(self.config.ts_per_iteration)
            )

            # Add the new experience to our buffer and compute the various
            # reinforcement learning quantities we need to
            # learn from (advantages, values, returns).
            self.add_new_experience(trajectories)

            # Let PPO compute updates using our experience buffer.
            ppo_report = self.ppo_learner.learn(
                self.experience_buffer, self.collect_ppo_metrics_fn
            )
            iteration_stop = time.perf_counter()
            iteration_time = iteration_stop - iteration_start

            self.metrics_logger.collect_learner_metrics(
                LearnerMetrics(
                    self.agent.cumulative_timesteps,
                    iteration_time,
                    steps_collected,
                    collection_time,
                )
            )

            if self.metrics_logger is not None:
                self.metrics_logger.report_metrics(
                    collected_metrics, self.wandb_run, self.agent.cumulative_timesteps
                )

            self.ts_since_last_save += steps_collected

            report.clear()
            ppo_report.clear()

            if "cuda" in self.device:
                torch.cuda.empty_cache()

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
                    self.save(self.agent.cumulative_timesteps)
                if c == "q":
                    return
                if c in ("c", "p"):
                    print("Resuming...\n")

            # Save if we've reached the next checkpoint timestep.
            if self.ts_since_last_save >= self.config.save_every_ts:
                self.save(self.agent.cumulative_timesteps)
                self.ts_since_last_save = 0

            self.iteration += 1

    @torch.no_grad()
    def add_new_experience(
        self,
        trajectories: List[
            Trajectory[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]]
        ],
    ):
        """
        Function to add timesteps to our experience buffer and compute the advantage
        function estimates, value function estimates, and returns.
        :param trajectories: list of Trajectory instances
        :return: None
        """

        # Unpack timestep data.
        traj_timestep_idx_ranges: List[Tuple[int, int]] = []
        start = 0
        stop = 0
        val_net_input: List[Tuple[AgentID, ObsType]] = []
        for trajectory in trajectories:
            traj_input = [
                (trajectory.agent_id, obs)
                for (obs, *_) in trajectory.complete_timesteps
            ]
            traj_input.append((trajectory.agent_id, trajectory.final_obs))
            stop = start + len(traj_input)
            traj_timestep_idx_ranges.append((start, stop))
            start = stop
            val_net_input += traj_input

        value_net = self.ppo_learner.value_net

        # Update the trajectories with the value predictions.
        val_preds: torch.Tensor = value_net(val_net_input).cpu().flatten()
        torch.cuda.empty_cache()
        for idx, (start, stop) in enumerate(traj_timestep_idx_ranges):
            val_preds_traj = val_preds[start : stop - 1]
            final_val_pred = val_preds[stop - 1]
            trajectories[idx].update_val_preds(val_preds_traj, final_val_pred)

        self.experience_buffer.submit_experience(trajectories)

    def save(self, cumulative_timesteps):
        """
        Function to save a checkpoint.
        :param cumulative_timesteps: Number of timesteps that have passed so far in the
        learning algorithm.
        :return: None
        """

        # Make the file path to which the checkpoint will be saved
        folder_path = os.path.join(
            self.checkpoints_save_folder, str(cumulative_timesteps)
        )
        os.makedirs(folder_path, exist_ok=True)

        # Check to see if we've run out of checkpoint space and remove the oldest
        # checkpoints
        print(f"Saving checkpoint {cumulative_timesteps}...")
        existing_checkpoints = [
            int(arg) for arg in os.listdir(self.checkpoints_save_folder)
        ]
        if len(existing_checkpoints) > self.config.n_checkpoints_to_keep:
            existing_checkpoints.sort()
            for checkpoint_name in existing_checkpoints[
                : -self.config.n_checkpoints_to_keep
            ]:
                shutil.rmtree(
                    os.path.join(self.checkpoints_save_folder, str(checkpoint_name))
                )

        os.makedirs(folder_path, exist_ok=True)

        # Save all the things that need saving.
        self.ppo_learner.save_to(folder_path)

        book_keeping_vars = {
            "cumulative_timesteps": self.agent.cumulative_timesteps,
            "cumulative_model_updates": self.ppo_learner.cumulative_model_updates,
            "policy_average_reward": self.average_reward,
            "iteration": self.iteration,
            "ts_since_last_save": self.ts_since_last_save,
        }
        # if self.config.standardize_returns:
        #     book_keeping_vars["reward_running_stats"] = self.return_stats.to_json()
        # TODO: handle saving TrajectoryProcessor
        if self.wandb_run is not None:
            book_keeping_vars["wandb_run_id"] = self.wandb_run.id
            book_keeping_vars["wandb_project"] = self.wandb_run.project
            book_keeping_vars["wandb_entity"] = self.wandb_run.entity
            book_keeping_vars["wandb_group"] = self.wandb_run.group
            book_keeping_vars["wandb_config"] = self.wandb_run.config.as_dict()

        book_keeping_table_path = os.path.join(folder_path, "BOOK_KEEPING_VARS.json")
        with open(book_keeping_table_path, "w") as f:
            json.dump(book_keeping_vars, f, indent=4)

        print(f"Checkpoint {cumulative_timesteps} saved!\n")

    def load(self, folder_path, load_wandb, new_policy_lr=None, new_critic_lr=None):
        """
        Function to load the learning algorithm from a checkpoint.

        :param folder_path: Path to the checkpoint folder that will be loaded.
        :param load_wandb: Whether to resume an existing weights and biases run that
        was saved with the checkpoint being loaded.
        :return: None
        """

        # Make sure the folder exists.
        assert os.path.exists(folder_path), f"UNABLE TO LOCATE FOLDER {folder_path}"
        print(f"Loading from checkpoint at {folder_path}")

        # Load stuff.
        self.ppo_learner.load_from(folder_path)

        wandb_loaded = False
        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), "r") as f:
            book_keeping_vars = dict(json.load(f))
            self.agent.cumulative_timesteps = book_keeping_vars["cumulative_timesteps"]
            self.average_reward = book_keeping_vars["policy_average_reward"]
            self.ppo_learner.cumulative_model_updates = book_keeping_vars[
                "cumulative_model_updates"
            ]
            self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            # TODO: handle loading trajectory processor
            # if (
            #     self.config.standardize_returns
            #     and "reward_running_stats" in book_keeping_vars.keys()
            # ):
            #     self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            self.iteration = book_keeping_vars["iteration"]

            # Update learning rates if new values are provided
            if new_policy_lr is not None or new_critic_lr is not None:
                self.update_learning_rate(new_policy_lr, new_critic_lr)

            # check here for backwards compatibility

            if "wandb_run_id" in book_keeping_vars and load_wandb:
                self.wandb_run = wandb.init(
                    settings=wandb.Settings(start_method="spawn"),
                    entity=book_keeping_vars["wandb_entity"],
                    project=book_keeping_vars["wandb_project"],
                    group=book_keeping_vars["wandb_group"],
                    id=book_keeping_vars["wandb_run_id"],
                    config=book_keeping_vars["wandb_config"],
                    resume="allow",
                    reinit=True,
                )
                wandb_loaded = True

        print("Checkpoint loaded!")
        return wandb_loaded

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """

        if self.wandb_run is not None:
            self.wandb_run.finish()
        if type(self.agent) == BatchedAgentManager:
            self.agent.cleanup()
        self.experience_buffer.clear()
