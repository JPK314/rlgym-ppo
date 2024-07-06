from typing import Any, Dict, List

import numpy as np
from rlgym.api import ActionType, AgentID, ObsType, RewardType, StateType
from wandb.wandb_run import Run

from rlgym_ppo.api import MetricsLogger
from rlgym_ppo.ppo.ppo_metrics import PPOMetrics
from rlgym_ppo.typing import LearnerMetrics
from rlgym_ppo.util import reporting


class PPOMetricsLogger(
    MetricsLogger[StateType, AgentID, ActionType, ObsType, RewardType]
):
    def collect_ppo_metrics(self, ppo_metrics):
        return {
            "PPO Batch Consumption Time": ppo_metrics.batch_consumption_time,
            "Cumulative Model Updates": ppo_metrics.cumulative_model_updates,
            "Policy Entropy": ppo_metrics.policy_entropy,
            "Mean KL Divergence": ppo_metrics.kl_divergence,
            "Value Function Loss": ppo_metrics.value_loss,
            "SB3 Clip Fraction": ppo_metrics.sb3_clip_fraction,
            "Policy Update Magnitude": ppo_metrics.policy_update_magnitude,
            "Value Function Update Magnitude": ppo_metrics.critic_update_magnitude,
        }

    def collect_learner_metrics(self, learner_metrics):
        return {
            "Cumulative Timesteps": learner_metrics.cumulative_timesteps,
            "Total Iteration Time": learner_metrics.iteration_time,
            "Timesteps Collected": learner_metrics.timesteps_collected,
            "Timestep Collection Time": learner_metrics.timestep_collection_time,
            "Timestep Consumption Time": learner_metrics.iteration_time
            - learner_metrics.timestep_collection_time,
            "Collected Steps per Second": learner_metrics.timesteps_collected
            / learner_metrics.timestep_collection_time,
            "Overall Steps per Second": learner_metrics.timesteps_collected
            / learner_metrics.iteration_time,
        }

    def collect_state_metrics(self, state: StateType) -> List[np.ndarray]:
        return []

    def report_metrics(
        self,
        collected_state_metrics: List[List[np.ndarray]],
        ppo_metrics: Dict[str, Any],
        learner_metrics: Dict[str, Any],
        wandb_run: Run,
    ):
        report = {**ppo_metrics, **learner_metrics, "Policy Reward": np.nan}
        reporting.report_metrics(report, wandb_run=wandb_run)
