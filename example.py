from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED
from rlgym.rocket_league.obs_builders import DefaultObs

from rlgym_ppo.api import MetricsLogger
from rlgym_ppo.standard_impl import (
    BasicActorCriticManager,
    BasicValueEstimator,
    DiscreteFF,
    FloatRewardTypeWrapper,
    FloatSerde,
    NumpyDynamicShapeSerde,
    RewardFunctionWrapper,
    RewardTypeWrapperSerde,
    StrIntTupleSerde,
    StrSerde,
)


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        tot_cars = 0
        lin_vel_sum = np.zeros(3)
        ang_vel_sum = np.zeros(3)
        for car_data in game_state.cars.values():
            lin_vel_sum += car_data.physics.linear_velocity
            ang_vel_sum += car_data.physics.angular_velocity
            tot_cars += 1

        return [
            lin_vel_sum / tot_cars,
            ang_vel_sum / tot_cars,
        ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {
            "x_vel": avg_linvel[0],
            "y_vel": avg_linvel[1],
            "z_vel": avg_linvel[2],
            "Cumulative Timesteps": cumulative_timesteps,
        }
        wandb_run.log(report)


class CustomObs(DefaultObs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_len = -1

    def get_obs_space(self, agent):
        if self.zero_padding is not None:
            return "real", 52 + 20 * self.zero_padding * 2
        else:
            return (
                "real",
                self.obs_len,
            )

    def build_obs(self, agents, state, shared_info):
        obs = super().build_obs(agents, state, shared_info)
        if self.obs_len == -1:
            self.obs_len = len(list(obs.values())[0])
        return obs


class VelocityPlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState):
        ball = state.ball
        car = state.cars[agent].physics

        car_to_ball = ball.position - car.position
        car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

        return np.dot(car_to_ball, car.linear_velocity) / CAR_MAX_SPEED


def policy_factory(
    obs_space: Tuple[str, int], action_space: Tuple[str, int], device: str
):
    return DiscreteFF(obs_space[1], action_space[1], (256, 256, 256), device)


def value_net_factory(obs_space: Tuple[str, int], device: str):
    return BasicValueEstimator(obs_space[1], (256, 256, 256), device)


def env_create_function():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition,
        NoTouchTimeoutCondition,
    )
    from rlgym.rocket_league.reward_functions import (
        CombinedReward,
        GoalReward,
        TouchReward,
    )
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    tick_skip = 8
    timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout=timeout_seconds)

    goal_reward_and_weight = (GoalReward(), 10)
    touch_reward_and_weight = (TouchReward(), 0.1)
    rewards_and_weights = (goal_reward_and_weight, touch_reward_and_weight)

    reward_fn = RewardFunctionWrapper(
        CombinedReward((TouchReward(), 1), (VelocityPlayerToBallReward(), 0.1)),
        FloatRewardTypeWrapper,
    )

    obs_builder = CustomObs(
        zero_padding=None,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer(),
    )


if __name__ == "__main__":
    from rlgym_ppo import Learner, LearnerConfig

    # 32 processes
    n_proc = 64

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    config: LearnerConfig = {
        "n_proc": n_proc,
        "min_inference_size": min_inference_size,
        "ppo_batch_size": 50000,
        "ts_per_iteration": 50000,
        "exp_buffer_size": 150000,
        "ppo_minibatch_size": 50000,
        "ppo_ent_coef": 0.001,
        "ppo_epochs": 1,
        "standardize_returns": True,
        "save_every_ts": 100000,
        "timestep_limit": 1000000000,
        "log_to_wandb": True,
        "wandb_group_name": "rlgym-ppo-testing",
        "render": False,
    }
    learner = Learner(
        env_create_function=env_create_function,
        actor_critic_manager=BasicActorCriticManager(),
        agent_id_serde=StrSerde(),
        action_type_serde=NumpyDynamicShapeSerde(dtype=np.int64),
        obs_type_serde=NumpyDynamicShapeSerde(dtype=np.float64),
        reward_type_serde=RewardTypeWrapperSerde(FloatRewardTypeWrapper, FloatSerde()),
        obs_space_type_serde=StrIntTupleSerde(),
        action_space_type_serde=StrIntTupleSerde(),
        policy_factory=policy_factory,
        value_net_factory=value_net_factory,
        metrics_logger=ExampleLogger(),
        config=config,
    )
    learner.learn()
