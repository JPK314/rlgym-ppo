"""
File: batched_agent_manager.py
Author: Matthew Allen and Jonathan Keegan

Description:
    A class to manage the multi-processed agents interacting with instances of the environment. This class is responsible
    for spawning and closing the individual processes, interacting with them through their respective pipes, and organizing
    the trajectories from each instance of the environment.
"""

import multiprocessing as mp
import pickle
import selectors
import socket
import time
from typing import Callable, Dict, Generic

import numpy as np
import torch
from numpy import frombuffer

from rlgym_ppo.api import ActorCriticManager, PPOPolicy, TypeSerde, ValueNet
from rlgym_ppo.batched_agents import BatchedTrajectory, Trajectory, comm_consts
from rlgym_ppo.batched_agents.batched_agent import batched_agent_process
from rlgym_ppo.batched_agents.comm_consts import PACKET_MAX_SIZE

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


from typing import Any, List, Optional, Tuple

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


class BatchedAgentManager(
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
        build_env_fn: Callable[
            [],
            RLGym[
                AgentID,
                ObsType,
                ActionType,
                EngineActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ],
        ],
        actor_critic_manager: ActorCriticManager,
        agent_id_serde: TypeSerde[AgentID],
        action_type_serde: TypeSerde[ActionType],
        obs_type_serde: TypeSerde[ObsType],
        reward_type_serde: TypeSerde[RewardType],
        obs_space_type_serde: TypeSerde[ObsSpaceType],
        action_space_type_serde: TypeSerde[ActionSpaceType],
        min_inference_size=8,
        seed=123,
        steps_per_obs_stats_increment=5,
        recalculate_agent_id_every_step=False,
    ):
        self.seed = seed
        self.processes: List[Tuple[Any, Any, Any, np.ndarray]] = []
        self.selector = selectors.DefaultSelector()

        self.build_env_fn = build_env_fn

        self.agent_id_serde = agent_id_serde
        self.action_type_serde = action_type_serde
        self.obs_type_serde = obs_type_serde
        self.reward_type_serde = reward_type_serde
        self.action_space_type_serde = action_space_type_serde
        self.obs_space_type_serde = obs_space_type_serde

        self.actor_critic_manager = actor_critic_manager

        self.current_obs: List[Dict[AgentID, ObsType]] = []
        self.current_actions: List[Dict[AgentID, ActionType]] = []
        self.current_log_probs: List[Dict[AgentID, torch.Tensor]] = []

        self.current_pids: List[int] = []
        self.average_reward = None
        self.cumulative_timesteps = 0
        self.min_inference_size = min_inference_size

        self.steps_per_obs_stats_increment = steps_per_obs_stats_increment
        self.steps_since_obs_stats_update = 0

        self.trajectory_map: List[BatchedTrajectory] = []
        self.prev_time = 0
        self.completed_trajectories: List[BatchedTrajectory] = []

        self.recalculate_agent_id_every_step = recalculate_agent_id_every_step

        self.n_procs = 0

        self.packed_actions_header = comm_consts.pack_header(
            comm_consts.POLICY_ACTIONS_HEADER
        )

    def collect_timesteps(self, n) -> Tuple[
        List[Trajectory[AgentID, ActionType, ObsType, RewardType]],
        List[np.ndarray],
        int,
        float,
    ]:
        """
        Collect a specified number of timesteps from the environment.

        :param n: Number of timesteps to collect.
        :return: A tuple containing the collected timesteps and additional information.
                - A list of trajectories to be used for learning
                - A list of metrics
                - Number of timesteps collected
                - Time spent inside this method
        """

        t1 = time.perf_counter()
        trajectories: List[Trajectory] = []

        n_collected = 0
        n_procs = max(1, len(self.processes))
        n_obs_per_inference = min(self.min_inference_size, n_procs)
        collected_metrics = []
        # Collect n timesteps.
        while n_collected < n:
            # Send actions for the current observations and collect new states to act on. Note that the next states
            # will not necessarily be from the environments that we just sent actions to. Whatever timestep data happens
            # to be lying around in the buffer will be collected and used in the next inference step.

            self._send_actions()
            (
                collected_metrics_this_pass,
                n_collected_this_pass,
            ) = self._collect_responses(n_obs_per_inference)
            n_collected += n_collected_this_pass
            collected_metrics += collected_metrics_this_pass

        # Organize our new timesteps into the appropriate lists.
        for proc_id, batched_trajectory in enumerate(self.trajectory_map):
            batched_trajectory.add_last_obs(self.current_obs[proc_id])
            self.completed_trajectories.append(batched_trajectory)
            self.trajectory_map[proc_id] = BatchedTrajectory(
                batched_trajectory.agent_trajectories.keys()
            )

        for batched_trajectory in self.completed_trajectories:
            trajectories += [
                trajectory
                for trajectory in batched_trajectory.get_all()
                if not self.actor_critic_manager.should_discard(trajectory.agent_id)
            ]

        self.cumulative_timesteps += n_collected
        t2 = time.perf_counter()
        self.completed_trajectories = []

        return (
            trajectories,
            collected_metrics,
            n_collected,
            t2 - t1,
        )

    @torch.no_grad()
    def _send_actions(self):
        """
        Send actions to environment processes based on current observations.
        """
        if len(self.current_pids) == 0:
            return

        obs_dict_list: List[Dict[AgentID, ObsType]] = []
        pids: List[int] = []
        for proc_id in self.current_pids:
            o = self.current_obs[proc_id]
            if o is None:
                continue

            obs_dict_list.append(o)
            pids.append(proc_id)

        if not obs_dict_list:
            return

        # Tricky bookkeeping. We can only assume agent ids are distinct within a proc_id
        obs_list: List[Tuple[AgentID, ObsType]] = []
        index_list: List[Tuple[int, AgentID]] = []
        for pid_idx, obs_dict in enumerate(obs_dict_list):
            for agent_id, obs in obs_dict.items():
                obs_list.append((agent_id, obs))
                index_list.append((pid_idx, agent_id))

        actions, log_probs = self.policy.get_action(obs_list)

        # convert back list of dicts
        action_dict_list: List[Dict[AgentID, ActionType]] = [{} for _ in pids]
        log_prob_dict_list: List[Dict[AgentID, torch.Tensor]] = [{} for _ in pids]
        for idx, action in enumerate(actions):
            (pid_idx, agent_id) = index_list[idx]
            action_dict_list[pid_idx][agent_id] = action
            log_prob_dict_list[pid_idx][agent_id] = log_probs[idx]

        for pid_idx, proc_id in enumerate(pids):
            process, parent_end, child_endpoint, shm_view = self.processes[proc_id]
            pid_actions = action_dict_list[pid_idx]
            actions_bytes = self.packed_actions_header
            for agent_id, action in pid_actions.items():
                actions_bytes += comm_consts.pack_bytes(
                    self.agent_id_serde.to_bytes(agent_id)
                )
                actions_bytes += comm_consts.pack_bytes(
                    self.action_type_serde.to_bytes(action)
                )
            assert (
                len(actions_bytes) < PACKET_MAX_SIZE
            ), "ACTIONS BYTES TOO LARGE TO SEND VIA SOCKET"
            # TODO: is it worth using shm? Kind of concerned about synchronicity and the upside seems small (test speed?)
            parent_end.sendto(actions_bytes, child_endpoint)
            self.current_actions[proc_id] = action_dict_list[pid_idx]
            self.current_log_probs[proc_id] = log_prob_dict_list[pid_idx]

        self.current_pids = []

    def _collect_responses(self, n_obs_per_inference):
        """
        Collect responses from environment processes and update trajectory data.
        :return: Number of responses collected.
        """

        n_collected = 0
        collected_metrics = []

        while n_collected < n_obs_per_inference:
            for key, event in self.selector.select():
                if not (event & selectors.EVENT_READ):
                    continue

                parent_end, fd, events, proc_id = key
                process, parent_end, child_endpoint, shm_view = self.processes[proc_id]
                n_collected += self._collect_response(
                    proc_id, parent_end, shm_view, collected_metrics
                )
                # print(n_collected, "|", len(collected_metrics))

        return collected_metrics, n_collected

    def _collect_response(
        self,
        proc_id: int,
        parent_end,
        shm_view: np.ndarray,
        collected_metrics: List[np.ndarray],
    ):
        socket_data = parent_end.recv(PACKET_MAX_SIZE)
        (header, _) = comm_consts.unpack_header(socket_data)
        if header[0] != comm_consts.ENV_STEP_DATA_HEADER[0]:
            return 0

        offset = 0
        new_episode_agent_ids: List[AgentID] = []
        obs_dict: Dict[AgentID, ObsType] = {}
        new_episode_obs_dict: Dict[AgentID, ObsType] = {}
        rew_dict: Dict[AgentID, RewardType] = {}
        terminated_dict: Dict[AgentID, bool] = {}
        new_episode_terminated_dict: Dict[AgentID, bool] = {}
        truncated_dict: Dict[AgentID, bool] = {}
        new_episode_truncated_dict: Dict[AgentID, bool] = {}
        (new_episode, offset) = comm_consts.retrieve_bool(shm_view, offset)
        if new_episode:
            # TODO: raise error if prev_n_agents = 0? Somewhere?
            (prev_n_agents, offset) = comm_consts.retrieve_int(shm_view, offset)
            n_collected = prev_n_agents
            for _ in range(prev_n_agents):
                (agent_id_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                agent_id = self.agent_id_serde.from_bytes(agent_id_bytes)
                (obs_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                obs_dict[agent_id] = self.obs_type_serde.from_bytes(obs_bytes)
                (rew_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                rew_dict[agent_id] = self.reward_type_serde.from_bytes(rew_bytes)
                (terminated, offset) = comm_consts.retrieve_bool(shm_view, offset)
                terminated_dict[agent_id] = terminated
                (truncated, offset) = comm_consts.retrieve_bool(shm_view, offset)
                truncated_dict[agent_id] = truncated

            (n_agents, offset) = comm_consts.retrieve_int(shm_view, offset)
            for _ in range(n_agents):
                (agent_id_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                agent_id = self.agent_id_serde.from_bytes(agent_id_bytes)
                new_episode_agent_ids.append(agent_id)
                (obs_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                new_episode_obs_dict[agent_id] = self.obs_type_serde.from_bytes(
                    obs_bytes
                )
                new_episode_terminated_dict[agent_id] = False
                new_episode_truncated_dict[agent_id] = False
        else:
            (n_agents, offset) = comm_consts.retrieve_int(shm_view, offset)
            n_collected = n_agents
            for _ in range(n_agents):
                (agent_id_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                agent_id = self.agent_id_serde.from_bytes(agent_id_bytes)
                (obs_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                obs_dict[agent_id] = self.obs_type_serde.from_bytes(obs_bytes)
                (rew_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                rew_dict[agent_id] = self.reward_type_serde.from_bytes(rew_bytes)
                (terminated, offset) = comm_consts.retrieve_bool(shm_view, offset)
                terminated_dict[agent_id] = terminated
                (truncated, offset) = comm_consts.retrieve_bool(shm_view, offset)
                truncated_dict[agent_id] = truncated

        (metrics_len, offset) = comm_consts.retrieve_int(shm_view, offset)

        if metrics_len != 0:
            metrics_shape = []
            for _ in range(metrics_len):
                (shape_dim, offset) = comm_consts.retrieve_int(shm_view, offset)
                metrics_shape.append(shape_dim)
            (metrics_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
            metrics = np.frombuffer(metrics_bytes, dtype=np.float32).reshape(
                metrics_shape
            )
        else:
            metrics_shape = (0,)
            metrics = np.empty(shape=metrics_shape, dtype=np.float32)

        collected_metrics.append(metrics.copy())

        if proc_id not in self.current_pids:
            self.current_pids.append(proc_id)

        if new_episode:
            self.trajectory_map[proc_id].add_timesteps(
                self.current_obs[proc_id],
                self.current_actions[proc_id],
                self.current_log_probs[proc_id],
                rew_dict,
                terminated_dict,
                truncated_dict,
            )
            self.trajectory_map[proc_id].add_last_obs(obs_dict)
            self.completed_trajectories.append(self.trajectory_map[proc_id])
            self.trajectory_map[proc_id] = BatchedTrajectory(new_episode_agent_ids)
        else:
            self.trajectory_map[proc_id].add_timesteps(
                self.current_obs[proc_id],
                self.current_actions[proc_id],
                self.current_log_probs[proc_id],
                rew_dict,
                terminated_dict,
                truncated_dict,
            )

        self.current_obs[proc_id] = obs_dict

        return n_collected

    def _get_initial_obs(self):
        """
        Retrieve initial observations from environment processes.
        :return: None.
        """

        self.current_pids = []
        for proc_id, proc_package in enumerate(self.processes):
            process, parent_end, child_endpoint, shm_view = proc_package

            socket_data = parent_end.recv(PACKET_MAX_SIZE)
            (header, _) = comm_consts.unpack_header(socket_data)
            if header == comm_consts.ENV_RESET_STATE_HEADER:
                self.current_pids.append(proc_id)
                offset = 0
                (n_agents, offset) = comm_consts.retrieve_int(shm_view, offset)
                obs_dict = {}
                new_episode_agent_ids: List[AgentID] = []
                for _ in range(n_agents):
                    (agent_id_bytes, offset) = comm_consts.retrieve_bytes(
                        shm_view, offset
                    )
                    agent_id = self.agent_id_serde.from_bytes(agent_id_bytes)
                    new_episode_agent_ids.append(agent_id)
                    (obs_bytes, offset) = comm_consts.retrieve_bytes(shm_view, offset)
                    obs = self.obs_type_serde.from_bytes(obs_bytes)
                    obs_dict[agent_id] = obs
                self.trajectory_map[proc_id] = BatchedTrajectory(new_episode_agent_ids)
                self.current_obs[proc_id] = obs_dict

    def _get_space_types(self) -> Tuple[ObsSpaceType, ActionSpaceType]:
        """
        Retrieve environment observation and action space shapes from one of the connected environment processes.
        :return: A tuple containing observation space type and action space type.
        """
        process, parent_end, child_endpoint, shm_view = self.processes[0]
        request_msg = comm_consts.pack_header(comm_consts.ENV_SHAPES_HEADER)
        parent_end.sendto(request_msg, child_endpoint)

        while True:
            socket_data = parent_end.recv(PACKET_MAX_SIZE)
            (header, socket_offset) = comm_consts.unpack_header(socket_data)
            if header == comm_consts.ENV_SHAPES_HEADER:
                (obs_space_bytes, socket_offset) = (
                    comm_consts.retrieve_bytes_from_message(socket_data, socket_offset)
                )
                obs_space = self.obs_space_type_serde.from_bytes(obs_space_bytes)
                (action_space_bytes, socket_offset) = (
                    comm_consts.retrieve_bytes_from_message(socket_data, socket_offset)
                )
                action_space = self.action_space_type_serde.from_bytes(
                    action_space_bytes
                )
                break

        return (obs_space, action_space)

    def init_processes(
        self,
        policy_factory: Callable[
            [ObsSpaceType, ActionSpaceType, str],
            PPOPolicy[AgentID, ObsType, ActionType],
        ],
        value_net_factory: Callable[[ObsSpaceType, str], ValueNet[AgentID, ObsType]],
        device: str,
        n_processes: int,
        collect_metrics_fn=None,
        spawn_delay=None,
        render=False,
        render_delay: Optional[float] = None,
        shm_buffer_size=8192,
    ) -> Tuple[PPOPolicy[AgentID, ObsType, ActionType], ValueNet[AgentID, ObsType]]:
        """
        Initialize and spawn environment processes.
        :param n_processes: Number of processes to spawn.
        :param collect_metrics_fn: A user-defined function that the environment processes will use to collect metrics
               about the environment at each timestep.
        :param spawn_delay: Delay between spawning environment instances. Defaults to None.
        :param render: Whether an environment should be rendered while collecting timesteps.
        :param render_delay: A period in seconds to delay a process between frames while rendering.
        :param shm_buffer_size: Size, in bytes, of shared memory allocated to each process. Defaults to 8192.
        :return: A tuple containing observation space type and action space type.
        """

        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = mp.get_context(start_method)
        self.n_procs = n_processes

        import multiprocessing.sharedctypes

        self.shm_size = shm_buffer_size
        self.shm_buffer = multiprocessing.sharedctypes.RawArray(
            "b", n_processes * self.shm_size
        )
        self.processes = [None for i in range(n_processes)]
        self.trajectory_map = [None for i in range(n_processes)]

        self.current_obs: List[Optional[Dict[AgentID, ObsType]]] = [
            None for i in range(n_processes)
        ]
        self.current_actions: List[Optional[Dict[AgentID, ActionType]]] = [
            None for i in range(n_processes)
        ]
        self.current_log_probs: List[Optional[Dict[AgentID, torch.Tensor]]] = [
            None for i in range(n_processes)
        ]

        # Spawn child processes
        for proc_id in tqdm(range(n_processes)):
            render_this_proc = proc_id == 0 and render

            # Create socket to communicate with child
            parent_end = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            parent_end.bind(("127.0.0.1", 0))

            shm_offset = proc_id * self.shm_size

            process = context.Process(
                target=batched_agent_process,
                args=(
                    proc_id,
                    parent_end.getsockname(),
                    self.agent_id_serde,
                    self.action_type_serde,
                    self.obs_type_serde,
                    self.reward_type_serde,
                    self.action_space_type_serde,
                    self.obs_space_type_serde,
                    self.shm_buffer,
                    shm_offset,
                    self.shm_size,
                    self.seed + proc_id,
                    render_this_proc,
                    render_delay,
                    self.recalculate_agent_id_every_step,
                ),
            )
            process.start()

            shm_view = frombuffer(
                buffer=self.shm_buffer,
                dtype=np.byte,
                offset=shm_offset,
                count=self.shm_size,
            )

            self.processes[proc_id] = (process, parent_end, None, shm_view)

            self.selector.register(parent_end, selectors.EVENT_READ, proc_id)

        # Initialize child processes
        for proc_id in range(n_processes):
            process, parent_end, _, shm_view = self.processes[proc_id]

            # Get child endpoint
            _, child_endpoint = parent_end.recvfrom(1)

            if spawn_delay is not None:
                time.sleep(spawn_delay)

            p = pickle.dumps(
                ("initialization_data", self.build_env_fn, collect_metrics_fn)
            )
            parent_end.sendto(p, child_endpoint)

            self.processes[proc_id] = (process, parent_end, child_endpoint, shm_view)

        self._get_initial_obs()
        (obs_space, action_space) = self._get_space_types()
        self.policy = policy_factory(obs_space, action_space, device)
        return (self.policy, value_net_factory(obs_space, device))

    def cleanup(self):
        """
        Clean up resources and terminate processes.
        """
        import traceback

        for proc_id, proc_package in enumerate(self.processes):
            process, parent_end, child_endpoint, shm_view = proc_package

            try:
                parent_end.sendto(
                    comm_consts.pack_header(comm_consts.STOP_MESSAGE_HEADER),
                    child_endpoint,
                )
            except Exception:
                print("Unable to join process")
                traceback.print_exc()
                print("Failed to send stop signal to child process!")
                traceback.print_exc()

            try:
                process.join()
            except Exception:
                print("Unable to join process")
                traceback.print_exc()

            try:
                parent_end.close()
            except Exception:
                print("Unable to close parent connection")
                traceback.print_exc()
