from abc import abstractmethod
from typing import Any, Dict, Generic, List

import numpy as np
from numpy import dtype
from rlgym.api import ActionType, AgentID, ObsType, RewardType, StateType
from wandb.wandb_run import Run

from rlgym_ppo import LearnerMetrics
from rlgym_ppo.ppo import PPOMetrics
from rlgym_ppo.standard_impl import NumpyDynamicShapeSerde
from rlgym_ppo.util import comm_consts


# TODO: update metrics logger to collect trajectory metrics
# TODO: docs
class MetricsLogger(Generic[StateType, AgentID, ActionType, ObsType, RewardType]):
    def __init__(self, dtype: dtype):
        self.numpy_serde = NumpyDynamicShapeSerde(dtype=dtype)
        self.state_metrics: List[np.ndarray] = []

    def encode_state_metrics(self, state: StateType) -> bytes:
        metrics_arrays = self.collect_state_metrics(state)
        byts = bytes()
        byts += comm_consts.pack_int(len(metrics_arrays))
        for arr in metrics_arrays:
            arr_bytes = self.numpy_serde.to_bytes(arr)
            byts += comm_consts.pack_int(len(arr_bytes))
            bytes += arr_bytes
        return byts

    def decode_state_metrics(self, byts: bytes) -> List[np.ndarray]:
        offset = 0
        (metrics_arrays_len, offset) = comm_consts.unpack_int(byts, 0)
        metrics_arrays = []
        for _ in range(metrics_arrays_len):
            (len_arr_bytes, offset) = comm_consts.unpack_int(byts, offset)
            stop = offset + len_arr_bytes
            metrics_arrays.append(self.numpy_serde.from_bytes(byts[offset:stop]))
            offset = stop

        return metrics_arrays

    @abstractmethod
    def collect_state_metrics(self, state: StateType) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def collect_ppo_metrics(self, ppo_metrics: PPOMetrics) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def collect_learner_metrics(
        self, learner_metrics: LearnerMetrics
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def report_metrics(
        self,
        collected_state_metrics: List[List[np.ndarray]],
        ppo_metrics: Dict[str, Any],
        learner_metrics: Dict[str, Any],
        wandb_run: Run,
    ):
        raise NotImplementedError
