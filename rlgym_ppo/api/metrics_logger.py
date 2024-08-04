from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from wandb.wandb_run import Run

from .agent import AgentData
from .typing import AgentData, StateMetrics


# TODO: docs
class MetricsLogger(
    Generic[
        StateMetrics,
        AgentData,
    ]
):
    @classmethod
    @abstractmethod
    def collect_agent_metrics(cls, data: AgentData) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def report_metrics(
        cls,
        collected_state_metrics: List[StateMetrics],
        agent_metrics: Dict[str, Any],
        wandb_run: Run,
    ):
        raise NotImplementedError
