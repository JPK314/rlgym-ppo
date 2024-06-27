from abc import abstractmethod
from typing import Generic

from rlgym.api import AgentID


class ActorCriticManager(Generic[AgentID]):

    @abstractmethod
    def should_discard(agent_id: AgentID) -> bool:
        """
        :agent_id: AgentID instance used to determine whether the associated trajectory should be discarded.
        :return: true if trajectory should be discarded, false otherwise.
        """
        raise NotImplementedError
