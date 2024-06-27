from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Type

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)


# TODO: add obs builder wrapper that includes standardization
class ObsTypeSerde(Generic[ObsType]):
    @staticmethod
    @abstractmethod
    def to_bytes(obs: ObsType) -> bytes:
        """
        Function to convert obs to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == obs.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_bytes(bytes: bytes) -> ObsType:
        """
        Function to convert bytes to ObsType, for passing between batched agent and the agent manager.
        :return: ObsType obs such that from_bytes(to_bytes(obs)) == obs.
        """
        raise NotImplementedError


class ActionTypeSerde(Generic[ActionType]):
    @staticmethod
    @abstractmethod
    def to_bytes(action: ActionType) -> bytes:
        """
        Function to convert action to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == action.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_bytes(bytes: bytes) -> ActionType:
        """
        Function to convert bytes to ActionType, for passing between batched agent and the agent manager.
        :return: ActionType action such that from_bytes(to_bytes(action)) == action.
        """
        raise NotImplementedError


class RewardTypeSerde(Generic[RewardType]):
    @staticmethod
    @abstractmethod
    def to_bytes(reward: RewardType) -> bytes:
        """
        Function to convert reward to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == reward.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_bytes(bytes: bytes) -> RewardType:
        """
        Function to convert bytes to RewardType, for passing between batched agent and the agent manager.
        :return: RewardType reward such that from_bytes(to_bytes(reward)) == reward.
        """
        raise NotImplementedError


class AgentIDSerde(Generic[AgentID]):
    @staticmethod
    @abstractmethod
    def to_bytes(agent_id: AgentID) -> bytes:
        """
        Function to convert agent_id to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == agent_id.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_bytes(bytes: bytes) -> AgentID:
        """
        Function to convert bytes to AgentID, for passing between batched agent and the agent manager.
        :return: AgentID agent_id such that from_bytes(to_bytes(agent_id)) == agent_id.
        """
        raise NotImplementedError


class ActionSpaceTypeSerde(Generic[ActionSpaceType]):
    @staticmethod
    @abstractmethod
    def to_bytes(action_space: ActionSpaceType) -> bytes:
        """
        Function to convert action_space to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == action_space.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_bytes(bytes: bytes) -> ActionSpaceType:
        """
        Function to convert bytes to ActionSpaceType, for passing between batched agent and the agent manager.
        :return: ActionSpaceType action_space such that from_bytes(to_bytes(action_space)) == action_space.
        """
        raise NotImplementedError


class ObsSpaceTypeSerde(Generic[ObsSpaceType]):
    @staticmethod
    @abstractmethod
    def to_bytes(obs_space: ObsSpaceType) -> bytes:
        """
        Function to convert obs_space to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == obs_Space.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_bytes(bytes: bytes) -> ObsSpaceType:
        """
        Function to convert bytes to ObsSpaceType, for passing between batched agent and the agent manager.
        :return: ObsSpaceType obs_Space such that from_bytes(to_bytes(obs_Space)) == obs_Space.
        """
        raise NotImplementedError
