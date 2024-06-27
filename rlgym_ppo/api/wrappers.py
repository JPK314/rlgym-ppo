from abc import abstractmethod
from typing import Generic

from rlgym.api import RewardType
from torch import Tensor


class RewardTypeWrapper(Generic[RewardType]):
    def __init__(self, reward: RewardType):
        super().__init__()
        self.reward = reward

    def __getattr__(self, name):
        return getattr(self.reward, name)

    @abstractmethod
    def as_tensor(self) -> Tensor:
        raise NotImplementedError
