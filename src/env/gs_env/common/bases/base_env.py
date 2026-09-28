from abc import ABC, abstractmethod
from typing import Any, Final

import torch


class BaseEnv(ABC):
    """Core simulator/task with a minimal, tensor-only API."""

    def __init__(self, device: torch.device) -> None:
        self.device: Final[torch.device] = device

    def reset(self) -> None:
        envs_idx = torch.IntTensor(range(self.num_envs))
        self.reset_idx(envs_idx=envs_idx.to(self.device))

    @abstractmethod
    def reset_idx(self, envs_idx: torch.IntTensor) -> None: ...

    @abstractmethod
    def apply_action(self, action: torch.Tensor) -> None: ...

    @abstractmethod
    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_extra_infos(self) -> dict[str, Any]: ...

    @abstractmethod
    def get_terminated(self) -> torch.Tensor: ...

    @abstractmethod
    def get_truncated(self) -> torch.Tensor: ...

    @abstractmethod
    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    @property
    @abstractmethod
    def num_envs(self) -> int: ...
