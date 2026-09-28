import abc

import torch


class BaseGymRobot(abc.ABC):
    """
    Abstract base class for robots in a gym-like environment.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        """
        Reset the robot.

        Args:
            envs_idx (torch.Tensor, optional): Indices of environments in the batch to reset.
        """

    @abc.abstractmethod
    def apply_action(self, action: torch.Tensor) -> None:
        """
        Apply an action to the robot.

        Args:
            action (torch.Tensor): Action to apply to the robot.
        """
