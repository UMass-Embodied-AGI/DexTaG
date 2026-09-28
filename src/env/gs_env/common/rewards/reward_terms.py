from abc import ABC, abstractmethod

import torch


class RewardTerm(ABC):
    """
    A reward term that *declares* the names of the tensors it needs.
    Nothing is inferred – you can read `.required_keys` and know exactly
    what the caller must supply.

    Return:
        A tensor of shape (B,) where B is the batch size.
    """

    #: Ordered list of keys the term expects to find in the `state` dict.
    required_keys: tuple[str, ...] = ()

    def __init__(self, scale: float = 1.0, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.scale = float(scale)

    def __call__(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = [state[k] for k in self.required_keys]
        return self.scale * self._compute(*tensors)

    @abstractmethod
    def _compute(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Actual reward computation.  Signature is fixed by `required_keys`."""

