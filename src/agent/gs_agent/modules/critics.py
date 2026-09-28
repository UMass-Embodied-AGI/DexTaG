import torch
from torch import nn

from gs_agent.bases.critic import BaseStateValueFunction
from gs_agent.bases.network_backbone import NetworkBackbone


class StateValueFunction(BaseStateValueFunction):
    """State value function V(s)."""

    def __init__(self, backbone: NetworkBackbone, output_dim: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim

        self.v_head = nn.Linear(self.backbone.output_dim, self.output_dim)
        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(obs)
        return self.v_head(feature)

