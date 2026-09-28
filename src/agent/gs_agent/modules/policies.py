import torch
from torch import nn
from torch.distributions import Normal

from gs_agent.bases.network_backbone import NetworkBackbone
from gs_agent.bases.policy import Policy


# === Gaussian Policy === #
class GaussianPolicy(Policy):
    def __init__(
        self,
        policy_backbone: NetworkBackbone,
        action_dim: int,
        initial_log_std: float = 0.0,
    ) -> None:
        super().__init__(policy_backbone, action_dim)
        self.mu = nn.Linear(self.backbone.output_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * initial_log_std)
        Normal.set_default_validate_args(False)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the policy.

        Args:
            obs: Observation tensor [B, obs_dim].
            deterministic: Whether to use deterministic action.

        Returns:
            tuple: (action, log_prob)
        """
        dist = self.dist_from_obs(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        # Sum per-dimension log-probs
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def forward_with_dist_params(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the policy with distribution parameters.

        Args:
            obs: Observation tensor [B, obs_dim].
            deterministic: Whether to use deterministic action.

        Returns:
            tuple: (action, log_prob, mu, sigma)
        """
        dist = self.dist_from_obs(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        # Sum per-dimension log-probs
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # Extract mu and sigma from the distribution
        mu = dist.mean
        sigma = dist.stddev

        return action, log_prob, mu, sigma

    def dist_from_obs(self, obs: torch.Tensor) -> Normal:
        feature = self.backbone(obs)
        action_mu = self.mu(feature)
        action_std = torch.exp(self.log_std)
        return Normal(action_mu, action_std.expand_as(action_mu))

    def evaluate_log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Evaluate the log probability of the action.

        Args:
            obs: Observation tensor [B, obs_dim].
            act: Action tensor.

        Returns:
            Log probability of the action.
        """
        dist = self.dist_from_obs(obs)
        return dist.log_prob(act).sum(-1, keepdim=True)

    def entropy_on(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the entropy of the action distribution.

        Args:
            obs: Observation tensor [B, obs_dim].

        Returns:
            Entropy of the action distribution.
        """
        dist = self.dist_from_obs(obs)
        return dist.entropy().sum(-1)

    @property
    def action_std(self) -> torch.Tensor:
        return self.log_std.exp()


class DeterministicPolicy(Policy):
    def __init__(self, policy_backbone: NetworkBackbone, action_dim: int) -> None:
        super().__init__(policy_backbone, action_dim)
        self.mu = nn.Linear(self.backbone.output_dim, self.action_dim)
        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the policy."""
        feature = self.backbone(obs)
        action = self.mu(feature)
        return action

