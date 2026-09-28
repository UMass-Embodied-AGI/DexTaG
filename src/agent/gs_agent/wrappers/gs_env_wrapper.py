from typing import Any, Final, TypeVar

import torch

from gs_agent.bases.env_wrapper import BaseEnvWrapper

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")

TGSEnv = TypeVar("TGSEnv")


class GenesisEnvWrapper(BaseEnvWrapper):
    def __init__(
        self,
        env: object,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(env, device)
        self.env.reset()
        self._curr_obs = self.env.get_observations()

    # ---------------------------
    # BaseEnvWrapper API (batch)
    # ---------------------------
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self.env.reset()
        self._curr_obs = self.env.get_observations()
        return self._curr_obs, self.env.get_extra_infos()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # apply action
        self.env.apply_action(action)
        # get terminated
        terminated = self.env.get_terminated()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)
        # get truncated
        truncated = self.env.get_truncated()
        if truncated.dim() == 1:
            truncated = truncated.unsqueeze(-1)
        # get reward
        reward, reward_terms = self.env.get_reward()
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        # get extra infos
        extra_infos = self.env.get_extra_infos()
        extra_infos["reward_terms"] = reward_terms
        # reset terminated envs (the env never truncates)
        done_idx = terminated.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.env.reset_idx(done_idx)
        # get observations
        next_actor_obs, next_critic_obs = self.env.get_observations()
        return next_actor_obs, next_critic_obs, reward, terminated, truncated, extra_infos

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get default (student) observations. Updates buffers and temporal state."""
        return self.env.get_observations()

    def get_teacher_observations(self, teacher_args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get teacher observations without shifting history buffers.

        Advances the teacher obs-delay queue when observation delay is enabled,
        so call it exactly once per step, after get_observations().
        """
        return self.env.get_teacher_observations(teacher_args=teacher_args)

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def actor_obs_dim(self) -> int:
        return self.env.actor_obs_dim

    @property
    def critic_obs_dim(self) -> int:
        return self.env.critic_obs_dim

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def close(self) -> None:
        self.env.close()

    def render(self) -> None:
        self.env.render()
