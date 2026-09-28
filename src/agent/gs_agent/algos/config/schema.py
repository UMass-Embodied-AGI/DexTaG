from pathlib import Path

from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat

from gs_agent.modules.config.schema import MLPConfig, NetworkBackboneConfig


class LearningRateType(GenesisEnum):
    FIXED = "FIXED"
    ADAPTIVE = "ADAPTIVE"


class PPOArgs(BaseModel):
    """Configuration for PPO algorithm."""

    model_config = genesis_pydantic_config(frozen=True)

    # Network architecture
    policy_backbone: NetworkBackboneConfig = MLPConfig()
    critic_backbone: NetworkBackboneConfig = MLPConfig()

    # Learning rates
    lr: PositiveFloat = 3e-4
    """Policy learning rate"""

    # Value function learning rate
    value_lr: PositiveFloat | None = None
    """None means use the same learning rate as the policy"""

    # Adaptive learning rate
    lr_type: LearningRateType = LearningRateType.FIXED
    lr_adaptive_factor: PositiveFloat = 1.5
    lr_min: PositiveFloat = 1e-5
    lr_max: PositiveFloat = 1e-2

    # Discount and GAE
    gamma: PositiveFloat = Field(default=0.98, ge=0, le=1)
    gae_lambda: PositiveFloat = Field(default=0.95, ge=0, le=1)

    # PPO specific
    clip_ratio: PositiveFloat = 0.2
    value_loss_coef: PositiveFloat = 1.0
    entropy_coef: NonNegativeFloat = 0.0
    max_grad_norm: PositiveFloat = 1.0
    target_kl: PositiveFloat = 0.02

    # Training
    num_epochs: NonNegativeInt = 10
    num_mini_batches: NonNegativeInt = 4
    rollout_length: NonNegativeInt = 1000

    # Kept so older configs load; nothing in this repo warm-starts PPO from a
    # pretrained actor.
    initial_log_std: float = 0.0
    """Initial value of the actor's log_std parameter. Default 0.0 gives std=1.0;
    log(0.5) ≈ -0.693 gives std=0.5."""

    critic_warmup_iters: NonNegativeInt = 0
    """Number of initial iterations during which the actor optimizer step is
    skipped (critic-only updates)."""


class BCArgs(BaseModel):
    """Configuration for BC algorithm."""

    model_config = genesis_pydantic_config(frozen=True)

    # Network architecture
    policy_backbone: NetworkBackboneConfig = MLPConfig()
    teacher_backbone: NetworkBackboneConfig = MLPConfig()

    # Learning rates
    lr: PositiveFloat = 3e-4
    """Policy learning rate"""
    max_grad_norm: PositiveFloat = 1.0

    # Teacher path and config
    teacher_path: Path
    teacher_config_path: Path | None = None
    """Path to teacher environment config yaml file. If None, uses student obs dim."""

    # Training
    num_epochs: NonNegativeInt = 10
    batch_size: NonNegativeInt = 256
    rollout_length: NonNegativeInt = 1000
    max_buffer_size: NonNegativeInt = 1_000_000
    max_num_batches: NonNegativeInt = 4
