from pathlib import Path

from gs_agent.algos.config.schema import BCArgs, LearningRateType, PPOArgs
from gs_agent.modules.config.registry import (
    TEMPORAL_ENCODER_BC,
    TEMPORAL_ENCODER_TACTILE_MAP_XARM,
    TEMPORAL_ENCODER_TACTILE_MAP_XARM_NO_HIST,
)

# Teacher PPO config: tactile map encoder with xArm. Actor has no history input,
# critic gets privileged history.
PPO_HAND_IMITATOR_TACTILE_MAP_XARM = PPOArgs(
    policy_backbone=TEMPORAL_ENCODER_TACTILE_MAP_XARM_NO_HIST,
    critic_backbone=TEMPORAL_ENCODER_TACTILE_MAP_XARM,
    lr=1e-4,
    lr_type=LearningRateType.ADAPTIVE,
    lr_adaptive_factor=1.5,
    lr_min=1e-5,
    lr_max=1e-2,
    value_lr=None,
    gamma=0.98,
    gae_lambda=0.95,
    clip_ratio=0.1,
    value_loss_coef=1.0,
    entropy_coef=0.0001,
    max_grad_norm=1.0,
    target_kl=0.01,
    num_epochs=5,
    num_mini_batches=4,
    rollout_length=32,
)

# BC config for single hand retargeting distillation (teacher -> student)
BC_SINGLE_HAND_RETARGETING_TEMPORAL = BCArgs(
    policy_backbone=TEMPORAL_ENCODER_BC,  # Student architecture (no privileged obs)
    teacher_backbone=TEMPORAL_ENCODER_TACTILE_MAP_XARM_NO_HIST,  # Teacher architecture (matches PPO training)
    lr=3e-4,
    teacher_path=Path(""),  # Will be set dynamically
    teacher_config_path=None,  # Will be set dynamically
    num_epochs=10,
    batch_size=256,
    rollout_length=24,
    max_buffer_size=24,
    max_num_batches=32,
    max_grad_norm=1.0,
)
