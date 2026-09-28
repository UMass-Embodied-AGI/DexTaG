from gs_agent.modules.config.schema import (
    ActivationType,
    TemporalEncoderConfig,
)

# ------------------------------------------------------------
# Temporal Encoder Configs
#
# The single-hand-retargeting pipeline uses exactly three backbones:
#   - teacher actor   : TEMPORAL_ENCODER_TACTILE_MAP_XARM_NO_HIST (no history input)
#   - teacher critic  : TEMPORAL_ENCODER_TACTILE_MAP_XARM         (with privileged history)
#   - student policy  : TEMPORAL_ENCODER_BC                       (deployment obs only)
# ------------------------------------------------------------

# Teacher critic: tactile-map encoder with privileged history, xArm proprio.
TEMPORAL_ENCODER_TACTILE_MAP_XARM = TemporalEncoderConfig(
    motion_dim=101,              # 20+20 (hand dof pos/vel) + 15+20 (5-tip pos+quat) + 3+4+3+3 (wrist) + 3+4+3+3 (object)
    proprio_dim=122,             # 106 (base) + 7 (arm_dof_pos) + 7 (arm_dof_vel) + 1 (table_contact) + 1 (last_reward, critic only)
    tactile_map_dim=768,         # 24 * 32
    tactile_map_h=24,
    tactile_map_w=32,
    tactile_map_latent_dim=128,
    add_target_tactile_map=True,
    history_dim=1184,            # 8 * 148
    future_dim=1616,             # 16 * 101
    motion_latent_dim=128,
    history_latent_dim=64,
    future_latent_dim=128,
    history_timesteps=8,
    history_per_step_dim=148,    # 121 (proprio) + 27 (action: 7 arm + 20 hand)
    future_timesteps=16,
    future_per_step_dim=101,
    mlp_hidden_dims=(512, 512, 256, 128),
    activation=ActivationType.SWISH,
    dropout_rate=0.1,
    use_layer_norm=True,
)

# Teacher actor: same tactile-map encoder but no history input (history_dim=0).
TEMPORAL_ENCODER_TACTILE_MAP_XARM_NO_HIST = TemporalEncoderConfig(
    motion_dim=101,
    proprio_dim=121,             # 106 (base) + 7 (arm_dof_pos) + 7 (arm_dof_vel) + 1 (table_contact)
    tactile_map_dim=768,         # 24 * 32
    tactile_map_h=24,
    tactile_map_w=32,
    tactile_map_latent_dim=128,
    add_target_tactile_map=False,
    history_dim=0,
    future_dim=1616,             # 16 * 101
    motion_latent_dim=128,
    history_latent_dim=64,
    future_latent_dim=128,
    history_timesteps=8,
    history_per_step_dim=148,    # 121 (proprio) + 27 (action: 7 arm + 20 hand)
    future_timesteps=16,
    future_per_step_dim=101,
    mlp_hidden_dims=(512, 512, 256, 128),
    activation=ActivationType.SWISH,
    dropout_rate=0.1,
    use_layer_norm=True,
)

# Student policy: deployment-only obs (no tactile map), wrist point cloud input.
TEMPORAL_ENCODER_BC = TemporalEncoderConfig(
    motion_dim=13,                # target_object_{pos(3), quat(4), vel(3), ang_vel(3)}
    proprio_dim=27,               # hand_dof_pos(20) + arm_dof_pos(7)
    tactile_map_dim=0,            # no tactile map for student
    history_dim=432,              # 8 * 54
    future_dim=208,               # 16 * 13
    pointcloud_dim=2048 * 3,      # 12288 (flattened point cloud: 2048 points × 3)
    pointcloud_num_points=2048,
    pointcloud_latent_dim=256,
    motion_latent_dim=64,
    history_latent_dim=128,
    future_latent_dim=128,
    history_timesteps=8,
    history_per_step_dim=54,      # 27 (action: 7 arm + 20 hand) + 27 (student proprio)
    future_timesteps=16,
    future_per_step_dim=13,       # matches TARGET_MOTION_TERMS_STUDENT
    mlp_hidden_dims=(512, 512, 256, 128),
    activation=ActivationType.SWISH,
    dropout_rate=0.1,
    use_layer_norm=True,
)
