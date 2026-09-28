from gs_env.sim.envs.config.schema import (
    EnvArgs,
    GenesisInitArgs,
    ObjectConfig,
    SingleHandRetargetingEnvArgs,
)
from gs_env.sim.robots.config.registry import RobotArgsRegistry
from gs_env.sim.scenes.config.registry import SceneArgsRegistry

# === Object registry ===
OBJECT_REGISTRY: dict[str, ObjectConfig] = {
    "marker_pen": ObjectConfig(
        trajectory_path="data/marker_pen/pickles_filtered",
        object_id="marker_pen_scanned",
        rotation_ignore_axis=[0, 1, 0],
        mass=0.03,
        perturbation_force_scale=20.0,
        perturbation_torque_scale=0.2,
        init_pos_noise_xy=0.03,
        init_rot_noise_deg=15.0,
        table_height_noise=0.015,
    ),
    "hammer": ObjectConfig(
        trajectory_path="data/hammer_mixed/pickles_filtered",
        object_id="hammer_scanned",
        mass=0.25,
        com_shift=[0.0, 0.04, 0.0],
        perturbation_force_scale=20.0,
        perturbation_torque_scale=0.2,
    ),
}

# === Reusable observation term lists ===
PROPRIO_TERMS_TACTILE_MAP = [
    "hand_dof_pos",        # 20
    "hand_dof_vel",        # 20
    "finger_link_pos",     # 15
    "finger_link_quat",    # 20
    "base_pos",            # 3
    "base_quat",           # 4
    "base_lin_vel",        # 3
    "base_ang_vel",        # 3
    "object_pos_rel",      # 3
    "fingertip_to_surface_dist",  # 5
    "object_quat",         # 4
    "object_lin_vel",      # 3
    "object_ang_vel",      # 3
]  # total = 106

PROPRIO_TERMS_TACTILE_MAP_XARM = PROPRIO_TERMS_TACTILE_MAP + [
    "arm_dof_pos",         # 7
    "arm_dof_vel",         # 7
    "table_contact_force_magnitude",  # 1
]  # total = 121

TARGET_MOTION_TERMS_TACTILE_MAP = [
    "target_hand_dof_pos",    # 20
    "target_hand_dof_vel",    # 20
    "target_mano_joint_pos",  # 15
    "target_mano_joint_quat", # 20
    "delta_wrist_pos",        # 3
    "target_wrist_quat",      # 4
    "target_wrist_vel",       # 3
    "target_wrist_ang_vel",   # 3
    "delta_object_pos",       # 3
    "target_object_quat",     # 4
    "target_object_vel",      # 3
    "target_object_ang_vel",  # 3
]  # total = 101

BASE_OBS_SCALES = {
    "hand_dof_pos": 1.0, "hand_dof_vel": 1.0,
    "finger_link_pos": 1.0, "finger_link_vel": 1.0, "finger_link_quat": 1.0,
    "base_pos": 1.0, "base_quat": 1.0, "base_lin_vel": 1.0, "base_ang_vel": 1.0,
    "object_pos_rel": 5.0, "fingertip_to_surface_dist": 10.0,
    "object_quat": 1.0, "object_lin_vel": 1.0, "object_ang_vel": 0.1, "object_to_finger_tips": 1.0,
    "arm_dof_pos": 1.0, "arm_dof_vel": 1.0, "dof_vel": 1.0,
    "target_hand_dof_pos": 1.0, "target_hand_dof_vel": 1.0,
    "target_mano_joint_pos": 1.0, "target_mano_joint_vel": 1.0, "target_mano_joint_quat": 1.0,
    "delta_wrist_pos": 10.0, "target_wrist_pos": 1.0, "target_wrist_quat": 1.0,
    "target_wrist_vel": 10.0, "target_wrist_ang_vel": 1.0,
    "delta_object_pos": 10.0, "target_object_pos": 1.0, "target_object_quat": 1.0,
    "target_object_vel": 10.0, "target_object_ang_vel": 1.0,
    "tactile_map_flat": 5.0, "bps": 10.0, "target_tactile_map_flat": 5.0,
    "action_proprio_history_flat": 1.0, "action_proprio_history_student_flat": 1.0,
    "target_action_flat": 1.0, "target_action_student_flat": 1.0,
    "wrist_pointcloud_flat": 1.0,
    "table_contact_force_magnitude": 0.05, "last_reward": 1.0,
}

WUJI_JOINT_MAPPING = {
    # Thumb (finger1 in URDF)
    "thumb_proximal": "finger1_link2",
    "thumb_intermediate": "finger1_link3",
    "thumb_distal": "finger1_link4",
    "thumb_tip": "finger1_link4",
    # Index (finger2 in URDF)
    "index_proximal": "finger2_link2",
    "index_intermediate": "finger2_link3",
    "index_distal": "finger2_link4",
    "index_tip": "finger2_link4",
    # Middle (finger3 in URDF)
    "middle_proximal": "finger3_link2",
    "middle_intermediate": "finger3_link3",
    "middle_distal": "finger3_link4",
    "middle_tip": "finger3_link4",
    # Ring (finger4 in URDF)
    "ring_proximal": "finger4_link2",
    "ring_intermediate": "finger4_link3",
    "ring_distal": "finger4_link4",
    "ring_tip": "finger4_link4",
    # Pinky (finger5 in URDF)
    "pinky_proximal": "finger5_link2",
    "pinky_intermediate": "finger5_link3",
    "pinky_distal": "finger5_link4",
    "pinky_tip": "finger5_link4",
}

# ------------------------------------------------------------
# Genesis init
# ------------------------------------------------------------


GenesisInitArgsRegistry: dict[str, GenesisInitArgs] = {}


GenesisInitArgsRegistry["default"] = GenesisInitArgs(
    seed=3,
    precision="32",
    logging_level="info",
    backend=None,
)


# ------------------------------------------------------------
# xArm7 + WUJI Hand Configuration
# ------------------------------------------------------------

EnvArgsRegistry: dict[str, EnvArgs] = {}

EnvArgsRegistry["single_hand_retargeting_tactile_map_xarm"] = SingleHandRetargetingEnvArgs(
    env_name="SingleHandRetargetingEnvTactileMap",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["xarm_wuji_hand"],
    reward_term="hand_imitator",
    skip_trajectories=[
        "20260201_164355_951590",
        "20260201_170118_968812",
        "20260201_142337_939403",
        "20260201_160302_406464",
    ],

    proprio_terms=PROPRIO_TERMS_TACTILE_MAP_XARM,
    target_motion_terms=TARGET_MOTION_TERMS_TACTILE_MAP,

    obs_future_length=16,

    joint_mapping=WUJI_JOINT_MAPPING,
    reward_args={
        ### Trajectory Tracking Rewards ###
        "WristPositionTrackingReward": {
            "scale": 30.0,
            "k": 40.0,
        },
        "WristRotationTrackingReward": {
            "scale": 45.0,
            "k": 2.0,
        },
        "FingertipPositionTrackingReward": {
            "scale": 50.0,
            "thumb_weight": 0.8,
            "index_weight": 0.8,
            "middle_weight": 0.8,
            "ring_weight": 0.6,
            "pinky_weight": 0.6,
            "thumb_k": 100.0,
            "index_k": 100.0,
            "middle_k": 100.0,
            "ring_k": 60.0,
            "pinky_k": 60.0,
        },
        "FingertipRotationTrackingReward": {
            "scale": 35.0,
            "thumb_weight": 0.8,
            "index_weight": 0.8,
            "middle_weight": 0.8,
            "ring_weight": 0.6,
            "pinky_weight": 0.6,
            "thumb_k": 3.0,
            "index_k": 3.0,
            "middle_k": 3.0,
            "ring_k": 3.0,
            "pinky_k": 3.0,
        },
        "HandDofPositionTrackingReward": {
            "scale": 25.0,
            "k": 1.0,
        },
        "ObjectPositionTrackingReward": {
            "scale": 100.0,
            "k": 40.0,
        },
        "ObjectRotationTrackingReward": {
            "scale": 125.0,
            "k": 4.0,
        },
        "WeightedGaussianBlurredTactileMapSimilarityReward": {
            "scale": 50.0,
            "k": 4.0,
        },
        "TactileGatedFingertipSurfaceProximityReward": {
            "scale": 25.0,
            "k": 50.0,
        },
        "TableContactPenalty": {
            "scale": 50.0,
            "k": 0.005,
            "threshold": 0.0,
        },
        "ObjectVelocityChangePenalty": {
            "scale": 50.0,
            "k": 2.0,
            "threshold": 0.05,
            "height_threshold": 0.05,
        },
        "ObjectAngularVelocityChangePenalty": {
            "scale": 50.0,
            "k": 0.5,
            "threshold": 3.0,
            "height_threshold": 0.05,
        },
        ### Power Penalties ###
        "AllDOFPowerPenalty": {
            "scale": 50.0,
            "k": 10.0,
        },
    },
    img_resolution=(480, 480),
    action_latency_range=(0, 1),
    obs_history_len=8,

    fingertip_tactile_center_z_offset=0.005,
    table_contact_threshold_2=50,

    # Training-only crutch: pulls the object along the reference, decaying to zero over
    # decay_steps. Skipped entirely when eval_mode=True.
    object_guide_force_decay_steps=5000,
    object_guide_force_kp=5.0,

    # Tactile sensor contact params (geometry paths now on robot_args)
    tactile_kn=2000.0,

    obs_scales=BASE_OBS_SCALES,
    obs_noises={},
    actor_obs_terms=[
        *TARGET_MOTION_TERMS_TACTILE_MAP,
        *PROPRIO_TERMS_TACTILE_MAP_XARM,
        "tactile_map_flat",
        "target_action_flat",
    ],
    critic_obs_terms=[
        *TARGET_MOTION_TERMS_TACTILE_MAP,
        *PROPRIO_TERMS_TACTILE_MAP_XARM,
        "last_reward",
        "tactile_map_flat",
        "target_tactile_map_flat",
        "action_proprio_history_flat",
        "target_action_flat",
    ],
)

# config for distilling a student BC policy
base_config = EnvArgsRegistry["single_hand_retargeting_tactile_map_xarm"]

# Student proprio: subset of teacher proprio (deployment-compatible sensors only).
PROPRIO_TERMS_STUDENT = [
    "hand_dof_pos",  # 20
    "arm_dof_pos",   # 7
]  # total = 27

# Student target motion: object-only future trajectory (no wrist/finger targets)
TARGET_MOTION_TERMS_STUDENT = [
    "target_object_pos",      # 3
    "target_object_quat",     # 4
    "target_object_vel",      # 3
    "target_object_ang_vel",  # 3
]  # total = 13

student_actor_obs_terms = [
    # Target object motion (current frame)
    *TARGET_MOTION_TERMS_STUDENT,

    # Current Proprioception
    *PROPRIO_TERMS_STUDENT,

    # History (student proprio + action)
    "action_proprio_history_student_flat",  # (action 27 + proprio 27) * history_len

    # Future (student terms only)
    "target_action_student_flat",  # 13 * future_len

    # Wrist point cloud
    "wrist_pointcloud_flat",  # 2048 * 3
]
# Critic can use same terms (or add privileged for training stability)
student_critic_obs_terms = student_actor_obs_terms.copy()
EnvArgsRegistry["single_hand_retargeting_bc_student"] = base_config.model_copy(
    update={
        "is_student": True,
        "proprio_terms_student": PROPRIO_TERMS_STUDENT,
        "target_motion_terms_student": TARGET_MOTION_TERMS_STUDENT,
        "actor_obs_terms": student_actor_obs_terms,
        "critic_obs_terms": student_critic_obs_terms,
        "use_wrist_depth_camera": True,
        "wrist_depth_camera_resolution": (128, 72),
        "use_wrist_pointcloud": True,
        "wrist_pointcloud_num_points": 2048,
        # Point cloud noise (also applied in eval)
        "pointcloud_depth_noise_std": 0.01,              # 10mm Gaussian depth noise
        "pointcloud_rect_dropout_num": 10,               # 10 random rectangles dropped per frame
        "pointcloud_rect_dropout_width": (5, 20),        # rectangle width range in pixels
        "pointcloud_rect_dropout_height": (5, 15),       # rectangle height range in pixels
        # Observation delay (also applied in eval)
        "student_obs_delay_max": 3,                  # stochastic delay 0-2 steps
        # Proprio noise (approximating real-hardware sensor errors). Only hand_dof_pos and
        # arm_dof_pos are observed by the student; the other entries are inert.
        "obs_noises": {
            "hand_dof_pos": 0.01,   # encoder quantization
            "hand_dof_vel": 0.1,    # finite-difference amplification
            "arm_dof_pos": 0.01,    # encoder quantization
            "arm_dof_vel": 0.1,     # finite-difference amplification
            "base_pos": 0.005,      # FK error from joint encoders (~5mm)
            "base_quat": 0.01,      # FK error (small-angle)
            "base_lin_vel": 0.05,   # differentiated FK / IMU
            "base_ang_vel": 0.05,   # differentiated FK / IMU
        },
    }
)
