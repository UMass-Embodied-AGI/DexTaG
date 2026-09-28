from gs_env.sim.robots.config.schema import (
    CtrlType,
    DomainRandomizationArgs,
    IKSolver,
    ManipulatorRobotArgs,
    RigidMaterialArgs,
    URDFMorphArgs,
)

MaterialArgsRegistry: dict[str, RigidMaterialArgs] = {}
MorphArgsRegistry: dict[str, URDFMorphArgs] = {}
RobotArgsRegistry: dict[str, ManipulatorRobotArgs] = {}

# ------------------------------------------------------------
# WUJI Hand Configuration
# ------------------------------------------------------------

# Material configuration for WUJI hand.
# Not applied: ManipulatorBase uses gs.materials.Rigid() defaults; kept for saved-config compatibility.
MaterialArgsRegistry["wuji_hand"] = RigidMaterialArgs(
    rho=200.0,
    friction=None,
    needs_coup=True,
    coup_friction=0.8,
    coup_softness=0.002,
    coup_restitution=0.0,
    sdf_cell_size=0.002,
    sdf_min_res=32,
    sdf_max_res=128,
    gravity_compensation=1,
)

# Default joint positions for WUJI hand
WUJI_default_dof_pos: dict[str, float] = {
    # Finger 1 (Thumb) - slightly spread and open
    "finger1_joint1": 0.1,
    "finger1_joint2": 0.6,
    "finger1_joint3": 0.9,
    "finger1_joint4": 0.4,
    # Finger 2 (Index) - open position
    "finger2_joint1": 0.2,
    "finger2_joint2": 0.0,
    "finger2_joint3": 1.2,
    "finger2_joint4": 0.4,
    # Finger 3 (Middle) - open position
    "finger3_joint1": 0.2,
    "finger3_joint2": 0.0,
    "finger3_joint3": 1.2,
    "finger3_joint4": 0.4,
    # Finger 4 (Ring) - open position
    "finger4_joint1": 0.2,
    "finger4_joint2": 0.0,
    "finger4_joint3": 1.2,
    "finger4_joint4": 0.4,
    # Finger 5 (Pinky) - open position
    "finger5_joint1": 0.3,
    "finger5_joint2": 0.0,
    "finger5_joint3": 1.2,
    "finger5_joint4": 0.4,
    }

# PD gains for WUJI hand
WUJI_kp_dict: dict[str, float] = {
    "finger1_joint1": 96.86,
    "finger1_joint2": 96.86,
    "finger1_joint3": 141.42,
    "finger1_joint4": 141.42,
    "finger2_joint1": 96.86,
    "finger2_joint2": 96.86,
    "finger2_joint3": 141.42,
    "finger2_joint4": 141.42,
    "finger3_joint1": 96.86,
    "finger3_joint2": 96.86,
    "finger3_joint3": 141.42,
    "finger3_joint4": 141.42,
    "finger4_joint1": 96.86,
    "finger4_joint2": 96.86,
    "finger4_joint3": 141.42,
    "finger4_joint4": 141.42,
    "finger5_joint1": 96.86,
    "finger5_joint2": 96.86,
    "finger5_joint3": 141.42,
    "finger5_joint4": 141.42,
}

WUJI_kd_dict: dict[str, float] = {
    "finger1_joint1": 5.18,
    "finger1_joint2": 5.18,
    "finger1_joint3": 7.20,
    "finger1_joint4": 7.20,
    "finger2_joint1": 5.18,
    "finger2_joint2": 5.18,
    "finger2_joint3": 7.20,
    "finger2_joint4": 7.20,
    "finger3_joint1": 5.18,
    "finger3_joint2": 5.18,
    "finger3_joint3": 7.20,
    "finger3_joint4": 7.20,
    "finger4_joint1": 5.18,
    "finger4_joint2": 5.18,
    "finger4_joint3": 7.20,
    "finger4_joint4": 7.20,
    "finger5_joint1": 5.18,
    "finger5_joint2": 5.18,
    "finger5_joint3": 7.20,
    "finger5_joint4": 7.20,
}

# Fingertip link names (stored as gripper_link_names; currently unused)
WUJI_fingertip_link_names: list[str] = [
    "finger1_link4",  # Thumb tip
    "finger2_link4",  # Index tip
    "finger3_link4",  # Middle tip
    "finger4_link4",  # Ring tip
    "finger5_link4",  # Pinky tip
]

# ------------------------------------------------------------
# xArm7 + WUJI Hand Configuration
# ------------------------------------------------------------

# PD gains: per-joint arm kp=1414-3015, kd=19-37; finger joints inherit from WUJI
XARM_WUJI_kp_dict: dict[str, float] = {
    "joint1": 1414.0,
    "joint2": 3015.0,
    "joint3": 2065.0,
    "joint4": 3015.0,
    "joint5": 1414.0,
    "joint6": 1414.0,
    "joint7": 1414.0,
    **WUJI_kp_dict,
}

XARM_WUJI_kd_dict: dict[str, float] = {
    "joint1": 19.0,
    "joint2": 37.0,
    "joint3": 27.0,
    "joint4": 37.0,
    "joint5": 19.0,
    "joint6": 19.0,
    "joint7": 19.0,
    **WUJI_kd_dict,
}

# Morph configuration for xArm7 + WujiHand (fixed base arm)
MorphArgsRegistry["xarm_wuji_hand"] = URDFMorphArgs(
    pos=(-0.2, 0.0, 0.0),
    euler=(0, 0, 0),
    quat=None,
    visualization=True,
    collision=True,
    requires_jac_and_IK=False,
    is_free=False,
    file="assets/robot/xarm/xarm7_with_wujihand_v5.urdf",
    scale=1.0,
    convexify=False,
    recompute_inertia=True,
    fixed=True,
    prioritize_urdf_material=False,
    merge_fixed_links=False,
    links_to_keep=[],
    decimate=True,
)

# Default arm DOF positions for xArm7
XARM_default_arm_dof: dict[str, float] = {
    "joint1": -0.54140069,
    "joint2": 0.28606658,
    "joint3": -0.39994904,
    "joint4": 0.47730433,
    "joint5": 2.18652979,
    "joint6": 1.65479239,
    "joint7": -1.44058947,
}

# Register xArm7 + WUJI hand robot configuration (7 DOF arm + 20 DOF fingers = 27 DOF total)
RobotArgsRegistry["xarm_wuji_hand"] = ManipulatorRobotArgs(
    material_args=MaterialArgsRegistry["wuji_hand"],
    morph_args=MorphArgsRegistry["xarm_wuji_hand"],
    dr_args=DomainRandomizationArgs(
        kp_range=(0.8, 1.2),
        kd_range=(0.8, 1.2),
        motor_strength_range=(0.9, 1.1),
        motor_offset_range=(-0.01, 0.01),
        friction_range=(0.7, 1.3),
        mass_range=(0.7, 1.3),
        com_displacement_range=(-0.15, 0.15),
        joint_damping_range=(0.7, 1.3),
        joint_armature_range=(0.7, 1.3),
        soft_dof_pos_range_range=(0.9, 1.0),
    ),
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.ARM_REL_HAND_ABS,
    ik_solver=IKSolver.GS,
    ee_link_name="palm_link",
    show_target=False,
    gripper_link_names=WUJI_fingertip_link_names,
    default_arm_dof=XARM_default_arm_dof,
    default_gripper_dof=WUJI_default_dof_pos,
    soft_dof_pos_range=0.95,
    action_scale=1.0,
    base_action_scale=0.025,
    action_ema=0.1,
    decimation=1,
    dof_kp=XARM_WUJI_kp_dict,
    dof_kd=XARM_WUJI_kd_dict,
    dof_max_force=[10000.0] * 7 + [10000.0, 10000.0, 10000.0, 10000.0] * 5,  # sequence following all_dof_idx_local
    tcp_yaw=2.3562,  # 135deg, matches wujihand_fix yaw in xarm7_with_wujihand_v5.urdf
    tactile_grid_path="assets/robot/wujihand-urdf/full_hand_tactile_v5.json",
    tactile_pixel_mapping_path="assets/robot/wujihand-urdf/tactile_pixel_mapping_v5.json",
)
