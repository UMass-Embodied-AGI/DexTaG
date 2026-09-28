from __future__ import annotations

import torch
from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel


class RigidMaterialArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    rho: float
    friction: float | None
    needs_coup: bool
    coup_friction: float
    coup_softness: float
    coup_restitution: float
    sdf_cell_size: float
    sdf_min_res: int
    sdf_max_res: int
    gravity_compensation: float


class URDFMorphArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    # Morph
    pos: tuple[float, float, float]
    euler: tuple[int, int, int]
    quat: tuple[float, float, float, float] | None
    visualization: bool
    collision: bool
    requires_jac_and_IK: bool
    is_free: bool

    # FileMorph
    file: str
    scale: float | tuple[float, float, float]
    convexify: bool
    recompute_inertia: bool

    # URDF
    fixed: bool
    prioritize_urdf_material: bool
    merge_fixed_links: bool
    links_to_keep: list[str]

    decimate: bool


class CtrlType(GenesisEnum):
    ARM_REL_HAND_ABS = "ARM_REL_HAND_ABS"  # Arm delta pose + hand absolute pose control


class IKSolver(GenesisEnum):
    GS = "GS"  # Genesis Solver
    PIN = "PIN"  # Pinocchio Solver


class ManipulatorRobotArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    material_args: RigidMaterialArgs
    morph_args: URDFMorphArgs
    dr_args: DomainRandomizationArgs
    visualize_contact: bool
    vis_mode: str
    ctrl_type: CtrlType
    ik_solver: IKSolver
    ee_link_name: str
    show_target: bool
    gripper_link_names: list[str]
    default_arm_dof: dict[str, float]
    default_gripper_dof: dict[str, float] | None = None
    soft_dof_pos_range: float = 1.0  # Soft limit range multiplier (1.0 = use full range)
    action_scale: float = 0.1  # Scale factor for hand (finger) actions
    base_action_scale: float = 0.01  # Scale factor for arm (delta-joint) actions
    action_ema: float = 1.0  # Hand-target blend toward current pos (1.0 = no smoothing)
    decimation: int = 4  # Number of simulation steps per action
    dof_kp: dict[str, float]
    dof_kd: dict[str, float]
    dof_max_force: float | list[float]
    tcp_yaw: float | None = None  # TCP yaw for IK (rad); xArm7 + WUJI right hand: 2.3562
    # Tactile sensor files (robot-side: geometry depends on hand morphology)
    tactile_grid_path: str | None = None
    tactile_pixel_mapping_path: str | None = None


class DomainRandomizationArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    kp_range: tuple[float, float]
    kd_range: tuple[float, float]
    # Unused: only the removed DR_JOINT_POSITION control path read these.
    # Kept because existing saved env_args.yaml files still carry them.
    motor_strength_range: tuple[float, float]
    motor_offset_range: tuple[float, float]
    friction_range: tuple[float, float]
    mass_range: tuple[float, float]
    com_displacement_range: tuple[float, float]
    joint_damping_range: tuple[float, float] = (1.0, 1.0)
    joint_armature_range: tuple[float, float] = (1.0, 1.0)
    soft_dof_pos_range_range: tuple[float, float] = (1.0, 1.0)


class JointPosAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    joint_pos: torch.Tensor  # (n_envs, n_dof)
