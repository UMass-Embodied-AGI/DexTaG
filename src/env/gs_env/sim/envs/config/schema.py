from typing import Literal

import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel, model_validator

from gs_env.sim.robots.config.schema import ManipulatorRobotArgs
from gs_env.sim.scenes.config.schema import SceneArgs


class GenesisInitArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    seed: int
    precision: str
    logging_level: str
    backend: (
        gs.constants.backend | None
    )  # Not read: gs.init() uses the device's backend; only seed is honored


class EnvArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    env_name: str
    gs_init_args: GenesisInitArgs
    scene_args: SceneArgs
    robot_args: ManipulatorRobotArgs
    reward_term: str = "reward"
    reward_args: dict[str, float] | dict[str, dict[str, float | list[float]]]
    img_resolution: tuple[int, int]


class ObjectConfig(BaseModel):
    """Per-object defaults for manipulation environments."""
    model_config = genesis_pydantic_config(frozen=True)

    trajectory_path: str
    object_id: str
    mass: float | None = None  # None = use physics engine default from mesh density
    com_shift: list[float] = [0.0, 0.0, 0.0]
    rotation_ignore_axis: list[float] | None = None
    trajectory_offset: list[float] = [0.0, 0.0, 0.1]
    # Domain randomization defaults
    mass_randomization_range: float = 0.3  # +/- fraction of base mass
    friction_range: tuple[float, float] = (0.7, 1.5)  # min/max friction ratio
    scale_range: tuple[float, float] = (0.9, 1.1)  # min/max object scale for heterogeneous morphs
    n_scale_variants: int = 10  # number of distinct scale variants
    # Random perturbation forces applied to object during training
    perturbation_force_scale: float = 0.0       # Force magnitude multiplier (0 = disabled)
    perturbation_force_prob_range: tuple[float, float] = (0.001, 0.05)  # Log-uniform prob range
    perturbation_torque_scale: float = 0.0      # Torque magnitude multiplier (0 = disabled)
    perturbation_torque_prob_range: tuple[float, float] = (0.001, 0.05)
    perturbation_lift_threshold: float = 0.05   # Only perturb when object is this far above table [m]
    perturbation_curriculum_start: int = 2000    # Training step at which perturbations begin
    perturbation_curriculum_end: int = 10000     # Training step at which perturbations reach full scale
    # Initial pose randomization (applied per reset)
    init_pos_noise_xy: float = 0.02              # uniform ± XY translation noise in meters (Z is not randomized)
    init_rot_noise_deg: float = 10.0             # uniform ± per-axis rotation noise in degrees
    # Table height randomization (sampled once per env at init)
    table_height_noise: float = 0.01             # uniform ± Z offset for table surface in meters


class TrajectoryAugmentationConfig(BaseModel):
    """Configuration for spatial augmentation of reference trajectories."""
    model_config = genesis_pydantic_config(frozen=True)
    enabled: bool = False
    translate_x_range: tuple[float, float] = (-0.05, 0.05)   # meters
    translate_y_range: tuple[float, float] = (-0.05, 0.05)   # meters
    rotate_z_range: tuple[float, float] = (-10.0, 10.0)      # degrees
    scale_range: tuple[float, float] = (0.95, 1.05)


class SingleHandRetargetingEnvArgs(EnvArgs):
    """Configuration for hand trajectory imitation environments."""
    action_latency_range: tuple[int, int] = (0, 1)  # (min, max) inclusive, randomized per env per step
    obs_history_len: int = 1
    obs_scales: dict[str, float]
    obs_noises: dict[str, float]
    actor_obs_terms: list[str]
    critic_obs_terms: list[str]
    proprio_terms: list[str]
    target_motion_terms: list[str]
    # Object configuration (set via --object_name CLI arg)
    object_config: ObjectConfig | None = None
    obs_future_length: int = 5  # Number of future trajectory frames to observe (K)
    joint_mapping: dict[str, str]
    object_domain_randomization: bool = True
    # Tactile sensor configuration (geometry paths now live on robot_args)
    tactile_kn: float = 3000.0
    tactile_kt: float = 200.0
    tactile_mu: float = 1.0
    # Links carrying fingertip tactile geometry in the v5 URDF (used for fingertip-to-surface SDF queries).
    tactile_fingertip_link_names: list[str] = [
        "finger1_link4", "finger2_link4", "finger3_link4",
        "finger4_link4", "finger5_link4",
    ]
    fingertip_tactile_center_z_offset: float = 0.0  # 0.005 or 0.01 seems a good value
    # Table contact configuration
    table_contact_threshold_0: float = 500.0  # Contact force magnitude at step 0 (start of curriculum)
    table_contact_threshold_1: float = 100.0  # Contact force magnitude at step 10000
    table_contact_threshold_2: float = 25.0   # Contact force magnitude at step 20000
    # Wrist camera configuration
    use_wrist_depth_camera: bool = False
    wrist_depth_camera_resolution: tuple[int, int] = (64, 64)
    # Wrist point cloud configuration (uses the same camera)
    use_wrist_pointcloud: bool = False
    wrist_pointcloud_num_points: int = 2048
    # Point cloud noise (also applied in eval)
    pointcloud_depth_noise_std: float = 0.0     # Gaussian noise on depth (meters), e.g. 0.002
    pointcloud_rect_dropout_num: int = 0        # Number of random rectangles to drop per env
    pointcloud_rect_dropout_width: tuple[int, int] = (5, 25)   # (min, max) rectangle width in pixels
    pointcloud_rect_dropout_height: tuple[int, int] = (5, 25)  # (min, max) rectangle height in pixels
    # Student observation delay (stochastic, also applied in eval)
    student_obs_delay_max: int = 0  # Delay-queue length N; per-step delay ~ U{0..N-1} (0 = disabled)
    # Student BC distillation: use separate proprio/target_motion terms for student observations
    is_student: bool = False
    proprio_terms_student: list[str] | None = None
    target_motion_terms_student: list[str] | None = None
    # Target visualization mode for eval: "ghost_hand", "fingertip_markers", "target_object", or None
    target_visualization: str | None = None
    # Object guide: use control_dofs_position to guide the object's translational DOFs
    # toward the reference trajectory. kp/kv decay linearly over training steps.
    # Set decay_steps to 0 to disable.
    object_guide_force_decay_steps: int = 0  # Number of training steps over which kp/kv decay to 0
    object_guide_force_kp: float = 1.0  # 0.5 seems to be the minimum for pen lifting
    skip_trajectories: list[str] = []  # Trajectory filenames to skip during loading
    # Train/eval split (deterministic, by trajectory ID prefix before "__"):
    # split_key = traj_file.stem.split("__")[0]; bucket = md5(split_key) -> [0, 1).
    # split_mode="train" keeps bucket < train_ratio; "eval" keeps bucket >= train_ratio;
    # "all" disables the split. Applied on top of skip_trajectories.
    train_ratio: float = 1.0
    split_mode: Literal["train", "eval", "all"] = "all"
    # Salts the split hash so different seeds give different train/eval partitions.
    split_seed: int = 0
    trajectory_augmentation: TrajectoryAugmentationConfig = TrajectoryAugmentationConfig()
    trajectory_playback_hz: float = 60.0  # Resample trajectories to this playback rate (Hz) if different from original 60 Hz

    @model_validator(mode="after")
    def _check_obs_and_student_terms(self) -> "SingleHandRetargetingEnvArgs":
        for name in ("actor_obs_terms", "critic_obs_terms", "proprio_terms", "target_motion_terms"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

        if self.is_student:
            if not self.proprio_terms_student:
                raise ValueError("is_student=True requires proprio_terms_student to be a non-empty list")
            if not self.target_motion_terms_student:
                raise ValueError("is_student=True requires target_motion_terms_student to be a non-empty list")
        else:
            if self.proprio_terms_student is not None:
                raise ValueError("proprio_terms_student must be None when is_student=False")
            if self.target_motion_terms_student is not None:
                raise ValueError("target_motion_terms_student must be None when is_student=False")

        scale_keys = set(self.obs_scales.keys())
        referenced: set[str] = set(self.actor_obs_terms) | set(self.critic_obs_terms) \
            | set(self.proprio_terms) | set(self.target_motion_terms)
        if self.is_student:
            referenced |= set(self.proprio_terms_student or [])
            referenced |= set(self.target_motion_terms_student or [])
        missing = sorted(referenced - scale_keys)
        if missing:
            raise ValueError(
                f"obs_scales is missing entries for referenced terms: {missing}. "
                f"Every term used in actor/critic/proprio/target_motion (and student variants "
                f"when is_student=True) must have an obs_scales entry."
            )
        return self
