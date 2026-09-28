from typing import Any

import genesis as gs
import torch
import numpy as np
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.common.utils.math_utils import quat_to_rotmat
from gs_env.sim.robots.config.schema import (
    JointPosAction,
    ManipulatorRobotArgs,
)
from gs_env.sim.robots.xarm.xarm7_kinematics import XArm7Kinematics


class ManipulatorBase(BaseGymRobot):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: ManipulatorRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__()
        # == set members ==
        self._device = device
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.URDF = gs.morphs.URDF(
            file=args.morph_args.file,
            merge_fixed_links=False,
            pos=args.morph_args.pos,
            euler=args.morph_args.euler,
            fixed=args.morph_args.fixed,
            is_free=args.morph_args.is_free,
            recompute_inertia=args.morph_args.recompute_inertia,
        )
        #
        robot_entity = scene.add_entity(
            material=material,
            morph=morph,
            visualize_contact=args.visualize_contact,
            vis_mode=args.vis_mode,
        )
        assert isinstance(robot_entity, RigidEntity), (
            "Robot entity must be an instance of gs.Entity"
        )
        self._robot_entity = robot_entity

        # == some buffer initialization ==
        self._init()

    def _init(self) -> None:
        all_dof_names = list(self._args.default_arm_dof.keys())
        if self._args.default_gripper_dof is not None:
            all_dof_names += list(self._args.default_gripper_dof.keys())
        self._all_dof_names = all_dof_names

        self._arm_dof_dim = sum(
            [self._robot_entity.get_joint(name).n_dofs for name in self._args.default_arm_dof.keys()]
        )
        self._gripper_dof_dim = sum(
            [self._robot_entity.get_joint(name).n_dofs for name in self._args.default_gripper_dof.keys()]
        )
        self._dof_dim = self._arm_dof_dim + self._gripper_dof_dim

        # About torque calculation and domain randomization
        dof_kp = [self._args.dof_kp[name] for name in all_dof_names]
        dof_kd = [self._args.dof_kd[name] for name in all_dof_names]
        self._dof_kp = torch.tensor(dof_kp, device=self._device)
        self._dof_kd = torch.tensor(dof_kd, device=self._device)
        self._batched_dof_kp = self._dof_kp[None, :].repeat(self._num_envs, 1)
        self._batched_dof_kd = self._dof_kd[None, :].repeat(self._num_envs, 1)

        default_dof_pos = list(self._args.default_arm_dof.values())
        if self._args.default_gripper_dof is not None:
            default_dof_pos += list(self._args.default_gripper_dof.values())
        self._default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float32, device=self._device)

        # Buffers
        self._dof_pos = torch.zeros((self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device)
        self._dof_vel = torch.zeros((self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device)

    def post_build_init(self, eval_mode: bool = False) -> None:
        """Initialize limits and constraints after scene is built."""
        # Store eval mode flag for later use
        self._eval_mode = eval_mode

        # Get DOF indices for all joints
        all_dofs_idx_local = []
        for name in self._all_dof_names:
            all_dofs_idx_local += self._robot_entity.get_joint(name).dofs_idx_local
        self._all_dof_idx_local = all_dofs_idx_local

        self._hand_dof_idx_local = []
        for name in self._args.default_gripper_dof.keys():
            self._hand_dof_idx_local += self._robot_entity.get_joint(name).dofs_idx_local

        self._base_dof_idx_local = []
        for name in self._args.default_arm_dof.keys():
            self._base_dof_idx_local += self._robot_entity.get_joint(name).dofs_idx_local

        assert self._all_dof_idx_local == self._base_dof_idx_local + self._hand_dof_idx_local, "DOF indices do not match!"

        # Cache base physics values for DR (before randomization)
        self._base_damping = self._robot_entity.get_dofs_damping(self._all_dof_idx_local)
        self._base_armature = self._robot_entity.get_dofs_armature(self._all_dof_idx_local)
        self._base_link_mass = self._robot_entity.get_links_inertial_mass()

        # Cache per-link AABB extents (x, y, z) for ratio-based COM displacement DR.
        # Compute directly from geom init_verts to avoid get_AABB() issues
        # with links that have geoms but no mesh vertices.
        link_extents = []
        for link in self._robot_entity.links:
            all_verts = [geom._init_verts for geom in link.geoms if len(geom._init_verts) > 0]
            if all_verts:
                verts = np.concatenate(all_verts, axis=0)  # (N, 3)
                extents = verts.max(axis=0) - verts.min(axis=0)  # (3,)
                link_extents.append(extents.tolist())
            else:
                link_extents.append([0.0, 0.0, 0.0])
        self._link_extents = torch.tensor(link_extents, device=self._device)  # (n_links, 3)

        # Cache hard DOF position limits for soft range DR
        hard_limits = torch.stack(
            self._robot_entity.get_dofs_limit(all_dofs_idx_local), dim=1
        )  # (n_dofs, 2)
        self._hard_dof_pos_limits = hard_limits
        self._limits_midpoint = (hard_limits[:, 0] + hard_limits[:, 1]) / 2  # (n_dofs,)
        self._limits_range = hard_limits[:, 1] - hard_limits[:, 0]            # (n_dofs,)
        self._limits_inf_mask = torch.isinf(hard_limits).any(dim=1)           # (n_dofs,)

        # Initialize domain randomization for control parameters
        if not eval_mode:
            self._init_domain_randomization()
        else:
            # set kp/kd to base values
            envs_idx = torch.arange(0, self._num_envs, device=self._device, dtype=torch.int32)
            self._robot_entity.set_dofs_kp(self._dof_kp, envs_idx=envs_idx)
            self._robot_entity.set_dofs_kv(self._dof_kd, envs_idx=envs_idx)
            # set soft DOF position limits to fixed soft_dof_pos_range
            self._build_soft_dof_pos_limits(
                torch.full((self._num_envs, 1), self._args.soft_dof_pos_range, device=self._device)
            )

        # Set up torque limits
        self._robot_entity.set_dofs_force_range(
            lower=np.array([-v for v in self._args.dof_max_force]),
            upper=np.array([v for v in self._args.dof_max_force]),
            dofs_idx_local=all_dofs_idx_local,
        )

    def _init_domain_randomization(self) -> None:
        """Initialize domain randomization for all environments."""
        envs_idx: torch.IntTensor = torch.arange(0, self._num_envs, device=self._device)  # type: ignore
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)

    def _build_soft_dof_pos_limits(self, soft_ranges: torch.Tensor) -> None:
        """Build per-env DOF position limits from soft range multipliers.

        Args:
            soft_ranges: (n_envs, 1) tensor of soft range multipliers.
        """
        m = self._limits_midpoint  # (n_dofs,)
        r = self._limits_range      # (n_dofs,)
        inf_mask = self._limits_inf_mask  # (n_dofs,)

        n_dofs = len(self._all_dof_idx_local)
        self._dof_pos_limits = torch.zeros(self._num_envs, n_dofs, 2, device=self._device)
        self._dof_pos_limits[:, :, 0] = m - 0.5 * r * soft_ranges
        self._dof_pos_limits[:, :, 1] = m + 0.5 * r * soft_ranges
        # For infinite limits, keep them infinite
        if inf_mask.any():
            self._dof_pos_limits[:, inf_mask, 0] = self._hard_dof_pos_limits[inf_mask, 0].unsqueeze(0)
            self._dof_pos_limits[:, inf_mask, 1] = self._hard_dof_pos_limits[inf_mask, 1].unsqueeze(0)

    def _randomize_rigids(self, envs_idx: torch.IntTensor) -> None:
        """Randomize rigid body properties (friction, damping, armature, link mass)."""
        # Friction
        min_friction, max_friction = self._args.dr_args.friction_range
        solver: RigidSolver = self._robot_entity.solver
        ratios = (
            torch.rand(len(envs_idx), solver.n_geoms) * (max_friction - min_friction)
            + min_friction
        )
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), envs_idx)

        # Joint damping
        lo, hi = self._args.dr_args.joint_damping_range
        if lo != hi:
            ratios = torch.rand(len(envs_idx), len(self._all_dof_idx_local)) * (hi - lo) + lo
            self._robot_entity.set_dofs_damping(
                self._base_damping * ratios, dofs_idx_local=self._all_dof_idx_local, envs_idx=envs_idx
            )

        # Joint armature
        lo, hi = self._args.dr_args.joint_armature_range
        if lo != hi:
            ratios = torch.rand(len(envs_idx), len(self._all_dof_idx_local)) * (hi - lo) + lo
            self._robot_entity.set_dofs_armature(
                self._base_armature * ratios, dofs_idx_local=self._all_dof_idx_local, envs_idx=envs_idx
            )

        # Link mass (ratio-based)
        lo, hi = self._args.dr_args.mass_range
        if lo != hi:
            n_links = self._base_link_mass.shape[-1]
            ratios = torch.rand(len(envs_idx), n_links) * (hi - lo) + lo
            self._robot_entity.set_links_inertial_mass(
                self._base_link_mass * ratios, envs_idx=envs_idx
            )

        # COM displacement (ratio of each link's per-axis AABB extent)
        min_com, max_com = self._args.dr_args.com_displacement_range
        if min_com != max_com:
            n_links = self._robot_entity.n_links
            # Sample ratios per env, per link, per axis in [min_com, max_com]
            ratios = torch.rand(len(envs_idx), n_links, 3, device=self._device) * (max_com - min_com) + min_com
            # Scale each axis by its own extent: _link_extents is (n_links, 3)
            displacement = ratios * self._link_extents[None, :, :]
            self._robot_entity.set_COM_shift(
                displacement,
                links_idx_local=list(range(n_links)),
                envs_idx=envs_idx,
            )

    def _randomize_controls(self, envs_idx: torch.IntTensor) -> None:
        """Randomize control parameters (kp, kd, soft DOF position limits)."""
        # Randomize kp
        min_kp, max_kp = self._args.dr_args.kp_range
        ratios = torch.rand(len(envs_idx), self._dof_dim) * (max_kp - min_kp) + min_kp
        self._batched_dof_kp[envs_idx] = ratios * self._dof_kp[None, :]

        # Randomize kd
        min_kd, max_kd = self._args.dr_args.kd_range
        ratios = torch.rand(len(envs_idx), self._dof_dim) * (max_kd - min_kd) + min_kd
        self._batched_dof_kd[envs_idx] = ratios * self._dof_kd[None, :]

        # Apply randomized kp/kd to Genesis (requires batch_dofs_info=True)
        self._robot_entity.set_dofs_kp(self._batched_dof_kp[envs_idx], envs_idx=envs_idx)
        self._robot_entity.set_dofs_kv(self._batched_dof_kd[envs_idx], envs_idx=envs_idx)

        # Randomize soft DOF position limits
        lo, hi = self._args.dr_args.soft_dof_pos_range_range
        if lo != hi:
            soft_ranges = torch.rand(self._num_envs, 1, device=self._device) * (hi - lo) + lo
        else:
            soft_ranges = torch.full((self._num_envs, 1), self._args.soft_dof_pos_range, device=self._device)
        self._build_soft_dof_pos_limits(soft_ranges)

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        if envs_idx is None or len(envs_idx) == 0:
            return
        self.go_home(envs_idx)

    def go_home(self, envs_idx: torch.IntTensor) -> None:
        self._robot_entity.set_dofs_position(
            self._default_dof_pos.repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            dofs_idx_local=self._all_dof_idx_local,
            zero_velocity=True,
        )
        self._dof_pos[envs_idx] = self._default_dof_pos[None, :].repeat(len(envs_idx), 1)
        self._dof_vel[envs_idx] = 0.0

    def apply_action(self, action: JointPosAction | torch.Tensor) -> None:
        """
        Apply the action to the robot.
        """
        if isinstance(action, torch.Tensor):
            action = JointPosAction(joint_pos=action)
        self._apply_arm_rel_hand_abs_control(action)

        self._dof_pos[:] = self._robot_entity.get_dofs_position(self._all_dof_idx_local)
        self._dof_vel[:] = self._robot_entity.get_dofs_velocity(self._all_dof_idx_local)

    # The only implemented control mode (CtrlType.ARM_REL_HAND_ABS): arm delta
    # position + hand absolute position. Subclasses provide the implementation.
    def _apply_arm_rel_hand_abs_control(self, act: JointPosAction) -> None:
        raise NotImplementedError("ARM_REL_HAND_ABS control not implemented for base manipulator.")


    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel

    @property
    def dofs_control_force(self) -> torch.Tensor:
        return self._robot_entity.get_dofs_control_force(dofs_idx_local=self._all_dof_idx_local)

    def __getattr__(self, item: str) -> Any:
        # Use object.__getattribute__ to avoid recursion with hasattr
        try:
            return object.__getattribute__(self._robot_entity, item)
        except AttributeError as err:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            ) from err


class XArmWUJIHand(ManipulatorBase):
    """
    xArm7 + WUJI Hand - 7 DOF arm + 20 DOF hand = 27 DOF total.
    Wrist pose is reached via IK (XArm7Kinematics) on the 7 revolute arm joints.
    """

    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: ManipulatorRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__(num_envs=num_envs, scene=scene, args=args, device=device)

        # Initialize IK solver with TCP offset for WujiHand palm.
        # TCP yaw matches the wujihand_fix joint yaw in xarm7_with_wujihand_v5.urdf
        # (2.3562 rad = 135deg).
        if args.tcp_yaw is None:
            raise ValueError(
                "XArmWUJIHand requires args.tcp_yaw to be set (2.3562 for the WUJI right hand)."
            )
        self._ik_solver = XArm7Kinematics(
            tcp_offset=[0, 0, 57, 0, 0, args.tcp_yaw],
        )

        # Store arm base position for coordinate transforms (morph position)
        self._arm_base_pos = torch.tensor(
            list(args.morph_args.pos), dtype=torch.float32, device=device
        )

    _ARM_LINK_NAMES = [
        "link_base", "link1", "link2", "link3", "link4",
        "link5", "link6", "link7", "link_eef",
    ]

    def post_build_init(self, eval_mode: bool = False) -> None:
        """Initialize limits and constraints after scene is built."""
        super().post_build_init(eval_mode=eval_mode)
        # Cache palm_link index for base_* property queries
        self._wrist_link_idx = self._robot_entity.get_link("palm_link").idx_local
        # Cache arm link local indices for contact queries
        self._arm_link_idx_local = [
            self._robot_entity.get_link(name).idx_local
            for name in self._ARM_LINK_NAMES
        ]

    def solve_ik(
        self,
        wrist_pos: torch.Tensor,
        wrist_quat: torch.Tensor,
        q_refs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve inverse kinematics for arm joints given wrist pose.

        Args:
            wrist_pos: Target wrist positions in world frame (B, 3), meters.
            wrist_quat: Target wrist quaternions (B, 4), [w, x, y, z].
            q_refs: Reference joint angles for IK solver (B, 7) or (7,).
                    If None, uses default arm DOF positions.

        Returns:
            arm_angles: Solved arm joint angles (B, 7). Falls back to defaults
                        for any batch element where IK fails.
        """
        batch_size = wrist_pos.shape[0]

        # Convert to arm base frame in mm
        pos_arm_frame = (wrist_pos - self._arm_base_pos.unsqueeze(0)) * 1000.0  # m -> mm

        # Build 4x4 homogeneous transform matrices
        rot_mat = quat_to_rotmat(wrist_quat)  # (B, 3, 3)
        target_mats = torch.zeros((batch_size, 4, 4), dtype=torch.float32, device=self._device)
        target_mats[:, :3, :3] = rot_mat
        target_mats[:, :3, 3] = pos_arm_frame
        target_mats[:, 3, 3] = 1.0

        # Resolve reference angles
        default_arm_angles = self._default_dof_pos[:self._arm_dof_dim]
        if q_refs is None:
            q_refs = default_arm_angles.unsqueeze(0).expand(batch_size, -1)

        arm_angles, ret_codes = self._ik_solver.inverse_kinematics_mat_batch(
            target_mats, q_refs=q_refs
        )
        arm_angles = torch.as_tensor(arm_angles, dtype=torch.float32, device=self._device)
        ret_codes = torch.as_tensor(ret_codes, dtype=torch.int32, device=self._device)

        # Fall back to default arm angles where IK failed
        failed_mask = ret_codes != 0
        if failed_mask.any():
            arm_angles[failed_mask] = default_arm_angles.unsqueeze(0).expand(failed_mask.sum(), -1)

        return arm_angles

    def reset_to_pose(
        self,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        wrist_pos: torch.Tensor,
        wrist_quat: torch.Tensor,
        envs_idx: torch.IntTensor,
    ) -> None:
        """Reset robot to a specific joint configuration using IK for arm joints."""
        batch_size = len(envs_idx)
        arm_angles = self.solve_ik(wrist_pos, wrist_quat)

        # Set arm DOF positions
        self._robot_entity.set_dofs_position(
            arm_angles,
            dofs_idx_local=self._base_dof_idx_local,
            envs_idx=envs_idx,
        )
        # Set finger DOF positions
        self._robot_entity.set_dofs_position(
            dof_pos,
            dofs_idx_local=self._hand_dof_idx_local,
            envs_idx=envs_idx,
        )
        # Set finger DOF velocities
        self._robot_entity.set_dofs_velocity(
            dof_vel,
            dofs_idx_local=self._hand_dof_idx_local,
            envs_idx=envs_idx,
        )
        # Zero arm velocities
        self._robot_entity.set_dofs_velocity(
            torch.zeros((batch_size, self._arm_dof_dim), device=self._device),
            dofs_idx_local=self._base_dof_idx_local,
            envs_idx=envs_idx,
        )

        # Update buffers
        self._dof_pos[envs_idx] = self._robot_entity.get_dofs_position(
            self._all_dof_idx_local, envs_idx=envs_idx
        )
        self._dof_vel[envs_idx] = self._robot_entity.get_dofs_velocity(
            self._all_dof_idx_local, envs_idx=envs_idx
        )

    # overwrite base method to implement combined arm+hand control
    def _apply_arm_rel_hand_abs_control(self, act: JointPosAction) -> None:
        assert act.joint_pos.shape == (
            self._num_envs,
            self._dof_dim,
        ), "Action must match the number of joints."
        action_ema = self._args.action_ema

        # for base, we use relative position control
        base_target_pos = self._dof_pos[:, :self._arm_dof_dim] + act.joint_pos[:, :self._arm_dof_dim]
        self._robot_entity.control_dofs_position(
            position=base_target_pos,
            dofs_idx_local=self._base_dof_idx_local,
        )
        self._last_base_target = base_target_pos

        # for hand, we use absolute position control
        hand_pos = act.joint_pos[:, self._arm_dof_dim:]
        hand_dofs_limits = self._dof_pos_limits[:, self._arm_dof_dim:, :]  # (n_envs, hand_dofs, 2)
        limits_lo = hand_dofs_limits[:, :, 0]  # (n_envs, hand_dofs)
        limits_hi = hand_dofs_limits[:, :, 1]  # (n_envs, hand_dofs)
        hand_target_pos = (hand_pos + 1.0) / 2.0 * (limits_hi - limits_lo) + limits_lo
        hand_target_pos = torch.clamp(hand_target_pos, limits_lo, limits_hi)
        hand_target_pos = action_ema * hand_target_pos + (1 - action_ema) * self._dof_pos[:, self._arm_dof_dim:]
        self._robot_entity.control_dofs_position(
            position=hand_target_pos,
            dofs_idx_local=self._hand_dof_idx_local,
        )
        self._last_hand_target = hand_target_pos

    @property
    def arm_links_contact_force(self) -> torch.Tensor:
        """Net contact force on each arm link. Shape: (n_envs, n_arm_links, 3)."""
        all_contact = self._robot_entity.get_links_net_contact_force()
        return all_contact[:, self._arm_link_idx_local, :]

    @property
    def base_pos(self) -> torch.Tensor:
        return self._robot_entity.get_links_pos(
            links_idx_local=[self._wrist_link_idx]
        ).squeeze(1)

    @property
    def base_quat(self) -> torch.Tensor:
        return self._robot_entity.get_links_quat(
            links_idx_local=[self._wrist_link_idx]
        ).squeeze(1)

    @property
    def base_lin_vel(self) -> torch.Tensor:
        return self._robot_entity.get_links_vel(
            links_idx_local=[self._wrist_link_idx]
        ).squeeze(1)

    @property
    def base_ang_vel(self) -> torch.Tensor:
        return self._robot_entity.get_links_ang(
            links_idx_local=[self._wrist_link_idx]
        ).squeeze(1)

