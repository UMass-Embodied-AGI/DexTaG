"""Trajectory-following environment for the xArm7 + WUJI hand.

Trains a policy to track reference hand-object trajectories from demonstrations,
with a tactile map as part of the observation.
"""
import hashlib
import importlib
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from genesis.utils.geom import inv_transform_by_quat
from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.utils.math_utils import (
    quat_angle_ignoring_axis,
    quat_apply,
    quat_conjugate,
    quat_from_euler,
    quat_mul,
)
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.sim.envs.config.schema import SingleHandRetargetingEnvArgs
from gs_env.sim.envs.manipulation.trajectory_utils import augment_trajectory, pad_and_stack, process_trajectory_data_tactile_map, query_object_sdf, resample_trajectory_lists, validate_on_table
from gs_env.sim.robots.manipulators import XArmWUJIHand
from gs_env.sim.scenes import FlatScene

_DEFAULT_DEVICE = torch.device("cpu")


class SingleHandRetargetingEnvTactileMap(BaseEnv):
    """Track a reference hand-object trajectory while manipulating the object.

    Observations and rewards are selected by name from ``args``
    (``actor_obs_terms`` / ``critic_obs_terms`` / ``reward_args``), so every
    such term is a buffer exposed as an attribute of this class.

    Actions (27D): 7 arm joint-position deltas followed by 20 absolute hand
    joint targets normalized to [-1, 1] over the joint limits (ARM_REL_HAND_ABS).
    """

    def __init__(
        self,
        args: SingleHandRetargetingEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = num_envs
        self._device = device
        self._show_viewer = show_viewer
        self._args = args
        self._eval_mode = eval_mode

        if not gs._initialized:  # noqa: SLF001
            gs.init(performance_mode=True, backend=getattr(gs.constants.backend, device.type))

        # == Load demonstration trajectory data ==
        self._load_trajectory_data()

        # == Apply trajectory offset ==
        self._trajectory_offset = np.array(args.object_config.trajectory_offset)
        if np.any(self._trajectory_offset != 0.0):
            self._wrist_positions += self._trajectory_offset
            self._obj_pose_matrices[:, :, :3, 3] += self._trajectory_offset
            # Also offset MANO fingertip positions in raw demo data
            for demo_data in self._demo_data_raw:
                for joint_name in demo_data["mano_reference"]["finger_joints"]:
                    demo_data["mano_reference"]["finger_joints"][joint_name] = (
                        demo_data["mano_reference"]["finger_joints"][joint_name] + self._trajectory_offset
                    )
            print(f"Applied trajectory offset: {self._trajectory_offset}")

        # == setup the scene ==
        self._scene = FlatScene(
            num_envs=self._num_envs,
            args=args.scene_args,
            show_viewer=self._show_viewer,
        )

        # == setup the robot ==
        self._robot = XArmWUJIHand(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self.device,
        )
        self._hand_dof_dim = self._robot._gripper_dof_dim
        self._base_dof_dim = self._robot._arm_dof_dim

        # == setup object ==
        obj_cfg = self._args.object_config
        obj_id = obj_cfg.object_id
        obj_mesh_file = f"{obj_cfg.trajectory_path}/{obj_id}_collision.obj"
        if self._args.object_domain_randomization and not eval_mode:
            # Heterogeneous morphs: different scales across environments
            scale_lo, scale_hi = obj_cfg.scale_range
            n_variants = min(obj_cfg.n_scale_variants, self._num_envs)
            obj_scales = np.linspace(scale_lo, scale_hi, n_variants).tolist()
            obj_morph = [
                gs.morphs.Mesh(
                    file=obj_mesh_file,
                    pos=[0.0, 0.0, 0.0],
                    euler=(90, 0, 0),
                    scale=s,
                )
                for s in obj_scales
            ]
            print(f"Object domain randomization: {n_variants} scale variants in [{scale_lo}, {scale_hi}]")
        else:
            obj_morph = gs.morphs.Mesh(
                file=obj_mesh_file,
                pos=[0.0, 0.0, 0.0],
                euler=(90, 0, 0),
                scale=1.0,
            )
        self._object = self._scene.scene.add_entity(
            morph=obj_morph,
            material=gs.materials.Rigid(
                rho=100.0,
                friction=0.8,
            ),
            vis_mode="visual"
        )

        # == Target visualization (controlled by --env.target_visualization) ==
        self._ghost_hand = None
        self._fingertip_markers = []
        self._target_object = None
        vis_mode = self._args.target_visualization
        if self._eval_mode and self._show_viewer and vis_mode is not None:
            if vis_mode == "ghost_hand":
                ghost_morph = gs.morphs.URDF(
                    file="assets/robot/xarm/wujihand_right_v5.urdf",
                    merge_fixed_links=False,
                    pos=args.robot_args.morph_args.pos,
                    euler=args.robot_args.morph_args.euler,
                    fixed=args.robot_args.morph_args.fixed,
                    is_free=args.robot_args.morph_args.is_free,
                    collision=False,
                    recompute_inertia=True
                )
                self._ghost_hand = self._scene.scene.add_entity(
                    morph=ghost_morph,
                    vis_mode="visual",
                )
            elif vis_mode == "fingertip_markers":
                for _ in range(5):
                    marker = self._scene.scene.add_entity(
                        gs.morphs.Mesh(
                            file="assets/scene/axis.obj",
                            scale=0.02,
                            collision=False,
                        ),
                    )
                    self._fingertip_markers.append(marker)
            elif vis_mode == "target_object":
                self._target_object = self._scene.scene.add_entity(
                    gs.morphs.Mesh(
                        file=f"{obj_cfg.trajectory_path}/{obj_id}_collision.obj",
                        pos=[0.0, 0.0, 0.0],
                        euler=(90, 0, 0),
                        scale=1.0,
                        collision=False,
                    ),
                )

        # == set up wrist camera (GUI window only in eval with viewer + use_wrist_depth_camera) ==
        self._wrist_camera = self._scene.scene.add_camera(
            res=self._args.wrist_depth_camera_resolution,
            fov=58,
            GUI=self._show_viewer and self._eval_mode and self._args.use_wrist_depth_camera,
            near=0.04,
            far=0.3
        )
        self._use_pointcloud = self._args.use_wrist_pointcloud
        self._pointcloud_num_points = self._args.wrist_pointcloud_num_points

        # == setup tactile sensors (always on) ==
        self._tactile_sensors = {}
        self._tactile_points = {}
        self._tactile_sensor_configs = {}
        self._setup_tactile_sensors(args)

        # Initialize TactileVisualizer for converting flat forces → 2D tactile map
        from genesis.vis import TactileVisualizer as GenesisTactileVisualizer
        self._tactile_map_converter = GenesisTactileVisualizer(
            tactile_grid_path=args.robot_args.tactile_grid_path,
            pixel_mapping_path=args.robot_args.tactile_pixel_mapping_path,
            num_envs=self._num_envs,
            show_viewer=False,
            device=self._device,
        )

        # == add table when trajectory offset is non-zero ==
        if np.any(self._trajectory_offset != 0.0):
            off = self._trajectory_offset
            self._table = self._scene.scene.add_entity(
                gs.morphs.Box(
                    size=(0.6, 1.0, 0.02),
                    pos=(off[0] + 0.5, off[1], off[2] - 0.01),
                    collision=True,
                    fixed=True,
                ),
            )

        # == build the scene ==
        self._scene.build()

        # Cache object geometries for SDF-based fingertip-to-surface distance queries
        if self._object._enable_heterogeneous:
            # Build per-variant geom lists for efficient heterogeneous SDF queries
            n_variants = len(self._object.variants_geom_start)
            n_envs = self._num_envs
            base = n_envs // n_variants
            extra = n_envs % n_variants
            sizes = np.concatenate([np.full(extra, base + 1, dtype=int),
                                    np.full(n_variants - extra, base, dtype=int)])
            link = self._object.links[0]
            link_geom_start = link._geom_start
            self._variant_blocks = []
            env_cursor = 0
            for v_idx, sz in enumerate(sizes):
                if sz > 0:
                    gs_start = self._object.variants_geom_start[v_idx] - link_geom_start
                    gs_end = self._object.variants_geom_end[v_idx] - link_geom_start
                    self._variant_blocks.append((
                        env_cursor,
                        env_cursor + sz,
                        list(link.geoms[gs_start:gs_end]),
                    ))
                    env_cursor += sz
            self._object_geoms = []  # Not used in heterogeneous path
            print(f"Heterogeneous SDF: {n_variants} variants, {len(link.geoms)} total geoms, "
                  f"{len(self._variant_blocks)} variant blocks")
        else:
            self._variant_blocks = None
            self._object_geoms = list(self._object.geoms)

        # == attach wrist camera to xArm link7 ==
        wrist_cam_transform = np.array([
            [-1.90808995e-01, -9.81627183e-01, -4.25007252e-17, -1.35471517e-02],
            [ 4.69711634e-01, -9.13026926e-02, -8.78085872e-01, -9.98060843e-02],
            [ 8.61952962e-01, -1.67546683e-01,  4.78503083e-01,  3.77781370e-02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ])
        additional_T = np.array([
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        wrist_cam_transform = wrist_cam_transform @ additional_T
        self._wrist_camera.attach(self._robot._robot_entity.get_link("link7"), wrist_cam_transform)

        # == initialize robot limits after scene is built ==
        self._robot.post_build_init(eval_mode=eval_mode)

        # == Randomize table height per-env (sampled once at init) ==
        table_height_noise = obj_cfg.table_height_noise
        if hasattr(self, '_table') and table_height_noise > 0.0 and not eval_mode:
            off = self._trajectory_offset
            self._table_height_offsets = (torch.rand(self._num_envs, device=self._device) * 2 - 1) * table_height_noise  # (num_envs,)
            table_pos = torch.tensor([off[0] + 0.5, off[1], off[2] - 0.01], device=self._device).unsqueeze(0).expand(self._num_envs, -1).clone()
            table_pos[:, 2] += self._table_height_offsets
            self._table.set_pos(table_pos)
            print(f"Table height randomization: ±{table_height_noise*100:.1f}cm, range [{self._table_height_offsets.min():.4f}, {self._table_height_offsets.max():.4f}]")
        else:
            self._table_height_offsets = torch.zeros(self._num_envs, device=self._device)

        # == Process trajectory data for RL training ==
        self._process_trajectory_data()

        # Apply object physics overrides from ObjectConfig
        if obj_cfg.mass is not None:
            print(f"Setting object mass to {obj_cfg.mass}")
            if self._object._enable_heterogeneous:
                mass_tensor = torch.full((self.num_envs, 1), obj_cfg.mass, device=self._device)
            else:
                mass_tensor = torch.tensor([obj_cfg.mass], device=self._device)
            self._object.set_links_inertial_mass(mass_tensor, links_idx_local=[0])
        if any(v != 0.0 for v in obj_cfg.com_shift):
            print(f"Setting object COM shift to {obj_cfg.com_shift}")
            self._object.set_COM_shift(
                torch.tensor(obj_cfg.com_shift).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, 1, 1)
            )
        if args.object_domain_randomization and not eval_mode:
            self._object_domain_randomization()

        # == Random perturbation setup ==
        self._perturbation_force_scale = obj_cfg.perturbation_force_scale if not eval_mode else 0.0
        self._perturbation_torque_scale = obj_cfg.perturbation_torque_scale if not eval_mode else 0.0

        if self._perturbation_force_scale > 0.0:
            lo, hi = obj_cfg.perturbation_force_prob_range
            self._force_prob = torch.exp(
                torch.log(torch.tensor(lo)) +
                (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo))) *
                torch.rand(self.num_envs, device=self._device)
            )  # (num_envs,) — per-env probability, sampled log-uniformly

        if self._perturbation_torque_scale > 0.0:
            lo, hi = obj_cfg.perturbation_torque_prob_range
            self._torque_prob = torch.exp(
                torch.log(torch.tensor(lo)) +
                (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo))) *
                torch.rand(self.num_envs, device=self._device)
            )

        # Table surface height (used for perturbation lift gating and the
        # height-gated velocity-change penalties). Defined unconditionally.
        self._table_surface_z = self._trajectory_offset[2] if np.any(self._trajectory_offset != 0.0) else 0.0

        if self._perturbation_force_scale > 0.0 or self._perturbation_torque_scale > 0.0:
            # Cache object mass for mass-scaled forces
            raw_mass = self._object.get_mass()
            if isinstance(raw_mass, (int, float)):
                self._object_mass = torch.full((self.num_envs, 1), raw_mass, device=self._device)
            else:
                self._object_mass = torch.tensor(raw_mass, device=self._device).reshape(self.num_envs, 1)

            # Global link index for the scene rigid solver (not entity-local)
            self._object_link_idx = self._object.links[0].idx

            self._perturbation_lift_threshold = obj_cfg.perturbation_lift_threshold
            self._perturbation_curriculum_start = obj_cfg.perturbation_curriculum_start
            self._perturbation_curriculum_end = obj_cfg.perturbation_curriculum_end

        # == setup reward functions ==
        dt = self._scene.scene.dt
        self._reward_functions = {}
        self._reward_required_keys: set[str] = set()

        # Load reward terms from config
        reward_term = getattr(args, 'reward_term', 'hand_imitator')
        module_name = f"gs_env.common.rewards.{reward_term}_terms"
        module = importlib.import_module(module_name)

        # Classes that should receive the ignore_axis parameter
        _ROTATION_REWARD_CLASSES = {"ObjectRotationTrackingReward"}

        reward_args = getattr(args, 'reward_args', {})
        for key in reward_args.keys():
            reward_cls = getattr(module, key, None)
            if reward_cls is None:
                raise ValueError(f"Reward {key} not found in rewards module {module_name}.")
            scale = reward_args[key]["scale"] * dt
            other_args = {k: v for k, v in reward_args[key].items() if k != "scale"}
            # Auto-inject ignore_axis for rotation-related reward terms
            if obj_cfg.rotation_ignore_axis is not None and key in _ROTATION_REWARD_CLASSES:
                other_args.setdefault("ignore_axis", obj_cfg.rotation_ignore_axis)
            reward_instance = reward_cls(scale=scale, **other_args)
            self._reward_functions[key] = reward_instance
            # Record declared inputs for this reward term
            if hasattr(reward_instance, "required_keys"):
                self._reward_required_keys.update(getattr(reward_instance, "required_keys", ()))

        # Cache ignore axis tensor for termination logic
        self._ignore_axis_tensor: torch.Tensor | None = None
        if obj_cfg.rotation_ignore_axis is not None:
            self._ignore_axis_tensor = torch.tensor(obj_cfg.rotation_ignore_axis, dtype=torch.float32, device=device)

        # Environment parameters
        self._obs_future_length = args.obs_future_length

        # Initialize buffers
        self._init()

        self.reset()

    def _load_trajectory_data(self) -> None:
        """Load multiple demonstration trajectory data from pickle files."""
        trajectory_path = Path(self._args.object_config.trajectory_path)

        # Load all .pkl files from directory
        traj_files = sorted(trajectory_path.glob("*.pkl"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files found in directory: {trajectory_path}")

        print(f"\n{'='*80}")
        print(f"Loading {len(traj_files)} trajectories from: {trajectory_path}")
        print(f"{'='*80}")

        all_demo_data = []
        all_wrist_positions = []
        all_wrist_rotations_aa = []
        all_dof_positions = []
        all_obj_pose_matrices = []
        all_tactile_maps = []
        all_traj_lengths = []
        all_traj_ids = []

        # Trajectories to skip (e.g. cause arm self-collision at starting pose)
        skip_filenames: set[str] = set(self._args.skip_trajectories)

        # Train/eval split: deterministic hash on the timestamp ID prefix
        split_mode = self._args.split_mode
        train_ratio = self._args.train_ratio
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be in [0, 1], got {train_ratio}")

        def _split_bucket(split_key: str) -> float:
            salt = f"{self._args.split_seed}:" if self._args.split_seed != 0 else ""
            digest = hashlib.md5((salt + split_key).encode()).digest()
            return int.from_bytes(digest[:8], "big") / 2**64  # in [0, 1)

        n_skipped_split: int = 0
        n_skipped_obj: int = 0
        n_skipped_list: int = 0

        for traj_idx, traj_file in enumerate(traj_files):
            with open(traj_file, "rb") as f:
                data = pickle.load(f)

            if data["metadata"]["obj_id"] != self._args.object_config.object_id:
                n_skipped_obj += 1
                continue  # skip trajectories not matching the specified object_id

            if any(skip_name in traj_file.name for skip_name in skip_filenames):
                print(f"  - SKIPPED (matches skip pattern): {traj_file.name}")
                n_skipped_list += 1
                continue

            # Train/eval split filter
            if split_mode != "all":
                # split_key = full stem when no "__" postfix; otherwise the prefix
                # before "__", so all postfix variants share a bucket.
                split_key = traj_file.stem.split("__")[0]
                bucket = _split_bucket(split_key)
                in_train = bucket < train_ratio
                if (split_mode == "train" and not in_train) or (
                    split_mode == "eval" and in_train
                ):
                    n_skipped_split += 1
                    continue

            all_demo_data.append(data)
            hand_traj = data["hand_trajectory"]

            wrist_pos = hand_traj["wrist_positions"]  # (T, 3)
            wrist_rot = hand_traj["wrist_rotations_aa"]  # (T, 3)
            dof_pos = hand_traj["dof_positions"]  # (T, n_dofs)
            obj_pose_matrix = data["object_trajectory"]["pose_matrices"]  # (T, 4, 4)

            # Load tactile map: (T, 24, 32) int32 -> zero values < 60, clamp [0,511], scale 1/60, flatten to (T, 768) float32
            tactile_map_raw = data["tactile_map"]  # (T, 24, 32) int32
            tactile_map_raw = np.where(tactile_map_raw < 60, 0, tactile_map_raw)  # filter noise below 60
            tactile_map = np.clip(tactile_map_raw, 0, 511).astype(np.float32) / 60.0  # (T, 24, 32) float32
            tactile_map = tactile_map.reshape(tactile_map.shape[0], -1)  # (T, 768)

            all_wrist_positions.append(wrist_pos)
            all_wrist_rotations_aa.append(wrist_rot)
            all_dof_positions.append(dof_pos)
            all_obj_pose_matrices.append(obj_pose_matrix)
            all_tactile_maps.append(tactile_map)
            all_traj_lengths.append(len(wrist_pos))

            all_traj_ids.append(traj_file.stem)

            print(f"  [{len(all_demo_data)-1}] {traj_file.name} — {len(wrist_pos)} timesteps")

        print(
            f"\nTrajectory split summary: kept={len(all_demo_data)}, "
            f"skipped_obj_id={n_skipped_obj}, skipped_skip_list={n_skipped_list}, "
            f"skipped_split={n_skipped_split} "
            f"(split_mode={split_mode}, train_ratio={train_ratio}, "
            f"split_seed={self._args.split_seed})"
        )

        if len(all_demo_data) == 0:
            raise ValueError(f"No trajectories found for object_id: {self._args.object_config.object_id}")

        # Resample trajectories to target playback rate if different from 60 Hz
        if self._args.trajectory_playback_hz != 60.0:
            resample_trajectory_lists(
                all_wrist_positions, all_wrist_rotations_aa, all_dof_positions,
                all_obj_pose_matrices, all_tactile_maps, all_traj_lengths, all_demo_data,
                target_hz=self._args.trajectory_playback_hz,
            )

        # Store raw data and individual trajectory lengths
        self._demo_data_raw = all_demo_data
        self._num_trajectories = len(all_demo_data)
        self._traj_lengths = np.array(all_traj_lengths)
        self._max_traj_length = max(all_traj_lengths)
        self._traj_ids = all_traj_ids

        # Pad and combine trajectories
        self._wrist_positions = pad_and_stack(all_wrist_positions)  # (num_traj, max_T, 3)
        self._wrist_rotations_aa = pad_and_stack(all_wrist_rotations_aa)  # (num_traj, max_T, 3)
        self._dof_positions = pad_and_stack(all_dof_positions)  # (num_traj, max_T, n_dofs)
        self._obj_pose_matrices = pad_and_stack(all_obj_pose_matrices)  # (num_traj, max_T, 4, 4)
        self._tactile_maps = pad_and_stack(all_tactile_maps)  # (num_traj, max_T, 768)

        print(f"\nCombined trajectory shapes:")
        print(f"  - Wrist positions: {self._wrist_positions.shape}")
        print(f"  - Wrist rotations: {self._wrist_rotations_aa.shape}")
        print(f"  - DOF positions: {self._dof_positions.shape}")
        print(f"  - Max trajectory length: {self._max_traj_length}")
        print(f"  - Trajectory lengths: {self._traj_lengths}")
        print(f"{'='*80}\n")

    def _process_trajectory_data(self) -> None:
        """Process raw trajectory data for RL training."""
        time_delta = self._scene.scene.dt * self._args.robot_args.decimation
        self._traj_data_template, self._tip_joint_mapping = process_trajectory_data_tactile_map(
            self._wrist_positions,
            self._wrist_rotations_aa,
            self._dof_positions,
            self._obj_pose_matrices,
            self._tactile_maps,
            self._demo_data_raw,
            self._args.joint_mapping,
            self._device,
            time_delta=time_delta,
        )

        # Move template to CPU to save GPU memory
        for key in self._traj_data_template:
            self._traj_data_template[key] = self._traj_data_template[key].cpu()

        # Build augmented per-env trajectory data if enabled
        aug_cfg = self._args.trajectory_augmentation
        if aug_cfg.enabled and not self._eval_mode:
            self._traj_data_augmented, self._augmented_traj_lengths = self._build_augmented_trajectories(aug_cfg)
        else:
            self._traj_data_augmented = None
            self._augmented_traj_lengths = None

        # Get the corresponding link idx for fingertips (MANO tracking)
        self._finger_link_idxs = []
        for link_name in self._tip_joint_mapping.values():
            self._finger_link_idxs.append(self._robot.get_link(link_name).idx_local)

        # Cache tactile fingertip link indices
        self._tactile_fingertip_link_names = self._args.tactile_fingertip_link_names
        self._tactile_fingertip_link_idxs = [
            self._robot.get_link(name).idx_local for name in self._tactile_fingertip_link_names
        ]

        # Precompute tactile center offsets for each tactile fingertip link.
        # For each link, find the tactile point closest to the centroid of all
        # tactile points on that link. This point's local offset will be used in
        # _compute_fingertip_to_surface_dist instead of the link COM.
        tactile_center_offsets = []
        for link_name in self._tactile_fingertip_link_names:
            if link_name in self._tactile_points:
                local_pts = self._tactile_points[link_name]  # (N, 3) numpy
                centroid = local_pts.mean(axis=0)  # (3,)
                assert "v5.urdf" in self._args.robot_args.morph_args.file
                centroid[2] += self._args.fingertip_tactile_center_z_offset
                dists = np.linalg.norm(local_pts - centroid, axis=1)
                closest_idx = int(np.argmin(dists))
                tactile_center_offsets.append(local_pts[closest_idx])
            else:
                # Fallback: zero offset (same as link origin)
                tactile_center_offsets.append(np.zeros(3, dtype=np.float32))
        # (5, 3) tensor on device
        self._fingertip_tactile_center_offsets = torch.tensor(
            np.stack(tactile_center_offsets, axis=0), dtype=torch.float32, device=self._device,
        )

    def _build_augmented_trajectories(self, aug_cfg):
        """Create per-env augmented trajectories from the raw numpy data.

        For each of num_envs environments, picks its round-robin base trajectory,
        applies random augmentation, validates that the object stays on the table,
        then processes through the full pipeline on CPU.

        Returns (traj_data_augmented dict on CPU, augmented_traj_lengths tensor).
        """
        num_envs = self._num_envs
        num_traj = self._num_trajectories
        max_T = self._max_traj_length

        # Round-robin assignment (same as _init)
        env_traj_idx = np.array([i % num_traj for i in range(num_envs)])

        # Table bounds for validation
        off = self._trajectory_offset
        table_x_range = (off[0] + 0.2, off[0] + 0.8)
        table_y_range = (off[1] - 0.5, off[1] + 0.5)

        # Pre-allocate augmented arrays
        aug_wrist_pos = np.zeros((num_envs, max_T, 3), dtype=np.float64)
        aug_wrist_rot = np.zeros((num_envs, max_T, 3), dtype=np.float64)
        aug_dof_pos = np.zeros_like(self._dof_positions[:1]).repeat(num_envs, axis=0)
        aug_obj_pose = np.zeros((num_envs, max_T, 4, 4), dtype=np.float64)
        aug_tactile = np.zeros_like(self._tactile_maps[:1]).repeat(num_envs, axis=0)
        aug_demo_data = []
        aug_traj_lengths = np.zeros(num_envs, dtype=np.int64)

        n_valid = 0
        n_rejected = 0
        max_retries = 10

        for env_idx in range(num_envs):
            traj_idx = env_traj_idx[env_idx]
            traj_len = self._traj_lengths[traj_idx]
            aug_traj_lengths[env_idx] = traj_len

            # Copy base arrays
            aug_dof_pos[env_idx] = self._dof_positions[traj_idx]
            aug_tactile[env_idx] = self._tactile_maps[traj_idx]

            # Deep copy MANO reference for this env
            base_demo = self._demo_data_raw[traj_idx]
            demo_entry = {
                "mano_reference": {
                    "finger_joints": {k: v.copy() for k, v in base_demo["mano_reference"]["finger_joints"].items()},
                    "finger_joints_quat": {k: v.copy() for k, v in base_demo["mano_reference"]["finger_joints_quat"].items()},
                }
            }

            # Sample and augment with rejection sampling
            for retry in range(max_retries):
                tx = np.random.uniform(*aug_cfg.translate_x_range)
                ty = np.random.uniform(*aug_cfg.translate_y_range)
                rz = np.random.uniform(*aug_cfg.rotate_z_range)
                sc = np.random.uniform(*aug_cfg.scale_range)

                # Copy base trajectory data trimmed to actual length (fresh each retry)
                wrist_pos = self._wrist_positions[traj_idx, :traj_len].copy()
                wrist_rot = self._wrist_rotations_aa[traj_idx, :traj_len].copy()
                obj_pose = self._obj_pose_matrices[traj_idx, :traj_len].copy()
                fj = {k: v.copy() for k, v in base_demo["mano_reference"]["finger_joints"].items()}
                fq = {k: v.copy() for k, v in base_demo["mano_reference"]["finger_joints_quat"].items()}

                aug_wp, aug_wr, aug_op, aug_fj, aug_fq = augment_trajectory(
                    wrist_pos, wrist_rot, obj_pose,
                    finger_joints=fj, finger_joints_quat=fq,
                    translate_x=tx, translate_y=ty, rotate_z_deg=rz, scale=sc,
                )

                if validate_on_table(aug_op[0], table_x_range, table_y_range):
                    break
                n_rejected += 1
            else:
                # All retries failed — use unaugmented
                aug_wp = self._wrist_positions[traj_idx, :traj_len].copy()
                aug_wr = self._wrist_rotations_aa[traj_idx, :traj_len].copy()
                aug_op = self._obj_pose_matrices[traj_idx, :traj_len].copy()
                aug_fj = {k: v.copy() for k, v in base_demo["mano_reference"]["finger_joints"].items()}
                aug_fq = {k: v.copy() for k, v in base_demo["mano_reference"]["finger_joints_quat"].items()}

            # Pad augmented arrays back to max_T (repeat last frame)
            aug_wrist_pos[env_idx, :traj_len] = aug_wp
            aug_wrist_pos[env_idx, traj_len:] = aug_wp[-1]
            aug_wrist_rot[env_idx, :traj_len] = aug_wr
            aug_wrist_rot[env_idx, traj_len:] = aug_wr[-1]
            aug_obj_pose[env_idx, :traj_len] = aug_op
            aug_obj_pose[env_idx, traj_len:] = aug_op[-1]

            # Store augmented MANO data (unpadded, trimmed to original length)
            for jn in aug_fj:
                demo_entry["mano_reference"]["finger_joints"][jn] = aug_fj[jn]
            for jn in aug_fq:
                demo_entry["mano_reference"]["finger_joints_quat"][jn] = aug_fq[jn]
            aug_demo_data.append(demo_entry)

            n_valid += 1

        print(f"Trajectory augmentation: {n_valid} envs augmented, {n_rejected} rejections")

        # Process augmented data through the pipeline on CPU to avoid GPU memory spike
        traj_data_augmented, _ = process_trajectory_data_tactile_map(
            aug_wrist_pos,
            aug_wrist_rot,
            aug_dof_pos,
            aug_obj_pose,
            aug_tactile,
            aug_demo_data,
            self._args.joint_mapping,
            torch.device("cpu"),
        )

        augmented_traj_lengths = torch.tensor(aug_traj_lengths, dtype=torch.long)
        return traj_data_augmented, augmented_traj_lengths

    def _init(self) -> None:
        """Initialize observation and action spaces and environment buffers."""

        # Assign trajectories to environments round-robin
        self.env_traj_idx = torch.tensor(
            [i % self._num_trajectories for i in range(self.num_envs)],
            dtype=torch.long,
            device=self._device
        )

        # Per-environment trajectory lengths
        self.env_traj_lengths = torch.tensor(
            [self._traj_lengths[i % self._num_trajectories] for i in range(self.num_envs)],
            dtype=torch.long,
            device=self._device
        )

        # Create per-environment trajectory data (duplicated from template)
        max_T = self._max_traj_length
        self._traj_data = {
            "wrist_pos": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "wrist_quat": torch.zeros((self.num_envs, max_T, 4), device=self._device),
            "wrist_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "wrist_ang_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "dof_pos": torch.zeros((self.num_envs, max_T, self._hand_dof_dim), device=self._device),
            "dof_vel": torch.zeros((self.num_envs, max_T, self._hand_dof_dim), device=self._device),
            "obj_pos": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "obj_quat": torch.zeros((self.num_envs, max_T, 4), device=self._device),
            "obj_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "obj_ang_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "tactile_map_ref": torch.zeros((self.num_envs, max_T, 24 * 32), device=self._device),
            "mano_joint_poses": torch.zeros((self.num_envs, max_T, 5, 3), device=self._device),
            "mano_joint_velocities": torch.zeros((self.num_envs, max_T, 5, 3), device=self._device),
            "mano_joint_quats": torch.zeros((self.num_envs, max_T, 5, 4), device=self._device),
        }

        # Initialize with trajectory data
        if self._traj_data_augmented is not None:
            # Use pre-computed augmented per-env data (CPU -> GPU)
            for key in self._traj_data.keys():
                self._traj_data[key] = self._traj_data_augmented[key].to(self._device)
            self.env_traj_lengths = self._augmented_traj_lengths.to(self._device)
        else:
            # Copy from template (CPU -> GPU) based on env_traj_idx
            env_traj_idx_cpu = self.env_traj_idx.cpu()
            for env_idx in range(self.num_envs):
                traj_idx = env_traj_idx_cpu[env_idx]
                for key in self._traj_data.keys():
                    self._traj_data[key][env_idx] = self._traj_data_template[key][traj_idx].to(self._device)

        # Current index into the trajectory timeline
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

        # Hand state buffers
        self.hand_dof_pos = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)
        self.hand_dof_vel = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)

        # Wrist state buffers
        self.base_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self._device)  # [w, x, y, z]
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self._device)

        self.arm_dof_pos = torch.zeros((self.num_envs, self._base_dof_dim), device=self._device)
        self.arm_dof_vel = torch.zeros((self.num_envs, self._base_dof_dim), device=self._device)
        self.dof_vel = torch.zeros((self.num_envs, self._hand_dof_dim + self._base_dof_dim), device=self._device)

        # Object state buffers
        self.object_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.object_pos_rel = torch.zeros((self.num_envs, 3), device=self._device)  # Relative to wrist base
        self.object_quat = torch.zeros((self.num_envs, 4), device=self._device)
        self.object_lin_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.object_ang_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.object_vel_change = torch.zeros((self.num_envs,), device=self._device)
        self.object_ang_vel_change = torch.zeros((self.num_envs,), device=self._device)
        self.object_height_above_table = torch.zeros((self.num_envs,), device=self._device)
        self.fingertip_to_surface_dist = torch.zeros((self.num_envs, 5), device=self._device)  # SDF-based distance
        self.table_contact_force_magnitude = torch.zeros((self.num_envs, 1), device=self._device)
        self.last_reward = torch.zeros((self.num_envs, 1), device=self._device)

        # Buffer related to target trajectory following
        # base
        self.target_wrist_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.delta_wrist_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.target_wrist_quat = torch.zeros((self.num_envs, 4), device=self._device)
        self.target_wrist_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.target_wrist_ang_vel = torch.zeros((self.num_envs, 3), device=self._device)
        # dof
        self.target_hand_dof_pos = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)
        self.target_hand_dof_vel = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)
        # object
        self.target_object_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.delta_object_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.target_object_quat = torch.zeros((self.num_envs, 4), device=self._device)
        self.target_object_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.target_object_ang_vel = torch.zeros((self.num_envs, 3), device=self._device)
        # tactile reference
        self.target_tactile_map_flat = torch.zeros((self.num_envs, 24 * 32), device=self._device)

        # Reference fingertip position buffers for reward computation (5 tips)
        K = self._obs_future_length
        self.target_mano_joint_pos = torch.zeros((self.num_envs, 5 * 3), device=self._device)
        self.finger_link_pos = torch.zeros((self.num_envs, 5 * 3), device=self._device)
        self.target_mano_joint_vel = torch.zeros((self.num_envs, 5 * 3), device=self._device)
        self.finger_link_vel = torch.zeros((self.num_envs, 5 * 3), device=self._device)
        self.target_mano_joint_quat = torch.zeros((self.num_envs, 5 * 4), device=self._device)
        self.finger_link_quat = torch.zeros((self.num_envs, 5 * 4), device=self._device)

        # Compute proprio_dim from config term list
        self._proprio_dim = sum(getattr(self, t).shape[-1] for t in self._args.proprio_terms)
        self.proprio_history_buf = torch.zeros(
            (self.num_envs, self._args.obs_history_len, self._proprio_dim), device=self._device
        )
        action_dim = self._hand_dof_dim + self._base_dof_dim
        self.action_history_buf = torch.zeros(
            (self.num_envs, self._args.obs_history_len, action_dim), device=self._device
        )
        self.action_proprio_history_flat = torch.zeros(
            (self.num_envs, (action_dim + self._proprio_dim) * self._args.obs_history_len), device=self._device
        )

        # Compute target_action_dim from config term list
        self._target_action_dim = sum(getattr(self, t).shape[-1] for t in self._args.target_motion_terms)
        self.target_action_flat = torch.zeros(
            (self.num_envs, self._target_action_dim * K), device=self._device
        )

        # Student-specific buffers for BC distillation
        if self._args.is_student:
            student_target_terms = self._args.target_motion_terms_student
            student_proprio_terms = self._args.proprio_terms_student
            assert student_target_terms and student_proprio_terms  # validator guarantees this

            self._student_target_action_dim = sum(getattr(self, t).shape[-1] for t in student_target_terms)
            self.target_action_student_flat = torch.zeros(
                (self.num_envs, self._student_target_action_dim * K), device=self._device
            )

            self._student_proprio_dim = sum(getattr(self, t).shape[-1] for t in student_proprio_terms)
            self.proprio_history_student_buf = torch.zeros(
                (self.num_envs, self._args.obs_history_len, self._student_proprio_dim), device=self._device
            )
            self.action_proprio_history_student_flat = torch.zeros(
                (self.num_envs, (action_dim + self._student_proprio_dim) * self._args.obs_history_len), device=self._device
            )
        else:
            self.target_action_student_flat = torch.zeros((self.num_envs, 0), device=self._device)
            self.action_proprio_history_student_flat = torch.zeros((self.num_envs, 0), device=self._device)

        # Environment state buffers
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.success_count_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

        # Training step counter (drives table-contact / perturbation / guide schedules)
        self.training_step = 0

        # Per-trajectory rollout length tracking (training only, for logging statistics).
        self._per_traj_ep_len_buf: list[deque] = [deque(maxlen=64) for _ in range(self._num_trajectories)]

        # Action scale
        self._action_scale = self._args.robot_args.action_scale
        self._base_action_scale = self._args.robot_args.base_action_scale

        # Action history length and latency
        self._action_history_len = self._args.obs_history_len
        self._action_latency_min, self._action_latency_max = self._args.action_latency_range
        assert self._action_latency_max < self._action_history_len, \
            f"max latency {self._action_latency_max} must be < history len {self._action_history_len}"

        # Torques buffer for tracking
        self.dof_force = torch.zeros((self.num_envs, self._hand_dof_dim + self._base_dof_dim), device=self._device)

        # Arm-link contact force buffer (any arm contact triggers termination)
        n_arm_links = len(self._robot._ARM_LINK_NAMES)
        self.arm_contact_force = torch.zeros((self.num_envs, n_arm_links, 3), device=self._device)

        # 2D tactile map buffer (flattened)
        h, w = self._tactile_map_converter.image_shape  # (24, 32)
        self.tactile_map_flat = torch.zeros((self.num_envs, h * w), device=self._device)
        # Build per-pixel weight tensor for weighted tactile reward
        self.tactile_pixel_weights = self._build_tactile_pixel_weights(h, w)
        # Build per-fingertip pixel masks for tactile-gated proximity reward
        self.fingertip_pixel_masks = self._build_fingertip_pixel_masks(h, w)

        # Point cloud buffers
        if self._use_pointcloud:
            self.wrist_pointcloud_flat = torch.zeros(
                (self.num_envs, self._pointcloud_num_points * 3), device=self._device
            )
        else:
            self.wrist_pointcloud_flat = torch.zeros((self.num_envs, 0), device=self._device)

        # ===== Build observation spaces =====
        actor_obs_dim = 0
        for term in self._args.actor_obs_terms:
            actor_obs_dim += getattr(self, term).shape[-1]
        critic_obs_dim = 0
        for term in self._args.critic_obs_terms:
            critic_obs_dim += getattr(self, term).shape[-1]

        # Build observation spaces
        self._actor_observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(actor_obs_dim,), dtype=np.float32
            ),
        })

        self._critic_observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(critic_obs_dim,), dtype=np.float32
            ),
        })

        self._extra_info = {}

        # Stochastic actor-observation delay queue (when student_obs_delay_max > 0; also active in eval).
        self._obs_delay_max = self._args.student_obs_delay_max
        if self._obs_delay_max > 0:
            self._obs_delay_queue = torch.zeros(
                self.num_envs, self._obs_delay_max, actor_obs_dim,
                dtype=torch.float32, device=self._device,
            )
        else:
            self._obs_delay_queue = None
        # Per-env delay index sampled once per step in get_observations().
        # Cached so get_teacher_observations() can reuse it for matched delay.
        self._step_delay_idx: torch.Tensor | None = None
        # Lazily allocated in get_teacher_observations(): teacher actor obs has
        # a different dim than the student's, so it needs its own queue.
        self._teacher_obs_delay_queue: torch.Tensor | None = None

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        """Reset the given environments to the first frame of their trajectory.

        Training applies initial-pose noise to the object; eval advances each
        env to the next reference trajectory instead.
        """
        if len(envs_idx) == 0:
            return

        num_reset = len(envs_idx)

        # Record per-trajectory rollout length for the episodes that just ended.
        if not self._eval_mode:
            old_traj_idx_cpu = self.env_traj_idx[envs_idx].cpu().numpy()
            ep_len_cpu = self.progress_buf[envs_idx].cpu().numpy()
            for i, t in enumerate(old_traj_idx_cpu):
                self._per_traj_ep_len_buf[int(t)].append(int(ep_len_cpu[i]))

        # Get trajectory indices and lengths for the environments being reset
        reset_traj_idx = self.env_traj_idx[envs_idx]
        reset_traj_lengths = self.env_traj_lengths[envs_idx]

        if self._eval_mode:
            # if in eval mode, select reference trajectory in a rotation manner
            assert num_reset == 1, "Eval mode only supports resetting one env at a time."
            reset_traj_idx = (reset_traj_idx + 1) % self._num_trajectories
            reset_traj_lengths = torch.tensor(
                [self._traj_lengths[reset_traj_idx]],
                dtype=torch.long,
                device=self._device
            )
            self.env_traj_idx[envs_idx] = reset_traj_idx
            self.env_traj_lengths[envs_idx] = reset_traj_lengths

            reset_traj_idx_cpu = reset_traj_idx.cpu()
            for key in self._traj_data.keys():
                self._traj_data[key][envs_idx] = self._traj_data_template[key][reset_traj_idx_cpu].to(self._device)

        # Always start from beginning of trajectory
        seq_idx = torch.zeros(num_reset, dtype=torch.long, device=self._device)


        # Get DOF positions / wrist pose from the first trajectory frame
        dof_pos = self._traj_data["dof_pos"][envs_idx, seq_idx]  # (num_reset, hand_dof_dim)
        dof_vel = self._traj_data["dof_vel"][envs_idx, seq_idx]  # (num_reset, hand_dof_dim)

        wrist_pos = self._traj_data["wrist_pos"][envs_idx, seq_idx]  # (num_reset, 3)
        wrist_quat = self._traj_data["wrist_quat"][envs_idx, seq_idx]  # (num_reset, 4)

        self._robot.reset_to_pose(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            wrist_pos=wrist_pos,
            wrist_quat=wrist_quat,
            envs_idx=envs_idx,
        )

        object_pos = self._traj_data["obj_pos"][envs_idx, seq_idx]  # (num_reset, 3)
        object_pos[:, 2] += 0.01  # small lift at reset to avoid initial table penetration (all objects)
        object_pos[:, 2] += self._table_height_offsets[envs_idx]  # compensate per-env table height noise
        object_quat = self._traj_data["obj_quat"][envs_idx, seq_idx]  # (num_reset, 4)
        object_lin_vel = self._traj_data["obj_vel"][envs_idx, seq_idx]  # (num_reset, 3)
        object_ang_vel = self._traj_data["obj_ang_vel"][envs_idx, seq_idx]  # (num_reset, 3)

        # Apply initial pose randomization (training only)
        if not self._eval_mode:
            obj_cfg = self._args.object_config
            if obj_cfg.init_pos_noise_xy > 0.0:
                xy_noise = (torch.rand(num_reset, 2, device=self._device) * 2 - 1) * obj_cfg.init_pos_noise_xy
                object_pos[:, :2] += xy_noise
            if obj_cfg.init_rot_noise_deg > 0.0:
                noise_rad = obj_cfg.init_rot_noise_deg * torch.pi / 180.0
                euler_noise = (torch.rand(num_reset, 3, device=self._device) * 2 - 1) * noise_rad
                noise_quat = quat_from_euler(euler_noise)
                object_quat = quat_mul(noise_quat, object_quat)

        self._object.set_pos(object_pos, envs_idx=envs_idx)
        self._object.set_quat(object_quat, envs_idx=envs_idx)
        self._object.set_dofs_velocity(
            torch.cat([object_lin_vel, object_ang_vel], dim=-1),
            envs_idx=envs_idx
        )

        # ===== Reset all buffers =====
        self.progress_buf[envs_idx] = seq_idx  # Trajectory position
        self.proprio_history_buf[envs_idx] = 0.0
        self.action_history_buf[envs_idx] = 0.0
        if self._args.is_student:
            self.proprio_history_student_buf[envs_idx] = 0.0
        if self._obs_delay_queue is not None:
            self._obs_delay_queue[envs_idx] = 0.0
        if self._teacher_obs_delay_queue is not None:
            self._teacher_obs_delay_queue[envs_idx] = 0.0
        self.reset_buf[envs_idx] = 0


    def _object_domain_randomization(self) -> None:
        obj_cfg = self._args.object_config
        mass_range = obj_cfg.mass_randomization_range
        friction_lo, friction_hi = obj_cfg.friction_range

        # randomize object mass (per-env absolute mass via set_links_inertial_mass)
        raw_mass = self._object.get_mass()  # numpy array or scalar
        if isinstance(raw_mass, (int, float)):
            obj_mass = torch.full((self.num_envs, 1), raw_mass, device=self._device)
        else:
            obj_mass = torch.tensor(raw_mass, device=self._device).reshape(self.num_envs, 1)
        perm = torch.randperm(self.num_envs, device=self._device)
        n_lower = self.num_envs // 2
        # lower half: mass * (1 - mass_range) to mass
        lower_mass = obj_mass[perm[:n_lower]] * (1.0 - mass_range * torch.rand(n_lower, 1, device=self._device))
        # upper half: mass to mass * (1 + mass_range)
        upper_mass = obj_mass[perm[n_lower:]] * (1.0 + mass_range * torch.rand(self.num_envs - n_lower, 1, device=self._device))
        random_mass = torch.cat([lower_mass, upper_mass], dim=0)
        self._object.set_links_inertial_mass(random_mass, links_idx_local=[0], envs_idx=perm)

        # randomize object friction
        friction_perm = torch.randperm(self.num_envs, device=self._device)
        n_lower_f = self.num_envs // 2
        # lower half: friction_lo to 1.0
        lower_friction = torch.rand(n_lower_f, 1, device=self._device) * (1.0 - friction_lo) + friction_lo
        # upper half: 1.0 to friction_hi
        upper_friction = torch.rand(self.num_envs - n_lower_f, 1, device=self._device) * (friction_hi - 1.0) + 1.0
        random_friction = torch.cat([lower_friction, upper_friction], dim=0)
        self._object.set_friction_ratio(random_friction, envs_idx=friction_perm)


    def get_per_trajectory_stats(self) -> dict[str, dict[str, float]]:
        """Return rolling mean rollout length per trajectory (training only).

        Keys are trajectory ids (derived from filename). Each value has:
          traj_idx, n_episodes, mean_length, full_length, n_assigned_envs.
        n_episodes counts recorded resets in the rolling window (deque maxlen=64),
        including zero-length entries from the initial full resets.
        Trajectories with no completed episodes yet report nan for mean_length.
        """
        stats: dict[str, dict[str, float]] = {}
        env_traj_counts = torch.bincount(
            self.env_traj_idx, minlength=self._num_trajectories
        ).cpu().tolist()
        for t in range(self._num_trajectories):
            len_buf = self._per_traj_ep_len_buf[t]
            n = len(len_buf)
            tid = self._traj_ids[t] if t < len(self._traj_ids) else f"traj_{t}"
            mean_len = (sum(len_buf) / n) if n > 0 else float("nan")
            stats[tid] = {
                "traj_idx": float(t),
                "n_episodes": float(n),
                "mean_length": float(mean_len),
                "full_length": float(self._traj_lengths[t]),
                "n_assigned_envs": float(env_traj_counts[t]),
            }
        return stats

    def log_per_trajectory_stats(self, header: str = "") -> None:
        """Print a table of mean rollout length per trajectory.

        No-op in eval mode (per-trajectory tracking is training-only).
        """
        if self._eval_mode:
            return
        title = "[Per-Trajectory Stats]"
        if header:
            title += f" {header}"
        print(f"\n{title}")
        print(
            f"{'Idx':>4} {'Traj ID':<40} {'Envs':>5} {'#Eps':>5} "
            f"{'MeanLen':>8} {'FullLen':>8}"
        )
        print("-" * 74)
        stats = self.get_per_trajectory_stats()
        # Iterate in trajectory-index order for stable output
        ordered = sorted(stats.items(), key=lambda kv: kv[1]["traj_idx"])
        for tid, s in ordered:
            n = int(s["n_episodes"])
            label = tid if len(tid) <= 40 else tid[:37] + "..."
            mean_len_str = f"{s['mean_length']:>8.1f}" if n > 0 else f"{'-':>8}"
            print(
                f"{int(s['traj_idx']):>4} {label:<40} {int(s['n_assigned_envs']):>5} "
                f"{n:>5} {mean_len_str} {int(s['full_length']):>8}"
            )

    def get_terminated(self) -> torch.Tensor:
        """Terminate on unrealistic velocities, tracking failure, or trajectory completion.

        There is no timeout truncation: episodes end only when the trajectory
        runs out or the policy fails.
        """
        # Initialize reset buffer
        reset_buf = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        # === Error conditions (sanity checks) ===
        error_wrist_lin_vel = torch.norm(self.base_lin_vel, dim=-1) > 100
        error_wrist_ang_vel = torch.norm(self.base_ang_vel, dim=-1) > 200
        error_hand_dof_vel = torch.abs(self.hand_dof_vel).mean(-1) > 200
        error_obj_lin_vel = torch.norm(self.object_lin_vel, dim=-1) > 100
        error_obj_ang_vel = torch.norm(self.object_ang_vel, dim=-1) > 200
        error_buf = (
            error_wrist_lin_vel
            | error_wrist_ang_vel
            | error_hand_dof_vel
            | error_obj_lin_vel
            | error_obj_ang_vel
        )

        # === Failed execution ===
        diff_joints_pos = self.target_mano_joint_pos.reshape(self.num_envs, 5, 3) - self.finger_link_pos.reshape(self.num_envs, 5, 3)  # (num_envs, 5, 3)
        diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)  # (num_envs, 5)

        diff_thumb_tip = diff_joints_pos_dist[:, 0]
        diff_index_tip = diff_joints_pos_dist[:, 1]
        diff_middle_tip = diff_joints_pos_dist[:, 2]
        diff_ring_tip = diff_joints_pos_dist[:, 3]
        diff_pinky_tip = diff_joints_pos_dist[:, 4]

        diff_obj_pos_dist = torch.norm(
            self.target_object_pos - self.object_pos,
            dim=-1
        )  # (num_envs,)
        if self._ignore_axis_tensor is not None:
            diff_obj_angle = quat_angle_ignoring_axis(
                self.target_object_quat[:, :4],
                self.object_quat,
                self._ignore_axis_tensor,
            )
        else:
            diff_obj_quat = quat_mul(
                self.target_object_quat[:, :4],
                quat_conjugate(self.object_quat)
            )  # (num_envs, 4)
            diff_obj_angle = 2 * torch.acos(
                torch.clamp(torch.abs(diff_obj_quat[:, 0]), 0.0, 1.0)
            )

        hand_scale_factor = 1.0
        object_scale_factor = 1.0
        thumb_failure = diff_thumb_tip > 0.04 / 0.7 * hand_scale_factor
        index_failure = diff_index_tip > 0.045 / 0.7 * hand_scale_factor
        middle_failure = diff_middle_tip > 0.05 / 0.7 * hand_scale_factor
        ring_failure = diff_ring_tip > 0.06 / 0.7 * hand_scale_factor
        pinky_failure = diff_pinky_tip > 0.06 / 0.7 * hand_scale_factor
        hand_pose_failure = (
            thumb_failure | index_failure | middle_failure | ring_failure | pinky_failure
        )
        obj_pos_failure = diff_obj_pos_dist > 0.02 / 0.343 * object_scale_factor**3
        obj_rot_failure = diff_obj_angle / np.pi * 180 > 15 / 0.343 * object_scale_factor**3
        object_pose_failure = obj_pos_failure | obj_rot_failure

        arm_contact_force_magnitude = torch.norm(self.arm_contact_force, dim=-1).sum(dim=-1)  # (num_envs,)
        arm_contact_failure = (arm_contact_force_magnitude > 0)

        # Table contact force termination
        if self.training_step < 10000:
            # linearly anneal from threshold_0 to threshold_1 over first 10k steps
            contact_threshold = self._args.table_contact_threshold_0 - (self._args.table_contact_threshold_0 - self._args.table_contact_threshold_1) * min(self.training_step / 10000, 1.0)
        else:
            # linearly anneal from threshold_1 to threshold_2 over next 10k steps, then stay at threshold_2
            contact_threshold = self._args.table_contact_threshold_1 - (self._args.table_contact_threshold_1 - self._args.table_contact_threshold_2) * min((self.training_step - 10000) / 10000, 1.0)
        table_contact_failure = (self.table_contact_force_magnitude.squeeze(-1) > contact_threshold)

        failed_execute = object_pose_failure | hand_pose_failure | arm_contact_failure | table_contact_failure | error_buf

        # === Success condition ===
        # Reached within lookahead steps of trajectory end without failure
        lookahead = 3
        succeeded = (self.progress_buf + 1 + lookahead >= self.env_traj_lengths) & ~failed_execute

        # Update reset buffer with failure and success conditions
        reset_buf |= succeeded | failed_execute
        self.reset_buf[:] = reset_buf
        # Update success count buffer
        self.success_count_buf += succeeded.long()

        termination_dict = {
            "error_termination": error_buf,
            "error/wrist_lin_vel": error_wrist_lin_vel,
            "error/wrist_ang_vel": error_wrist_ang_vel,
            "error/hand_dof_vel": error_hand_dof_vel,
            "error/obj_lin_vel": error_obj_lin_vel,
            "error/obj_ang_vel": error_obj_ang_vel,
            "failed_execution": failed_execute,
            "succeeded": succeeded,
            "any": reset_buf,
            "trajectory_updated": self.success_count_buf >= 100,
            "hand_pose_failure": hand_pose_failure,
            "hand/thumb_failure": thumb_failure,
            "hand/index_failure": index_failure,
            "hand/middle_failure": middle_failure,
            "hand/ring_failure": ring_failure,
            "hand/pinky_failure": pinky_failure,
            "object_pose_failure": object_pose_failure,
            "object/pos_failure": obj_pos_failure,
            "object/rot_failure": obj_rot_failure,
            "arm_contact_failure": arm_contact_failure,
            "table_contact_failure": table_contact_failure,
        }
        self._extra_info["termination"] = termination_dict

        return reset_buf

    def get_truncated(self) -> torch.Tensor:
        """Always False: episodes end on completion or failure, never on a time limit.

        Also advances progress_buf, because the env wrapper drives the env
        through these hooks rather than through step().
        """
        time_out_buf = torch.zeros_like(self.time_out_buf, dtype=torch.bool)
        self.time_out_buf[:] = time_out_buf

        self.progress_buf += 1

        return time_out_buf

    def _compute_fingertip_to_surface_dist(self) -> None:
        """Compute SDF-based distance from each fingertip tactile center to the object surface.

        For each fingertip link, the query point is the tactile point closest to
        the centroid of all tactile points on that link (precomputed as
        _fingertip_tactile_center_offsets).  The local offset is transformed to
        world coordinates via the link's pose, then into the object mesh frame
        for SDF queries.
        Results are stored in self.fingertip_to_surface_dist (B, 5).
        """
        B = self.num_envs

        # Get tactile fingertip link poses in world frame
        link_pos = self._robot._robot_entity.get_links_pos(
            links_idx_local=self._tactile_fingertip_link_idxs,
        )  # (B, 5, 3)
        link_quat = self._robot._robot_entity.get_links_quat(
            links_idx_local=self._tactile_fingertip_link_idxs,
        )  # (B, 5, 4)

        # Transform precomputed local tactile center offsets to world frame
        offsets = self._fingertip_tactile_center_offsets.unsqueeze(0).expand(B, -1, -1)  # (B, 5, 3)
        offsets_world = quat_apply(link_quat, offsets)  # (B, 5, 3)
        fingertip_tactile_center = link_pos + offsets_world  # (B, 5, 3)

        # Object pose in world frame
        obj_pos = self.object_pos    # (B, 3)
        obj_quat = self.object_quat  # (B, 4)

        # Transform tactile center positions to object mesh frame
        points_rel = fingertip_tactile_center - obj_pos.unsqueeze(1)  # (B, 5, 3)
        obj_quat_expanded = obj_quat.unsqueeze(1).expand(-1, 5, -1)  # (B, 5, 4)
        points_mesh = inv_transform_by_quat(points_rel, obj_quat_expanded)  # (B, 5, 3)

        if self._variant_blocks is not None:
            # Heterogeneous path: query only each variant's geoms for its env block
            result = torch.full((B, 5), float('inf'), device=points_mesh.device,
                                dtype=points_mesh.dtype)
            for env_start, env_end, geoms in self._variant_blocks:
                pts = points_mesh[env_start:env_end].reshape(-1, 3)  # (n_v*5, 3)
                min_sdf = None
                for geom in geoms:
                    sdf_values = query_object_sdf(geom, pts)
                    if min_sdf is None:
                        min_sdf = sdf_values
                    else:
                        min_sdf = torch.minimum(min_sdf, sdf_values)
                result[env_start:env_end] = min_sdf.reshape(-1, 5)
            self.fingertip_to_surface_dist[:] = result
        else:
            # Homogeneous path: query all geoms for all envs
            points_flat = points_mesh.reshape(-1, 3)  # (B*5, 3)
            min_sdf = None
            for geom in self._object_geoms:
                sdf_values = query_object_sdf(geom, points_flat)  # (B*5,)
                if min_sdf is None:
                    min_sdf = sdf_values
                else:
                    min_sdf = torch.minimum(min_sdf, sdf_values)
            self.fingertip_to_surface_dist[:] = min_sdf.reshape(B, 5)

    def update_buffers(self) -> None:
        """Update all state buffers from simulator."""
        # Current Proprio
        # hand state
        self.hand_dof_pos[:] = self._robot.get_dofs_position(dofs_idx_local=self._robot._hand_dof_idx_local)
        self.hand_dof_vel[:] = self._robot.get_dofs_velocity(dofs_idx_local=self._robot._hand_dof_idx_local)

        # base state (wrist pose)
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        self.base_lin_vel[:] = self._robot.base_lin_vel
        self.base_ang_vel[:] = self._robot.base_ang_vel

        self.arm_dof_pos[:] = self._robot.dof_pos[:, :self._base_dof_dim]
        self.arm_dof_vel[:] = self._robot.dof_vel[:, :self._base_dof_dim]
        self.dof_vel[:] = self._robot.dof_vel

        # object state
        self.object_pos[:] = self._object.get_pos()
        self.object_pos_rel[:] = self.object_pos - self.base_pos
        self.object_quat[:] = self._object.get_quat()

        self.object_lin_vel[:] = self._object.get_vel()
        self.object_ang_vel[:] = self._object.get_ang()

        # Height of object above the (per-env) table top, used to gate
        # velocity-change penalties so table contacts don't trigger them.
        self.object_height_above_table[:] = (
            self.object_pos[:, 2] - (self._table_surface_z + self._table_height_offsets)
        )

        # SDF-based fingertip-to-surface distance
        self._compute_fingertip_to_surface_dist()

        # Target motion
        future_index = torch.clamp(
            self.progress_buf + 1, max=self.env_traj_lengths - 1
        )  # (num_envs,)
        # Use advanced indexing: select different timestep for each environment
        batch_indices = torch.arange(self.num_envs, device=self._device)  # (num_envs,)

        self.target_hand_dof_pos[:] = self._traj_data["dof_pos"][batch_indices, future_index]  # (num_envs, hand_dof_dim)
        self.target_hand_dof_vel[:] = self._traj_data["dof_vel"][batch_indices, future_index]  # (num_envs, hand_dof_dim)

        self.target_wrist_pos[:] = self._traj_data["wrist_pos"][batch_indices, future_index]  # (num_envs, 3)
        self.delta_wrist_pos[:] = self.target_wrist_pos - self.base_pos  # (num_envs, 3)
        self.target_wrist_quat[:] = self._traj_data["wrist_quat"][batch_indices, future_index]  # (num_envs, 4)
        self.target_wrist_vel[:] = self._traj_data["wrist_vel"][batch_indices, future_index]  # (num_envs, 3)
        self.target_wrist_ang_vel[:] = self._traj_data["wrist_ang_vel"][batch_indices, future_index]  # (num_envs, 3)

        self.target_object_pos[:] = self._traj_data["obj_pos"][batch_indices, future_index]  # (num_envs, 3)
        self.delta_object_pos[:] = self.target_object_pos - self.object_pos  # (num_envs, 3)
        self.target_object_quat[:] = self._traj_data["obj_quat"][batch_indices, future_index]  # (num_envs, 4)
        self.target_object_vel[:] = self._traj_data["obj_vel"][batch_indices, future_index]  # (num_envs, 3)
        self.target_object_ang_vel[:] = self._traj_data["obj_ang_vel"][batch_indices, future_index]  # (num_envs, 3)

        # Update reference tactile map
        self.target_tactile_map_flat[:] = self._traj_data["tactile_map_ref"][batch_indices, future_index]  # (num_envs, 768)

        # Update fingertip positions (5 tips)
        self.target_mano_joint_pos[:] = self._traj_data["mano_joint_poses"][batch_indices, future_index].reshape(self.num_envs, -1)  # (num_envs, 5 * 3)
        self.finger_link_pos[:] = self._robot.get_links_pos(links_idx_local=self._finger_link_idxs, ref="link_com").reshape(self.num_envs, -1)  # (num_envs, 5 * 3)
        self.finger_link_vel[:] = self._robot.get_links_vel(links_idx_local=self._finger_link_idxs, ref="link_com").reshape(self.num_envs, -1)  # (num_envs, 5 * 3)
        self.target_mano_joint_vel[:] = self._traj_data["mano_joint_velocities"][batch_indices, future_index].reshape(self.num_envs, -1)  # (num_envs, 5 * 3)

        # Update fingertip quaternions (5 tips)
        self.target_mano_joint_quat[:] = self._traj_data["mano_joint_quats"][batch_indices, future_index].reshape(self.num_envs, -1)  # (num_envs, 5 * 4)
        self.finger_link_quat[:] = self._robot._robot_entity.get_links_quat(links_idx_local=self._finger_link_idxs).reshape(self.num_envs, -1)  # (num_envs, 5 * 4)

        # Update tactile sensor data
        tactile_forces_list = []
        # Use JSON insertion order for TactileVisualizer compatibility
        for link_name in self._tactile_link_names_json_order:
            sensor = self._tactile_sensors[link_name]
            config = self._tactile_sensor_configs[link_name]

            # Read sensor data (returns num_points * 3 forces)
            force_field_full = sensor.read()
            num_points = config['num_points']

            # Reshape to (num_envs, num_points, 3), in world coordinate
            force_field_3d = force_field_full.reshape(self.num_envs, num_points, 3)

            # Get force magnitudes: shape (num_envs, num_points)
            force_mag = torch.norm(force_field_3d, dim=-1)

            tactile_forces_list.append(force_mag)

        # Concatenate all sensor readings in JSON order: shape (num_envs, total_tactile_points)
        tactile_forces_flat = torch.cat(tactile_forces_list, dim=-1)

        # Convert to 2D tactile map
        tactile_map = self._tactile_map_converter.update(tactile_forces_flat)  # (B, 24, 32)
        self.tactile_map_flat[:] = tactile_map.reshape(self.num_envs, -1)  # (B, 768)

        self.dof_force[:] = self._robot.dofs_control_force

        self.arm_contact_force[:] = self._robot.arm_links_contact_force

        # Update table contact force magnitude
        if hasattr(self, '_table'):
            table_force = self._table.get_links_net_contact_force()  # (n_envs, n_links, 3)
            self.table_contact_force_magnitude[:] = torch.norm(table_force, dim=-1).sum(dim=-1, keepdim=True)
        else:
            self.table_contact_force_magnitude[:] = 0.0


    def update_temporal_buffers(self) -> None:
        # Update history buffers
        current_proprio = torch.cat([
            getattr(self, term) * self._args.obs_scales[term]
            for term in self._args.proprio_terms
        ], dim=-1)

        self.proprio_history_buf[:] = torch.cat(
            [self.proprio_history_buf[:, 1:], current_proprio.unsqueeze(1)],
            dim=1
        )  # (num_envs, history_len, prop_dim)
        self.action_proprio_history_flat[:] = torch.cat(
            [self.action_history_buf, self.proprio_history_buf], dim=-1
        ).reshape(self.num_envs, -1)  # (num_envs, history_len * (action_dim + prop_dim))

        # Future trajectory targets
        K = self._obs_future_length
        cur_idx = self.progress_buf  # (num_envs,) Current timestep in trajectory
        future_indices = cur_idx.unsqueeze(-1) + torch.arange(2, 2 * K + 1, 2, device=self._device).unsqueeze(0)
        # Clamp using per-environment trajectory lengths
        max_indices = (self.env_traj_lengths.unsqueeze(-1) - 1).expand(-1, K)  # (num_envs, K)
        future_indices = torch.min(future_indices, max_indices)  # (num_envs, K)

        # Index trajectory data using environment index and timestep
        env_indices = torch.arange(self.num_envs, device=self._device).unsqueeze(-1).expand(-1, K)  # (num_envs, K)

        future_hand_dof_pos = self._traj_data["dof_pos"][env_indices, future_indices]  # (num_envs, K, hand_dof_dim)
        future_hand_dof_vel = self._traj_data["dof_vel"][env_indices, future_indices]  # (num_envs, K, hand_dof_dim)
        future_mano_joint_pos = self._traj_data["mano_joint_poses"][env_indices, future_indices].reshape(self.num_envs, K, -1)  # (num_envs, K, 15)
        future_mano_joint_quat = self._traj_data["mano_joint_quats"][env_indices, future_indices].reshape(self.num_envs, K, -1)  # (num_envs, K, 20)

        future_absolute_wrist_pos = self._traj_data["wrist_pos"][env_indices, future_indices]  # (num_envs, K, 3)
        future_delta_wrist_pos = future_absolute_wrist_pos - self.base_pos.unsqueeze(1)  # (num_envs, K, 3)
        future_wrist_quat = self._traj_data["wrist_quat"][env_indices, future_indices]  # (num_envs, K, 4)
        future_wrist_vel = self._traj_data["wrist_vel"][env_indices, future_indices]  # (num_envs, K, 3)
        future_wrist_ang_vel = self._traj_data["wrist_ang_vel"][env_indices, future_indices]  # (num_envs, K, 3)

        future_absolute_object_pos = self._traj_data["obj_pos"][env_indices, future_indices]  # (num_envs, K, 3)
        future_delta_object_pos = future_absolute_object_pos - self.object_pos.unsqueeze(1)  # (num_envs, K, 3)
        future_object_quat = self._traj_data["obj_quat"][env_indices, future_indices]  # (num_envs, K, 4)
        future_object_vel = self._traj_data["obj_vel"][env_indices, future_indices]  # (num_envs, K, 3)
        future_object_ang_vel = self._traj_data["obj_ang_vel"][env_indices, future_indices]  # (num_envs, K, 3)

        future_data = {
            "target_hand_dof_pos": future_hand_dof_pos,
            "target_hand_dof_vel": future_hand_dof_vel,
            "target_mano_joint_pos": future_mano_joint_pos,
            "target_mano_joint_quat": future_mano_joint_quat,
            "delta_wrist_pos": future_delta_wrist_pos,
            "target_wrist_pos": future_absolute_wrist_pos,
            "target_wrist_quat": future_wrist_quat,
            "target_wrist_vel": future_wrist_vel,
            "target_wrist_ang_vel": future_wrist_ang_vel,
            "delta_object_pos": future_delta_object_pos,
            "target_object_pos": future_absolute_object_pos,
            "target_object_quat": future_object_quat,
            "target_object_vel": future_object_vel,
            "target_object_ang_vel": future_object_ang_vel,
        }
        future_target_action = torch.cat([
            future_data[term] * self._args.obs_scales[term]
            for term in self._args.target_motion_terms
        ], dim=-1)  # (num_envs, K, target_dim)
        self.target_action_flat[:] = future_target_action.reshape(self.num_envs, -1)

        # Student-specific temporal buffers for BC distillation
        if self._args.is_student:
            # Student future: use student target_motion_terms
            student_future = torch.cat([
                future_data[term] * self._args.obs_scales[term]
                for term in self._args.target_motion_terms_student
            ], dim=-1)  # (num_envs, K, student_target_dim)
            self.target_action_student_flat[:] = student_future.reshape(self.num_envs, -1)

            # Student history: use student proprio_terms (with same noise as current obs)
            proprio_noise = getattr(self, '_step_proprio_noise', {})
            current_student_proprio = torch.cat([
                getattr(self, term) * self._args.obs_scales[term] + proprio_noise.get(term, 0)
                for term in self._args.proprio_terms_student
            ], dim=-1)
            self.proprio_history_student_buf[:] = torch.cat(
                [self.proprio_history_student_buf[:, 1:], current_student_proprio.unsqueeze(1)],
                dim=1
            )
            self.action_proprio_history_student_flat[:] = torch.cat(
                [self.action_history_buf, self.proprio_history_student_buf], dim=-1
            ).reshape(self.num_envs, -1)


    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get actor and critic observations. Updates buffers and temporal state.

        Returns:
            Tuple of (actor_obs, critic_obs)
        """
        self.update_buffers()
        self._render_wrist_camera()

        # Generate proprio noise ONCE per step (shared between current obs and history buffer).
        self._step_proprio_noise: dict[str, torch.Tensor] = {}
        if self._args.is_student:
            for term in self._args.proprio_terms_student:
                noise_std = self._args.obs_noises.get(term, 0.0)
                if noise_std > 0:
                    self._step_proprio_noise[term] = (
                        torch.randn_like(getattr(self, term)) * noise_std
                    )

        self.update_temporal_buffers()

        # Build actor observation from configured terms
        obs_components = []
        for key in self._args.actor_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales[key]
            if key in self._step_proprio_noise:
                # Reuse the same noise that was applied to the history buffer
                obs_noise = self._step_proprio_noise[key]
            else:
                obs_noise = torch.randn_like(obs_gt) * self._args.obs_noises.get(key, 0.0)
            obs_components.append(obs_gt + obs_noise)
        actor_obs = torch.cat(obs_components, dim=-1)

        # Stochastic observation delay
        if self._obs_delay_queue is not None:
            # Shift queue: slot 1..T-1 ← slot 0..T-2, then insert current at slot 0
            self._obs_delay_queue[:, 1:] = self._obs_delay_queue[:, :-1].clone()
            self._obs_delay_queue[:, 0] = actor_obs
            # Sample a random delay index per env
            self._step_delay_idx = torch.randint(
                0, self._obs_delay_max, (self.num_envs,), device=self._device
            )
            actor_obs = self._obs_delay_queue[
                torch.arange(self.num_envs, device=self._device), self._step_delay_idx
            ].clone()
        else:
            self._step_delay_idx = None

        # Build critic observation from configured terms
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales[key]
            obs_components.append(obs_gt)
        critic_obs = torch.cat(obs_components, dim=-1)

        return actor_obs, critic_obs

    def get_teacher_observations(
        self, teacher_args: SingleHandRetargetingEnvArgs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build teacher observations from already-updated buffers (no state mutation).

        Unlike get_observations(), this does NOT call update_buffers() or
        update_temporal_buffers(), so history buffers are not shifted again.
        Must be called after get_observations() in the same step.

        Args:
            teacher_args: Teacher environment args specifying observation terms.

        Returns:
            Tuple of (actor_obs, critic_obs)
        """
        # Build actor observation (no noise for teacher)
        obs_components = []
        for key in teacher_args.actor_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales[key]
            obs_components.append(obs_gt)
        actor_obs = torch.cat(obs_components, dim=-1)

        # Lazy-create the teacher delay queue on the first call after
        # get_observations() has sampled a delay index (in practice BC's
        # dim-discovery call).
        if (
            self._teacher_obs_delay_queue is None
            and self._obs_delay_queue is not None
            and self._step_delay_idx is not None
        ):
            self._teacher_obs_delay_queue = torch.zeros(
                self.num_envs, self._obs_delay_max, actor_obs.shape[-1],
                dtype=torch.float32, device=self._device,
            )

        # Apply the same per-env delay index sampled in get_observations() so
        # the teacher and student see temporally aligned observations.
        if self._teacher_obs_delay_queue is not None:
            assert self._step_delay_idx is not None
            self._teacher_obs_delay_queue[:, 1:] = self._teacher_obs_delay_queue[:, :-1].clone()
            self._teacher_obs_delay_queue[:, 0] = actor_obs
            actor_obs = self._teacher_obs_delay_queue[
                torch.arange(self.num_envs, device=self._device), self._step_delay_idx
            ].clone()

        # Build critic observation (no delay)
        obs_components = []
        for key in teacher_args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales[key]
            obs_components.append(obs_gt)
        critic_obs = torch.cat(obs_components, dim=-1)

        return actor_obs, critic_obs

    def _apply_object_guide_control(
        self,
        target_pos: torch.Tensor,
        decay: float,
    ) -> None:
        """Guide the object position toward the reference trajectory.

        Only controls the translational DOFs (first 3). Rotation is left free.
        Decays kp/kv over training steps so the guide force vanishes.
        """
        kp_val = self._args.object_guide_force_kp * decay
        kp = torch.tensor([kp_val, kp_val, kp_val, 0.0, 0.0, 0.0], dtype=torch.float32)
        kv = kp * 0.1
        self._object.set_dofs_kp(kp)
        self._object.set_dofs_kv(kv)
        self._object.control_dofs_position(target_pos, dofs_idx_local=[0, 1, 2])

    def _apply_random_perturbations(self) -> None:
        """Apply random force/torque kicks to the object (called each physics substep).

        Only applies when the object is lifted above the table surface by at least
        perturbation_lift_threshold meters, preventing kicks while the object rests on the table.
        Scales linearly from 0 to full between curriculum_start and curriculum_end steps.
        """
        # Curriculum: ramp from 0 at start to 1.0 at end
        step = self.training_step
        if step < self._perturbation_curriculum_start:
            return
        ramp = min((step - self._perturbation_curriculum_start) /
                   max(self._perturbation_curriculum_end - self._perturbation_curriculum_start, 1), 1.0)

        solver = self._scene.scene.rigid_solver

        # Check which envs have the object lifted above the table
        obj_z = self._object.get_pos()[:, 2]  # (num_envs,)
        lifted = obj_z > (self._table_surface_z + self._perturbation_lift_threshold)  # (num_envs,)

        if self._perturbation_force_scale > 0.0:
            # Per-env coin flip AND lifted gate
            selected = (
                (torch.rand(self.num_envs, device=self._device) < self._force_prob) & lifted
            ).nonzero(as_tuple=False).squeeze(-1)
            if selected.numel() > 0:
                force = (
                    torch.randn(selected.numel(), 1, 3, device=self._device)
                    * self._object_mass[selected].unsqueeze(-1)
                    * self._perturbation_force_scale
                    * ramp
                )
                solver.apply_links_external_force(
                    force=force, links_idx=self._object_link_idx, envs_idx=selected, ref="link_com"
                )

        if self._perturbation_torque_scale > 0.0:
            selected = (
                (torch.rand(self.num_envs, device=self._device) < self._torque_prob) & lifted
            ).nonzero(as_tuple=False).squeeze(-1)
            if selected.numel() > 0:
                torque = (
                    torch.randn(selected.numel(), 1, 3, device=self._device)
                    * self._object_mass[selected].unsqueeze(-1)
                    * self._perturbation_torque_scale
                    * ramp
                )
                solver.apply_links_external_torque(
                    torque=torque, links_idx=self._object_link_idx, envs_idx=selected, ref="link_com"
                )

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply an ARM_REL_HAND_ABS action: first 7 DOFs are arm joint deltas, the rest normalized absolute hand targets.

        Arm and hand parts are scaled separately (base_action_scale / action_scale)
        before entering the action history.
        """
        action = action.detach().to(self._device)

        # Scale action before storing in history buffer
        scaled_action = action.clone()
        scaled_action[:, :self._base_dof_dim] *= self._base_action_scale
        scaled_action[:, self._base_dof_dim:] *= self._action_scale

        # Update action history buffer (shift and add scaled action)
        self.action_history_buf[:] = torch.cat(
            [self.action_history_buf[:, 1:], scaled_action.unsqueeze(1)],
            dim=1
        )

        # Get action from history buffer with randomized per-env latency
        latency = torch.randint(
            self._action_latency_min, self._action_latency_max + 1,
            (self.num_envs,), device=self._device
        )
        buf_idx = self._action_history_len - 1 - latency
        exec_action = self.action_history_buf[
            torch.arange(self.num_envs, device=self._device), buf_idx
        ]

        # Precompute object guide force targets (before physics loop)
        guide_decay_steps = self._args.object_guide_force_decay_steps
        apply_guide = (
            not self._eval_mode
            and guide_decay_steps > 0
            and self.training_step < guide_decay_steps
        )
        if apply_guide:
            all_idx = torch.arange(self.num_envs, device=self._device)
            future_idx = torch.clamp(self.progress_buf + 1, max=self.env_traj_lengths - 1)
            guide_target_pos = self._traj_data["obj_pos"][all_idx, future_idx]  # (B, 3)
            guide_decay = max(1.0 - self.training_step / guide_decay_steps, 0.0)

        prev_obj_vel = self._object.get_vel()  # (num_envs, 3)
        prev_obj_ang_vel = self._object.get_ang()  # (num_envs, 3)

        # Apply actions and simulate physics with decimation
        for _ in range(self._args.robot_args.decimation):
            self._robot.apply_action(action=exec_action)

            if apply_guide:
                self._apply_object_guide_control(guide_target_pos, guide_decay)

            if self._perturbation_force_scale > 0.0 or self._perturbation_torque_scale > 0.0:
                self._apply_random_perturbations()

            self._scene.scene.step()

        curr_obj_vel = self._object.get_vel()
        curr_obj_ang_vel = self._object.get_ang()
        self.object_vel_change[:] = torch.norm(curr_obj_vel - prev_obj_vel, dim=-1)
        self.object_ang_vel_change[:] = torch.norm(curr_obj_ang_vel - prev_obj_ang_vel, dim=-1)

        self.update_buffers()

        # === Update target visualization markers ===
        vis_mode = self._args.target_visualization
        if self._eval_mode and self._show_viewer and vis_mode is not None:
            env_indices = torch.arange(self.num_envs, device=self._device)
            if vis_mode == "ghost_hand" and self._ghost_hand is not None:
                self._ghost_hand.set_pos(self.target_wrist_pos, envs_idx=env_indices)
                self._ghost_hand.set_quat(self.target_wrist_quat, envs_idx=env_indices)
                self._ghost_hand.set_dofs_position(
                    self.target_hand_dof_pos,
                    dofs_idx_local=[idx - 7 for idx in self._robot._hand_dof_idx_local],
                    envs_idx=env_indices,
                )
            elif vis_mode == "fingertip_markers" and self._fingertip_markers:
                tgt_pos = self.target_mano_joint_pos.reshape(self.num_envs, 5, 3)
                tgt_quat = self.target_mano_joint_quat.reshape(self.num_envs, 5, 4)
                for i, marker in enumerate(self._fingertip_markers):
                    marker.set_pos(tgt_pos[:, i, :], envs_idx=env_indices)
                    marker.set_quat(tgt_quat[:, i, :], envs_idx=env_indices)
            elif vis_mode == "target_object" and self._target_object is not None:
                self._target_object.set_pos(self.target_object_pos, envs_idx=env_indices)
                self._target_object.set_quat(self.target_object_quat, envs_idx=env_indices)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment."""
        # Apply action
        self.apply_action(action)

        # Get terminated
        terminated = self.get_terminated()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)

        # Get truncated
        truncated = self.get_truncated()
        if truncated.dim() == 1:
            truncated = truncated.unsqueeze(-1)

        # Get reward
        reward, reward_terms = self.get_reward()
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)

        # Get extra infos
        extra_infos = self.get_extra_infos()
        extra_infos["reward_terms"] = reward_terms

        # Reset if terminated or truncated
        done_idx = terminated.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.reset_idx(done_idx)

        # Get observations
        next_obs, _ = self.get_observations()

        return next_obs, reward, terminated, truncated, extra_infos

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute trajectory following rewards using modular reward terms."""
        reward_total = torch.zeros(self.num_envs, device=self._device)
        reward_dict = {}

        # Prepare state dict for reward functions
        state_dict = {key: getattr(self, key) for key in self._reward_required_keys}

        # Compute all configured rewards and sum them
        # Each reward term already has its scale applied in the RewardTerm class
        for key, func in self._reward_functions.items():
            reward = func(state_dict)
            reward_total += reward
            reward_dict[f"{key}"] = reward.clone()

        reward_dict["Total"] = reward_total

        # Store for critic observation
        self.last_reward[:] = reward_total.unsqueeze(-1)

        return reward_total, reward_dict

    def get_extra_infos(self) -> dict[str, Any]:
        # Note: update_buffers() is not called here because it's already called
        # in apply_action() which precedes get_extra_infos() in the step() flow.
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales[key]
            obs_components.append(obs_gt)
        obs_tensor = torch.cat(obs_components, dim=-1)
        self._extra_info["observations"] = {"critic": obs_tensor}
        self._extra_info["time_outs"] = self.time_out_buf.clone()[:, None]
        return self._extra_info

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @property
    def action_dim(self) -> int:
        return self._base_dof_dim + self._hand_dof_dim  # 7 arm + 20 hand

    @property
    def actor_obs_dim(self) -> int:
        return get_space_dim(self._actor_observation_space)

    @property
    def critic_obs_dim(self) -> int:
        return get_space_dim(self._critic_observation_space)

    @property
    def scene(self) -> FlatScene:
        return self._scene

    @property
    def robot(self) -> XArmWUJIHand:
        return self._robot

    @property
    def dt(self) -> float:
        """Environment timestep (accounts for decimation)."""
        return self._scene.scene.dt * self._args.robot_args.decimation

    def _render_wrist_camera(self) -> None:
        """Render the wrist-camera point cloud into wrist_pointcloud_flat (no-op unless use_wrist_pointcloud)."""
        if self._use_pointcloud:
            # Use CPU path (render_pointcloud) in eval mode for devices without batch renderer
            if self._eval_mode:
                pc_np, mask_np = self._wrist_camera.render_pointcloud(world_frame=False)
                pc = torch.as_tensor(pc_np, dtype=torch.float32, device=self._device)
                mask = torch.as_tensor(mask_np, device=self._device)
                # CPU path may return (H, W, 3) / (H, W) without batch dim
                if pc.ndim == 3:
                    pc = pc.unsqueeze(0)
                    mask = mask.unsqueeze(0)
            else:
                pc, mask = self._wrist_camera.render_pointcloud_gpu(world_frame=False)
                # pc: (n_envs, H, W, 3), mask: (n_envs, H, W)

            # --- Point cloud noise ---
            # Ensure mask is a writable bool tensor (not shared with numpy)
            valid_mask = mask.bool().clone()
            B, H, W = valid_mask.shape
            depth_noise_std = self._args.pointcloud_depth_noise_std
            n_rects = self._args.pointcloud_rect_dropout_num
            N = self._pointcloud_num_points

            # Full-image noise + rect dropout + topk sub-sampling
            mask = valid_mask
            if depth_noise_std > 0:
                noise = torch.randn(B, H, W, device=self._device) * depth_noise_std
                pc = pc.clone()
                pc[..., 2] += noise * mask

            if n_rects > 0:
                w_min, w_max = self._args.pointcloud_rect_dropout_width
                h_min, h_max = self._args.pointcloud_rect_dropout_height
                cx = torch.randint(0, W, (B, n_rects), device=self._device)
                cy = torch.randint(0, H, (B, n_rects), device=self._device)
                rw = torch.randint(w_min, w_max + 1, (B, n_rects), device=self._device)
                rh = torch.randint(h_min, h_max + 1, (B, n_rects), device=self._device)
                dropout = torch.zeros(B, H, W, dtype=torch.bool, device=self._device)
                gy = torch.arange(H, device=self._device).view(1, H, 1)
                gx = torch.arange(W, device=self._device).view(1, 1, W)
                for k in range(n_rects):
                    x0 = (cx[:, k] - rw[:, k] // 2).clamp(min=0).view(B, 1, 1)
                    x1r = (cx[:, k] + rw[:, k] // 2).clamp(max=W).view(B, 1, 1)
                    y0 = (cy[:, k] - rh[:, k] // 2).clamp(min=0).view(B, 1, 1)
                    y1r = (cy[:, k] + rh[:, k] // 2).clamp(max=H).view(B, 1, 1)
                    dropout |= (gx >= x0) & (gx < x1r) & (gy >= y0) & (gy < y1r)
                mask[dropout] = False

            pc = pc.reshape(self.num_envs, -1, 3)
            mask = mask.reshape(self.num_envs, -1)

            pc = pc * mask.unsqueeze(-1)

            scores = torch.rand(self.num_envs, pc.shape[1], device=self._device)
            scores[~mask] = -1.0
            _, indices = scores.topk(N, dim=1)
            idx = indices.unsqueeze(-1).expand(-1, -1, 3)
            sampled = torch.gather(pc, 1, idx)
            self.wrist_pointcloud_flat[:] = sampled.reshape(self.num_envs, -1)

            if self._eval_mode and self._show_viewer:
                self._visualize_pointcloud(sampled[0].cpu().numpy())

    def _visualize_pointcloud(self, points: np.ndarray) -> None:
        """Render subsampled point cloud as a debug image and save to /tmp/.

        Draws two panels side-by-side: XY (front view) and XZ (top-down view),
        with points colored by depth (z).

        Args:
            points: (N, 3) array in camera frame.
        """
        H, W = 360, 640
        half_w = W // 2
        img = np.zeros((H, W, 3), dtype=np.uint8)

        valid = np.linalg.norm(points, axis=-1) > 1e-6
        pts = points[valid]
        if len(pts) == 0:
            cv2.imwrite("/tmp/pointcloud_debug.png", img)
            return

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Color by depth
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / max(z_max - z_min, 1e-6)
        colors = cv2.applyColorMap(
            (z_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS
        ).reshape(-1, 3)

        def project(u_vals, v_vals, w, h):
            u_min, u_max = u_vals.min(), u_vals.max()
            v_min, v_max = v_vals.min(), v_vals.max()
            span = max(u_max - u_min, v_max - v_min, 1e-6)
            margin = 0.1 * span
            px = ((u_vals - u_min + margin) / (span + 2 * margin) * (w - 1)).astype(np.int32)
            py = ((v_vals - v_min + margin) / (span + 2 * margin) * (h - 1)).astype(np.int32)
            return np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)

        # Left panel: XY (front view)
        px, py = project(x, y, half_w, H)
        for i in range(len(px)):
            cv2.circle(img, (px[i], py[i]), 1, colors[i].tolist(), -1)

        # Right panel: XZ (top-down view)
        px2, py2 = project(x, z, half_w, H)
        for i in range(len(px2)):
            cv2.circle(img, (px2[i] + half_w, py2[i]), 1, colors[i].tolist(), -1)

        n_valid = int(valid.sum())
        cv2.putText(img, f"XY (front) [{n_valid} pts]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"XZ (top) [{n_valid} pts]", (half_w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite("/tmp/pointcloud_debug.png", img)

    def close(self) -> None:
        """Clean up resources."""
        pass

    def _build_fingertip_pixel_masks(self, h: int, w: int) -> torch.Tensor:
        """Build a (5, h*w) boolean mask: row i marks pixels belonging to fingertip i."""
        tip_link_groups = [
            ["finger1_link4"],
            ["finger2_link4"],
            ["finger3_link4"],
            ["finger4_link4"],
            ["finger5_link4"],
        ]
        masks = torch.zeros((5, h * w), dtype=torch.bool, device=self._device)
        per_link_mapping = self._tactile_map_converter._pixel_mapping["per_link_mapping"]
        for i, link_names in enumerate(tip_link_groups):
            for link_name in link_names:
                if link_name not in per_link_mapping:
                    continue
                for _local_idx, info in per_link_mapping[link_name].items():
                    r, c = info["pixel"]
                    masks[i, r * w + c] = True
        print(f"Fingertip pixel masks: {masks.sum(dim=-1).tolist()} pixels per finger")
        return masks

    def _build_tactile_pixel_weights(self, h: int, w: int, fingertip_weight: float = 3.0) -> torch.Tensor:
        """Build a (h*w,) weight tensor where fingertip pixels get higher weight.

        Uses the per_link_mapping from the tactile pixel mapping JSON (already
        loaded by self._tactile_map_converter) to identify which pixels belong
        to fingertip links.
        """
        weights = torch.ones(h * w, device=self._device)

        tip_links = [
            "finger1_link4", "finger2_link4", "finger3_link4", "finger4_link4", "finger5_link4",
        ]

        per_link_mapping = self._tactile_map_converter._pixel_mapping["per_link_mapping"]
        tip_pixels: set[tuple[int, int]] = set()
        for link_name in tip_links:
            if link_name not in per_link_mapping:
                continue
            for _local_idx, info in per_link_mapping[link_name].items():
                r, c = info["pixel"]
                tip_pixels.add((r, c))

        for r, c in tip_pixels:
            weights[r * w + c] = fingertip_weight

        # normalize to mean 1.0
        weights *= (h * w) / weights.sum()

        print(f"Tactile pixel weights: {len(tip_pixels)} fingertip pixels -> weight {fingertip_weight}")
        return weights

    def _setup_tactile_sensors(self, args: SingleHandRetargetingEnvArgs) -> None:
        """Setup tactile sensors from JSON grid file."""
        import json

        if args.robot_args.tactile_grid_path is None:
            raise ValueError("robot_args.tactile_grid_path must be set")

        tactile_grid_path = Path(args.robot_args.tactile_grid_path)
        if not tactile_grid_path.exists():
            raise FileNotFoundError(f"Tactile grid file not found: {tactile_grid_path}")

        print(f"\n{'='*70}")
        print(f"Loading tactile grid from: {tactile_grid_path}")
        print(f"{'='*70}")

        with open(tactile_grid_path, 'r') as f:
            tactile_data = json.load(f)

        links_data = tactile_data['links']
        sensor_link_names = list(links_data.keys())
        # Store JSON insertion order for TactileVisualizer compatibility
        self._tactile_link_names_json_order = sensor_link_names

        print(f"Configuring tactile sensors for links: {sensor_link_names}")

        # Validate sensor links and store tactile points
        for link_name in sensor_link_names:
            if link_name not in links_data:
                raise ValueError(f"Link '{link_name}' not found in tactile grid JSON")

            link_data = links_data[link_name]
            points = link_data.get('points', [])

            if not points:
                raise ValueError(f"No tactile points found for link '{link_name}'")

            # Store local positions
            local_positions = np.array(points, dtype=np.float32)
            self._tactile_points[link_name] = local_positions

            print(f"  {link_name}: {len(local_positions)} tactile points")

        # Add TactileFieldSensors
        print(f"\n{'='*70}")
        print(f"Adding TactileFieldSensors")
        print(f"{'='*70}")

        for link_name in sensor_link_names:
            link_data = links_data[link_name]
            if link_name == "base":
                link_idx_local = self.robot.get_link("palm_link").idx_local
            else:
                link_idx_local = self.robot.get_link(link_name).idx_local
            num_points = link_data['num_points']
            local_positions = self._tactile_points[link_name]

            print(f"\nSensor for {link_name}:")
            print(f"  Link index (local): {link_idx_local}")
            print(f"  Tactile points: {num_points}")

            # Create TactileFieldSensor with custom tactile points
            sensor = self._scene.scene.add_sensor(
                gs.sensors.TactileField(
                    entity_idx=self._robot._robot_entity.idx,
                    link_idx_local=link_idx_local,
                    indenter_entity_idx=self._object.idx,
                    indenter_link_idx_local=0,
                    tactile_points_local=local_positions,
                    kn=args.tactile_kn,
                )
            )

            self._tactile_sensors[link_name] = sensor
            self._tactile_sensor_configs[link_name] = {
                'num_points': num_points,
                'link_idx_local': link_idx_local,
            }

            print(f"  ✓ TactileFieldSensor added")
