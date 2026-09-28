"""Trajectory loading/processing/augmentation utilities for SingleHandRetargetingEnvTactileMap."""

import genesis as gs
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as ScipyR

from gs_env.common.utils.math_utils import (
    quat_from_angle_axis,
    quat_to_angle_axis,
    rotmat_to_quat,
)
from gs_env.common.utils.maniptrans_util import (
    compute_velocity,
    compute_angular_velocity,
    compute_dof_velocity,
)


def pad_and_stack(arrays: list[np.ndarray]) -> np.ndarray:
    """Pad arrays to same length and stack them.

    Args:
        arrays: List of arrays with shape (T_i, ...) where T_i can vary

    Returns:
        Stacked array with shape (num_arrays, max_T, ...)
    """
    max_len = max(arr.shape[0] for arr in arrays)
    padded_arrays = []

    for arr in arrays:
        if arr.shape[0] < max_len:
            # Pad by repeating the last frame
            pad_len = max_len - arr.shape[0]
            last_frame = arr[-1:].repeat(pad_len, axis=0)
            padded_arr = np.concatenate([arr, last_frame], axis=0)
        else:
            padded_arr = arr
        padded_arrays.append(padded_arr)

    return np.stack(padded_arrays, axis=0)


def resample_trajectory_lists(
    all_wrist_positions: list[np.ndarray],
    all_wrist_rotations_aa: list[np.ndarray],
    all_dof_positions: list[np.ndarray],
    all_obj_pose_matrices: list[np.ndarray],
    all_tactile_maps: list[np.ndarray],
    all_traj_lengths: list[int],
    all_demo_data: list[dict],
    target_hz: float,
) -> None:
    """Resample trajectory data from 60 Hz to target playback rate.

    Lower Hz = slower playback = more interpolated frames for the same motion.
    Modifies all lists in-place.
    """
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    ratio = 60.0 / target_hz  # >1 for slower, <1 for faster

    print(f"\nResampling trajectories: 60 Hz -> {target_hz} Hz (ratio={ratio:.3f})")

    for i in range(len(all_traj_lengths)):
        L = all_traj_lengths[i]
        new_L = max(int(round(L * ratio)), 2)

        old_t = np.linspace(0, 1, L)
        new_t = np.linspace(0, 1, new_L)

        # Wrist positions (L, 3) -> linear interp
        f = interp1d(old_t, all_wrist_positions[i], axis=0)
        all_wrist_positions[i] = f(new_t).astype(np.float32)

        # Wrist rotations axis-angle (L, 3) -> Slerp
        rotations = Rotation.from_rotvec(all_wrist_rotations_aa[i])
        slerp_fn = Slerp(old_t, rotations)
        all_wrist_rotations_aa[i] = slerp_fn(new_t).as_rotvec().astype(np.float32)

        # DOF positions (L, n_dofs) -> linear interp
        f = interp1d(old_t, all_dof_positions[i], axis=0)
        all_dof_positions[i] = f(new_t).astype(np.float32)

        # Object pose matrices (L, 4, 4) -> decompose, linear interp pos + slerp rot
        obj_matrices = all_obj_pose_matrices[i]
        obj_pos = obj_matrices[:, :3, 3]
        obj_rot = Rotation.from_matrix(obj_matrices[:, :3, :3])
        f = interp1d(old_t, obj_pos, axis=0)
        new_obj_pos = f(new_t).astype(np.float32)
        slerp_fn = Slerp(old_t, obj_rot)
        new_obj_rot = slerp_fn(new_t)
        new_matrices = np.zeros((new_L, 4, 4), dtype=np.float32)
        new_matrices[:, :3, :3] = new_obj_rot.as_matrix().astype(np.float32)
        new_matrices[:, :3, 3] = new_obj_pos
        new_matrices[:, 3, 3] = 1.0
        all_obj_pose_matrices[i] = new_matrices

        # Tactile maps (L, 768) -> linear interp
        f = interp1d(old_t, all_tactile_maps[i], axis=0)
        all_tactile_maps[i] = f(new_t).astype(np.float32)

        # MANO fingertip data in demo_data
        mano_ref = all_demo_data[i]["mano_reference"]
        for joint_name in list(mano_ref["finger_joints"].keys()):
            # Positions (L, 3) -> linear interp
            f = interp1d(old_t, mano_ref["finger_joints"][joint_name], axis=0)
            mano_ref["finger_joints"][joint_name] = f(new_t).astype(np.float32)

            # Quaternions (L, 4) [w,x,y,z] -> Slerp
            quat_wxyz = mano_ref["finger_joints_quat"][joint_name]
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # scipy expects xyzw
            rotations = Rotation.from_quat(quat_xyzw)
            slerp_fn = Slerp(old_t, rotations)
            new_quat_xyzw = slerp_fn(new_t).as_quat()
            mano_ref["finger_joints_quat"][joint_name] = (
                new_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)  # back to wxyz
            )

        all_traj_lengths[i] = new_L
        print(f"  Trajectory {i}: {L} -> {new_L} frames")


def _process_wrist_and_object(
    wrist_positions: np.ndarray,
    wrist_rotations_aa: np.ndarray,
    dof_positions: np.ndarray,
    obj_pose_matrices: np.ndarray,
    device: torch.device,
    time_delta: float = 1 / 60.0,
) -> dict[str, torch.Tensor]:
    """Process wrist, hand-DOF and object trajectory data (poses, quats, velocities).

    Args:
        time_delta: Time between consecutive trajectory frames (scene_dt * decimation).

    Returns a dict with wrist and object tensors for _traj_data_template.
    """
    wrist_pos = torch.from_numpy(wrist_positions).float().to(device)
    wrist_rot_aa = torch.from_numpy(wrist_rotations_aa).float().to(device)
    dof_pos = torch.from_numpy(dof_positions).float().to(device)

    num_traj, max_T, n_dofs = dof_pos.shape

    wrist_pos_T = wrist_pos.transpose(0, 1)
    wrist_rot_aa_T = wrist_rot_aa.transpose(0, 1)
    dof_pos_T = dof_pos.transpose(0, 1)

    wrist_vel_T = compute_velocity(wrist_pos_T, time_delta, gaussian_filter=True)
    wrist_ang_vel_T = compute_angular_velocity(wrist_rot_aa_T, time_delta, gaussian_filter=True)
    dof_vel_T = compute_dof_velocity(dof_pos_T, time_delta, gaussian_filter=True)

    wrist_vel = wrist_vel_T.transpose(0, 1)
    wrist_ang_vel = wrist_ang_vel_T.transpose(0, 1)
    dof_vel = dof_vel_T.transpose(0, 1)

    # Convert wrist rotations to quaternions
    wrist_rot_aa_flat = wrist_rot_aa.reshape(-1, 3)
    angle = torch.norm(wrist_rot_aa_flat, dim=-1)
    axis = wrist_rot_aa_flat / (angle.unsqueeze(-1) + 1e-8)
    wrist_quat = quat_from_angle_axis(angle, axis)
    wrist_quat = wrist_quat.reshape(num_traj, max_T, 4)

    # Process object pose and rotation
    obj_pos = torch.from_numpy(obj_pose_matrices[:, :, :3, 3]).float().to(device)
    obj_rot_mat = torch.from_numpy(obj_pose_matrices[:, :, :3, :3]).float().to(device)
    obj_rot_quat = rotmat_to_quat(obj_rot_mat)
    # Enforce quaternion continuity
    for t in range(1, obj_rot_quat.shape[1]):
        dot = (obj_rot_quat[:, t] * obj_rot_quat[:, t - 1]).sum(dim=-1, keepdim=True)
        obj_rot_quat[:, t] = torch.where(dot < 0, -obj_rot_quat[:, t], obj_rot_quat[:, t])

    add_smoothing = False
    if add_smoothing:
        # Smooth object position and rotation trajectories
        from scipy.ndimage import gaussian_filter1d as _gf1d
        _sigma = 3
        # Smooth positions: (num_traj, T, 3) along time axis
        obj_pos_np = obj_pos.cpu().numpy()
        obj_pos = torch.from_numpy(_gf1d(obj_pos_np, _sigma, axis=1, mode="nearest")).float().to(device)
        # Smooth quaternions: normalize after filtering to stay on unit sphere
        obj_quat_np = obj_rot_quat.cpu().numpy()
        obj_quat_smoothed = _gf1d(obj_quat_np, _sigma, axis=1, mode="nearest")
        obj_quat_norms = np.linalg.norm(obj_quat_smoothed, axis=-1, keepdims=True)
        obj_quat_smoothed = obj_quat_smoothed / (obj_quat_norms + 1e-8)
        obj_rot_quat = torch.from_numpy(obj_quat_smoothed).float().to(device)

    obj_rot_aa = quat_to_angle_axis(obj_rot_quat)

    obj_pos_T = obj_pos.transpose(0, 1)
    obj_rot_aa_T = obj_rot_aa.transpose(0, 1)
    obj_vel_T = compute_velocity(obj_pos_T, time_delta, gaussian_filter=True)
    obj_ang_vel_T = compute_angular_velocity(obj_rot_aa_T, time_delta, gaussian_filter=True)
    obj_vel = obj_vel_T.transpose(0, 1)
    obj_ang_vel = obj_ang_vel_T.transpose(0, 1)

    return {
        "wrist_pos": wrist_pos,
        "wrist_rot": wrist_rot_aa,
        "wrist_quat": wrist_quat,
        "wrist_vel": wrist_vel,
        "wrist_ang_vel": wrist_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "obj_pos": obj_pos,
        "obj_quat": obj_rot_quat,
        "obj_vel": obj_vel,
        "obj_ang_vel": obj_ang_vel,
    }


def process_trajectory_data_tactile_map(
    wrist_positions: np.ndarray,
    wrist_rotations_aa: np.ndarray,
    dof_positions: np.ndarray,
    obj_pose_matrices: np.ndarray,
    tactile_maps: np.ndarray,
    demo_data_raw: list[dict],
    joint_mapping: dict[str, str],
    device: torch.device,
    time_delta: float = 1 / 60.0,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Process trajectory data for the tactile map environment.

    Uses only fingertip joints (ending with '_tip'). Includes tactile maps
    and fingertip quaternions with continuity enforcement.

    Args:
        time_delta: Time between consecutive trajectory frames in seconds.
                    Should be scene_dt * decimation (default 1/60 for decimation=1).

    Returns:
        Tuple of (_traj_data_template dict, tip_joint_mapping).
    """
    traj_data = _process_wrist_and_object(
        wrist_positions, wrist_rotations_aa, dof_positions, obj_pose_matrices, device,
        time_delta=time_delta,
    )

    # Filter to fingertip joints only
    tip_joint_mapping = {k: v for k, v in joint_mapping.items() if k.endswith("_tip")}

    # Load fingertip positions and velocities from MANO reference (5 tips only)
    all_mano_joint_poses = []
    all_mano_joint_velocities = []

    for demo_data in demo_data_raw:
        mano_joint_poses = []
        for joint_name in tip_joint_mapping.keys():
            mano_joint_poses.append(
                demo_data["mano_reference"]["finger_joints"][joint_name]  # (T, 3)
            )
        mano_joint_poses = np.stack(mano_joint_poses, axis=1)  # (T, 5, 3)
        mano_joint_velocities = compute_velocity(
            torch.tensor(mano_joint_poses), time_delta, gaussian_filter=True,
        ).cpu().numpy()  # (T, 5, 3)

        all_mano_joint_poses.append(mano_joint_poses)
        all_mano_joint_velocities.append(mano_joint_velocities)

    mano_joint_poses_padded = pad_and_stack(all_mano_joint_poses)
    mano_joint_velocities_padded = pad_and_stack(all_mano_joint_velocities)

    traj_data["mano_joint_poses"] = torch.from_numpy(mano_joint_poses_padded).float().to(device)
    traj_data["mano_joint_velocities"] = torch.from_numpy(mano_joint_velocities_padded).float().to(device)

    # Load fingertip quaternions from MANO reference (5 tips only)
    all_mano_joint_quats = []
    for demo_data in demo_data_raw:
        mano_joint_quats = []
        for joint_name in tip_joint_mapping.keys():
            mano_joint_quats.append(
                demo_data["mano_reference"]["finger_joints_quat"][joint_name]  # (T, 4)
            )
        mano_joint_quats = np.stack(mano_joint_quats, axis=1)  # (T, 5, 4)
        all_mano_joint_quats.append(mano_joint_quats)

    mano_joint_quats_padded = pad_and_stack(all_mano_joint_quats)
    mano_joint_quats_tensor = torch.from_numpy(mano_joint_quats_padded).float().to(device)

    # Enforce quaternion continuity for fingertip quats (same hemisphere fix)
    for finger_idx in range(5):
        for t in range(1, mano_joint_quats_tensor.shape[1]):
            dot = (mano_joint_quats_tensor[:, t, finger_idx] * mano_joint_quats_tensor[:, t - 1, finger_idx]).sum(dim=-1, keepdim=True)
            mano_joint_quats_tensor[:, t, finger_idx] = torch.where(
                dot < 0, -mano_joint_quats_tensor[:, t, finger_idx], mano_joint_quats_tensor[:, t, finger_idx]
            )

    traj_data["mano_joint_quats"] = mano_joint_quats_tensor
    traj_data["tactile_map_ref"] = torch.from_numpy(tactile_maps).float().to(device)

    print(f"Processed trajectory data:")
    print(f"  - Wrist positions: {traj_data['wrist_pos'].shape}")
    print(f"  - Wrist rotations (axis-angle): {traj_data['wrist_rot'].shape}")
    print(f"  - Wrist rotations (quaternion): {traj_data['wrist_quat'].shape}")
    print(f"  - Wrist velocities: {traj_data['wrist_vel'].shape}")
    print(f"  - DOF positions: {traj_data['dof_pos'].shape}")
    print(f"  - DOF velocities: {traj_data['dof_vel'].shape}")
    print(f"  - MANO joint poses: {traj_data['mano_joint_poses'].shape}")
    print(f"  - MANO joint velocities: {traj_data['mano_joint_velocities'].shape}")
    print(f"  - MANO joint quats: {traj_data['mano_joint_quats'].shape}")

    return traj_data, tip_joint_mapping


def augment_trajectory(
    wrist_positions: np.ndarray,
    wrist_rotations_aa: np.ndarray,
    object_pose_matrices: np.ndarray,
    finger_joints: dict[str, np.ndarray] | None = None,
    finger_joints_quat: dict[str, np.ndarray] | None = None,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    rotate_z_deg: float = 0.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Augment a single hand-object trajectory with exact specified values.

    Applies scale (relative to object first frame), Z-axis rotation, and XY
    translation to wrist/object/MANO-fingertip data.  Finger DOFs and tactile
    maps are left unchanged (intrinsic).

    Copied from scripts/visualization/replay_xarm_wujihand_trajectory.py.
    """
    wrist_positions = wrist_positions.copy()
    wrist_rotations_aa = wrist_rotations_aa.copy()
    object_pose_matrices = object_pose_matrices.copy()
    if finger_joints is not None:
        finger_joints = {k: v.copy() for k, v in finger_joints.items()}
    if finger_joints_quat is not None:
        finger_joints_quat = {k: v.copy() for k, v in finger_joints_quat.items()}

    object_positions = object_pose_matrices[:, :3, 3].copy()
    orig_wrist_positions = wrist_positions.copy()

    translation = np.array([translate_x, translate_y, 0.0])

    theta_z = np.deg2rad(rotate_z_deg)
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    Rz = np.array([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [0,      0,     1],
    ])

    # Scale object trajectory relative to its first frame
    obj_origin = object_positions[0].copy()
    scaled_obj = obj_origin + scale * (object_positions - obj_origin)

    # Reconstruct wrist by preserving original hand-to-object offset
    hand_to_object = object_positions - wrist_positions
    scaled_wrist = scaled_obj - hand_to_object

    # Rotate and translate both
    rotated_wrist = (Rz @ scaled_wrist.T).T
    wrist_positions = rotated_wrist + translation

    rotated_obj = (Rz @ scaled_obj.T).T
    object_pose_matrices[:, :3, 3] = rotated_obj + translation

    # Rotate wrist orientations
    for t in range(len(wrist_rotations_aa)):
        aa = wrist_rotations_aa[t]
        angle = np.linalg.norm(aa)
        if angle > 1e-6:
            R_wrist = ScipyR.from_rotvec(aa).as_matrix()
        else:
            R_wrist = np.eye(3)
        wrist_rotations_aa[t] = ScipyR.from_matrix(Rz @ R_wrist).as_rotvec()

    # Rotate object orientations
    for t in range(len(object_pose_matrices)):
        object_pose_matrices[t, :3, :3] = Rz @ object_pose_matrices[t, :3, :3]

    # Augment fingertip positions: preserve offset from wrist, then rotate+translate
    if finger_joints is not None:
        for joint_name, joint_pos in finger_joints.items():
            finger_to_wrist = joint_pos - orig_wrist_positions
            scaled_finger = scaled_wrist + finger_to_wrist
            finger_joints[joint_name] = (Rz @ scaled_finger.T).T + translation

    # Augment fingertip quaternions: apply Rz rotation
    if finger_joints_quat is not None:
        Rz_scipy = ScipyR.from_matrix(Rz)
        for joint_name, joint_quat_wxyz in finger_joints_quat.items():
            q_xyzw = joint_quat_wxyz[:, [1, 2, 3, 0]]
            r_new = Rz_scipy * ScipyR.from_quat(q_xyzw)
            q_new_xyzw = r_new.as_quat()
            finger_joints_quat[joint_name] = q_new_xyzw[:, [3, 0, 1, 2]]

    return wrist_positions, wrist_rotations_aa, object_pose_matrices, finger_joints, finger_joints_quat


def validate_on_table(
    obj_pose_matrix_first_frame: np.ndarray,
    table_x_range: tuple[float, float],
    table_y_range: tuple[float, float],
    offset: float = 0.05,
) -> bool:
    """Check if object's first frame XY is within table bounds."""
    x = obj_pose_matrix_first_frame[0, 3]
    y = obj_pose_matrix_first_frame[1, 3]
    return (
        table_x_range[0] + offset <= x <= table_x_range[1] - offset and
        table_y_range[0] + offset <= y <= table_y_range[1] - offset
    )


def query_object_sdf(geom, points_mesh_frame: torch.Tensor) -> torch.Tensor:
    """Query a geom's precomputed SDF for points in mesh-local frame.

    Adapted from Genesis's TactileFieldSensor._query_genesis_sdf_gpu. Uses GPU trilinear
    interpolation via F.grid_sample.

    Args:
        geom: RigidGeom object with precomputed SDF.
        points_mesh_frame: (N, 3) points in the object's mesh coordinate frame.

    Returns:
        sdf_values: (N,) signed distances (positive = outside, negative = penetrating).
    """
    N = points_mesh_frame.shape[0]
    device = points_mesh_frame.device

    # Cache SDF tensors on GPU (first call only)
    # NOTE: Must cache ALL fields that Genesis's TactileFieldSensor._query_genesis_sdf_gpu expects,
    # because both code paths share the same geom object and use hasattr('_sdf_val_torch')
    # as the initialization guard.
    if not hasattr(geom, '_sdf_val_torch'):
        geom._sdf_val_torch = torch.from_numpy(geom.sdf_val).to(device=device, dtype=gs.tc_float)
        geom._sdf_grad_torch = torch.from_numpy(geom.sdf_grad).to(device=device, dtype=gs.tc_float)
        geom._T_mesh_to_sdf_torch = torch.from_numpy(geom.T_mesh_to_sdf).to(device=device, dtype=gs.tc_float)
        geom._sdf_val_grid = geom._sdf_val_torch.unsqueeze(0).unsqueeze(0)
        geom._sdf_grad_grid = geom._sdf_grad_torch.permute(3, 0, 1, 2).unsqueeze(0)

    # Transform to SDF grid coordinates
    T = geom._T_mesh_to_sdf_torch
    points_homo = torch.cat([points_mesh_frame, torch.ones((N, 1), device=device, dtype=gs.tc_float)], dim=1)
    points_sdf = (T @ points_homo.T).T[:, :3]  # (N, 3)

    res = torch.tensor(geom.sdf_res, device=device, dtype=gs.tc_float)
    cell_size = geom.sdf_cell_size

    # Identify points outside grid
    outside_mask = (points_sdf >= res - 1).any(dim=1) | (points_sdf < 0).any(dim=1)
    inside_mask = ~outside_mask

    sdf_values = torch.zeros(N, device=device, dtype=gs.tc_float)

    # Outside points: metric distance estimate
    # Clamp to grid boundary, measure how far beyond the boundary the point is
    # (in grid units), convert to meters, and add the boundary SDF (~sdf_max).
    if outside_mask.any():
        points_outside = points_sdf[outside_mask]
        clamped = torch.clamp(points_outside, min=torch.zeros(3, device=device, dtype=gs.tc_float),
                              max=res - 1)
        beyond_dist_metric = torch.norm(points_outside - clamped, dim=1) * cell_size
        sdf_values[outside_mask] = beyond_dist_metric + geom.sdf_max

    # Inside points: GPU trilinear interpolation
    if inside_mask.any():
        points_inside = points_sdf[inside_mask]  # (M, 3)
        grid_coords_normalized = (points_inside / (res - 1)) * 2.0 - 1.0
        grid_coords_normalized = grid_coords_normalized.flip(-1)  # (z, y, x) order for grid_sample
        grid_coords_5d = grid_coords_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, M, 3)

        sdf_vals_sampled = F.grid_sample(
            geom._sdf_val_grid,
            grid_coords_5d,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (1, 1, 1, 1, M)

        sdf_values[inside_mask] = sdf_vals_sampled.squeeze()

    return sdf_values
