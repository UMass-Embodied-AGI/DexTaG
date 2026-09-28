import numpy as np
import torch


def compute_velocity(p: torch.Tensor, time_delta: float, gaussian_filter: bool = True) -> torch.Tensor:
    """Compute velocity from positions using gradient and optional Gaussian smoothing.

    Following ManipTrans implementation.

    Args:
        p: Positions tensor of shape [T, K, 3] where K is number of points
        time_delta: Time step for velocity computation
        gaussian_filter: Whether to apply Gaussian smoothing

    Returns:
        velocity: Velocity tensor of shape [T, K, 3]
    """
    from scipy.ndimage import gaussian_filter1d

    # [T, K, 3]
    velocity = np.gradient(p.cpu().numpy(), axis=0) / time_delta
    if gaussian_filter:
        velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
    return torch.from_numpy(velocity).to(p)


def compute_angular_velocity(
    r: torch.Tensor, time_delta: float, gaussian_filter: bool = True
) -> torch.Tensor:
    """Compute angular velocity from axis-angle rotations.

    Following ManipTrans implementation. Converts axis-angle to rotation matrices,
    computes rotation differences, then extracts angular velocity.

    Args:
        r: Axis-angle rotations of shape [T, K, 3]
        time_delta: Time step for velocity computation
        gaussian_filter: Whether to apply Gaussian smoothing

    Returns:
        angular_velocity: Angular velocity tensor of shape [T, K, 3]
    """
    from scipy.ndimage import gaussian_filter1d

    # Convert axis-angle to rotation matrices [T, K, 3, 3]
    r_mat = axis_angle_to_rotmat_batch(r)

    # Compute rotation differences: R[t+1] @ R[t]^T
    diff_r = r_mat[1:] @ r_mat[:-1].transpose(-1, -2)  # [T-1, K, 3, 3]

    # Convert back to axis-angle
    diff_aa = rotmat_to_axis_angle_batch(diff_r).cpu().numpy()  # [T-1, K, 3]

    # Extract angle and axis
    diff_angle = np.linalg.norm(diff_aa, axis=-1)  # [T-1, K]
    diff_axis = diff_aa / (diff_angle[:, :, None] + 1e-8)  # [T-1, K, 3]

    # Angular velocity = axis * angle / dt
    angular_velocity = diff_axis * diff_angle[:, :, None] / time_delta  # [T-1, K, 3]

    # Duplicate last timestep
    angular_velocity = np.concatenate([angular_velocity, angular_velocity[-1:]], axis=0)  # [T, K, 3]

    if gaussian_filter:
        angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="nearest")

    return torch.from_numpy(angular_velocity).to(r)


def compute_dof_velocity(dof: torch.Tensor, time_delta: float, gaussian_filter: bool = True) -> torch.Tensor:
    """Compute DOF velocities from DOF positions.

    Following ManipTrans implementation.

    Args:
        dof: DOF positions of shape [T, ..., n_dofs] (time on axis 0)
        time_delta: Time step for velocity computation
        gaussian_filter: Whether to apply Gaussian smoothing

    Returns:
        velocity: DOF velocity tensor with the same shape as ``dof``
    """
    from scipy.ndimage import gaussian_filter1d

    # [T, ..., n_dofs]
    velocity = np.gradient(dof.cpu().numpy(), axis=0) / time_delta
    if gaussian_filter:
        velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
    return torch.from_numpy(velocity).to(dof)


def axis_angle_to_rotmat_batch(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix (batch version).

    Args:
        axis_angle: [T, K, 3] axis-angle vectors

    Returns:
        rotmat: [T, K, 3, 3] rotation matrices
    """
    # Flatten to [T*K, 3]
    orig_shape = axis_angle.shape[:-1]
    axis_angle_flat = axis_angle.reshape(-1, 3)

    angle = torch.norm(axis_angle_flat, dim=-1, keepdim=True)
    axis = axis_angle_flat / (angle + 1e-8)

    # Rodrigues' formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # Cross product matrix
    K = torch.zeros((axis_angle_flat.shape[0], 3, 3), device=axis_angle.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # R = I + sin(θ)K + (1 - cos(θ))K^2
    I = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * (K @ K)

    # Reshape back to [T, K, 3, 3]
    return R.reshape(*orig_shape, 3, 3)


def rotmat_to_axis_angle_batch(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle (batch version).

    Args:
        rotmat: [T, K, 3, 3] rotation matrices

    Returns:
        axis_angle: [T, K, 3] axis-angle vectors
    """
    # Flatten to [T*K, 3, 3]
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)

    # Compute angle
    trace = rotmat_flat[:, 0, 0] + rotmat_flat[:, 1, 1] + rotmat_flat[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    # Compute axis
    axis = torch.stack([
        rotmat_flat[:, 2, 1] - rotmat_flat[:, 1, 2],
        rotmat_flat[:, 0, 2] - rotmat_flat[:, 2, 0],
        rotmat_flat[:, 1, 0] - rotmat_flat[:, 0, 1],
    ], dim=-1)
    axis = axis / (2 * torch.sin(angle).unsqueeze(-1) + 1e-8)

    # axis-angle = axis * angle
    axis_angle = axis * angle.unsqueeze(-1)

    # Reshape back to [T, K, 3]
    return axis_angle.reshape(*orig_shape, 3)
