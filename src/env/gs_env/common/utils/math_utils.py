import torch


@torch.jit.script
def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for batched tensors [w, x, y, z]."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternion(s) to rotation matrix (3x3).
    Input:
        q: Tensor of shape [..., 4] where the quaternion is in (w, x, y, z) format.
    Output:
        Rotation matrix of shape [..., 3, 3]
    """
    assert q.shape[-1] == 4, "Quaternion must be of shape [..., 4]"

    w, x, y, z = q.unbind(-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    rot = torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )  # shape: [..., 3, 3]

    return rot


def rotmat_to_quat(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrix/matrices to quaternion(s).
    Input:
        rot_mat: Tensor of shape [..., 3, 3] rotation matrix.
    Output:
        Quaternion tensor of shape [..., 4] in (w, x, y, z) format.

    Note: Not JIT-compiled due to dynamic reshaping requirements.
    """
    assert rot_mat.shape[-2:] == (3, 3), "Rotation matrix must be of shape [..., 3, 3]"

    batch_shape = rot_mat.shape[:-2]
    rot_mat_flat = rot_mat.reshape(-1, 3, 3)
    N = rot_mat_flat.shape[0]

    quat = torch.zeros((N, 4), device=rot_mat.device, dtype=rot_mat.dtype)

    # Compute quaternion components using Shepperd's method
    trace = rot_mat_flat[:, 0, 0] + rot_mat_flat[:, 1, 1] + rot_mat_flat[:, 2, 2]

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        quat[mask1, 0] = 0.25 * s
        quat[mask1, 1] = (rot_mat_flat[mask1, 2, 1] - rot_mat_flat[mask1, 1, 2]) / s
        quat[mask1, 2] = (rot_mat_flat[mask1, 0, 2] - rot_mat_flat[mask1, 2, 0]) / s
        quat[mask1, 3] = (rot_mat_flat[mask1, 1, 0] - rot_mat_flat[mask1, 0, 1]) / s

    # Case 2: rot_mat[0,0] is largest diagonal
    mask2 = (~mask1) & (rot_mat_flat[:, 0, 0] > rot_mat_flat[:, 1, 1]) & (rot_mat_flat[:, 0, 0] > rot_mat_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + rot_mat_flat[mask2, 0, 0] - rot_mat_flat[mask2, 1, 1] - rot_mat_flat[mask2, 2, 2]) * 2
        quat[mask2, 0] = (rot_mat_flat[mask2, 2, 1] - rot_mat_flat[mask2, 1, 2]) / s
        quat[mask2, 1] = 0.25 * s
        quat[mask2, 2] = (rot_mat_flat[mask2, 0, 1] + rot_mat_flat[mask2, 1, 0]) / s
        quat[mask2, 3] = (rot_mat_flat[mask2, 0, 2] + rot_mat_flat[mask2, 2, 0]) / s

    # Case 3: rot_mat[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (rot_mat_flat[:, 1, 1] > rot_mat_flat[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + rot_mat_flat[mask3, 1, 1] - rot_mat_flat[mask3, 0, 0] - rot_mat_flat[mask3, 2, 2]) * 2
        quat[mask3, 0] = (rot_mat_flat[mask3, 0, 2] - rot_mat_flat[mask3, 2, 0]) / s
        quat[mask3, 1] = (rot_mat_flat[mask3, 0, 1] + rot_mat_flat[mask3, 1, 0]) / s
        quat[mask3, 2] = 0.25 * s
        quat[mask3, 3] = (rot_mat_flat[mask3, 1, 2] + rot_mat_flat[mask3, 2, 1]) / s

    # Case 4: rot_mat[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + rot_mat_flat[mask4, 2, 2] - rot_mat_flat[mask4, 0, 0] - rot_mat_flat[mask4, 1, 1]) * 2
        quat[mask4, 0] = (rot_mat_flat[mask4, 1, 0] - rot_mat_flat[mask4, 0, 1]) / s
        quat[mask4, 1] = (rot_mat_flat[mask4, 0, 2] + rot_mat_flat[mask4, 2, 0]) / s
        quat[mask4, 2] = (rot_mat_flat[mask4, 1, 2] + rot_mat_flat[mask4, 2, 1]) / s
        quat[mask4, 3] = 0.25 * s

    # Normalize quaternions
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    return quat.reshape(*batch_shape, 4)


@torch.jit.script
def quat_to_angle_axis(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_from_euler(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to quaternions in (w, x, y, z) format.

    Args:
        euler: Tensor of shape (N, 3), where each row is (roll, pitch, yaw) in radians.

    Returns:
        Quaternion tensor of shape (N, 4) in (w, x, y, z) format.
    """
    half_euler = euler * 0.5
    c = half_euler.cos()
    s = half_euler.sin()

    cr, cp, cy = c.unbind(-1)
    sr, sp, sy = s.unbind(-1)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = torch.stack([w, x, y, z], dim=-1)
    return normalize(quat)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def swing_twist_decomposition(
    q: torch.Tensor, twist_axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose a quaternion into swing and twist components around a given axis.

    Args:
        q: Quaternion in (w, x, y, z) format. Shape is (..., 4).
        twist_axis: Unit vector for the twist axis. Shape is (3,).

    Returns:
        A tuple (swing, twist) quaternions, each of shape (..., 4).
    """
    # Project the quaternion's vector part onto the twist axis
    # q = (w, x, y, z), vector part = (x, y, z)
    projection = (q[..., 1:] * twist_axis).sum(dim=-1, keepdim=True) * twist_axis  # (..., 3)
    twist = torch.cat([q[..., 0:1], projection], dim=-1)  # (..., 4)
    twist = normalize(twist)
    # swing = q * twist^{-1}
    swing = quat_mul(q, quat_conjugate(twist))
    return swing, twist


@torch.jit.script
def quat_angle_ignoring_axis(
    q1: torch.Tensor, q2: torch.Tensor, ignore_axis: torch.Tensor
) -> torch.Tensor:
    """Compute the rotation angle between two quaternions, ignoring rotation around a specified axis.

    The decomposition is done in the local frame of q2, so ignore_axis should be a
    fixed object-local vector (e.g. [0, 1, 0]).

    Args:
        q1: First quaternion in (w, x, y, z). Shape is (..., 4).
        q2: Second quaternion in (w, x, y, z). Shape is (..., 4).
        ignore_axis: Unit vector for the axis to ignore. Shape is (3,).

    Returns:
        Angular error (swing component only) in radians. Shape is (...,).
    """
    # Compute relative rotation in q2's local frame: d_local = q2^{-1} * q1
    d_local = quat_mul(quat_conjugate(q2), q1)
    # Decompose into swing and twist around the ignore axis
    swing, _twist = swing_twist_decomposition(d_local, ignore_axis)
    # Swing angle from the w component
    angle = 2.0 * torch.acos(torch.clamp(torch.abs(swing[..., 0]), 0.0, 1.0))
    return angle


