"""Reward terms for hand-object trajectory imitation tasks.

Tracking terms are adapted from ManipTrans
(maniptrans_envs/lib/envs/tasks/dexhandimitator.py); the tactile, contact and
smoothness terms are specific to this project.
"""
import torch
import torch.nn.functional as F

from gs_env.common.utils.math_utils import (
    quat_angle_ignoring_axis,
    quat_conjugate,
    quat_mul,
)

from .reward_terms import RewardTerm


### ---- Trajectory Following Rewards ---- ###


class WristPositionTrackingReward(RewardTerm):
    """
    Reward for tracking target wrist position.
    Uses exponential reward: exp(-k * position_error)

    Required state keys:
        base_pos: Current wrist position (B, 3)
        target_wrist_pos: Target wrist position from trajectory (B, 3)
    """

    required_keys = ("base_pos", "target_wrist_pos")

    def __init__(self, scale: float = 0.1, k: float = 40.0, name: str | None = None):
        """
        Args:
            scale: Overall reward scale
            k: Exponential decay rate (higher = sharper reward)
        """
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_pos: torch.Tensor, target_wrist_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_wrist_pos[..., :3] - base_pos
        dist = torch.norm(diff, dim=-1)
        return torch.exp(-self.k * dist)


class ObjectPositionTrackingReward(RewardTerm):
    """
    Reward for tracking target object position.
    Uses exponential reward: exp(-k * position_error)

    Required state keys:
        object_pos: Current object position (B, 3)
        target_object_pos: Target object position from trajectory (B, 3)
    """

    required_keys = ("object_pos", "target_object_pos")

    def __init__(self, scale: float = 5.0, k: float = 80.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, object_pos: torch.Tensor, target_object_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        if len(target_object_pos.shape) == 3:
            target_object_pos = target_object_pos[:, 0]  # (B, 3)
        diff = target_object_pos - object_pos
        dist = torch.norm(diff, dim=-1)
        return torch.exp(-self.k * dist)


class WristRotationTrackingReward(RewardTerm):
    """
    Reward for tracking target wrist rotation.
    Uses exponential reward based on rotation angle difference.

    Required state keys:
        base_quat: Current wrist quaternion (B, 4) [w, x, y, z]
        target_wrist_quat: Target wrist quaternion (B, 4) [w, x, y, z]
    """

    required_keys = ("base_quat", "target_wrist_quat")

    def __init__(self, scale: float = 0.6, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_quat: torch.Tensor, target_wrist_quat: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Compute delta quaternion: q_delta = q_target * q_current^{-1}
        # Note: Using standard quaternion multiplication
        diff_quat = quat_mul(target_wrist_quat[:, :4], quat_conjugate(base_quat))

        # Extract rotation angle from quaternion
        # Use abs(w) so the angle is invariant to quaternion sign (q and -q are the same rotation)
        w = diff_quat[..., 0]
        angle = 2 * torch.acos(torch.clamp(torch.abs(w), 0.0, 1.0))

        return torch.exp(-self.k * angle)


class ObjectRotationTrackingReward(RewardTerm):

    required_keys = ("object_quat", "target_object_quat")

    def __init__(self, scale: float = 1.0, k: float = 3.0, ignore_axis: list[float] | None = None, name: str | None = None):
        super().__init__(scale, name)
        self.k = k
        self._ignore_axis_cfg = ignore_axis
        self._ignore_axis_tensor: torch.Tensor | None = None

    def _compute(self, object_quat: torch.Tensor, target_object_quat: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self._ignore_axis_cfg is not None:
            if self._ignore_axis_tensor is None:
                self._ignore_axis_tensor = torch.tensor(self._ignore_axis_cfg, dtype=object_quat.dtype, device=object_quat.device)
            angle = quat_angle_ignoring_axis(target_object_quat[:, :4], object_quat, self._ignore_axis_tensor)
        else:
            diff_quat = quat_mul(target_object_quat[:, :4], quat_conjugate(object_quat))
            w = diff_quat[..., 0]
            angle = 2 * torch.acos(torch.clamp(torch.abs(w), 0.0, 1.0))

        return torch.exp(-self.k * angle)


class FingertipPositionTrackingReward(RewardTerm):
    """
    Reward for tracking 5 fingertip positions in 3D space.
    Supports per-finger weighting (thumb, index, middle, ring, pinky).

    Required state keys:
        finger_link_pos: Current fingertip positions (B, 5 * 3)
        target_mano_joint_pos: Target fingertip positions (B, 5 * 3)
    """

    required_keys = ("finger_link_pos", "target_mano_joint_pos")

    def __init__(
        self,
        scale: float = 1.0,
        thumb_weight: float = 0.9,
        index_weight: float = 0.8,
        middle_weight: float = 0.75,
        ring_weight: float = 0.6,
        pinky_weight: float = 0.6,
        thumb_k: float = 100.0,
        index_k: float = 90.0,
        middle_k: float = 80.0,
        ring_k: float = 60.0,
        pinky_k: float = 60.0,
        name: str | None = None,
    ):
        super().__init__(scale, name)
        self.thumb_weight = thumb_weight
        self.index_weight = index_weight
        self.middle_weight = middle_weight
        self.ring_weight = ring_weight
        self.pinky_weight = pinky_weight
        self.thumb_k = thumb_k
        self.index_k = index_k
        self.middle_k = middle_k
        self.ring_k = ring_k
        self.pinky_k = pinky_k

    def _compute(self, finger_link_pos: torch.Tensor, target_mano_joint_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        num_envs = target_mano_joint_pos.shape[0]
        diff = target_mano_joint_pos.reshape(num_envs, 5, 3) - finger_link_pos.reshape(num_envs, 5, 3)
        dist = torch.norm(diff, dim=-1)  # (B, 5)

        # Order: thumb(0), index(1), middle(2), ring(3), pinky(4)
        total_reward = (
            self.thumb_weight * torch.exp(-self.thumb_k * dist[:, 0])
            + self.index_weight * torch.exp(-self.index_k * dist[:, 1])
            + self.middle_weight * torch.exp(-self.middle_k * dist[:, 2])
            + self.ring_weight * torch.exp(-self.ring_k * dist[:, 3])
            + self.pinky_weight * torch.exp(-self.pinky_k * dist[:, 4])
        )

        return total_reward


class FingertipRotationTrackingReward(RewardTerm):
    """
    Reward for tracking 5 fingertip orientations (quaternions) in SO(3).
    Supports per-finger weighting (thumb, index, middle, ring, pinky).

    Required state keys:
        finger_link_quat: Current fingertip quaternions (B, 5 * 4) [w, x, y, z]
        target_mano_joint_quat: Target fingertip quaternions (B, 5 * 4) [w, x, y, z]
    """

    required_keys = ("finger_link_quat", "target_mano_joint_quat")

    def __init__(
        self,
        scale: float = 1.0,
        thumb_weight: float = 0.9,
        index_weight: float = 0.8,
        middle_weight: float = 0.75,
        ring_weight: float = 0.6,
        pinky_weight: float = 0.6,
        thumb_k: float = 3.0,
        index_k: float = 3.0,
        middle_k: float = 3.0,
        ring_k: float = 3.0,
        pinky_k: float = 3.0,
        name: str | None = None,
    ):
        super().__init__(scale, name)
        self.thumb_weight = thumb_weight
        self.index_weight = index_weight
        self.middle_weight = middle_weight
        self.ring_weight = ring_weight
        self.pinky_weight = pinky_weight
        self.thumb_k = thumb_k
        self.index_k = index_k
        self.middle_k = middle_k
        self.ring_k = ring_k
        self.pinky_k = pinky_k

    def _compute(self, finger_link_quat: torch.Tensor, target_mano_joint_quat: torch.Tensor) -> torch.Tensor:  # type: ignore
        num_envs = target_mano_joint_quat.shape[0]
        cur_quat = finger_link_quat.reshape(num_envs, 5, 4)  # (B, 5, 4)
        ref_quat = target_mano_joint_quat.reshape(num_envs, 5, 4)  # (B, 5, 4)

        # Compute quaternion difference: q_diff = q_ref * q_cur^{-1}
        # For unit quaternions, inverse = conjugate
        cur_conj = cur_quat.clone()
        cur_conj[:, :, 1:] = -cur_conj[:, :, 1:]  # negate xyz
        # quat_mul expects (B, 4), so reshape to (B*5, 4) and back
        diff_quat = quat_mul(
            ref_quat.reshape(-1, 4), cur_conj.reshape(-1, 4)
        ).reshape(num_envs, 5, 4)  # (B, 5, 4)

        # Rotation angle from quaternion: angle = 2 * arccos(|w|)
        angle = 2.0 * torch.acos(torch.clamp(torch.abs(diff_quat[:, :, 0]), 0.0, 1.0))  # (B, 5)

        # Per-finger weighted exponential reward (same pattern as position tracking)
        # Order: thumb(0), index(1), middle(2), ring(3), pinky(4)
        total_reward = (
            self.thumb_weight * torch.exp(-self.thumb_k * angle[:, 0])
            + self.index_weight * torch.exp(-self.index_k * angle[:, 1])
            + self.middle_weight * torch.exp(-self.middle_k * angle[:, 2])
            + self.ring_weight * torch.exp(-self.ring_k * angle[:, 3])
            + self.pinky_weight * torch.exp(-self.pinky_k * angle[:, 4])
        )

        return total_reward


def _normalize_and_gamma_tactile(
    x: torch.Tensor, min_max: float, gamma: float
) -> torch.Tensor:
    """Mirror the tactile-map *visualization* preprocessing so the reward compares
    the same quantities a human sees in the rendered maps.

    See ``render_tactile_pair`` in
    ``scripts/visualization/render_rollout_with_tactile_frames.py``: each map (B, 1,
    24, 32) is independently normalized by its own per-env peak readout — floored at
    ``min_max`` so near-empty frames aren't amplified into noise — then passed
    through a ``gamma`` curve (gamma < 1 compresses the dynamic range so faint
    readouts still register while peaks don't dominate; 0.5 = sqrt).

    Note this makes the comparison scale-invariant per map: it rewards matching the
    *spatial contact pattern* rather than absolute force magnitude.
    """
    peak = x.amax(dim=(-2, -1), keepdim=True).clamp(min=min_max)
    x = (x / peak).clamp(0.0, 1.0)
    if gamma != 1.0:
        x = x.pow(gamma)
    return x


class WeightedGaussianBlurredTactileMapSimilarityReward(RewardTerm):
    """
    Compares tactile maps after visualization-matching preprocessing (per-map
    normalization + gamma, see _normalize_and_gamma_tactile) and a 3x3 Gaussian
    blur, applies per-pixel weights (e.g. higher weight on fingertip pixels)
    before computing the L1 error, AND masks the error to pixels where the *raw*
    target has contact.

    The masking is essential: without it the term is non-monotonic on real data
    (a no-contact policy scores a lower L1 against a sparse reference than a real
    contact pattern that is spatially offset from it), so it would reward making
    no contact. Restricting the error to target-contact pixels — plus a smooth
    contact gate — makes it actually reward matching the reference pattern.

    Required state keys:
        tactile_map_flat: Current tactile map (B, 768)
        target_tactile_map_flat: Target tactile map (B, 768)
        tactile_pixel_weights: Per-pixel weights (768,)
    """

    required_keys = ("tactile_map_flat", "target_tactile_map_flat", "tactile_pixel_weights")

    def __init__(self, scale: float = 1.0, k: float = 1.0, sigma: float = 1.0,
                 gamma: float = 0.5, min_max: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k
        # Visualization-matching preprocessing (see _normalize_and_gamma_tactile)
        self.gamma = gamma
        self.min_max = min_max
        # Build normalised 3x3 Gaussian kernel
        ax = torch.arange(-1, 2, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        # Shape for conv2d: (out_channels=1, in_channels=1, 3, 3)
        self._kernel = kernel.reshape(1, 1, 3, 3)

    def _compute(self, tactile_map_flat: torch.Tensor, target_tactile_map_flat: torch.Tensor, tactile_pixel_weights: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Lazy device move
        if self._kernel.device != tactile_map_flat.device:
            self._kernel = self._kernel.to(tactile_map_flat.device)
        # Contact mask + gate from the RAW target (before blur spreads to neighbors)
        mask = target_tactile_map_flat > 0  # (B, 768)
        num_contact = mask.sum(dim=-1)  # (B,)
        # Smooth gate: 0 with no contact, linear ramp 0->1 over 1->10 pixels, 1 above
        contact_scale = (num_contact.float() / 10.0).clamp(0.0, 1.0)
        # Reshape to spatial maps
        cur = tactile_map_flat.reshape(-1, 1, 24, 32)
        tgt = target_tactile_map_flat.reshape(-1, 1, 24, 32)
        # Match the visualization preprocessing: per-map normalize + gamma...
        cur = _normalize_and_gamma_tactile(cur, self.min_max, self.gamma)
        tgt = _normalize_and_gamma_tactile(tgt, self.min_max, self.gamma)
        # ...then Gaussian blur (padding=1 keeps spatial dims)
        cur = F.conv2d(cur, self._kernel, padding=1)
        tgt = F.conv2d(tgt, self._kernel, padding=1)
        # Weighted + target-masked L1 error
        diff = torch.abs((tgt - cur).reshape(tactile_map_flat.shape[0], -1))  # (B, 768)
        weights = tactile_pixel_weights * mask  # (768,) * (B, 768) -> (B, 768)
        weight_sum = weights.sum(dim=-1).clamp(min=1)  # (B,)
        error = (diff * weights).sum(dim=-1) / weight_sum  # (B,)
        return torch.exp(-self.k * error) * contact_scale


class HandDofPositionTrackingReward(RewardTerm):
    required_keys = ("hand_dof_pos", "target_hand_dof_pos")

    def __init__(self, scale: float = 1.0, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, hand_dof_pos: torch.Tensor, target_hand_dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_hand_dof_pos - hand_dof_pos
        error = torch.abs(diff).mean(dim=-1)
        return torch.exp(-self.k * error)


class AllDOFPowerPenalty(RewardTerm):
    """Penalize |torque * velocity| across all DOFs (arm + fingers)."""
    required_keys = ("dof_force", "dof_vel")

    def __init__(self, scale: float = 0.5, k: float = 10.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, dof_force: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        power = torch.abs(dof_force * dof_vel).sum(dim=-1)
        return torch.exp(-self.k * power)


class TableContactPenalty(RewardTerm):
    """Penalize table contact force as a soft reward penalty.

    Uses exp(-k * max(force - threshold, 0)) so that forces below the
    threshold receive full reward (1.0) and larger forces are smoothly
    penalized. This is gentler than hard termination and gives the policy
    a gradient to learn from.

    Required state keys:
        table_contact_force_magnitude: Contact force on the table (B,) or (B, 1)
    """

    required_keys = ("table_contact_force_magnitude",)

    def __init__(self, scale: float = 1.0, k: float = 0.01, threshold: float = 100.0, name: str | None = None):
        """
        Args:
            scale: Overall reward scale (positive; reward drops as excess force grows)
            k: Exponential decay rate for force above threshold
            threshold: Force magnitude below which no penalty is applied
        """
        super().__init__(scale, name)
        self.k = k
        self.threshold = threshold

    def _compute(self, table_contact_force_magnitude: torch.Tensor) -> torch.Tensor:  # type: ignore
        force = table_contact_force_magnitude.squeeze(-1)  # (B,)
        excess = torch.clamp(force - self.threshold, min=0.0)
        return torch.exp(-self.k * excess)


class ObjectVelocityChangePenalty(RewardTerm):
    """
    Penalize sudden changes in object velocity (acceleration).
    Returns exp(-k * max(||vel_change|| - threshold, 0)) - 1, gated to zero
    when the object is at or below ``height_threshold`` above the table top
    (so table contacts don't trigger the penalty).

    Required state keys:
        object_vel_change: Norm of object velocity change per step (B,)
        object_height_above_table: Object z minus table-top z per env (B,)
    """

    required_keys = ("object_vel_change", "object_height_above_table")

    def __init__(self, scale: float = 1.0, k: float = 2.0, threshold: float = 0.15,
                 height_threshold: float = 0.05, name: str | None = None):
        super().__init__(scale, name)
        self.k = k
        self.threshold = threshold
        self.height_threshold = height_threshold

    def _compute(self, object_vel_change: torch.Tensor, object_height_above_table: torch.Tensor) -> torch.Tensor:  # type: ignore
        excess = torch.clamp(object_vel_change - self.threshold, min=0.0)
        penalty = torch.exp(-self.k * excess) - 1.0
        above = (object_height_above_table > self.height_threshold).to(penalty.dtype)
        return penalty * above


class ObjectAngularVelocityChangePenalty(RewardTerm):
    """
    Penalize sudden changes in object angular velocity (angular acceleration).
    Returns exp(-k * max(||ang_vel_change|| - threshold, 0)) - 1, gated to zero
    when the object is at or below ``height_threshold`` above the table top
    (so table contacts don't trigger the penalty).

    Required state keys:
        object_ang_vel_change: Norm of object angular velocity change per step (B,)
        object_height_above_table: Object z minus table-top z per env (B,)
    """

    required_keys = ("object_ang_vel_change", "object_height_above_table")

    def __init__(self, scale: float = 1.0, k: float = 0.5, threshold: float = 3.0,
                 height_threshold: float = 0.05, name: str | None = None):
        super().__init__(scale, name)
        self.k = k
        self.threshold = threshold
        self.height_threshold = height_threshold

    def _compute(self, object_ang_vel_change: torch.Tensor, object_height_above_table: torch.Tensor) -> torch.Tensor:  # type: ignore
        excess = torch.clamp(object_ang_vel_change - self.threshold, min=0.0)
        penalty = torch.exp(-self.k * excess) - 1.0
        above = (object_height_above_table > self.height_threshold).to(penalty.dtype)
        return penalty * above


class TactileGatedFingertipSurfaceProximityReward(RewardTerm):
    """
    Per-finger reward for being close to the object surface, gated by whether
    the target tactile map has readout on that finger's pixels.

    For each finger i (0..4):
      - Check if target tactile map has any nonzero value on finger i's pixels
      - If yes: reward_i = exp(-k * max(sdf_i, 0))  (only penalize distance, not penetration)
      - If no: reward_i = 0
    If none of the 5 fingertip regions have any readout, total reward = 0.
    Final reward = sum of per-finger rewards.

    Required state keys:
        fingertip_to_surface_dist: SDF distance from each fingertip to object surface (B, 5)
        target_tactile_map_flat: Target tactile map (B, 768)
        fingertip_pixel_masks: Boolean masks mapping fingers to pixels (5, 768)
    """

    required_keys = ("fingertip_to_surface_dist", "target_tactile_map_flat", "fingertip_pixel_masks")

    def __init__(self, scale: float = 1.0, k: float = 50.0,
                 tactile_threshold: float = 0.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k
        self.tactile_threshold = tactile_threshold

    def _compute(self, fingertip_to_surface_dist: torch.Tensor,  # type: ignore
                 target_tactile_map_flat: torch.Tensor,
                 fingertip_pixel_masks: torch.Tensor) -> torch.Tensor:
        # fingertip_to_surface_dist: (B, 5)
        # target_tactile_map_flat: (B, 768)
        # fingertip_pixel_masks: (5, 768)

        # For each finger, check if target tactile map has readout on its pixels
        # (B, 1, 768) * (1, 5, 768) -> (B, 5, 768) -> max over pixels -> (B, 5)
        masked_tactile = target_tactile_map_flat.unsqueeze(1) * fingertip_pixel_masks.unsqueeze(0).float()
        finger_has_contact = masked_tactile.max(dim=-1).values > self.tactile_threshold  # (B, 5) bool

        # Proximity reward: exp(-k * max(sdf, 0)) — clamp so penetration gives full reward
        proximity = torch.exp(-self.k * torch.clamp(fingertip_to_surface_dist, min=0.0))  # (B, 5)

        # Gate by contact: zero reward for fingers with no target readout
        gated_reward = proximity * finger_has_contact.float()  # (B, 5)

        # Sum per-finger rewards -> (B,)
        return gated_reward.sum(dim=-1)
