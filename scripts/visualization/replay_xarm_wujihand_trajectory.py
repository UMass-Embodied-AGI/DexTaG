"""Replay a WujiHand trajectory using the combined xArm7 + WujiHand URDF.

Instead of a free-floating hand, the wrist pose is achieved by solving IK
for the 7 xArm joints while the hand is mounted on the arm's end-effector.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import genesis as gs


URDF = "assets/robot/xarm/xarm7_with_wujihand_v5.urdf"
GHOST_URDF = "assets/robot/wujihand-urdf/urdf/right.urdf"
# wujihand_fix yaw on the combined URDF
TCP_YAW = 2.3562  # 135 deg
# Warm-start arm seed for IK
ARM_SEED = np.array([
    -0.2276704, 0.17346721, -0.30160318, 0.53979892,
    2.35584248, 1.32476175, 2.97201047,
])


def augment_trajectory(
    wrist_positions,
    wrist_rotations_aa,
    object_pose_matrices,
    finger_joints=None,
    finger_joints_quat=None,
    translate_x=0.0,
    translate_y=0.0,
    rotate_z_deg=0.0,
    scale=1.0,
):
    """
    Augment hand-object trajectory with exact specified values.

    Args:
        translate_x: Translation in X (meters).
        translate_y: Translation in Y (meters).
        rotate_z_deg: Rotation around Z-axis (degrees).
        scale: Scales the object's motion about its first-frame position; the
            wrist keeps its per-frame offset to the object.
        finger_joints: Dict of joint_name -> (T, 3) world-frame positions.
        finger_joints_quat: Dict of joint_name -> (T, 4) wxyz quaternions.
    Finger DOFs are left unchanged (intrinsic).
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
            R_wrist = R.from_rotvec(aa).as_matrix()
        else:
            R_wrist = np.eye(3)
        wrist_rotations_aa[t] = R.from_matrix(Rz @ R_wrist).as_rotvec()

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
        Rz_scipy = R.from_matrix(Rz)
        for joint_name, joint_quat_wxyz in finger_joints_quat.items():
            q_xyzw = joint_quat_wxyz[:, [1, 2, 3, 0]]
            r_new = Rz_scipy * R.from_quat(q_xyzw)
            q_new_xyzw = r_new.as_quat()
            finger_joints_quat[joint_name] = q_new_xyzw[:, [3, 0, 1, 2]]

    return wrist_positions, wrist_rotations_aa, object_pose_matrices, finger_joints, finger_joints_quat


def main():
    parser = argparse.ArgumentParser(
        description="Replay hand-object trajectory with xArm7 + WujiHand",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="Path to a demonstration trajectory pickle file",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the trajectory playback",
    )
    parser.add_argument(
        "--xarm-base-pos",
        type=float,
        nargs=3,
        default=[-0.2, 0.0, 0.0],
        help="xArm base position in meters (x y z)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.1],
        help="Global offset to apply to the entire trajectory (x y z in meters)",
    )
    parser.add_argument(
        "--translate-x",
        type=float,
        default=None,
        help="Augmentation: translate trajectory in X (meters)",
    )
    parser.add_argument(
        "--translate-y",
        type=float,
        default=None,
        help="Augmentation: translate trajectory in Y (meters)",
    )
    parser.add_argument(
        "--rotate-z",
        type=float,
        default=None,
        help="Augmentation: rotate trajectory around Z-axis (degrees)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Augmentation: scale the object's motion about its first frame (wrist follows the object)",
    )
    args = parser.parse_args()

    ########################## Load trajectory data ##########################
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    print(f"\n{'='*80}")
    print(f"Loading trajectory from: {trajectory_path}")
    print(f"{'='*80}")

    with open(trajectory_path, "rb") as f:
        data = pickle.load(f)

    shadow_traj = data["hand_trajectory"]
    object_traj = data["object_trajectory"]
    metadata = data["metadata"]

    wrist_positions = shadow_traj["wrist_positions"]  # (T, 3) meters
    wrist_rotations = shadow_traj["wrist_rotations_aa"]  # (T, 3) axis-angle
    dof_positions = shadow_traj["dof_positions"]  # (T, N) finger DOFs

    object_pose_matrices = object_traj["pose_matrices"]  # (T, 4, 4)

    # Load fingertip data from mano_reference
    mano_ref = data.get("mano_reference", {})
    all_finger_joints = mano_ref.get("finger_joints", {})
    all_finger_joints_quat = mano_ref.get("finger_joints_quat", {})
    # Filter to 5 fingertip joints (names ending with _tip)
    tip_names = sorted([k for k in all_finger_joints.keys() if k.endswith("_tip")])
    if not tip_names:
        tip_names = sorted(all_finger_joints.keys())
    fingertip_positions = {k: all_finger_joints[k] for k in tip_names}  # name -> (T, 3)
    fingertip_quats = {k: all_finger_joints_quat[k] for k in tip_names}  # name -> (T, 4) wxyz
    print(f"Loaded {len(tip_names)} fingertip joints: {tip_names}")

    # Save original trajectory for ghost visualization before any modification
    orig_wrist_positions = wrist_positions.copy()
    orig_wrist_rotations = wrist_rotations.copy()
    orig_dof_positions = dof_positions.copy()
    orig_object_pose_matrices = object_pose_matrices.copy()
    orig_fingertip_positions = {k: v.copy() for k, v in fingertip_positions.items()}
    orig_fingertip_quats = {k: v.copy() for k, v in fingertip_quats.items()}

    # Apply augmentation before offset (so offset is not distorted by scale/rotation)
    aug_tx = args.translate_x or 0.0
    aug_ty = args.translate_y or 0.0
    aug_rz = args.rotate_z or 0.0
    aug_scale = args.scale or 1.0
    has_augmentation = any(v is not None for v in [args.translate_x, args.translate_y, args.rotate_z, args.scale])

    if has_augmentation:
        print(f"{'='*80}")
        print("Applying trajectory augmentation...")
        print(f"  Translate X: {aug_tx:.3f}m, Y: {aug_ty:.3f}m")
        print(f"  Rotate Z: {aug_rz:.1f}°")
        print(f"  Scale: {aug_scale:.3f}")
        print(f"{'='*80}")

        wrist_positions, wrist_rotations, object_pose_matrices, fingertip_positions, fingertip_quats = augment_trajectory(
            wrist_positions,
            wrist_rotations,
            object_pose_matrices,
            finger_joints=fingertip_positions,
            finger_joints_quat=fingertip_quats,
            translate_x=aug_tx,
            translate_y=aug_ty,
            rotate_z_deg=aug_rz,
            scale=aug_scale,
        )
        print("Augmentation applied successfully!\n")

    # Apply global offset after augmentation (to both augmented and original)
    offset = np.array(args.offset)
    wrist_positions += offset
    object_pose_matrices[:, :3, 3] += offset
    orig_wrist_positions += offset
    orig_object_pose_matrices[:, :3, 3] += offset
    for k in fingertip_positions:
        fingertip_positions[k] += offset
    for k in orig_fingertip_positions:
        orig_fingertip_positions[k] += offset

    T = len(wrist_positions)
    print(f"Trajectory length: {T} timesteps")
    print(f"Finger DOF count in trajectory: {dof_positions.shape[1]}")
    print(f"{'='*80}\n")

    ########################## Init Genesis ##########################
    gs.init(backend=gs.cuda, logging_level="error")

    ########################## Init IK solver ##########################
    # TCP offset accounts for the wujihand_fix joint (xyz "0 0 0.057", yaw 135°).
    # IK library units: mm and radians
    from gs_env.sim.robots.xarm.xarm7_kinematics import XArm7Kinematics
    kin = XArm7Kinematics(tcp_offset=[0, 0, 57, 0, 0, TCP_YAW])

    base_pos = args.xarm_base_pos

    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0.5, -0.5, 0.5),
        camera_lookat=tuple(base_pos),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, 0),
            enable_collision=True,
        ),
        show_viewer=True,
    )

    ########################## Entities ##########################
    scene.add_entity(gs.morphs.Plane())

    # Object mesh
    load_data_dir = str(trajectory_path.parent)
    obj_id = metadata["obj_id"]
    object_mesh_path = Path(load_data_dir) / f"{obj_id}_collision.obj"
    print(f"Loading object mesh from: {object_mesh_path}")

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(object_mesh_path),
            pos=object_pose_matrices[0][:3, 3].tolist(),
            euler=(90, 0, 0),
            collision=False
        ),
    )

    # xArm7 + WujiHand (fixed base)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=URDF,
            merge_fixed_links=False,
            fixed=True,
            pos=base_pos,
            collision=False,
            recompute_inertia=True,
        ),
    )

    scene.add_entity(
        gs.morphs.Box(
            size=(0.6, 1.0, 0.02),
            pos=(offset[0] + 0.5, offset[1], offset[2]-0.01),
            collision=True,
            fixed=True,
        ),
    )

    # Ghost entities for original (un-augmented) trajectory comparison
    ghost_hand = None
    ghost_obj = None
    if has_augmentation:
        ghost_obj = scene.add_entity(
            gs.morphs.Mesh(
                file=str(object_mesh_path),
                pos=orig_object_pose_matrices[0][:3, 3].tolist(),
                euler=(90, 0, 0),
                collision=False,
            ),
        )
        ghost_hand = scene.add_entity(
            gs.morphs.URDF(
                file=GHOST_URDF,
                merge_fixed_links=False,
                fixed=False,
                is_free=True,
                collision=False,
            ),
        )

    # Axis mesh entities for augmented fingertip visualization
    tip_axis_entities = []
    for name in tip_names:
        ent = scene.add_entity(
            gs.morphs.Mesh(
                file="assets/scene/axis.obj",
                scale=0.03,
                pos=fingertip_positions[name][0].tolist(),
                collision=False,
            ),
        )
        tip_axis_entities.append(ent)

    # Ghost axis mesh entities for original fingertip visualization
    orig_tip_axis_entities = []
    if has_augmentation:
        for name in tip_names:
            ent = scene.add_entity(
                gs.morphs.Mesh(
                    file="assets/scene/axis.obj",
                    scale=0.03,
                    pos=orig_fingertip_positions[name][0].tolist(),
                    collision=False,
                ),
            )
            orig_tip_axis_entities.append(ent)

    ########################## Build ##########################
    scene.build()

    # Arm DOF indices: joint1 – joint7
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_dof_idx = []
    for name in arm_joint_names:
        arm_dof_idx.extend(robot.get_joint(name).dofs_idx_local)

    # Finger DOF indices: finger{1-5}_joint{1-4}
    finger_joint_names = [
        f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)
    ]
    finger_dof_idx = []
    for name in finger_joint_names:
        finger_dof_idx.extend(robot.get_joint(name).dofs_idx_local)

    # Ghost hand finger DOF indices
    ghost_finger_dof_idx = []
    if ghost_hand is not None:
        for name in finger_joint_names:
            ghost_finger_dof_idx.extend(ghost_hand.get_joint(name).dofs_idx_local)

    print(f"Robot total DOFs: {robot.n_dofs}")
    print(f"Arm DOFs: {len(arm_dof_idx)} (indices: {arm_dof_idx})")
    print(f"Finger DOFs: {len(finger_dof_idx)} (indices: {finger_dof_idx})")

    robot.set_dofs_kp([6300] * 7 + [20.0] * 20)
    robot.set_dofs_kv([315.8] * 7 + [5.0] * 20)

    n_finger_dofs = min(len(finger_dof_idx), dof_positions.shape[1])
    if len(finger_dof_idx) != dof_positions.shape[1]:
        print(
            f"WARNING: DOF count mismatch! Robot has {len(finger_dof_idx)} finger DOFs "
            f"but trajectory has {dof_positions.shape[1]}. Using first {n_finger_dofs}."
        )


    ########################## Replay ##########################
    base_pos_np = np.array(base_pos)
    prev_arm_angles = np.zeros(7)
    ik_failures = 0

    def replay():
        nonlocal prev_arm_angles, ik_failures

        while True:
            ik_failures = 0
            prev_arm_angles = ARM_SEED.copy()

            for t in range(T):
                # --- Build 4x4 target matrix for IK ---
                # Rotation from axis-angle
                axis_angle = wrist_rotations[t]
                angle = np.linalg.norm(axis_angle)
                if angle > 1e-6:
                    rot_mat = R.from_rotvec(axis_angle).as_matrix()
                else:
                    rot_mat = np.eye(3)

                # Position: metres (world) → mm (arm base frame)
                target_pos_mm = (wrist_positions[t] - base_pos_np) * 1000.0

                target_mat = np.eye(4)
                target_mat[:3, :3] = rot_mat
                target_mat[:3, 3] = target_pos_mm

                # --- Solve IK ---
                try:
                    arm_angles = kin.inverse_kinematics_mat(
                        target_mat, q_ref=prev_arm_angles,
                    )
                    if t != 0:
                        robot.set_dofs_position(arm_angles, arm_dof_idx)
                    else:
                        robot.set_dofs_position(arm_angles, arm_dof_idx)
                    prev_arm_angles = arm_angles
                except RuntimeError as e:
                    ik_failures += 1
                    if ik_failures <= 10:
                        print(f"[t={t}] IK failed: {e}")
                    elif ik_failures == 11:
                        print("(suppressing further IK warnings...)")

                # --- Set finger DOFs ---
                finger_dofs = dof_positions[t, :n_finger_dofs]
                robot.set_dofs_position(finger_dofs, finger_dof_idx[:n_finger_dofs])


                # --- Set object pose ---
                pose_matrix = object_pose_matrices[t]
                obj_pos = pose_matrix[:3, 3]
                obj_rot_matrix = pose_matrix[:3, :3]
                obj_quat = gs.utils.geom.R_to_quat(obj_rot_matrix)
                obj.set_qpos(np.concatenate([obj_pos, obj_quat]))

                # --- Update ghost (original trajectory) ---
                if ghost_hand is not None:
                    # Ghost hand wrist pose
                    orig_aa = orig_wrist_rotations[t]
                    orig_angle = np.linalg.norm(orig_aa)
                    if orig_angle > 1e-6:
                        orig_quat = gs.utils.geom.axis_angle_to_quat(
                            np.array(orig_angle), orig_aa / orig_angle,
                        )
                    else:
                        orig_quat = np.array([1.0, 0.0, 0.0, 0.0])
                    ghost_hand.set_pos(orig_wrist_positions[t])
                    ghost_hand.set_quat(orig_quat)
                    ghost_hand.set_dofs_position(
                        orig_dof_positions[t, :len(ghost_finger_dof_idx)],
                        ghost_finger_dof_idx,
                    )

                if ghost_obj is not None:
                    orig_pose = orig_object_pose_matrices[t]
                    orig_obj_quat = gs.utils.geom.R_to_quat(orig_pose[:3, :3])
                    ghost_obj.set_qpos(np.concatenate([orig_pose[:3, 3], orig_obj_quat]))

                # --- Update augmented fingertip axes ---
                for i, name in enumerate(tip_names):
                    tip_axis_entities[i].set_pos(fingertip_positions[name][t])
                    tip_axis_entities[i].set_quat(fingertip_quats[name][t])

                # --- Update ghost (original) fingertip axes ---
                for i, name in enumerate(tip_names):
                    if i < len(orig_tip_axis_entities):
                        orig_tip_axis_entities[i].set_pos(orig_fingertip_positions[name][t])
                        orig_tip_axis_entities[i].set_quat(orig_fingertip_quats[name][t])

                scene.step()

                if t % 50 == 0:
                    progress = (t / T) * 100
                    print(f"Progress: {progress:.1f}% ({t}/{T})", end="\r")

            print(f"\nTrajectory replay completed ({T} steps)")
            if ik_failures > 0:
                print(f"Total IK failures: {ik_failures}/{T}")

            if not args.loop:
                break

    replay()


if __name__ == "__main__":
    main()
