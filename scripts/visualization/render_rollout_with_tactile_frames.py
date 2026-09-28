"""Render a rollout episode's full scene + side-by-side tactile maps per frame.

Loads a single recorded episode npz from
`deploy/logs/<exp>/trajectories/episode_*.npz` (produced by
`run_ppo_single_hand_retargeting.py --eval=True --save_trajectory=True`) and
replays it kinematically: robot driven by `arm_dof_pos`/`hand_dof_pos`, object
driven by the recorded `object_pos`/`object_quat`. Outputs:

  <output-dir>/XXXXX_scene.png   — full scene RGB (robot + tables + object)
  <output-dir>/XXXXX_tactile.png — current vs. target tactile map (24x32)
  <output-dir>/scene.mp4         — combined scene video
  <output-dir>/tactile.mp4       — combined tactile-map video

Kinematic replay: no physics, and no trajectory pickles needed (only the
object mesh).

Example:
  python scripts/visualization/render_rollout_with_tactile_frames.py \
      --npz deploy/logs/my_run/trajectories/episode_0000_<traj_id>.npz \
      --output-dir /tmp/rollout_demo/ \
      --object-mesh-dir /path/to/pickles_filtered --object-id marker_pen_scanned
"""

import argparse
from pathlib import Path

import cv2
import genesis as gs
import imageio.v2 as imageio
import numpy as np


DEFAULT_URDF = "assets/robot/xarm/xarm7_with_wujihand_v5.urdf"
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = [f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]


class H264VideoWriter:
    """Drop-in for cv2.VideoWriter that encodes H.264 (yuv420p) via the ffmpeg
    bundled with imageio-ffmpeg (which includes libx264 — the OpenCV build here
    does not). Output opens natively in VS Code, browsers, and QuickTime.
    Accepts BGR frames, like cv2.VideoWriter.
    """

    def __init__(self, path: str, fps: float) -> None:
        self._writer = imageio.get_writer(
            path, format="FFMPEG", mode="I", fps=float(fps),
            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_log_level="error", output_params=["-crf", "20", "-preset", "medium"],
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.append_data(frame_bgr[:, :, ::-1])  # BGR -> RGB for ffmpeg

    def release(self) -> None:
        self._writer.close()


def collect_entity_idxc(seg_color_map, entity_idx: int) -> set[int]:
    """All seg_idxc values that map to the given entity (handles entity/link/geom levels)."""
    idxcs: set[int] = set()
    for idxc, key in seg_color_map.idxc_map.items():
        if key == -1:
            continue
        eid = key[0] if isinstance(key, tuple) else key
        if eid == entity_idx:
            idxcs.add(idxc)
    return idxcs


def render_tactile_pair(
    cur_map: np.ndarray,
    tgt_map: np.ndarray,
    gamma: float = 0.5,
    min_max: float = 1.0,
) -> np.ndarray:
    """Side-by-side current/target tactile maps. Inputs: (24, 32) float arrays.

    Each map is independently normalized by its own per-frame max (floored at
    `min_max` so empty frames don't get amplified into noise) — so the brightest
    taxel of each map maps to the same JET top color regardless of absolute
    magnitude. `gamma < 1` then compresses the dynamic range so faint readouts
    are still visible while peaks don't dominate (0.5 = sqrt).
    """
    cur_max = max(float(cur_map.max()), float(min_max))
    tgt_max = max(float(tgt_map.max()), float(min_max))
    cur_norm = (np.asarray(cur_map, dtype=np.float32) / cur_max).clip(0.0, 1.0)
    tgt_norm = (np.asarray(tgt_map, dtype=np.float32) / tgt_max).clip(0.0, 1.0)
    if gamma != 1.0:
        cur_norm = np.power(cur_norm, gamma)
        tgt_norm = np.power(tgt_norm, gamma)
    cur_u8 = (cur_norm * 255).astype(np.uint8)
    tgt_u8 = (tgt_norm * 255).astype(np.uint8)
    cur_u8 = cv2.GaussianBlur(cur_u8, (3, 3), 0)
    tgt_u8 = cv2.GaussianBlur(tgt_u8, (3, 3), 0)
    cur_color = cv2.applyColorMap(cur_u8, cv2.COLORMAP_JET)
    tgt_color = cv2.applyColorMap(tgt_u8, cv2.COLORMAP_JET)
    cur_color = cv2.resize(cur_color, (32 * 16, 24 * 16), interpolation=cv2.INTER_NEAREST)
    tgt_color = cv2.resize(tgt_color, (32 * 16, 24 * 16), interpolation=cv2.INTER_NEAREST)
    spacer = np.ones((24 * 16, 16, 3), dtype=np.uint8) * 255
    combined = np.hstack([cur_color, spacer, tgt_color])
    cv2.putText(combined, "Current", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Target", (32 * 16 + 16 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to a rollout episode npz with arm_dof_pos, hand_dof_pos, "
                             "object_pos, object_quat, tactile_map, target_tactile_map.")
    parser.add_argument("--output-dir", type=str, default="/tmp/rollout_demo",
                        help="Frames are written as <output-dir>/XXXXX_scene.png and "
                             "<output-dir>/XXXXX_tactile.png. Videos are written to "
                             "<output-dir>/scene.mp4 and <output-dir>/tactile.mp4.")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frame rate of the scene video (default 60 = env.trajectory_playback_hz).")
    parser.add_argument("--tactile-fps", type=int, default=15,
                        help="Frame rate of the tactile video (slower than scene for readability).")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing the combined mp4 videos.")
    parser.add_argument("--object-mesh-dir", type=str, required=True,
                        help="Directory containing {object-id}_collision.obj.")
    parser.add_argument("--object-id", type=str, default="marker_pen_scanned",
                        help="Used to locate {object-mesh-dir}/{object_id}_collision.obj.")
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    parser.add_argument("--arm-base-pos", type=float, nargs=3, default=[-0.2, 0.0, 0.0])
    parser.add_argument("--trajectory-offset", type=float, nargs=3, default=[0.0, 0.0, 0.1],
                        help="World offset for the workpiece table (matches env_args.yaml).")
    parser.add_argument("--res", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--cam-offset", type=float, nargs=3, default=[0.45, 0.45, 0.30],
                        help="World-frame translation from tracked link to camera (m). "
                             "Direction sets the fixed orientation; magnitude sets framing.")
    parser.add_argument("--track-link", type=str, default="palm_link",
                        help="Robot link the camera should track (default: WUJI hand base).")
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--object-color", type=float, nargs=3, default=[0.55, 0.72, 0.92])
    parser.add_argument("--object-roughness", type=float, default=0.9)
    parser.add_argument("--table-color", type=float, nargs=3, default=[0.82, 0.68, 0.50])
    parser.add_argument("--table-roughness", type=float, default=0.8)
    parser.add_argument("--bg-color", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--transparent", action="store_true",
                        help="Save 4-channel PNGs with alpha (background → fully transparent).")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="-1 = full length")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--tactile-gamma", type=float, default=0.5,
                        help="Gamma for tactile-map dynamic-range compression (<1 brightens "
                             "small readouts, =1 disables). Try 0.35 for stronger, 0.7 for weaker.")
    parser.add_argument("--tactile-min-max", type=float, default=1.0,
                        help="Floor for the per-map normalization peak — prevents empty "
                             "frames from being amplified into noise. Raise if low-amplitude "
                             "frames still look too bright.")
    args = parser.parse_args()

    # ----------------------- load rollout npz -----------------------
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    required = ("arm_dof_pos", "hand_dof_pos", "object_pos", "object_quat",
                "tactile_map", "target_tactile_map")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"{npz_path} missing keys: {missing}. Available: {list(data.files)}")

    arm_dof_raw = np.asarray(data["arm_dof_pos"])               # (T, 7)
    hand_dof_raw = np.asarray(data["hand_dof_pos"])             # (T, 20)
    object_pos_raw = np.asarray(data["object_pos"], dtype=np.float64)    # (T, 3)
    object_quat_raw = np.asarray(data["object_quat"], dtype=np.float64)  # (T, 4) WXYZ
    tactile_map_raw = np.asarray(data["tactile_map"]).reshape(-1, 24, 32)
    target_tactile_map_raw = np.asarray(data["target_tactile_map"]).reshape(-1, 24, 32)
    T_raw = arm_dof_raw.shape[0]
    for name, arr in [
        ("hand_dof_pos", hand_dof_raw),
        ("object_pos", object_pos_raw), ("object_quat", object_quat_raw),
        ("tactile_map", tactile_map_raw), ("target_tactile_map", target_tactile_map_raw),
    ]:
        if arr.shape[0] != T_raw:
            raise ValueError(f"{name} has length {arr.shape[0]}, expected {T_raw}")

    # In the env, arm_dof_pos[t] is read from the robot's cached _dof_pos which is
    # filled BEFORE the per-control-step scene.step(), while hand_dof_pos / object_pos
    # / object_quat / tactile_map are read AFTER it. With decimation=1 that's exactly
    # one control step of lag for the arm, which shows up in playback as the object
    # subtly leading the hand. Shift arm_dof forward by one step (and drop the last
    # frame of everything else) so all arrays at index t correspond to the same
    # post-step simulation moment.
    arm_dof = arm_dof_raw[1:]
    hand_dof = hand_dof_raw[:-1]
    object_pos = object_pos_raw[:-1]
    object_quat = object_quat_raw[:-1]
    tactile_map = tactile_map_raw[:-1]
    target_tactile_map = target_tactile_map_raw[:-1]
    T_total = arm_dof.shape[0]

    t_end = T_total if args.end < 0 else min(args.end, T_total)
    frame_idxs = list(range(args.start, t_end, max(1, args.step)))
    print(f"npz: {npz_path.name}  T={T_total}, rendering {len(frame_idxs)} frames")

    obj_mesh_path = Path(args.object_mesh_dir) / f"{args.object_id}_collision.obj"
    if not obj_mesh_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {obj_mesh_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offset = np.asarray(args.trajectory_offset, dtype=np.float64)

    # ----------------------- init genesis (kinematic playback) -----------------------
    gs.init(backend=gs.cuda, logging_level="error")

    # No gravity, no collisions: robot and object are driven entirely by set_dofs_position /
    # set_pos / set_quat from the logged trajectory.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, 0.0), enable_collision=False),
        vis_options=gs.options.VisOptions(
            segmentation_level="entity",
            show_world_frame=False,
            shadow=True,
            plane_reflection=False,
            background_color=tuple(args.bg_color),
            ambient_light=(0.20, 0.20, 0.20),
            lights=[
                {"type": "directional", "dir": (-1.0, -1.0, -1.5),
                 "color": (1.0, 1.0, 1.0), "intensity": 4.5},
                {"type": "directional", "dir": (1.0, 1.0, -0.5),
                 "color": (1.0, 1.0, 1.0), "intensity": 1.5},
            ],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=tuple(args.arm_base_pos),
            camera_lookat=tuple(args.arm_base_pos),
            camera_fov=args.fov, max_FPS=60,
        ),
        show_viewer=args.show_viewer,
    )

    arm_base_pos = tuple(args.arm_base_pos)

    table_surface = gs.surfaces.Plastic(
        color=tuple(args.table_color), roughness=args.table_roughness, metallic=0.0,
    )
    robot_table = scene.add_entity(
        gs.morphs.Box(size=(0.6, 1.0, 0.1), pos=(-0.44, 0.0, -0.053),
                      collision=False, fixed=True),
        surface=table_surface,
    )
    work_table = scene.add_entity(
        gs.morphs.Box(size=(0.6, 1.0, 0.02),
                      pos=(offset[0] + 0.5, offset[1], offset[2] - 0.01),
                      collision=False, fixed=True),
        surface=table_surface,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=args.urdf, merge_fixed_links=False,
            fixed=True, pos=arm_base_pos, recompute_inertia=True,
        ),
    )

    object_surface = gs.surfaces.Plastic(
        color=tuple(args.object_color), roughness=args.object_roughness, metallic=0.0,
    )
    obj = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(obj_mesh_path),
            pos=object_pos[0].tolist(),
            euler=(90, 0, 0),  # overwritten by set_quat() in the loop below.
            collision=False,
        ),
        surface=object_surface,
    )

    width, height = args.res
    camera = scene.add_camera(
        res=(width, height),
        pos=tuple(args.arm_base_pos),
        lookat=tuple(args.arm_base_pos),
        up=tuple(args.up),
        fov=args.fov, GUI=False, near=0.01, far=10.0,
    )

    scene.build()

    # Link the camera follows each frame (palm of the WUJI hand by default).
    track_link = robot.get_link(args.track_link)

    # ----------------------- DOF indices -----------------------
    arm_dof_idx: list[int] = []
    for name in ARM_JOINT_NAMES:
        arm_dof_idx.extend(robot.get_joint(name).dofs_idx_local)
    finger_dof_idx: list[int] = []
    for name in FINGER_JOINT_NAMES:
        finger_dof_idx.extend(robot.get_joint(name).dofs_idx_local)
    if len(arm_dof_idx) != arm_dof.shape[1]:
        raise ValueError(f"Robot has {len(arm_dof_idx)} arm DOFs, npz has {arm_dof.shape[1]}.")
    n_hand_dofs = min(len(finger_dof_idx), hand_dof.shape[1])
    if len(finger_dof_idx) != hand_dof.shape[1]:
        print(f"WARNING: robot has {len(finger_dof_idx)} finger DOFs, "
              f"npz has {hand_dof.shape[1]}. Using first {n_hand_dofs}.")
    all_dof_idx = arm_dof_idx + finger_dof_idx[:n_hand_dofs]

    # ----------------------- segmentation mask lookup -----------------------
    seg_color_map = scene.visualizer._context.seg_color_map
    keep: set[int] = set()
    for ent in (robot, obj, robot_table, work_table):
        keep |= collect_entity_idxc(seg_color_map, ent.idx)
    keep_idxcs = np.array(sorted(keep), dtype=np.int64)
    if keep_idxcs.size == 0:
        raise RuntimeError("Empty segmentation lookup — has the scene been built?")
    print(f"Segmentation: keeping {len(keep_idxcs)} idxc(s) across robot+object+tables.")

    bg = (np.asarray(args.bg_color) * 255.0).clip(0, 255).astype(np.uint8)
    cam_offset = np.asarray(args.cam_offset, dtype=np.float64)
    up = np.asarray(args.up, dtype=np.float64)

    scene_writer = None
    tactile_writer = None
    if not args.no_video:
        # H.264 (yuv420p) so the videos open directly in VS Code / browsers; the
        # scene video is always RGB (no alpha) — transparent PNGs still get composited
        # against bg for the video so videos render correctly in any player.
        scene_writer = H264VideoWriter(str(out_dir / "scene.mp4"), args.fps)
        tactile_writer = H264VideoWriter(str(out_dir / "tactile.mp4"), args.tactile_fps)

    # ----------------------- render loop -----------------------
    for out_i, t in enumerate(frame_idxs):
        q = np.concatenate([arm_dof[t], hand_dof[t, :n_hand_dofs]])
        robot.set_dofs_position(q, all_dof_idx)
        obj.set_pos(object_pos[t])
        obj.set_quat(object_quat[t])
        scene.step()

        # Camera tracks the hand base in world frame (fixed offset, fixed orientation).
        center = np.asarray(track_link.get_pos().detach().cpu()).reshape(-1)[-3:]
        camera.set_pose(pos=(center + cam_offset), lookat=center, up=up)

        rgb, _, seg, _ = camera.render(rgb=True, segmentation=True)
        rgb = np.asarray(rgb)
        seg = np.asarray(seg)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        mask = np.isin(seg, keep_idxcs)

        # Always build a BGR composite for the video (alpha is added on top for the PNG).
        comp = np.where(mask[..., None], rgb, bg[None, None, :]).astype(np.uint8)
        comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)

        if args.transparent:
            alpha = (mask.astype(np.uint8) * 255)
            bgra = np.concatenate([comp_bgr, alpha[..., None]], axis=-1)
            cv2.imwrite(str(out_dir / f"{out_i:05d}_scene.png"), bgra)
        else:
            cv2.imwrite(str(out_dir / f"{out_i:05d}_scene.png"), comp_bgr)

        tactile_frame = render_tactile_pair(
            tactile_map[t], target_tactile_map[t],
            gamma=args.tactile_gamma, min_max=args.tactile_min_max,
        )
        cv2.imwrite(str(out_dir / f"{out_i:05d}_tactile.png"), tactile_frame)

        if scene_writer is not None:
            scene_writer.write(comp_bgr)
        if tactile_writer is not None:
            tactile_writer.write(tactile_frame)

        if out_i % 20 == 0:
            print(f"  frame {out_i + 1}/{len(frame_idxs)}  (t={t})")

    if scene_writer is not None:
        scene_writer.release()
    if tactile_writer is not None:
        tactile_writer.release()
    
    print(f"Saved {len(frame_idxs)} scene+tactile frames to {out_dir}/")
    if not args.no_video:
        print(f"Saved scene video to {out_dir / 'scene.mp4'} ({args.fps} fps)")
        print(f"Saved tactile video to {out_dir / 'tactile.mp4'} ({args.tactile_fps} fps)")


if __name__ == "__main__":
    main()
