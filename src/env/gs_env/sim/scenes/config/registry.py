import genesis as gs

from gs_env.sim.scenes.config.schema import (
    FlatSceneArgs,
    SceneArgs,
)

# NOTE: break dependencies on default values from Genesis to avoid silent bugs
# due to any changes in the Genesis repo.


# ------------------------------------------------------------
# Sim
# ------------------------------------------------------------

SimArgsRegistry: dict[str, gs.options.SimOptions] = {}

SimArgsRegistry["dexhand"] = gs.options.SimOptions(
    dt=1/60.0,  # match the reference data framerate
    substeps=5,
    substeps_local=None,
    gravity=(0.0, 0.0, -9.81),
    floor_height=0.0,
    requires_grad=False,
)


# ------------------------------------------------------------
# Tool
# ------------------------------------------------------------

ToolArgsRegistry: dict[str, gs.options.ToolOptions] = {}

ToolArgsRegistry["default"] = gs.options.ToolOptions(
    dt=None,
    floor_height=0.0,
)


# ------------------------------------------------------------
# MPM
# ------------------------------------------------------------

MPMArgsRegistry: dict[str, gs.options.MPMOptions] = {}

MPMArgsRegistry["default"] = gs.options.MPMOptions(
    dt=None,
    gravity=None,
    particle_size=None,
    grid_density=64,
    enable_CPIC=False,
    lower_bound=(-1.0, -1.0, 0.0),
    upper_bound=(1.0, 1.0, 1.0),
    use_sparse_grid=False,
    leaf_block_size=8,
)


# ------------------------------------------------------------
# FEM
# ------------------------------------------------------------

FEMArgsRegistry: dict[str, gs.options.FEMOptions] = {}

FEMArgsRegistry["default"] = gs.options.FEMOptions(
    dt=None,
    gravity=None,
    damping=0.0,
    floor_height=0.0,
)


# ------------------------------------------------------------
# SF
# ------------------------------------------------------------

SFArgsRegistry: dict[str, gs.options.SFOptions] = {}

SFArgsRegistry["default"] = gs.options.SFOptions(
    dt=None,
    res=128,
    solver_iters=500,
    decay=0.99,
    T_low=1.0,
    T_high=0.0,
    inlet_pos=(0, 0, 0),
    inlet_vel=(0, 0, 1),
    inlet_quat=(1, 0, 0, 0),
    inlet_s=400.0,
)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

VisArgsRegistry: dict[str, gs.options.VisOptions] = {}

VisArgsRegistry["default"] = gs.options.VisOptions(
    show_world_frame=True,
    world_frame_size=1.0,
    show_link_frame=False,
    link_frame_size=0.2,
    show_cameras=False,
    shadow=True,
    plane_reflection=False,
    env_separate_rigid=False,
    background_color=(0.04, 0.08, 0.12),
    ambient_light=(0.1, 0.1, 0.1),
    visualize_mpm_boundary=False,
    visualize_sph_boundary=False,
    visualize_pbd_boundary=False,
    segmentation_level="link",
    render_particle_as="sphere",
    particle_size_scale=1.0,
    contact_force_scale=0.01,
    n_support_neighbors=12,
    n_rendered_envs=None,
    lights=[
        {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
    ],
)


# ------------------------------------------------------------
# Scene
# ------------------------------------------------------------


SceneArgsRegistry: dict[str, SceneArgs] = {}


SceneArgsRegistry["flat_scene_default"] = FlatSceneArgs(
    show_viewer=False,
    show_FPS=False,
    center_envs_at_origin=True,
    compile_kernels=True,
    sim_options=SimArgsRegistry["dexhand"],
    tool_options=ToolArgsRegistry["default"],
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=True,
        enable_collision=True,
        gravity=(0, 0, -9.8),
        box_box_detection=True,
        noslip_iterations=0,
        batch_dofs_info=True,  # Enable per-env kp/kv/damping/armature
    ),
    mpm_options=MPMArgsRegistry["default"],
    fem_options=FEMArgsRegistry["default"],
    sf_options=SFArgsRegistry["default"],
    vis_options=VisArgsRegistry["default"],
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(-0.6, 0.0, 0.7),
        camera_lookat=(0.2, 0.0, 0.1),
        camera_fov=50,
        max_FPS=60,
    ),
    normal=(0.0, 0.0, 1.0),
)
