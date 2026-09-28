from typing import Any

import genesis as gs

from gs_env.common.bases.base_scene import BaseSimScene
from gs_env.sim.scenes.config.schema import FlatSceneArgs


class FlatScene(BaseSimScene):
    def __init__(
        self,
        num_envs: int,
        args: FlatSceneArgs,
        show_viewer: bool = False,
        show_fps: bool = False,
        n_envs_per_row: int | None = None,
        env_spacing: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        #
        self._scene = gs.Scene(
            sim_options=args.sim_options,
            tool_options=args.tool_options,
            rigid_options=args.rigid_options,
            mpm_options=args.mpm_options,
            fem_options=args.fem_options,
            sf_options=args.sf_options,
            vis_options=args.vis_options,
            viewer_options=args.viewer_options,
            show_FPS=show_fps,
            show_viewer=show_viewer,
            renderer=gs.options.renderers.BatchRenderer(use_rasterizer=True),
        )
        # BatchRenderer requires at least one light
        self._scene.add_light(
            pos=(0, 0, 3), dir=(0, 0, -1), color=(1, 1, 1),
            directional=True, intensity=1.0,
        )

        self._scene.add_entity(
            gs.morphs.Box(
                size=(0.6, 1.0, 0.1),
                pos=(-0.44, 0.0, -0.053),
                collision=False,
                fixed=True,
            ),
        )
        #
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._n_envs_per_row = n_envs_per_row
        self._center_envs_at_origin = args.center_envs_at_origin
        self._compile_kernels = args.compile_kernels

    def __getattr__(self, item: str) -> Any:
        if hasattr(self._scene, item):
            return getattr(self._scene, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

