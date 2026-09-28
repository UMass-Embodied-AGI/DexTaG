import abc

import genesis as gs


class BaseSimScene(abc.ABC):
    """
    Base class for simulated scenes; wraps gs.Scene and handles the batched build().
    """

    _scene: gs.Scene
    _num_envs: int
    _env_spacing: tuple[float, float]
    _n_envs_per_row: int | None
    _center_envs_at_origin: bool
    _compile_kernels: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build(self) -> None:
        self._scene.build(
            n_envs=self._num_envs,
            env_spacing=self._env_spacing,
            n_envs_per_row=self._n_envs_per_row,
            center_envs_at_origin=self._center_envs_at_origin,
            compile_kernels=self._compile_kernels,
        )

    @property
    def scene(self) -> gs.Scene:
        return self._scene

    @property
    def num_envs(self) -> int:
        return self._num_envs
