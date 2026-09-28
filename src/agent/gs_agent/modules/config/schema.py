from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel


class ActivationType(GenesisEnum):
    RELU = "RELU"
    TANH = "TANH"
    GELU = "GELU"
    SWISH = "SWISH"


class NetworkBackboneType(GenesisEnum):
    MLP = "MLP"
    TEMPORAL_ENCODER = "TEMPORAL_ENCODER"


class MLPConfig(BaseModel):
    """Configuration for Multi-Layer Perceptron networks."""

    model_config = genesis_pydantic_config(frozen=True)

    hidden_dims: tuple[int, ...] = (256, 256, 128)
    activation: ActivationType = ActivationType.RELU

    network_type: NetworkBackboneType = NetworkBackboneType.MLP


class TemporalEncoderConfig(BaseModel):
    """Configuration for Temporal Encoder with motion/history/future processing."""

    model_config = genesis_pydantic_config(frozen=True)

    network_type: NetworkBackboneType = NetworkBackboneType.TEMPORAL_ENCODER

    # Input dimension splits (must sum to total input_dim)
    motion_dim: int = 66
    proprio_dim: int = 71
    history_dim: int = 776  # 8 * 97
    future_dim: int = 1056  # 16 * 66

    # Tactile map encoder settings (Conv2D)
    tactile_map_dim: int = 0       # h * w flattened, 0 = no tactile map
    tactile_map_h: int = 24
    tactile_map_w: int = 32
    tactile_map_latent_dim: int = 128
    add_target_tactile_map: bool = False  # If True, read an extra h*w for target tactile and use 2-channel Conv2D

    # Latent dimensions
    motion_latent_dim: int = 128
    history_latent_dim: int = 128
    future_latent_dim: int = 128

    # History encoder settings
    history_timesteps: int = 8
    history_per_step_dim: int = 97

    # Future encoder settings
    future_timesteps: int = 16
    future_per_step_dim: int = 66

    # Point cloud encoder settings (PointNet)
    pointcloud_dim: int = 0           # num_points * 3 (flattened), 0 = disabled
    pointcloud_num_points: int = 2048
    pointcloud_latent_dim: int = 128

    # Final MLP settings
    mlp_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: ActivationType = ActivationType.SWISH
    dropout_rate: float = 0.1
    use_layer_norm: bool = True


NetworkBackboneConfig = MLPConfig | TemporalEncoderConfig
