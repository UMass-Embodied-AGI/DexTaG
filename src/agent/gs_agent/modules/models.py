import torch
from torch import nn

from gs_agent.bases.network_backbone import NetworkBackbone
from gs_agent.modules.config.schema import (
    ActivationType,
    MLPConfig,
    NetworkBackboneConfig,
    NetworkBackboneType,
    TemporalEncoderConfig,
)


def get_activation(
    activation: ActivationType,
) -> nn.Module:
    """Get activation function by typed name."""
    match activation:
        case ActivationType.RELU:
            return nn.ReLU()
        case ActivationType.TANH:
            return nn.Tanh()
        case ActivationType.GELU:
            return nn.GELU()
        case ActivationType.SWISH:
            return nn.SiLU()


class MLPBackbone(NetworkBackbone):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        config: MLPConfig,
        device: torch.device,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.device = device
        self._input_dim = input_dim
        self._hidden_dims = config.hidden_dims
        self._activation = config.activation

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in self._hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(get_activation(self._activation))
            prev_dim = hidden_dim

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim, bias=True))

        self._output_dim = output_dim if output_dim is not None else prev_dim

        self._network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self._network(obs)

    @property
    def output_dim(self) -> int:
        """Output dimension of the network."""
        return self._output_dim

    @property
    def input_dim(self) -> int:
        """Input dimension of the network."""
        return self._input_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        """Hidden dimensions of the network."""
        return self._hidden_dims

    @property
    def activation(self) -> ActivationType:
        """Activation function of the network."""
        return self._activation


class TemporalEncoderBackbone(NetworkBackbone):
    """Temporal encoder with separate motion, history, future, tactile map, and point cloud processing.

    Architecture processes observations as:
    - Target motion → Motion encoder → latent (raw motion also passed through)
    - Proprioception → Raw features (passed through)
    - Tactile map [+ target map] (optional, 24×32) → Conv2D encoder → latent
    - History (optional, timesteps × per_step) → History encoder (conv1d) → latent
    - Future (timesteps × per_step) → Future encoder (conv1d) → latent
    - Wrist point cloud (optional) → PointNet encoder → latent
    - Combined features → MLP backbone → output
    """

    def __init__(
        self,
        input_dim: int,
        config: TemporalEncoderConfig,
        device: torch.device,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.device = device
        self._input_dim = input_dim
        self._config = config

        # Validate input dimensions
        self._add_target_tactile_map = config.add_target_tactile_map
        target_tactile_dim = config.tactile_map_dim if self._add_target_tactile_map else 0
        expected_dim = (
            config.motion_dim
            + config.proprio_dim
            + config.tactile_map_dim
            + target_tactile_dim
            + config.history_dim
            + config.future_dim
            + config.pointcloud_dim
        )
        assert input_dim == expected_dim, (
            f"Input dim {input_dim} doesn't match expected {expected_dim}. "
            f"Got: motion({config.motion_dim}) + proprio({config.proprio_dim}) + "
            f"tactile_map({config.tactile_map_dim}) + "
            f"target_tactile_map({target_tactile_dim}) + "
            f"history({config.history_dim}) + future({config.future_dim}) + "
            f"pointcloud({config.pointcloud_dim})"
        )

        # Module 1: Motion Encoder (current motion → latent)
        self.motion_encoder = nn.Sequential(
            nn.Linear(config.motion_dim, 60),
            nn.SiLU(),
            nn.Linear(60, config.motion_latent_dim),
        )

        # Module 1.5: Tactile Map Encoder (Conv2D)
        self._use_tactile_map = config.tactile_map_dim > 0
        if self._use_tactile_map:
            self._tactile_map_h = config.tactile_map_h
            self._tactile_map_w = config.tactile_map_w
            tactile_in_channels = 2 if self._add_target_tactile_map else 1
            self.tactile_map_encoder = nn.Sequential(
                nn.Conv2d(tactile_in_channels, 16, kernel_size=4, stride=2, padding=1),   # (24,32) → (12,16)
                nn.SiLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # (12,16) → (6,8)
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (6,8) → (3,4)
                nn.SiLU(),
                nn.Flatten(),                                            # → 768
                nn.Linear(64 * 3 * 4, config.tactile_map_latent_dim),
                nn.SiLU(),
            )

        # Module 2: History Encoder (8 timesteps → conv1d → latent)
        # Only create if history_dim > 0
        self._use_history = config.history_dim > 0
        if self._use_history:
            self.history_pre_conv = nn.Sequential(
                nn.Linear(config.history_per_step_dim, 60),
                nn.SiLU(),
            )
            self.history_conv = nn.Sequential(
                nn.Conv1d(in_channels=60, out_channels=40, kernel_size=4, stride=2),
                nn.SiLU(),
                nn.Conv1d(in_channels=40, out_channels=20, kernel_size=2, stride=1),
                nn.SiLU(),
            )
            # Calculate conv output size for history_timesteps=8:
            # (8-4)/2+1=3, then (3-2)/1+1=2 → 2*20=40
            self.history_post_conv = nn.Linear(40, config.history_latent_dim)

        # Module 3: Future Motion Encoder (future timesteps → conv1d → latent)
        self.future_pre_conv = nn.Sequential(
            nn.Linear(config.future_per_step_dim, 60),
            nn.SiLU(),
        )
        self.future_conv = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=40, kernel_size=2, stride=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=40, out_channels=20, kernel_size=2, stride=2),
            nn.SiLU(),
        )
        # Calculate conv output size for future_timesteps=16:
        # (16-2)/1+1=15, then (15-2)/2+1=7 → 7*20=140
        self.future_post_conv = nn.Linear(140, config.future_latent_dim)

        # Module 3.6: Point Cloud Encoder (PointNet: shared MLP + max pool)
        self._use_pointcloud = config.pointcloud_dim > 0
        if self._use_pointcloud:
            self._pc_num_points = config.pointcloud_num_points
            self.pc_shared_mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.SiLU(),
                nn.Linear(64, 128),
                nn.SiLU(),
                nn.Linear(128, 256),
                nn.SiLU(),
            )
            self.pc_projection = nn.Sequential(
                nn.Linear(256, config.pointcloud_latent_dim),
                nn.SiLU(),
            )

        # Module 4: MLP trunk (actor or critic)
        concatenated_dim = (
            config.motion_dim
            + config.proprio_dim
            + config.motion_latent_dim
            + (config.tactile_map_latent_dim if self._use_tactile_map else 0)
            + (config.history_latent_dim if self._use_history else 0)
            + config.future_latent_dim
            + (config.pointcloud_latent_dim if self._use_pointcloud else 0)
        )

        mlp_layers = []
        prev_dim = concatenated_dim
        for i, hidden_dim in enumerate(config.mlp_hidden_dims):
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.SiLU())

            # Add LayerNorm after 256-dim layer
            if hidden_dim == 256 and config.use_layer_norm:
                mlp_layers.append(nn.LayerNorm(hidden_dim))

            prev_dim = hidden_dim

        if output_dim is not None:
            mlp_layers.append(nn.Linear(prev_dim, output_dim))

        self._output_dim = output_dim if output_dim is not None else prev_dim
        self.mlp_backbone = nn.Sequential(*mlp_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal encoder.

        Args:
            obs: (batch, input_dim) observation tensor laid out as
                [motion | proprio | tactile | target_tactile | history | future | pointcloud]
                (disabled parts omitted)

        Returns:
            (batch, output_dim) encoded features
        """
        batch_size = obs.shape[0]

        # Split observation into components
        idx = 0
        motion_obs = obs[:, idx : idx + self._config.motion_dim]
        idx += self._config.motion_dim

        proprio_obs = obs[:, idx : idx + self._config.proprio_dim]
        idx += self._config.proprio_dim

        if self._use_tactile_map:
            tactile_map_obs = obs[:, idx : idx + self._config.tactile_map_dim]
            idx += self._config.tactile_map_dim
            if self._add_target_tactile_map:
                target_tactile_map_obs = obs[:, idx : idx + self._config.tactile_map_dim]
                idx += self._config.tactile_map_dim

        if self._use_history:
            history_obs = obs[:, idx : idx + self._config.history_dim]
            idx += self._config.history_dim

        future_obs = obs[:, idx : idx + self._config.future_dim]
        idx += self._config.future_dim

        if self._use_pointcloud:
            pointcloud_obs = obs[:, idx : idx + self._config.pointcloud_dim]
            idx += self._config.pointcloud_dim

        # 1. Encode current motion
        motion_latent = self.motion_encoder(motion_obs)  # (batch, motion_latent_dim)

        # 1.5. Encode tactile map with Conv2D
        if self._use_tactile_map:
            observed_map = tactile_map_obs.reshape(batch_size, 1, self._tactile_map_h, self._tactile_map_w)
            if self._add_target_tactile_map:
                target_map = target_tactile_map_obs.reshape(batch_size, 1, self._tactile_map_h, self._tactile_map_w)
                tactile_map_2d = torch.cat([observed_map, target_map], dim=1)  # (batch, 2, h, w)
            else:
                tactile_map_2d = observed_map  # (batch, 1, h, w)
            tactile_map_latent = self.tactile_map_encoder(tactile_map_2d)  # (batch, tactile_map_latent_dim)

        # 2. Encode history (reshape to timesteps) - only if history is used
        if self._use_history:
            history_reshaped = history_obs.reshape(
                batch_size, self._config.history_timesteps, self._config.history_per_step_dim
            )  # (batch, 8, history_per_step_dim)

            # Process each timestep
            history_processed = self.history_pre_conv(history_reshaped)  # (batch, 8, 60)

            # Conv1d expects (batch, channels, length)
            history_transposed = history_processed.transpose(1, 2)  # (batch, 60, 8)
            history_conv_out = self.history_conv(history_transposed)  # (batch, 20, 2)

            # Flatten and project
            history_flat = history_conv_out.reshape(batch_size, -1)  # (batch, 40)
            history_latent = self.history_post_conv(history_flat)  # (batch, history_latent_dim)

        # 3. Encode future (reshape to timesteps)
        future_reshaped = future_obs.reshape(
            batch_size, self._config.future_timesteps, self._config.future_per_step_dim
        )  # (batch, 16, future_per_step_dim)

        # Process each timestep
        future_processed = self.future_pre_conv(future_reshaped)  # (batch, 16, 60)

        # Conv1d expects (batch, channels, length)
        future_transposed = future_processed.transpose(1, 2)  # (batch, 60, 16)
        future_conv_out = self.future_conv(future_transposed)  # (batch, 20, 7)

        # Flatten and project
        future_flat = future_conv_out.reshape(batch_size, -1)  # (batch, 140)
        future_latent = self.future_post_conv(future_flat)  # (batch, 128)

        # 3.6. Encode point cloud (PointNet: shared MLP + max pool)
        if self._use_pointcloud:
            pc = pointcloud_obs.reshape(batch_size, self._pc_num_points, 3)  # (batch, N, 3)
            pc_features = self.pc_shared_mlp(pc)  # (batch, N, 256)
            pc_global = pc_features.max(dim=1)[0]  # (batch, 256) symmetric max pool
            pc_latent = self.pc_projection(pc_global)  # (batch, pointcloud_latent_dim)

        # 4. Concatenate and process through MLP
        features_to_concat = [
            motion_obs,  # motion_dim
            proprio_obs,  # proprio_dim
            motion_latent,  # motion_latent_dim
        ]
        if self._use_tactile_map:
            features_to_concat.append(tactile_map_latent)  # tactile_map_latent_dim
        if self._use_history:
            features_to_concat.append(history_latent)  # history_latent_dim
        features_to_concat.append(future_latent)  # future_latent_dim
        if self._use_pointcloud:
            features_to_concat.append(pc_latent)  # pointcloud_latent_dim

        combined = torch.cat(features_to_concat, dim=-1)

        output = self.mlp_backbone(combined)
        return output

    @property
    def output_dim(self) -> int:
        """Output dimension of the network."""
        return self._output_dim

    @property
    def input_dim(self) -> int:
        """Input dimension of the network."""
        return self._input_dim


class NetworkFactory:
    """Factory for creating different types of neural networks."""

    @staticmethod
    def create_network(
        network_backbone_args: NetworkBackboneConfig,
        input_dim: int,
        device: torch.device,
        output_dim: int | None = None,
    ) -> NetworkBackbone:
        """Create a network based on the provided configuration.

        Args:
            network_backbone_args: Network configuration
            input_dim: Input dimension
            output_dim: Output dimension
            device: Device to place the network on

        Returns:
            Neural network module
        """
        match network_backbone_args.network_type:
            case NetworkBackboneType.MLP:
                return MLPBackbone(
                    input_dim=input_dim,
                    config=network_backbone_args,
                    device=device,
                    output_dim=output_dim,
                )
            case NetworkBackboneType.TEMPORAL_ENCODER:
                return TemporalEncoderBackbone(
                    input_dim=input_dim,
                    config=network_backbone_args,
                    device=device,
                    output_dim=output_dim,
                )
            case _:
                raise ValueError(f"Unsupported network type: {network_backbone_args.network_type}")
