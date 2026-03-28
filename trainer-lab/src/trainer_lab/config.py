"""Pydantic configuration models for trainer-lab."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """Architecture hyper-parameters for the ResNet policy-value network."""

    in_channels: int = Field(default=6, description="Number of input feature planes")
    board_max: int = Field(default=16, description="Maximum board dimension (padded)")
    res_blocks: int = Field(default=8, description="Number of residual blocks in the tower")
    res_filters: int = Field(default=128, description="Filter count per residual block")
    policy_filters: int = Field(default=2, description="Filters in the policy head conv layer")
    value_fc: int = Field(default=128, description="Hidden units in the value head FC layer")

    model_config = {"env_prefix": "MODEL_"}


class TrainConfig(BaseSettings):
    """Training loop hyper-parameters."""

    batch_size: int = Field(default=256, description="Mini-batch size")
    epochs: int = Field(default=30, description="Total training epochs")
    lr: float = Field(default=1e-3, description="Initial learning rate")
    weight_value: float = Field(default=0.5, description="Weight for the value loss term")
    mixed_precision: bool = Field(default=True, description="Enable AMP mixed-precision training")
    checkpoint_dir: Path = Field(default=Path("checkpoints"), description="Directory for model checkpoints")
    data_dir: Path = Field(default=Path("data"), description="Directory containing training data")

    model_config = {"env_prefix": "TRAIN_"}


class SelfPlayConfig(BaseSettings):
    """Self-play generation parameters."""

    games: int = Field(default=200, description="Number of self-play games per generation")
    simulations: int = Field(default=400, description="MCTS simulations per move")
    replay_buffer_max: int = Field(default=20_000, description="Maximum positions in the replay buffer")

    model_config = {"env_prefix": "SELFPLAY_"}
