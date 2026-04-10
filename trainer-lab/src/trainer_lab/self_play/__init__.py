"""Self-play modules for AlphaZero-style generation and replay."""

from trainer_lab.self_play.mixed_replay import MixedReplay
from trainer_lab.self_play.replay_buffer import ReplayBuffer

__all__ = ["MixedReplay", "ReplayBuffer"]
