"""Tests for PolicyValueResNet forward pass."""

import torch
from trainer_lab.models.resnet import PolicyValueResNet


def _make_model(**kwargs) -> PolicyValueResNet:
    defaults = dict(
        in_channels=6, res_filters=32, res_blocks=2,
        policy_filters=2, value_fc=32, board_max=16,
    )
    defaults.update(kwargs)
    return PolicyValueResNet(**defaults)


def test_forward_shape():
    model = _make_model()
    x = torch.randn(4, 6, 16, 16)
    policy, value = model(x)
    assert policy.shape == (4, 256), f"policy shape {policy.shape}"
    assert value.shape == (4, 1), f"value shape {value.shape}"


def test_single_sample():
    model = _make_model()
    x = torch.randn(1, 6, 16, 16)
    policy, value = model(x)
    assert policy.shape == (1, 256)
    assert value.shape == (1, 1)


def test_value_range():
    """Value head uses tanh, so output should be in [-1, 1]."""
    model = _make_model()
    x = torch.randn(8, 6, 16, 16)
    _, value = model(x)
    assert value.min().item() >= -1.0
    assert value.max().item() <= 1.0


def test_default_config():
    """Full-size model with default config should forward-pass."""
    model = PolicyValueResNet()
    x = torch.randn(2, 6, 16, 16)
    policy, value = model(x)
    assert policy.shape == (2, 256)
    assert value.shape == (2, 1)


def test_gradients_flow():
    model = _make_model()
    x = torch.randn(2, 6, 16, 16)
    policy, value = model(x)
    loss = policy.sum() + value.sum()
    loss.backward()
    # Check that at least the first conv has gradients
    assert model.input_conv.weight.grad is not None
    assert model.input_conv.weight.grad.abs().sum().item() > 0
