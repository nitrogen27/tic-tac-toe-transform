"""Tests for D4 symmetry augmentation."""

import torch
from trainer_lab.data.augmentation import augment_sample


def test_produces_eight_variants():
    planes = torch.randn(6, 16, 16)
    policy = torch.randn(256)
    value = 0.5
    results = augment_sample(planes, policy, value)
    assert len(results) == 8


def test_identity_is_first():
    planes = torch.randn(6, 16, 16)
    policy = torch.randn(256)
    value = -0.3
    results = augment_sample(planes, policy, value)
    aug_planes, aug_policy, aug_value = results[0]
    assert torch.allclose(aug_planes, planes)
    assert torch.allclose(aug_policy, policy)


def test_value_preserved():
    planes = torch.randn(6, 16, 16)
    policy = torch.randn(256)
    value = 0.75
    results = augment_sample(planes, policy, value)
    for _, _, v in results:
        assert v.item() == value


def test_shapes_preserved():
    planes = torch.randn(6, 16, 16)
    policy = torch.randn(256)
    value = 0.0
    results = augment_sample(planes, policy, value)
    for p, pol, v in results:
        assert p.shape == (6, 16, 16), f"planes shape {p.shape}"
        assert pol.shape == (256,), f"policy shape {pol.shape}"


def test_transforms_differ():
    """At least some of the 8 transforms should produce different planes."""
    planes = torch.randn(6, 16, 16)
    policy = torch.randn(256)
    results = augment_sample(planes, policy, 0.0)
    # Compare flattened tensors — transforms rearrange elements, so
    # at least some should not be element-wise equal.
    distinct = 0
    base = results[0][0]
    for p, _, _ in results[1:]:
        if not torch.allclose(base, p):
            distinct += 1
    assert distinct >= 3, "Expected transforms to produce distinct outputs"


def test_rot180_involution():
    """Applying rot180 twice should give back the original."""
    planes = torch.randn(6, 16, 16)
    policy = torch.randn(256)
    results = augment_sample(planes, policy, 0.0)
    # index 2 is rot180
    rot180_planes = results[2][0]
    # apply augment again to the rot180 result
    results2 = augment_sample(rot180_planes, results[2][1], 0.0)
    # index 2 again -> should be back to original
    assert torch.allclose(results2[2][0], planes, atol=1e-6)
