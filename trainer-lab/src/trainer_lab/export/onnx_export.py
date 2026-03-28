"""Export a trained PolicyValueResNet to ONNX format."""

from __future__ import annotations

from pathlib import Path

import torch

from trainer_lab.models.resnet import PolicyValueResNet


def export_to_onnx(
    model: PolicyValueResNet,
    output_path: str | Path,
    opset_version: int = 17,
    board_max: int = 16,
    in_channels: int = 6,
) -> Path:
    """Export *model* to ONNX with a fixed ``[1, 6, 16, 16]`` input shape.

    Parameters
    ----------
    model        : trained PolicyValueResNet (will be set to eval mode)
    output_path  : destination ``.onnx`` file
    opset_version: ONNX opset (default 17)
    board_max    : spatial dimension (16)
    in_channels  : number of input planes (6)

    Returns
    -------
    Path to the written ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.zeros(1, in_channels, board_max, board_max)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["board_planes"],
        output_names=["policy_logits", "value"],
        dynamic_axes=None,  # fixed shape
    )

    return output_path
