"""Quantization-aware training for YOLOX-Tiny (ADR-0003).

BatchNorm is folded into each conv first and frozen; every conv then fake-quantizes its input, its
per-channel int8 weights and its output with the same observers ONNX Runtime's QDQ uses (uint8
affine activations, int8 symmetric weights). Hand-rolled instead of torch.ao FX because YOLOX's head
branches on `training` and cannot be symbolically traced.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

from rtod._yolox.network_blocks import BaseConv


def _act_fq() -> FakeQuantize:
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )


def _weight_fq() -> FakeQuantize:
    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )


class QConv(nn.Module):
    """Conv with fake-quantized input, weights and output. BN, if any, is already folded in."""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d | None = None) -> None:
        super().__init__()
        w, b = conv.weight.detach().clone(), conv.bias
        b = torch.zeros(w.shape[0]) if b is None else b.detach().clone()
        if bn is not None:
            scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
            w = w * scale.view(-1, 1, 1, 1)
            b = bn.bias.detach() + (b - bn.running_mean) * scale
        self.weight, self.bias = nn.Parameter(w), nn.Parameter(b)
        self.cfg = dict(
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups
        )
        self.in_fq, self.w_fq, self.out_fq = _act_fq(), _weight_fq(), _act_fq()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.conv2d(self.in_fq(x), self.w_fq(self.weight), self.bias, **self.cfg)
        return self.out_fq(y)


def prepare_qat(model: nn.Module) -> nn.Module:
    """Replace every conv (folding BN where present) with a QConv, in place."""
    for m in list(model.modules()):
        for name, child in list(m.named_children()):
            if isinstance(m, BaseConv) and name == "conv":
                setattr(m, "conv", QConv(child, m.bn))  # noqa: B010
                m.bn = nn.Identity()
            elif isinstance(child, nn.Conv2d) and not isinstance(m, QConv):
                setattr(m, name, QConv(child))
    # head conv lists hold their convs as ModuleList items, not named children
    for lst in (m for m in model.modules() if isinstance(m, nn.ModuleList)):
        for i, child in enumerate(lst):
            if isinstance(child, nn.Conv2d):
                lst[i] = QConv(child)
    return model


def set_fake_quant(model: nn.Module, on: bool) -> None:
    """Switch every fake-quantizer on (QAT / QDQ export) or off (folded float model)."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.enable_fake_quant() if on else m.disable_fake_quant()


def calibrate(model: nn.Module, batches: Iterable[torch.Tensor]) -> None:
    """Let the observers see real inputs, then freeze their ranges."""
    model.eval()
    with torch.no_grad():
        for b in batches:
            model(b)
    model.apply(torch.ao.quantization.disable_observer)
