"""Structured channel pruning of YOLOX-Tiny with torch-pruning (needs the `prune` extra)."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch_pruning as tp

from rtod.models import INPUT_SIZE

# channel counts rounded to a multiple of this: unaligned widths gave no CPU speedup at all
ROUND_TO = 16
PRUNE_THREADS = 4


def macs_params(model: torch.nn.Module, size: int = INPUT_SIZE) -> tuple[float, float]:
    macs, params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, size, size))
    return float(macs), float(params)


def _prune_channels(model: torch.nn.Module, ratio: float) -> torch.nn.Module:
    m = copy.deepcopy(model).eval()
    head: Any = m.head
    pruner = tp.pruner.BasePruner(
        m,
        torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE),
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio,
        # the output convs fix the 80 + 4 + 1 channels of the (N, 3549, 85) contract
        ignored_layers=[head.cls_preds, head.reg_preds, head.obj_preds],
        iterative_steps=1,
        round_to=ROUND_TO,
    )
    pruner.step()
    return m


def prune_to_macs(model: torch.nn.Module, macs_removed: float) -> torch.nn.Module:
    """Prune whole channels so about `macs_removed` of the multiply-adds are gone.

    Rounding makes the achievable MACs coarse, so this returns the closest one it finds.

    Removing a fraction r of every layer's channels removes about 1-(1-r)^2 of the MACs, so
    bisect on r until the measured MACs reach the target.
    """
    if not 0 < macs_removed < 1:
        raise ValueError(f"macs_removed must be in (0, 1), got {macs_removed}")
    # Tiny ops on many threads are slower, not faster: 7 s at 4 threads, 24 s at 16, and about
    # 35 minutes on a 64-thread pod. Cap the threads while pruning, then restore.
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, PRUNE_THREADS))
    try:
        return _bisect(model, macs_removed)
    finally:
        torch.set_num_threads(previous)


def _bisect(model: torch.nn.Module, macs_removed: float) -> torch.nn.Module:
    base, _ = macs_params(model)
    lo, hi, best, gap = 0.0, 0.9, model, 1.0
    for _ in range(10):
        mid = (lo + hi) / 2
        cand = _prune_channels(model, mid)
        removed = 1 - macs_params(cand)[0] / base
        if abs(removed - macs_removed) < gap:
            best, gap = cand, abs(removed - macs_removed)
        lo, hi = (mid, hi) if removed < macs_removed else (lo, mid)
    return best
