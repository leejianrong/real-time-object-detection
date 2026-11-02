"""Training metrics: a JSONL file every run, optional TensorBoard, and matplotlib plots.

Weights & Biases is deliberately not wired in (it needs an account and a key on the pod); add a
second sink to `MetricsWriter.log` if it is wanted later.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_KV = re.compile(r"(\w+)=(-?[\d.]+(?:e[-+]?\d+)?)(?:/(\d+))?")


def parse_train_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Recover metric rows from `TRAIN step=.. loss=..` console lines (e.g. a saved pod log).

    Smoke-run lines (`step=1/3`) are dropped by keeping only the longest run in the log.
    """
    runs: list[list[dict[str, Any]]] = []
    for line in lines:
        if not line.startswith("TRAIN "):
            continue
        row: dict[str, Any] = {}
        total = 0
        for key, val, denom in _KV.findall(line):
            row[key] = float(val)
            if key == "step":
                total = int(denom)
        row["step"] = int(row["step"])
        if not runs or row["step"] < runs[-1][-1]["step"] or total != runs[-1][0]["_total"]:
            runs.append([])
        row["_total"] = total
        runs[-1].append(row)
    best = max(runs, key=lambda r: r[0]["_total"], default=[])
    return [{k: v for k, v in r.items() if k != "_total"} for r in best]


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows from a metrics.jsonl, or from a raw pod/console log containing TRAIN lines."""
    text = path.read_text().splitlines()
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in text if x.strip()]
    return parse_train_lines(text)


class MetricsWriter:
    """Append one JSON row per step to <dir>/metrics.jsonl; mirror to TensorBoard if installed."""

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._f = (out_dir / "metrics.jsonl").open("a")
        self._tb: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(out_dir / "tb"))
        except ImportError:
            pass

    def log(self, step: int, **values: float) -> None:
        self._f.write(json.dumps({"step": step, **values}) + "\n")
        self._f.flush()
        if self._tb is not None:
            for k, v in values.items():
                self._tb.add_scalar(f"train/{k}", v, step)

    def close(self) -> None:
        self._f.close()
        if self._tb is not None:
            self._tb.close()


def plot_training(rows: list[dict[str, Any]], out: Path, smooth: int = 10) -> Path:
    """One figure: total loss (raw + moving average), loss components, grad norm, learning rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.array([r["step"] for r in rows])

    def col(name: str) -> Any:
        return np.array([r.get(name, np.nan) for r in rows], dtype=float)

    def avg(y: Any) -> Any:
        k = max(1, min(smooth, len(y)))
        return np.convolve(y, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    loss = col("loss")
    ax[0, 0].plot(steps, loss, alpha=0.35, label="raw")
    ax[0, 0].plot(steps[len(steps) - len(avg(loss)) :], avg(loss), label=f"{smooth}-pt mean")
    ax[0, 0].set(title="total loss")
    ax[0, 0].legend()
    for name in ("iou", "conf", "cls"):
        if not np.isnan(col(name)).all():
            y = col(name)
            ax[0, 1].plot(steps[len(steps) - len(avg(y)) :], avg(y), label=name)
    ax[0, 1].set(title="loss components")
    ax[0, 1].legend()
    ax[1, 0].plot(steps, col("gnorm"))
    ax[1, 0].set(title="grad norm (before clipping)", xlabel="step", yscale="log")
    ax[1, 1].plot(steps, col("lr"))
    ax[1, 1].set(title="learning rate", xlabel="step")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
