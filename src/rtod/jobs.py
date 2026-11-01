"""Cost-safe RunPod job runner: a $5 hard cap, enforced in code (ADR-0004).

The pod-side dead-man's-switch (max lifetime + idle watchdog) and teardown mechanics belong to the
runpod-jobs `rp` script; `RpProvider` drives it. This module adds what `rp` does not know: a
project-wide spend ledger and a refusal to launch a job that could push total spend over the cap.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

CAP_USD = 5.00
LEDGER = Path("results/spend.json")


class CapExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class JobSpec:
    name: str
    image: str
    cmd: str  # what the pod runs; it must exit when done so the pod can terminate
    gpu_types: str  # comma-separated RunPod GPU ids, any acceptable
    max_hourly_usd: float
    max_hours: float  # hard lifetime ceiling baked into the pod

    @property
    def worst_case_usd(self) -> float:
        return self.max_hourly_usd * self.max_hours


class Ledger:
    def __init__(self, path: Path = LEDGER, cap: float = CAP_USD) -> None:
        self.path, self.cap = path, cap

    def entries(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def spent(self) -> float:
        return sum(float(e["usd"]) for e in self.entries())

    def check(self, spec: JobSpec) -> None:
        if self.spent() + spec.worst_case_usd > self.cap:
            raise CapExceeded(
                f"{spec.name}: spent ${self.spent():.2f} + worst case ${spec.worst_case_usd:.2f} "
                f"exceeds the ${self.cap:.2f} cap"
            )

    def record(self, spec: JobSpec, hours: float) -> None:
        """Charge at the hourly ceiling, so the ledger over-estimates rather than under."""
        usd = round(spec.max_hourly_usd * hours, 4)
        entry = {"job": spec.name, "hours": round(hours, 4), "usd": usd, **asdict(spec)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([*self.entries(), entry], indent=2) + "\n")


class Provider(Protocol):
    def up(self, spec: JobSpec) -> None: ...
    def follow(self) -> None: ...
    def fetch(self, dest: Path) -> None: ...
    def down(self) -> None: ...
    def active(self) -> bool: ...


def run_job(provider: Provider, ledger: Ledger, spec: JobSpec, dest: Path) -> None:
    """Check the cap, run the job, always tear down and charge, verify no pod is left."""
    ledger.check(spec)
    start = time.monotonic()
    try:
        provider.up(spec)
        provider.follow()
        provider.fetch(dest)
    finally:
        provider.down()
        ledger.record(spec, min((time.monotonic() - start) / 3600, spec.max_hours))
    if provider.active():
        raise RuntimeError(f"{spec.name}: a pod is still running after teardown; stop it by hand")


class RpProvider:
    """Drives the runpod-jobs `rp` script. RUNPOD_API_KEY is inherited from the environment only."""

    def __init__(self, rp: str = "rp") -> None:
        self.rp = rp

    @staticmethod
    def env(spec: JobSpec) -> dict[str, str]:
        return {
            **os.environ,
            "RP_POD_NAME": f"rtod-{spec.name}",
            "RP_POD_IMAGE": spec.image,
            "RP_JOB_CMD": spec.cmd,
            "RP_COMPUTE_TYPE": "GPU",
            "RP_GPU_TYPE": spec.gpu_types,
            "RP_MAX_HOURLY_USD": str(spec.max_hourly_usd),
            "RP_MAX_LIFETIME_SECS": str(int(spec.max_hours * 3600)),
        }

    def _rp(self, *args: str, spec: JobSpec | None = None) -> None:
        env = self.env(spec) if spec else None
        subprocess.run([self.rp, *args], check=True, env=env)

    def up(self, spec: JobSpec) -> None:
        self._spec = spec
        self._rp("up", spec=spec)

    def follow(self) -> None:
        self._rp("logs", "-f", spec=self._spec)

    def fetch(self, dest: Path) -> None:
        self._rp("relay-get", str(dest), spec=self._spec)

    def down(self) -> None:
        subprocess.run([self.rp, "down"], check=False, env=self.env(self._spec))

    def active(self) -> bool:
        out = subprocess.run(["runpodctl", "pod", "list"], capture_output=True, text=True)
        return out.stdout.strip() not in ("", "[]")
