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


class JobFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class Event:
    level: str  # info | warn | fatal | artifact
    msg: str


FATAL_MARKERS = (
    "RTOD_FAIL",
    "RTOD_FATAL",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "non-finite loss",
    "No space left on device",
)
# Per-phase budgets in seconds: an overrun is a warning (the lifetime cap is the hard stop).
PHASE_BUDGET = {"install": 600, "preflight": 120, "smoke": 600, "train": 2700, "eval": 1800}


class LogMonitor:
    """Decides, from pod log lines alone, whether a job is healthy. No I/O, so it is unit-tested.

    Feed it every poll (an empty list is a pure time check). It reports phase changes and training
    progress, warns when a phase overruns its budget or output is slow to start, and goes fatal on
    an error marker, a lost heartbeat, or no output at all.
    """

    def __init__(
        self, start: float, boot_warn: float = 180, boot_fatal: float = 420, hb_stall: float = 180
    ) -> None:
        self.start, self.boot_warn, self.boot_fatal, self.hb_stall = (
            start,
            boot_warn,
            boot_fatal,
            hb_stall,
        )
        self.first_output: float | None = None
        self.last_hb: float | None = None
        self.phase, self.phase_since = "none", start
        self._seen: set[str] = set()
        self._warned: set[str] = set()

    def feed(self, lines: list[str], now: float) -> list[Event]:
        out: list[Event] = []
        for line in lines:
            if line in self._seen:
                continue
            self._seen.add(line)
            if self.first_output is None:
                self.first_output = now
            if "RTOD_HB" in line:
                self.last_hb = now
            if " phase=" in line and line.startswith("RTOD t+"):
                self.phase, self.phase_since = line.split("phase=")[1].split()[0], now
                out.append(Event("info", line))
            elif line.startswith(("TRAIN ", "DATA ")) or "RTOD_ARTIFACT" in line:
                out.append(Event("info", line))
            if any(m in line for m in FATAL_MARKERS):
                out.append(Event("fatal", line))
            if "RP_ARTIFACT_CODE=" in line and line.split("=")[-1].strip():
                out.append(Event("artifact", line.split("=")[-1].strip()))
        out += self._time_checks(now)
        return out

    def _warn_once(self, key: str, msg: str) -> list[Event]:
        if key in self._warned:
            return []
        self._warned.add(key)
        return [Event("warn", msg)]

    def _time_checks(self, now: float) -> list[Event]:
        if self.first_output is None:
            waited = now - self.start
            if waited > self.boot_fatal:
                return [
                    Event(
                        "fatal",
                        f"no output from the pod after {waited:.0f}s (image pull or start failed?)",
                    )
                ]
            if waited > self.boot_warn:
                return self._warn_once(
                    "boot", f"no output yet after {waited:.0f}s; check system logs"
                )
            return []
        out: list[Event] = []
        if self.last_hb is not None and now - self.last_hb > self.hb_stall:
            out.append(
                Event(
                    "fatal", f"heartbeat lost for {now - self.last_hb:.0f}s in phase {self.phase}"
                )
            )
        budget = PHASE_BUDGET.get(self.phase)
        if budget and now - self.phase_since > budget:
            out += self._warn_once(
                f"budget-{self.phase}", f"phase {self.phase} over its {budget}s budget"
            )
        return out


class Provider(Protocol):
    def up(self, spec: JobSpec) -> None: ...
    def follow(self) -> str: ...
    def fetch(self, code: str, dest: Path) -> None: ...
    def down(self) -> None: ...
    def active(self) -> bool: ...


def run_job(provider: Provider, ledger: Ledger, spec: JobSpec, dest: Path) -> None:
    """Check the cap, run the job, always tear down and charge, verify no pod is left."""
    ledger.check(spec)
    start = time.monotonic()
    try:
        provider.up(spec)
        provider.fetch(provider.follow(), dest)
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
            "RP_CLOUD_TYPE": "COMMUNITY",
            "RP_INTERRUPTIBLE": "false",  # spot pods are no longer offered (runpod-jobs skill)
            "RP_CONTAINER_DISK_GB": "40",
            "RP_MIN_RAM_GB": "16",
            "RP_GPU_TYPE": spec.gpu_types,
            "RP_MAX_HOURLY_USD": str(spec.max_hourly_usd),
            "RP_MAX_LIFETIME_SECS": str(int(spec.max_hours * 3600)),
        }

    def _rp(self, *args: str, spec: JobSpec | None = None) -> str:
        env = self.env(spec) if spec else None
        done = subprocess.run([self.rp, *args], check=True, env=env, capture_output=True, text=True)
        return done.stdout.strip()

    def up(self, spec: JobSpec) -> None:
        self._spec = spec
        self.pod_id = self._rp("up", spec=spec).splitlines()[-1]
        print(f"[job] pod {self.pod_id} created", flush=True)

    def _logs(self, source: str) -> list[str]:
        done = subprocess.run(
            [
                "runpodctl",
                "pod",
                "logs",
                self.pod_id,
                "--source",
                source,
                "--tail",
                "3000",
                "--max-wait",
                "5s",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        lines = []
        for raw in done.stdout.splitlines():
            try:
                rec = json.loads(raw)
            except ValueError:
                lines.append(raw)
                continue
            if "error" in rec:
                if rec.get("code") == "not_found":
                    raise JobFailed("pod is gone before the job finished")
                continue
            lines.append(str(rec.get("line", "")))
        return lines

    def follow(self, poll: float = 15, snapshot: Path | None = None) -> str:
        """Poll pod logs, show progress, persist the raw log, and return the artifact code.

        Raises JobFailed the moment the monitor sees a fatal condition.
        """
        snapshot = snapshot or Path(".cache/jobs") / f"{self._spec.name}.pod.log"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        mon, shown = LogMonitor(time.monotonic()), set[str]()
        limit = time.monotonic() + self._spec.max_hours * 3600
        while time.monotonic() < limit:
            try:
                lines = self._logs("container")
            except subprocess.TimeoutExpired:
                lines = []
            new = [x for x in lines if x not in shown]
            shown.update(new)
            if new:
                with snapshot.open("a") as f:
                    f.write("\n".join(new) + "\n")
            for ev in mon.feed(lines, time.monotonic()):
                print(f"[job:{ev.level}] {ev.msg}", flush=True)
                if ev.level == "warn" and "system logs" in ev.msg:
                    print("\n".join(self._logs("system")[-15:]), flush=True)
                if ev.level == "fatal":
                    print("\n".join(lines[-40:]), flush=True)
                    raise JobFailed(ev.msg)
                if ev.level == "artifact":
                    return ev.msg
            time.sleep(poll)
        raise JobFailed("lifetime reached without an artifact")

    def fetch(self, code: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["runpodctl", "receive", code], cwd=dest, check=True, timeout=900)

    def down(self) -> None:
        subprocess.run([self.rp, "down"], check=False, env=self.env(self._spec))

    def active(self) -> bool:
        out = subprocess.run(["runpodctl", "pod", "list"], capture_output=True, text=True)
        return out.stdout.strip() not in ("", "[]")
