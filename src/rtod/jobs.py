"""Cost-safe RunPod job runner: a $5 hard cap, enforced in code (ADR-0004).

The pod-side dead-man's-switch (max lifetime + idle watchdog) and teardown mechanics belong to the
runpod-jobs `rp` script; `RpProvider` drives it. This module adds what `rp` does not know: a
project-wide spend ledger and a refusal to launch a job that could push total spend over the cap.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Iterator
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
    # (cloud, comma-separated GPU ids), tried in order. Every GPU must fit under max_hourly_usd:
    # RunPod may hand out any listed one and rp only checks the price after creating the pod.
    offers: tuple[tuple[str, str], ...]
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


# RunPod words "no capacity" at least two ways: "...currently available" and "no longer any ...".
NO_STOCK = re.compile(r"no (longer any )?instances (currently )?available", re.I)
RAW = "https://raw.githubusercontent.com/leejianrong/real-time-object-detection"


# One-factor-at-a-time ablations for the QAT recipe: every config shares seed, lr, clip and
# steps and changes one thing relative to its neighbour. The 40k run is last so its extra
# images are fetched only after the cheaper configs have already produced results.
SWEEP = {
    "float": "--no-fake-quant",
    "float_ema": "--no-fake-quant --ema",
    "qat": "",
    "qat_ema": "--ema",
    "qat_ema_40k": "--ema --subset train40k",
}


def job_env(
    steps: int, lr: float, clip: float, configs: dict[str, str] | None = None
) -> dict[str, str]:
    env = {"QAT_STEPS": str(steps), "QAT_LR": str(lr), "QAT_CLIP": str(clip)}
    if configs:  # "name:flags,name:flags": no pipes or semicolons, which the shell would eat
        env["QAT_CONFIGS"] = ",".join(f"{n}:{f}" for n, f in configs.items())
    return env


def bootstrap_cmd(sha: str, env: dict[str, str], base: str | None = None) -> str:
    """The pod's job command: fetch the pinned script (after rp's watchdog is armed) and run it.

    It must stay ONE simple command. rp runs it as `( ${JOB_CMD} 2>&1 | tee LOG )`, so a bare
    `a && b || c` would bind the pipe to `c` only: the job's output would skip rp's log and rp's
    idle watchdog would kill a healthy pod after 900 s of apparent silence (seen on the first run).
    """
    base = base or f"{RAW}/{sha}"
    assigns = " ".join(f'{k}="{v}"' for k, v in {"RTOD_SHA": sha, **env}.items())
    inner = (
        f"(timeout 120 curl -fsSL {base}/scripts/job_qat.sh -o /tmp/job.sh "
        f"&& {assigns} bash /tmp/job.sh) "
        '|| echo "RTOD_FATAL phase=bootstrap job script fetch/run failed exit=$?"'
    )
    return f"bash -c '{inner}'"


class JobFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class Event:
    level: str  # info | warn | fatal | artifact | done
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
PHASE_BUDGET = {
    "install": 600,
    "preflight": 120,
    "smoke": 600,
    "train": 2700,
    "send": 600,
    "ptq": 1800,
    "eval": 1800,
}


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
            if "RTOD_DONE" in line:
                out.append(Event("done", line))
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
    def follow(self) -> Iterator[str]: ...
    def fetch(self, code: str, dest: Path) -> None: ...
    def down(self) -> None: ...
    def active(self) -> bool: ...


def run_job(provider: Provider, ledger: Ledger, spec: JobSpec, dest: Path) -> None:
    """Check the cap, run the job, always tear down and charge, verify no pod is left."""
    ledger.check(spec)
    start = time.monotonic()
    try:
        provider.up(spec)
        for n, code in enumerate(provider.follow(), 1):  # the job may publish several artifacts
            provider.fetch(code, dest / f"artifact{n}")
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
    def env(spec: JobSpec, offer: int = 0) -> dict[str, str]:
        cloud, gpus = spec.offers[offer]
        return {
            **os.environ,
            "RP_POD_NAME": f"rtod-{spec.name}",
            "RP_POD_IMAGE": spec.image,
            "RP_JOB_CMD": spec.cmd,
            "RP_COMPUTE_TYPE": "GPU",
            "RP_CLOUD_TYPE": cloud,
            "RP_INTERRUPTIBLE": "false",  # spot pods are no longer offered (runpod-jobs skill)
            "RP_CONTAINER_DISK_GB": "40",
            "RP_MIN_RAM_GB": "16",
            "RP_GPU_TYPE": gpus,
            "RP_MAX_HOURLY_USD": str(spec.max_hourly_usd),
            "RP_MAX_LIFETIME_SECS": str(int(spec.max_hours * 3600)),
        }

    def _rp(self, *args: str, spec: JobSpec | None = None, offer: int = 0) -> str:
        env = self.env(spec, offer) if spec else None
        done = subprocess.run([self.rp, *args], env=env, capture_output=True, text=True)
        if done.returncode:  # rp explains itself on stderr (stock, price cap, rejected field)
            raise JobFailed(
                f"rp {args[0]} failed ({done.returncode}): {done.stderr.strip()[-600:]}"
            )
        return done.stdout.strip()

    def up(self, spec: JobSpec, wait_stock_secs: float = 600, retry_every: float = 60) -> None:
        """Create the pod, trying each offer, and waiting out 'no instances available'."""
        self._spec, self._offer = spec, 0
        deadline = time.monotonic() + wait_stock_secs
        while True:
            for i, (cloud, gpus) in enumerate(spec.offers):
                self._offer = i
                try:
                    self.pod_id = self._rp("up", spec=spec, offer=i).splitlines()[-1]
                except JobFailed as e:
                    if not NO_STOCK.search(str(e)):
                        raise
                    print(f"[job:warn] no stock: {cloud} {gpus}", flush=True)
                    continue
                print(f"[job] pod {self.pod_id} created ({cloud}: {gpus})", flush=True)
                return
            if time.monotonic() > deadline:
                raise JobFailed("no GPU offer had stock within the wait window")
            time.sleep(retry_every)

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

    def _pod_exists(self) -> bool:
        out = subprocess.run(
            ["runpodctl", "pod", "list"], capture_output=True, text=True, timeout=60
        )
        return self.pod_id in out.stdout

    def follow(self, poll: float = 15, snapshot: Path | None = None) -> Iterator[str]:
        """Poll pod logs, show progress, persist the raw log, and yield each artifact code.

        The caller fetches a code before asking for the next, while the job waits for the transfer.
        Returns when the job reports RTOD_DONE; raises JobFailed on any fatal condition.
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
                    yield ev.msg
                if ev.level == "done":
                    return
            if not self._pod_exists():
                print("\n".join(lines[-40:]), flush=True)
                raise JobFailed(
                    "pod no longer exists (self-terminated or reclaimed) before RTOD_DONE"
                )
            time.sleep(poll)
        raise JobFailed("lifetime reached before the job reported RTOD_DONE")

    def fetch(self, code: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["runpodctl", "receive", code], cwd=dest, check=True, timeout=900)

    def down(self) -> None:
        subprocess.run([self.rp, "down"], check=False, env=self.env(self._spec, self._offer))

    def active(self) -> bool:
        out = subprocess.run(["runpodctl", "pod", "list"], capture_output=True, text=True)
        return out.stdout.strip() not in ("", "[]")
