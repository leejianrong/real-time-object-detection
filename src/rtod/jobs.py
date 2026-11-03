"""Cost-safe RunPod job runner: a $5 hard cap, enforced in code (ADR-0004).

The pod-side dead-man's-switch (max lifetime + idle watchdog) and teardown mechanics belong to the
runpod-jobs `rp` script; `RpProvider` drives it. This module adds what `rp` does not know: a
project-wide spend ledger and a refusal to launch a job that could push total spend over the cap.
"""

from __future__ import annotations

import json
import os
import re
import signal
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
    min_ram_gb: int = 16  # rp terminates a pod with less; the GPU benchmark needs far less

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
# Offers worth skipping: out of stock, or a pod that came up with too little RAM (rp terminates it)
NO_STOCK = re.compile(
    r"no (longer any )?instances (currently )?available|pod had \d+ GB RAM <", re.I
)
RAW = "https://raw.githubusercontent.com/leejianrong/real-time-object-detection"


# Named ablation sweeps. Every config shares seed, lr, clip and steps with its neighbours unless it
# overrides them, and changes one thing. Run order matters: configs on the 10k subset come first so
# the extra images of a 40k config are fetched only after cheaper results exist.
SWEEPS: dict[str, dict[str, str]] = {
    "recipe": {
        "float": "--no-fake-quant",
        "float_ema": "--no-fake-quant --ema",
        "qat": "",
        "qat_ema": "--ema",
        "qat_ema_40k": "--ema --subset train40k",
    },
    # Float-only (no fake-quant): where does the ~2 pt loss of a plain fine-tune come from? (The
    # --steps 0 control, which must reproduce FP32, was checked locally: 38.1189 vs 38.12.)
    "float2": {
        "float_steps250": "--no-fake-quant --steps 250",
        "float_lr1e5": "--no-fake-quant --lr 1e-5",
        "float_ema999": "--no-fake-quant --ema --ema-decay 0.999",
        "float_ema9998": "--no-fake-quant --ema --ema-decay 0.9998",
        "float_lr1e5_ema9998": "--no-fake-quant --lr 1e-5 --ema --ema-decay 0.9998",
    },
    # QAT at the learning rates float2 showed keep accuracy (5e-5 lost ~2.4 pt on val_dev, 1e-5 lost
    # 0.3). BN is folded before training, so weights carry the BN scale and need small steps.
    # Pruned float models (no fake-quant: QAT on pruned models is a separate step). EMA is left off:
    # started from the damaged pruned weights, a slow average would hold on to them.
    "prune": {
        "prune30": "--prune-macs 0.3 --no-fake-quant",
        "prune50": "--prune-macs 0.5 --no-fake-quant",
        "prune70": "--prune-macs 0.7 --no-fake-quant",
        "prune50_lr5e5": "--prune-macs 0.5 --no-fake-quant --lr 5e-5",
        "prune50_3750": "--prune-macs 0.5 --no-fake-quant --steps 3750",
    },
    # Pruned models recovered with ordinary training (BN kept, lr 2.5e-3, weight decay, EMA). The
    # `prune` sweep's folded-BN recipe at lr 1e-5 left every pruned model at 0.00 mAP.
    "prune_std": {
        "p30_std": "--prune-macs 0.3 --plain --lr 2.5e-3 --warmup 200 --steps 6000 --ema",
        "p50_std": "--prune-macs 0.5 --plain --lr 2.5e-3 --warmup 200 --steps 6000 --ema",
    },
    "qat3": {
        "qat_lr1e5": "--lr 1e-5",
        "qat_lr5e6": "--lr 5e-6",
        "qat_lr2e6": "--lr 2e-6",
        "qat_lr1e5_40k": "--lr 1e-5 --subset train40k",
    },
}


# prune30 finished in the first attempt (its artifact was received); this is the rest.
SWEEPS["prune_rest"] = {k: v for k, v in SWEEPS["prune"].items() if k != "prune30"}


def job_env(
    steps: int, lr: float, clip: float, configs: dict[str, str] | None = None
) -> dict[str, str]:
    env = {"QAT_STEPS": str(steps), "QAT_LR": str(lr), "QAT_CLIP": str(clip)}
    if configs:  # "name:flags,name:flags": no pipes or semicolons, which the shell would eat
        env["QAT_CONFIGS"] = ",".join(f"{n}:{f}" for n, f in configs.items())
    return env


def gpu_env(only: str | None = None) -> dict[str, str]:
    """Environment for the pod's GPU benchmark; `only` runs one stage ("cuda" or "tensorrt")."""
    return {"RTOD_JOB": "gpu", **({"RTOD_GPU_ONLY": only} if only else {})}


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
    level: str  # info | status | warn | fatal | artifact | done
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
# Phases that must keep printing progress (TRAIN/DATA/PRUNE lines): (warn, fatal) seconds quiet.
# Heartbeats alone only prove the shell is alive: a job can sit idle for 20 minutes with a healthy
# heartbeat (seen when pruning ran at 64 threads). Pruning, data fetch and training all print.
PROGRESS_SILENCE = {"train": (300.0, 900.0), "bench": (300.0, 900.0)}
GPU_IDLE_BEATS = 4  # consecutive heartbeats with the GPU unused during train (about 2 minutes)
STATUS_EVERY = 120.0
GPU_FIELD = re.compile(r"gpu\(util%,memMiB,totMiB,C\)=(\d+),(\d+),")

PHASE_BUDGET = {
    "install": 600,
    "preflight": 120,
    "smoke": 600,
    "train": 2700,
    "bench": 2400,
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
        self.last_progress, self.last_progress_line = start, ""
        self.gpu_idle_beats, self.last_gpu = 0, "na"
        self._last_status = start
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
                self._beat(line)
            if line.startswith(("TRAIN ", "DATA ", "PRUNE ")):
                self.last_progress, self.last_progress_line = now, line[:120]
            if " phase=" in line and line.startswith("RTOD t+"):
                self.phase, self.phase_since = line.split("phase=")[1].split()[0], now
                self.last_progress, self.gpu_idle_beats = now, 0
                out.append(Event("info", line))
            elif line.startswith(("TRAIN ", "DATA ", "PRUNE ")) or "RTOD_ARTIFACT" in line:
                out.append(Event("info", line))
            if "RTOD_DONE" in line:
                out.append(Event("done", line))
            if any(m in line for m in FATAL_MARKERS):
                out.append(Event("fatal", line))
            if "RP_ARTIFACT_CODE=" in line and line.split("=")[-1].strip():
                out.append(Event("artifact", line.split("=")[-1].strip()))
        out += self._time_checks(now)
        return out

    def _beat(self, line: str) -> None:
        m = GPU_FIELD.search(line)
        if not m:
            return
        util, mem = int(m[1]), int(m[2])
        self.last_gpu = f"util {util}% mem {mem}MiB"
        idle = util <= 2 and mem <= 100  # nothing resident: the model is not on the GPU
        self.gpu_idle_beats = self.gpu_idle_beats + 1 if idle and self.phase == "train" else 0

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
        silence = PROGRESS_SILENCE.get(self.phase)
        if silence:
            quiet = now - max(self.last_progress, self.phase_since)
            if quiet > silence[1]:
                out.append(
                    Event(
                        "fatal",
                        f"no progress for {quiet:.0f}s in phase {self.phase} "
                        f"(last: {self.last_progress_line or 'none'}; gpu {self.last_gpu})",
                    )
                )
            elif quiet > silence[0]:
                out += self._warn_once(
                    f"quiet-{self.phase}-{self.phase_since}",
                    f"no progress lines for {quiet:.0f}s in phase {self.phase}; "
                    f"gpu {self.last_gpu}",
                )
        if self.gpu_idle_beats >= GPU_IDLE_BEATS:
            out += self._warn_once(
                f"gpu-idle-{self.phase_since}",
                f"GPU unused for {self.gpu_idle_beats} heartbeats during train "
                "(model not on the GPU?)",
            )
        if now - self._last_status >= STATUS_EVERY:
            self._last_status = now
            out.append(
                Event(
                    "status",
                    f"t+{now - self.start:.0f}s phase={self.phase} "
                    f"for {now - self.phase_since:.0f}s, "
                    f"gpu {self.last_gpu}, last progress {now - self.last_progress:.0f}s ago: "
                    f"{self.last_progress_line or '-'}",
                )
            )
        return out


class Provider(Protocol):
    def up(self, spec: JobSpec) -> None: ...
    def follow(self) -> Iterator[str]: ...
    def fetch(self, code: str, dest: Path) -> None: ...
    def down(self) -> None: ...
    def active(self) -> bool: ...


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def run_job(provider: Provider, ledger: Ledger, spec: JobSpec, dest: Path) -> None:
    """Check the cap, run the job, always tear down and charge, verify no pod is left.

    SIGTERM is turned into KeyboardInterrupt so `kill <driver>` also tears the pod down and
    charges the ledger (a plain SIGTERM skipped the finally block and left the pod running).
    """
    ledger.check(spec)
    start = time.monotonic()
    old = signal.signal(signal.SIGTERM, _interrupt)
    try:
        provider.up(spec)
        for n, code in enumerate(provider.follow(), 1):  # the job may publish several artifacts
            provider.fetch(code, dest / f"artifact{n}")
    finally:
        provider.down()
        ledger.record(spec, min((time.monotonic() - start) / 3600, spec.max_hours))
        signal.signal(signal.SIGTERM, old)
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
            "RP_MIN_RAM_GB": str(spec.min_ram_gb),
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
                    print(f"[job:warn] no stock or too little RAM: {cloud} {gpus}", flush=True)
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
