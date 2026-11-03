from pathlib import Path

import pytest

from rtod.jobs import (
    NO_STOCK,
    SWEEPS,
    CapExceeded,
    JobSpec,
    Ledger,
    LogMonitor,
    RpProvider,
    bootstrap_cmd,
    gpu_env,
    job_env,
    run_job,
)

SPEC = JobSpec(
    "qat", "img:tag", "python train.py", (("COMMUNITY", "A"), ("SECURE", "B,C")), 0.34, 2.0
)


class Fake:
    def __init__(self, fail_follow=False, leak=False):
        self.calls, self.fail_follow, self.leak = [], fail_follow, leak

    def up(self, spec):
        self.calls.append("up")

    def follow(self):
        self.calls.append("follow")
        yield "c1"
        if self.fail_follow:
            raise RuntimeError("job crashed")
        yield "c2"

    def fetch(self, code, dest):
        self.calls.append(f"fetch:{code}:{dest.name}")

    def down(self):
        self.calls.append("down")

    def active(self):
        return self.leak


def test_job_over_cap_is_refused_before_any_launch(tmp_path):
    ledger = Ledger(tmp_path / "s.json", cap=0.5)  # worst case is 0.34 * 2 = 0.68
    fake = Fake()
    with pytest.raises(CapExceeded):
        run_job(fake, ledger, SPEC, tmp_path)
    assert fake.calls == []


def test_ledger_accumulates_and_blocks_the_next_job(tmp_path):
    ledger = Ledger(tmp_path / "s.json", cap=1.0)
    run_job(Fake(), ledger, SPEC, tmp_path)
    assert 0 <= ledger.spent() < 0.01  # near-instant fake run, charged at the ceiling rate
    ledger.record(SPEC, 2.0)  # a full-length run costs 0.68
    with pytest.raises(CapExceeded):
        ledger.check(SPEC)


def test_teardown_and_charge_happen_even_when_the_job_crashes(tmp_path):
    ledger = Ledger(tmp_path / "s.json")
    fake = Fake(fail_follow=True)
    with pytest.raises(RuntimeError, match="crashed"):
        run_job(fake, ledger, SPEC, tmp_path)
    assert fake.calls == ["up", "follow", "fetch:c1:artifact1", "down"]  # artifact 1 is kept
    assert len(ledger.entries()) == 1


def test_leftover_pod_after_teardown_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="still running"):
        run_job(Fake(leak=True), Ledger(tmp_path / "s.json"), SPEC, Path(tmp_path))


def test_rp_env_always_carries_the_lifetime_and_price_ceilings():
    env = RpProvider.env(SPEC)
    assert env["RP_MAX_LIFETIME_SECS"] == "7200"
    assert env["RP_MAX_HOURLY_USD"] == "0.34"
    assert env["RP_JOB_CMD"] == "python train.py"


def feed(mon, now, *lines):
    return [(e.level, e.msg) for e in mon.feed(list(lines), now)]


def test_healthy_run_reports_phases_progress_and_artifact():
    mon = LogMonitor(start=0)
    ev = feed(mon, 10, "RTOD t+0001s phase=boot host=x", "RTOD_HB t+0030s phase=boot gpu=na")
    assert ("info", "RTOD t+0001s phase=boot host=x") in ev and all(lvl != "fatal" for lvl, _ in ev)
    ev = feed(mon, 40, "TRAIN step=10/1250 loss=7.1", "RTOD_HB t+0060s phase=train")
    assert ev == [("info", "TRAIN step=10/1250 loss=7.1")]
    assert feed(mon, 50, "RP_ARTIFACT_CODE=rtodqat-12-3") == [("artifact", "rtodqat-12-3")]


@pytest.mark.parametrize(
    "line",
    [
        "RTOD_FAIL phase=train line=9 exit=1",
        "RTOD_FATAL phase=preflight CUDA is not available to torch",
        "Traceback (most recent call last):",
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate",
        "FloatingPointError: non-finite loss nan at step 3",
    ],
)
def test_error_markers_are_fatal_immediately(line):
    assert ("fatal", line) in feed(LogMonitor(start=0), 5, line)


def test_silent_pod_warns_then_goes_fatal():
    mon = LogMonitor(start=0, boot_warn=180, boot_fatal=420)
    assert feed(mon, 100) == []
    assert feed(mon, 200)[0][0] == "warn"
    assert feed(mon, 210) == []  # warned once only
    assert feed(mon, 500)[0][0] == "fatal"


def test_lost_heartbeat_is_fatal():
    mon = LogMonitor(start=0, hb_stall=180)
    feed(mon, 10, "RTOD_HB t+0010s phase=train")
    assert feed(mon, 100) == []
    assert feed(mon, 300)[0][0] == "fatal"


def test_phase_over_budget_warns_once():
    mon = LogMonitor(start=0)
    feed(mon, 1, "RTOD t+0001s phase=install sha=x", "RTOD_HB a")
    assert feed(mon, 650, "RTOD_HB b")[0][0] == "warn"
    assert feed(mon, 700, "RTOD_HB c") == []


def test_replayed_log_lines_are_not_reported_twice():
    mon = LogMonitor(start=0)
    assert len(feed(mon, 1, "TRAIN step=10/20 loss=1")) == 1
    assert feed(mon, 2, "TRAIN step=10/20 loss=1") == []


def test_rp_env_follows_the_selected_offer():
    assert RpProvider.env(SPEC, 0)["RP_CLOUD_TYPE"] == "COMMUNITY"
    env = RpProvider.env(SPEC, 1)
    assert env["RP_CLOUD_TYPE"] == "SECURE" and env["RP_GPU_TYPE"] == "B,C"


def test_bootstrap_output_reaches_rps_log_through_its_exact_wrapper(tmp_path):
    """Regression: a trailing `|| echo` once made `| tee LOG` bind to the echo only, so the job's
    output skipped rp's log and rp's idle watchdog killed the healthy pod."""
    import subprocess

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/job_qat.sh").write_text("echo JOB_OUTPUT_$RTOD_SHA\n")
    cmd = bootstrap_cmd("abc123", job_env(5, 1e-5, 100.0), base=f"file://{tmp_path}")
    assert 'QAT_LR="1e-05"' in cmd
    log = tmp_path / "rp-job.log"
    # the same shape as rp's launch_command: ( ${JOB_CMD} 2>&1 | tee "$LOG" )
    subprocess.run(["bash", "-c", f'( {cmd} 2>&1 | tee "{log}" )'], check=True, capture_output=True)
    assert "JOB_OUTPUT_abc123" in log.read_text()


def test_bootstrap_failure_is_reported_in_the_log_too(tmp_path):
    import subprocess

    cmd = bootstrap_cmd("abc123", job_env(5, 1e-5, 100.0), base=f"file://{tmp_path}/missing")
    log = tmp_path / "rp-job.log"
    subprocess.run(["bash", "-c", f'( {cmd} 2>&1 | tee "{log}" )'], check=True, capture_output=True)
    assert "RTOD_FATAL phase=bootstrap" in log.read_text()


def test_every_published_artifact_is_fetched_in_order(tmp_path):
    fake = Fake()
    run_job(fake, Ledger(tmp_path / "s.json"), SPEC, tmp_path)
    assert fake.calls == ["up", "follow", "fetch:c1:artifact1", "fetch:c2:artifact2", "down"]


def test_done_marker_is_reported_and_is_not_an_error():
    mon = LogMonitor(start=0)
    assert ("done", "RTOD t+0900s RTOD_DONE") in feed(mon, 5, "RTOD t+0900s RTOD_DONE")


@pytest.mark.parametrize(
    "msg",
    [
        "create pod: There are no instances currently available",
        "create pod: There are no longer any instances available with the requested specs.",
        "rp: error: pod had 15 GB RAM < 16; terminated",  # already terminated: next offer
    ],
)
def test_both_runpod_stock_wordings_are_retried(msg):
    assert NO_STOCK.search(msg)


def test_other_create_errors_are_not_mistaken_for_stock():
    assert not NO_STOCK.search("create pod: invalid field gpuTypeIds")


def test_sweep_configs_reach_the_script_intact_through_rps_wrapper(tmp_path):
    """The config list carries spaces, dashes and colons; none may be eaten by the shell layers."""
    import subprocess

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/job_qat.sh").write_text('echo "GOT:$QAT_CONFIGS"\n')
    cmd = bootstrap_cmd("abc", job_env(5, 5e-5, 100.0, SWEEPS["recipe"]), base=f"file://{tmp_path}")
    log = tmp_path / "log"
    subprocess.run(["bash", "-c", f'( {cmd} 2>&1 | tee "{log}" )'], check=True, capture_output=True)
    got = log.read_text().split("GOT:")[1].strip()
    assert got.split(",") == [f"{n}:{f}" for n, f in SWEEPS["recipe"].items()]


def test_each_sweep_config_differs_from_the_previous_in_one_factor():
    flags = [set(f.split()) for f in SWEEPS["recipe"].values()]
    # qat_ema_40k adds a data flag pair on top of qat_ema; the rest add or drop one flag
    assert flags[2] == set() and flags[3] == {"--ema"}
    assert flags[4] == {"--ema", "--subset", "train40k"}


def test_every_named_sweep_encodes_cleanly_and_has_unique_config_names():
    for name, configs in SWEEPS.items():
        env = job_env(5, 5e-5, 100.0, configs)["QAT_CONFIGS"]
        assert len(env.split(",")) == len(configs), name
        assert all(c and ":" in c for c in env.split(",")), name
        assert not set("|;'\n") & set(env), name  # characters the shell layers would eat


HB = "RTOD_HB t+0100s phase=train gpu(util%,memMiB,totMiB,C)=0,2,16380,29 load=3 last=-"


def test_train_phase_without_progress_lines_warns_then_goes_fatal_despite_heartbeats():
    """Regression: a pod sat in `train` for 23 minutes (pruning at 64 threads) with a healthy
    heartbeat and an idle GPU; heartbeats alone only prove the shell is alive."""
    mon = LogMonitor(start=0)
    feed(mon, 1, "RTOD t+0001s phase=train config=a")
    assert feed(mon, 100, "RTOD_HB 1") == []
    assert [lvl for lvl, _ in feed(mon, 320, "RTOD_HB 2") if lvl != "status"] == ["warn"]
    assert [lvl for lvl, _ in feed(mon, 400, "RTOD_HB 3") if lvl != "status"] == []  # warned once
    level, msg = feed(mon, 950, "RTOD_HB 4")[0]
    assert level == "fatal" and "no progress" in msg


@pytest.mark.parametrize(
    "line",
    ["TRAIN step=10/1250 loss=7", "PRUNE start macs_removed=0.5", "DATA split=train fetched=10/20"],
)
def test_progress_lines_keep_a_long_train_phase_healthy(line):
    mon = LogMonitor(start=0)
    feed(mon, 1, "RTOD t+0001s phase=train config=a")
    for t in range(200, 2000, 200):
        evs = feed(mon, t, f"{line} t={t}", f"RTOD_HB {t}")
        assert all(lvl in ("info", "status") for lvl, _ in evs)


def test_gpu_idle_during_train_is_reported_while_data_is_still_flowing():
    mon = LogMonitor(start=0)
    feed(mon, 1, "RTOD t+0001s phase=train config=a")
    out = []
    for i in range(4):
        out += feed(mon, 10 + i, f"{HB} n={i}", f"DATA split=train fetched={i}/9")
    assert any(lvl == "warn" and "GPU" in msg for lvl, msg in out)


def test_busy_gpu_is_not_reported():
    mon = LogMonitor(start=0)
    feed(mon, 1, "RTOD t+0001s phase=train config=a")
    busy = HB.replace("=0,2,", "=85,1168,")
    assert all(lvl != "warn" for i in range(6) for lvl, _ in feed(mon, 5 + i, f"{busy} n={i}"))


def test_a_status_line_summarises_the_job_every_two_minutes():
    mon = LogMonitor(start=0)
    feed(mon, 1, "RTOD t+0001s phase=train config=a", "TRAIN step=10/1250 loss=7")
    evs = feed(mon, 130, "RTOD_HB 1")
    status = [m for lvl, m in evs if lvl == "status"]
    assert len(status) == 1 and "phase=train" in status[0] and "step=10/1250" in status[0]


def test_sigterm_to_the_driver_still_tears_down_and_charges(tmp_path):
    """Regression: `kill <driver>` skipped the finally block, leaving the pod running."""
    import os
    import signal

    class Killed(Fake):
        def follow(self):
            self.calls.append("follow")
            os.kill(os.getpid(), signal.SIGTERM)
            yield "never"

    fake, ledger = Killed(), Ledger(tmp_path / "s.json")
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        run_job(fake, ledger, SPEC, tmp_path)
    assert "down" in fake.calls and len(ledger.entries()) == 1
    assert signal.getsignal(signal.SIGTERM) == before


def test_job_kind_reaches_the_pod_script_through_rps_wrapper(tmp_path):
    import subprocess

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/job_qat.sh").write_text('echo "KIND:${RTOD_JOB:-qat}"\n')
    cmd = bootstrap_cmd("abc", {"RTOD_JOB": "gpu"}, base=f"file://{tmp_path}")
    log = tmp_path / "log"
    subprocess.run(["bash", "-c", f'( {cmd} 2>&1 | tee "{log}" )'], check=True, capture_output=True)
    assert "KIND:gpu" in log.read_text()


def test_min_ram_follows_the_job_spec():
    assert RpProvider.env(SPEC)["RP_MIN_RAM_GB"] == "16"
    small = JobSpec("gpu", "img", "x", (("SECURE", "A"),), 0.4, 1.0, min_ram_gb=8)
    assert RpProvider.env(small)["RP_MIN_RAM_GB"] == "8"


def test_gpu_stage_selection_reaches_the_pod_script(tmp_path):
    """Regression: `--only tensorrt` was parsed but never put in the pod's environment, so the pod
    repeated the whole CUDA benchmark first."""
    import subprocess

    assert gpu_env() == {"RTOD_JOB": "gpu"}
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/job_qat.sh").write_text('echo "ONLY:${RTOD_GPU_ONLY:-all}"\n')
    cmd = bootstrap_cmd("abc", gpu_env("tensorrt"), base=f"file://{tmp_path}")
    log = tmp_path / "log"
    subprocess.run(["bash", "-c", f'( {cmd} 2>&1 | tee "{log}" )'], check=True, capture_output=True)
    assert "ONLY:tensorrt" in log.read_text()
