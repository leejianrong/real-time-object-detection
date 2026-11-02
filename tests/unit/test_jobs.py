from pathlib import Path

import pytest

from rtod.jobs import CapExceeded, JobSpec, Ledger, LogMonitor, RpProvider, bootstrap_cmd, run_job

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
    cmd = bootstrap_cmd("abc123", 5, base=f"file://{tmp_path}")
    log = tmp_path / "rp-job.log"
    # the same shape as rp's launch_command: ( ${JOB_CMD} 2>&1 | tee "$LOG" )
    subprocess.run(["bash", "-c", f'( {cmd} 2>&1 | tee "{log}" )'], check=True, capture_output=True)
    assert "JOB_OUTPUT_abc123" in log.read_text()


def test_bootstrap_failure_is_reported_in_the_log_too(tmp_path):
    import subprocess

    cmd = bootstrap_cmd("abc123", 5, base=f"file://{tmp_path}/missing")
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
