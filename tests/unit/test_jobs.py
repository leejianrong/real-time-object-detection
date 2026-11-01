from pathlib import Path

import pytest

from rtod.jobs import CapExceeded, JobSpec, Ledger, RpProvider, run_job

SPEC = JobSpec("qat", "img:tag", "python train.py", "NVIDIA GeForce RTX 4090", 0.34, 2.0)


class Fake:
    def __init__(self, fail_follow=False, leak=False):
        self.calls, self.fail_follow, self.leak = [], fail_follow, leak

    def up(self, spec):
        self.calls.append("up")

    def follow(self):
        self.calls.append("follow")
        if self.fail_follow:
            raise RuntimeError("job crashed")

    def fetch(self, dest):
        self.calls.append("fetch")

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
    assert fake.calls == ["up", "follow", "down"]
    assert len(ledger.entries()) == 1


def test_leftover_pod_after_teardown_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="still running"):
        run_job(Fake(leak=True), Ledger(tmp_path / "s.json"), SPEC, Path(tmp_path))


def test_rp_env_always_carries_the_lifetime_and_price_ceilings():
    env = RpProvider.env(SPEC)
    assert env["RP_MAX_LIFETIME_SECS"] == "7200"
    assert env["RP_MAX_HOURLY_USD"] == "0.34"
    assert env["RP_JOB_CMD"] == "python train.py"
