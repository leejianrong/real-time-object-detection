import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_pruning")

from rtod.models import build_yolox_tiny  # noqa: E402
from rtod.prune import macs_params, prune_to_macs  # noqa: E402


@pytest.mark.parametrize("bad", [0, 1, -0.2, 1.5])
def test_ratios_outside_zero_one_are_rejected(bad):
    with pytest.raises(ValueError, match="macs_removed"):
        prune_to_macs(build_yolox_tiny(), bad)


def test_pruned_model_hits_the_mac_target_and_keeps_the_output_contract():
    model = build_yolox_tiny()
    pruned = prune_to_macs(model, 0.4)
    base, _ = macs_params(model)
    macs, params = macs_params(pruned)
    assert abs((1 - macs / base) - 0.4) < 0.08
    assert params < macs_params(model)[1]
    assert tuple(pruned(torch.randn(1, 3, 416, 416)).shape) == (1, 3549, 85)


def test_pruning_caps_torch_threads_and_restores_them(monkeypatch):
    """Regression: on a 64-thread pod pruning took ~35 minutes per model; it needs few threads."""
    import rtod.prune as prune

    seen = []
    monkeypatch.setattr(prune, "_bisect", lambda m, r: seen.append(torch.get_num_threads()) or m)
    before = torch.get_num_threads()
    torch.set_num_threads(max(before, 8))
    try:
        prune.prune_to_macs(build_yolox_tiny(), 0.3)
        assert seen == [prune.PRUNE_THREADS] and torch.get_num_threads() == max(before, 8)
    finally:
        torch.set_num_threads(before)
