from pathlib import Path

import pytest

from rtod import cli


def test_quantize_options_reach_quantize_ptq(monkeypatch, tmp_path):
    """Regression: `--method` was read as args.method but never declared, so `rtod quantize` crashed
    with AttributeError on main (CI had no test through the CLI)."""
    seen = {}
    monkeypatch.setattr("rtod.coco.read_ids", lambda name: list(range(10)))
    monkeypatch.setattr(
        "rtod.coco.fetch_images", lambda ids, split: [Path(f"{i}.jpg") for i in ids]
    )

    def fake(fp32, out, images, op_types, method):
        seen.update(images=len(images), op_types=op_types, method=method)
        return out

    monkeypatch.setattr("rtod.quant.quantize_ptq", fake)
    rc = cli.main(
        ["quantize", "a.onnx", str(tmp_path / "o.onnx"), "--calib", "4", "--conv-only",
         "--method", "moving_avg"]
    )  # fmt: skip
    assert rc == 0 and seen == {"images": 4, "op_types": ["Conv"], "method": "moving_avg"}


def test_unknown_calibration_method_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["quantize", "a.onnx", "b.onnx", "--method", "nope"])


def test_calib_start_selects_a_different_slice_of_the_list(monkeypatch, tmp_path):
    picked = {}
    monkeypatch.setattr("rtod.coco.read_ids", lambda name: list(range(100)))
    monkeypatch.setattr("rtod.coco.fetch_images", lambda ids, split: picked.setdefault("ids", ids))
    monkeypatch.setattr("rtod.quant.quantize_ptq", lambda *a: tmp_path / "o.onnx")
    cli.main(["quantize", "a.onnx", "b.onnx", "--calib", "5", "--calib-start", "20"])
    assert picked["ids"] == [20, 21, 22, 23, 24]


def test_export_qdq_loads_the_checkpoint_and_exports_with_decode_in_graph(monkeypatch, tmp_path):
    calls = {}

    class Model:
        class head:  # noqa: N801
            decode_in_inference = False

    monkeypatch.setattr("rtod.qat.load_qat_checkpoint", lambda p: Model)
    monkeypatch.setattr(
        "rtod.models.export_onnx",
        lambda m, out: calls.update(decode=m.head.decode_in_inference) or out,
    )
    assert cli.main(["export-qdq", "qat_model.pt", str(tmp_path / "o.onnx")]) == 0
    assert calls == {"decode": True}


def test_prune_rejects_a_ratio_outside_zero_one(tmp_path):
    pytest.importorskip("torch_pruning")
    assert cli.main(["prune", "--macs-removed", "1.5", "--out", str(tmp_path / "p.pt")]) == 2


def test_qat_prune_macs_hands_a_pruned_model_to_the_fine_tune(monkeypatch, tmp_path):
    pytest.importorskip("torch_pruning")
    from rtod.models import build_yolox_tiny
    from rtod.prune import macs_params

    seen = {}
    base = build_yolox_tiny()
    monkeypatch.setattr("rtod.coco.read_ids", lambda name: [1])
    monkeypatch.setattr("rtod.models.load_official_weights", lambda path=None: base)
    monkeypatch.setattr("rtod.manifest.write_manifest", lambda *a, **k: None)
    monkeypatch.setattr("rtod.train.export_folded", lambda m, out: out)

    def fake_finetune(ids, split, **kw):
        seen["macs"] = macs_params(kw["init"])[0]
        return base

    monkeypatch.setattr("rtod.train.qat_finetune", fake_finetune)
    rc = cli.main(
        ["qat", "--prune-macs", "0.4", "--no-fake-quant", "--no-ptq", "--out", str(tmp_path)]
    )
    assert rc == 0 and seen["macs"] < 0.8 * macs_params(base)[0]
