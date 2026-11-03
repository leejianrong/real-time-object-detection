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
