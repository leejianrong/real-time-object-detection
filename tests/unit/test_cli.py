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
