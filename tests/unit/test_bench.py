import json

import pytest

from rtod.bench import Result, _percentiles, bench, load_result, write_result


def make(**kw):
    base = dict(
        variant="yolox-tiny/fp32", runtime="ort-cpu", hardware={"cpu": "x"}, threads=1,
        input_size=416, warmup=20, iters=200, quick=False, model_ms_p50=10.0, model_ms_p95=12.0,
        e2e_ms_p50=11.0, e2e_ms_p95=13.0, fps=90.9, size_bytes=1, git_sha="abc",
    )  # fmt: skip
    return Result(**{**base, **kw})


def test_percentiles():
    assert _percentiles([float(i) for i in range(1, 101)]) == pytest.approx((50.5, 95.05))


def test_result_round_trips(tmp_path):
    path = write_result(make(map50_95=0.33), tmp_path)
    assert path.name == "yolox-tiny_fp32__ort-cpu__t1.json"
    assert load_result(path) == make(map50_95=0.33)


def test_unknown_schema_version_rejected(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({**json.loads(json.dumps(make().__dict__)), "schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version"):
        load_result(p)


def test_too_few_iterations_refused_unless_quick(tmp_path):
    with pytest.raises(ValueError, match="at least 200"):
        bench(tmp_path / "nope.onnx", "v", iters=50)
