import json

from rtod.metrics import MetricsWriter, parse_train_lines, plot_training, read_rows

SMOKE = "TRAIN step=1/3 loss=4.5 iou=1.6 conf=1.8 cls=1.0 lr=2.00e-04 gnorm=189.79 it_s=0.64"
REST = "iou=1.5 conf=1.8 cls=0.7 lr=1.97e-04 gnorm=140.3 it_s=4.4"
FULL = [f"TRAIN step={s}/1250 loss={4.0 + s / 1000} {REST}" for s in (1, 10, 20)]


def test_parse_keeps_the_longest_run_and_drops_the_smoke_run():
    rows = parse_train_lines(["noise", SMOKE, "RTOD_HB x", *FULL])
    assert [r["step"] for r in rows] == [1, 10, 20]
    assert rows[1]["loss"] == 4.01 and rows[0]["lr"] == 1.97e-4 and "_total" not in rows[0]


def test_writer_appends_jsonl_and_read_rows_round_trips(tmp_path):
    w = MetricsWriter(tmp_path)
    w.log(1, loss=4.0, gnorm=2.0)
    w.log(2, loss=3.5, gnorm=1.0)
    w.close()
    assert [
        json.loads(x)["loss"] for x in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ] == [4.0, 3.5]
    assert read_rows(tmp_path / "metrics.jsonl")[1]["step"] == 2


def test_plot_writes_a_png(tmp_path):
    rows = parse_train_lines(FULL)
    png = plot_training(rows, tmp_path / "t.png", smooth=2)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
