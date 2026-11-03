import numpy as np
import pytest

pytest.importorskip("indah")

from rtod.dashboard import BUDGET_MS, jpeg_uri, percentile, split_bar  # noqa: E402


def test_percentile_of_nothing_is_zero_and_picks_a_sample_otherwise():
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_a_latency_splits_into_inside_and_over_the_frame_budget():
    assert split_bar(10.0) == [10.0, 0.0]
    inside, over = split_bar(BUDGET_MS + 20)
    assert inside == pytest.approx(BUDGET_MS) and over == pytest.approx(20)


def test_frames_are_sent_as_small_jpeg_data_uris():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    uri = jpeg_uri(frame)
    assert uri.startswith("data:image/jpeg;base64,") and len(uri) < 5_000  # a black frame is tiny
