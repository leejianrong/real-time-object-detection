import pytest

pytestmark = pytest.mark.integration


def test_placeholder() -> None:
    pytest.importorskip("numpy")
