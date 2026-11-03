import pytest

from rtod.resources import status_mb

STATUS = "Name:\tpython3\nVmHWM:\t  204800 kB\nVmRSS:\t  102400 kB\n"


def test_status_fields_are_converted_from_kb_to_mb():
    assert status_mb("VmHWM", STATUS) == 200.0
    assert status_mb("VmRSS", STATUS) == 100.0


def test_unknown_field_is_an_error():
    with pytest.raises(KeyError):
        status_mb("VmSwap", STATUS)
