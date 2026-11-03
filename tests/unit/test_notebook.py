import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[2] / "notebooks"


def _generator():
    spec = importlib.util.spec_from_file_location("make_notebook", ROOT / "make_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_notebook_is_what_the_generator_builds():
    """Edit notebooks/make_notebook.py and re-run it; never hand-edit the .ipynb."""
    assert json.loads((ROOT / "rtod_showcase.ipynb").read_text()) == _generator().build()


def test_notebook_ships_with_no_outputs_and_the_heavy_switch_off():
    nb = json.loads((ROOT / "rtod_showcase.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] for c in code)
    assert any("LIVE_TRAIN = False" in "".join(c["source"]) for c in code)
