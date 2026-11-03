"""Run the showcase notebook top to bottom, as Colab would (a fresh kernel, the published models
and the repo's results fetched from the network). Slow: about 5 minutes on a CPU runner."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.notebook, pytest.mark.slow]
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK = Path(__file__).parents[2] / "notebooks/rtod_showcase.ipynb"


def test_the_notebook_runs_without_errors_and_prints_its_tables(tmp_path):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(
        nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    ).execute()
    text = "\n".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs
    )
    assert "LIVE_TRAIN is off" in text  # the heavy path is off by default
    assert "33.47" in text and "32.75" in text  # the held-out CPU table
    assert "18.44" in text  # the pruning table
    assert "FP16" in text and "indah is running" in text  # GPU table and the live dashboard
