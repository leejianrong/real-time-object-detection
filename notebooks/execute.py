"""Run the showcase notebook and keep the executed copy (with outputs) next to it.

    make notebook          # writes notebooks/out/rtod_showcase.executed.ipynb (and .html if nbconvert is installed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent
OUT = HERE / "out"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    nb = nbformat.read(HERE / "rtod_showcase.ipynb", as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
    executed = OUT / "rtod_showcase.executed.ipynb"
    nbformat.write(nb, executed)
    print(executed)
    try:
        from nbconvert import HTMLExporter
    except ImportError:
        return 0
    body, _ = HTMLExporter().from_notebook_node(nb)
    html = OUT / "rtod_showcase.executed.html"
    html.write_text(body)
    print(html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
