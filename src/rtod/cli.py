"""`rtod` command line. JSON on stdout with --json; non-zero exit on failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rtod import coco


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rtod")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="COCO mAP of an ONNX model")
    e.add_argument("model", type=Path)
    e.add_argument("--subset", choices=["val200", "val"], default="val")
    e.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    ids = coco.read_ids("val200") if args.subset == "val200" else None
    res = {"model": str(args.model), "subset": args.subset, **coco.evaluate(args.model, ids)}
    print(json.dumps(res) if args.json else res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
