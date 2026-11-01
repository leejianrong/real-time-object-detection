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
    q = sub.add_parser("quantize", help="PTQ INT8 (QDQ) from an FP32 ONNX")
    q.add_argument("fp32", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--calib", type=int, default=5000, help="calibration images from train10k")
    args = p.parse_args(argv)

    if args.cmd == "quantize":
        from rtod.quant import quantize_ptq

        calib = coco.read_ids("train10k")[: args.calib]
        print(quantize_ptq(args.fp32, args.out, coco.fetch_images(calib, "train")))
        return 0
    val_ids = coco.read_ids("val200") if args.subset == "val200" else None
    res = {"model": str(args.model), "subset": args.subset, **coco.evaluate(args.model, val_ids)}
    print(json.dumps(res) if args.json else res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
