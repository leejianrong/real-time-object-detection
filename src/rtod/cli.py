"""`rtod` command line. JSON on stdout with --json; non-zero exit on failure."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rtod import bench, coco


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
    b = sub.add_parser("bench", help="CPU latency benchmark, writes results JSON")
    b.add_argument("model", type=Path)
    b.add_argument("--variant", required=True, help="e.g. yolox-tiny/fp32")
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--iters", type=int, default=bench.MIN_ITERS)
    b.add_argument("--quick", action="store_true", help="allow <200 iterations; labelled in output")
    b.add_argument("--eval", choices=["val200", "val"], help="also compute mAP and merge it")
    b.add_argument("--out", type=Path, default=Path("results"))
    b.add_argument("--json", action="store_true")
    t = sub.add_parser("qat", help="QAT fine-tune, export folded float, PTQ it (the pod job)")
    t.add_argument("--steps", type=int, default=1250)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default="cpu")
    t.add_argument("--subset", choices=["train10k", "val200"], default="train10k")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", type=Path, default=Path("qat_out"))
    args = p.parse_args(argv)

    if args.cmd == "qat":
        from rtod.quant import quantize_ptq
        from rtod.train import export_folded, qat_finetune

        split = "train" if args.subset == "train10k" else "val"
        ids = coco.read_ids(args.subset)
        args.out.mkdir(parents=True, exist_ok=True)
        model = qat_finetune(
            ids, split, steps=args.steps, batch=args.batch, device=args.device, workers=args.workers
        )
        folded = export_folded(model, args.out / "yolox_tiny_qat_folded.onnx")
        calib_imgs = coco.fetch_images(ids[:5000], split)
        print(quantize_ptq(folded, args.out / "yolox_tiny_qat_int8.onnx", calib_imgs))
        return 0

    if args.cmd == "bench":
        try:
            r = bench.bench(
                args.model, args.variant, threads=args.threads, iters=args.iters, quick=args.quick
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.eval:
            m = coco.evaluate(
                args.model, coco.read_ids("val200") if args.eval == "val200" else None
            )
            r.map50_95, r.map50 = m["map50_95"], m["map50"]
        path = bench.write_result(r, args.out)
        print(json.dumps(asdict(r)) if args.json else path)
        return 0

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
