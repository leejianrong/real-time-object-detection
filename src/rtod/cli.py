"""`rtod` command line. JSON on stdout with --json; non-zero exit on failure."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rtod import bench, coco


def _job_qat(args: argparse.Namespace) -> int:
    import subprocess

    from rtod import jobs

    sha = (
        args.sha
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    on_main = (
        subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"]).returncode == 0
    )
    if not on_main:
        print(f"error: {sha} is not on origin/main; the pod installs from GitHub", file=sys.stderr)
        return 2
    spec = jobs.JobSpec(
        name="qat",
        image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        cmd=jobs.bootstrap_cmd(sha, args.steps, args.lr, args.clip),
        offers=(
            ("COMMUNITY", "NVIDIA GeForce RTX 4090,NVIDIA RTX 4000 Ada Generation"),
            ("SECURE", "NVIDIA RTX 4000 Ada Generation,NVIDIA RTX 2000 Ada Generation"),
        ),
        max_hourly_usd=0.40,
        max_hours=1.0,
    )
    ledger = jobs.Ledger()
    print(
        f"job qat sha={sha[:8]} steps={args.steps} worst case ${spec.worst_case_usd:.2f}; "
        f"spent ${ledger.spent():.2f} of ${ledger.cap:.2f}"
    )
    ledger.check(spec)
    if args.dry_run:
        print("dry run: cap check passed, nothing launched")
        return 0
    jobs.run_job(jobs.RpProvider(), ledger, spec, args.dest)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rtod")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="COCO mAP of an ONNX model")
    e.add_argument("model", type=Path)
    e.add_argument("--subset", choices=["val200", "val", "val_dev", "val_test"], default="val")
    e.add_argument("--jackknife", type=int, default=0, help="B subsets for a standard error")
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
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--clip", type=float, default=10.0, help="grad-norm clip threshold")
    t.add_argument("--device", default="cpu")
    t.add_argument("--subset", choices=["train10k", "train40k", "val200"], default="train10k")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ema", action="store_true", help="YOLOX-style EMA of the weights")
    t.add_argument(
        "--no-fake-quant", action="store_true", help="float control: same recipe, no QAT"
    )
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--no-ptq", action="store_true", help="stop after exporting the folded model")
    t.add_argument("--out", type=Path, default=Path("qat_out"))
    j = sub.add_parser("job", help="launch a RunPod job under the spend cap")
    j.add_argument("which", choices=["qat"])
    j.add_argument("--sha", help="pushed commit to run (default: HEAD)")
    j.add_argument("--steps", type=int, default=1250)
    j.add_argument("--lr", type=float, default=2e-4)
    j.add_argument("--clip", type=float, default=10.0)
    j.add_argument(
        "--dry-run", action="store_true", help="print the plan and cap check, launch nothing"
    )
    j.add_argument("--dest", type=Path, default=Path(".cache/jobs/qat_artifact"))
    g = sub.add_parser("plot", help="plot training curves from metrics.jsonl or a pod/console log")
    g.add_argument("source", type=Path)
    g.add_argument("out", type=Path, help="output PNG")
    args = p.parse_args(argv)

    if args.cmd == "plot":
        from rtod.metrics import plot_training, read_rows

        rows = read_rows(args.source)
        if not rows:
            print(f"error: no TRAIN rows in {args.source}", file=sys.stderr)
            return 2
        print(plot_training(rows, args.out))
        return 0

    if args.cmd == "job":
        return _job_qat(args)

    if args.cmd == "qat":
        from rtod.quant import quantize_ptq
        from rtod.train import export_folded, qat_finetune

        split = "val" if args.subset == "val200" else "train"
        ids = coco.read_ids(args.subset)
        args.out.mkdir(parents=True, exist_ok=True)
        model = qat_finetune(
            ids,
            split,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            clip=args.clip,
            device=args.device,
            workers=args.workers,
            metrics_dir=args.out,
            seed=args.seed,
            ema=args.ema,
            fake_quant=not args.no_fake_quant,
        )
        import torch

        from rtod.manifest import write_manifest

        write_manifest(args.out, vars(args), args.subset)
        torch.save(model.state_dict(), args.out / "qat_model.pt")  # incl. observer ranges (path A)
        folded = export_folded(model, args.out / "yolox_tiny_qat_folded.onnx")
        if args.no_ptq:
            print(folded)
            return 0
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
    val_ids = None if args.subset == "val" else coco.read_ids(args.subset)
    scores = coco.evaluate(args.model, val_ids, jackknife=args.jackknife)
    res = {"model": str(args.model), "subset": args.subset, **scores}
    print(json.dumps(res) if args.json else res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
