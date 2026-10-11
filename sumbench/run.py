"""Benchmark entrypoint: python -m sumbench.run --config configs/default.yaml"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import yaml

from .data import load_test_sample
from .metrics import ROUGE_KEYS, bootstrap_ci, latency_stats, paired_bootstrap_diff, rouge_per_article
from .report import render_markdown


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def hardware() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        chip = ""
    return f"{chip or platform.processor() or platform.machine()} ({platform.system()} {platform.release()})"


def generate_rows(cfg: dict, spec: dict, articles: list[dict], pred_path: Path) -> None:
    """Resumable generation: appends to pred_path one JSON line per article."""
    from .models import Summarizer, pick_device

    done = read_jsonl(pred_path)
    n_warm = cfg["warmup_articles"]
    if len(done) >= len(articles):
        return
    device = pick_device()
    s = Summarizer(spec, device)
    meta = {"trainable_params": s.trainable, "total_params": s.total, "device": device}
    (pred_path.with_suffix(".meta.json")).write_text(json.dumps(meta))
    for a in articles[:n_warm]:  # warm up kernels/allocator, not recorded
        s.summarize(a["article"])
    with pred_path.open("a") as f:
        for i, a in enumerate(articles[len(done):], start=len(done)):
            text, dt = s.summarize(a["article"])
            f.write(json.dumps({"id": a["id"], "prediction": text, "latency_s": round(dt, 4)}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{spec['name']}] {i + 1}/{len(articles)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-generate", action="store_true",
                    help="only recompute metrics/tables from committed predictions")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["out_dir"])
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    ids_path = out / "sample_ids.json"
    articles = None
    refs_path = out / "references.jsonl"
    if not args.skip_generate:
        articles = load_test_sample(cfg["n_samples"], cfg["seed"])
        ids_path.write_text(json.dumps({"seed": cfg["seed"], "n": len(articles),
                                        "ids": [a["id"] for a in articles]}, indent=1))
        refs_path.write_text("".join(json.dumps({"id": a["id"], "highlights": a["highlights"]}) + "\n"
                                     for a in articles))
        t = cfg["train"]
        if any(m["kind"] == "t5_lora" and m.get("enabled", True) for m in cfg["models"]) \
                and not (Path(t["adapter_path"]) / "adapter_model.safetensors").exists():
            from .train import train
            train(cfg, cfg["seed"])
    refs = {r["id"]: r["highlights"] for r in read_jsonl(refs_path)}

    results, per_article = [], {}
    for spec in cfg["models"]:
        if not spec.get("enabled", True):
            continue
        pred_path = out / "predictions" / f"{spec['name']}.jsonl"
        if not args.skip_generate:
            generate_rows(cfg, spec, articles, pred_path)
        preds = read_jsonl(pred_path)
        if not preds:
            continue
        meta = json.loads(pred_path.with_suffix(".meta.json").read_text())
        r = rouge_per_article([p["prediction"] for p in preds], [refs[p["id"]] for p in preds])
        per_article[spec["name"]] = r
        row = {"name": spec["name"], "label": spec["label"], "n": len(preds), **meta,
               "decoding": spec.get("generate", {}), "max_input_tokens": spec["max_input_tokens"],
               "latency": latency_stats([p["latency_s"] for p in preds])}
        for k in ROUGE_KEYS:
            m, lo, hi = bootstrap_ci(r[k], cfg["bootstrap_resamples"], cfg["seed"])
            row[k] = {"mean": m, "ci95": [lo, hi]}
        results.append(row)
        (out / f"{spec['name']}.json").write_text(json.dumps(row, indent=2))

    paired = {}
    for a, b in [("t5_small_lora", "t5_small_zeroshot"), ("bart_large_cnn", "t5_small_lora")]:
        if a in per_article and b in per_article:
            paired[f"{a}_minus_{b}"] = {k: dict(zip(("mean", "lo", "hi"), paired_bootstrap_diff(
                per_article[a][k], per_article[b][k], cfg["bootstrap_resamples"], cfg["seed"])))
                for k in ROUGE_KEYS}
    train_info = Path(cfg["train"]["adapter_path"]) / "train_info.json"
    summary = {"hardware": hardware(), "batch_size": cfg["batch_size"], "seed": cfg["seed"],
               "n_samples": cfg["n_samples"], "bootstrap_resamples": cfg["bootstrap_resamples"],
               "results": results, "paired_diffs": paired,
               "lora_training": json.loads(train_info.read_text()) if train_info.exists() else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "results.md").write_text(render_markdown(summary))
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
