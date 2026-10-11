"""Markdown table rendering from a summary dict (no model deps)."""
from __future__ import annotations

from .metrics import ROUGE_KEYS

_HEAD = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeLsum": "ROUGE-Lsum"}


def _fmt_params(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n < 1e8 else f"{n / 1e6:.0f}M"


def render_markdown(summary: dict) -> str:
    lines = [
        f"N = {summary['n_samples']} CNN/DailyMail test articles (seed {summary['seed']}); "
        f"ROUGE F1 x100 with {summary['bootstrap_resamples']}-resample bootstrap 95% CI; "
        f"latency per article at batch size {summary['batch_size']} on {summary['hardware']}.",
        "",
        "| Model | " + " | ".join(_HEAD[k] for k in ROUGE_KEYS)
        + " | Latency p50 (ms) | Latency p95 (ms) | Trainable params | Total params |",
        "|---|" + "---|" * (len(ROUGE_KEYS) + 4),
    ]
    for r in summary["results"]:
        cells = [f"{r[k]['mean'] * 100:.2f} [{r[k]['ci95'][0] * 100:.2f}, {r[k]['ci95'][1] * 100:.2f}]"
                 for k in ROUGE_KEYS]
        trainable = _fmt_params(r["trainable_params"]) if r["name"] == "t5_small_lora" else "0 (inference only)"
        lat = r["latency"]
        lines.append(f"| {r['label']} | " + " | ".join(cells)
                     + f" | {lat['p50_ms']:.0f} | {lat['p95_ms']:.0f} | {trainable} "
                     f"| {_fmt_params(r['total_params'])} |")
    if summary.get("paired_diffs"):
        lines += ["", "Paired bootstrap difference (ROUGE F1 x100, 95% CI):", "",
                  "| Comparison | " + " | ".join(_HEAD[k] for k in ROUGE_KEYS) + " |",
                  "|---|" + "---|" * len(ROUGE_KEYS)]
        for name, d in summary["paired_diffs"].items():
            a, b = name.split("_minus_")
            lines.append(f"| {a} - {b} | " + " | ".join(
                f"{d[k]['mean'] * 100:+.2f} [{d[k]['lo'] * 100:+.2f}, {d[k]['hi'] * 100:+.2f}]"
                for k in ROUGE_KEYS) + " |")
    return "\n".join(lines) + "\n"
