"""ROUGE scoring, bootstrap confidence intervals and latency percentiles."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from rouge_score import rouge_scorer

ROUGE_KEYS = ("rouge1", "rouge2", "rougeLsum")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> str:
    """Put each sentence on its own line (rougeLsum expects newline-separated sentences)."""
    return "\n".join(s for s in _SENT_SPLIT.split(text.strip()) if s)


def rouge_per_article(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, np.ndarray]:
    """ROUGE-1/2/Lsum F1 for every (prediction, reference) pair, stemmer on."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    out = {k: [] for k in ROUGE_KEYS}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, split_sentences(pred))
        for k in ROUGE_KEYS:
            out[k].append(scores[k].fmeasure)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI over articles."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) with a paired bootstrap 95% CI (same articles in both)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_resamples, seed)


def latency_stats(latencies_s: Sequence[float]) -> dict[str, float]:
    """p50 / p95 / mean per-article latency in milliseconds."""
    ms = np.asarray(latencies_s, dtype=float) * 1000.0
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
    }
