import numpy as np
import pytest

from sumbench.metrics import (
    bootstrap_ci,
    latency_stats,
    paired_bootstrap_diff,
    rouge_per_article,
    split_sentences,
)
from sumbench.report import render_markdown

REFS = ["The cat sat on the mat.\nIt was happy.", "Stocks fell sharply on Monday."]


def test_identical_text_scores_one():
    r = rouge_per_article(["The cat sat on the mat. It was happy.", "Stocks fell sharply on Monday."], REFS)
    for k in ("rouge1", "rouge2", "rougeLsum"):
        assert np.allclose(r[k], 1.0)


def test_disjoint_text_scores_zero():
    r = rouge_per_article(["zzz qqq", "xxx yyy"], REFS)
    assert all(np.allclose(v, 0.0) for v in r.values())


def test_partial_overlap_known_value():
    # unigram overlap: pred={the,cat} ref={the,cat,sat} -> P=1, R=2/3, F=0.8
    r = rouge_per_article(["the cat"], ["the cat sat"])
    assert r["rouge1"][0] == pytest.approx(0.8)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        rouge_per_article(["a"], ["a", "b"])


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == "One.\nTwo!\nThree?"


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    m, lo, hi = bootstrap_ci(v, 500, seed=1)
    assert m == pytest.approx(0.35) and lo < m < hi
    assert bootstrap_ci(v, 500, seed=1) == (m, lo, hi)


def test_bootstrap_constant_has_zero_width():
    m, lo, hi = bootstrap_ci([0.5] * 10, 100)
    assert m == lo == hi == pytest.approx(0.5)


def test_paired_diff():
    m, lo, hi = paired_bootstrap_diff([0.5, 0.6, 0.7], [0.4, 0.5, 0.6], 200)
    assert m == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_latency_percentiles():
    s = latency_stats([0.1] * 90 + [1.0] * 10)
    assert s["p50_ms"] == pytest.approx(100) and s["p95_ms"] == pytest.approx(1000)


def test_render_markdown_smoke():
    row = {"name": "t5_small_lora", "label": "L", "trainable_params": 294912, "total_params": 60506624,
           "latency": {"p50_ms": 1, "p95_ms": 2, "mean_ms": 1},
           **{k: {"mean": 0.4, "ci95": [0.39, 0.41]} for k in ("rouge1", "rouge2", "rougeLsum")}}
    md = render_markdown({"n_samples": 2, "seed": 1, "bootstrap_resamples": 10, "batch_size": 1,
                          "hardware": "hw", "results": [row], "paired_diffs": {}})
    assert "40.00 [39.00, 41.00]" in md and "0.29M" in md
