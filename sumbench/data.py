"""CNN/DailyMail loading and deterministic test-set sampling."""
from __future__ import annotations

import random
from typing import Sequence

DATASET = "abisee/cnn_dailymail"
VERSION = "3.0.0"
T5_PREFIX = "summarize: "


def sample_indices(n_total: int, n: int, seed: int) -> list[int]:
    """Fixed-seed sample of n distinct indices out of n_total (sorted, reproducible)."""
    if n > n_total:
        raise ValueError(f"cannot sample {n} from {n_total}")
    return sorted(random.Random(seed).sample(range(n_total), n))


def sample_ids(ids: Sequence[str], n: int, seed: int) -> list[str]:
    return [ids[i] for i in sample_indices(len(ids), n, seed)]


def load_test_sample(n: int, seed: int) -> list[dict]:
    """Return n test articles ({id, article, highlights}) chosen with a fixed seed."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, VERSION, split="test")
    idx = sample_indices(len(ds), n, seed)
    return [
        {"id": r["id"], "article": r["article"], "highlights": r["highlights"]}
        for r in (ds[i] for i in idx)
    ]
