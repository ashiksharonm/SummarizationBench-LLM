"""
utils.py — Preprocessing and text-cleaning utilities for SummarizationBench-LLM.

This module handles:
  1. Loading and subsetting the CNN/DailyMail dataset
  2. Tokenizing article + highlight pairs for T5/BART
  3. Basic text cleaning (whitespace, control characters)
  4. Batch preprocessing for Trainer-compatible datasets
"""

import re
import logging
from typing import Optional

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
# T5 uses a task prefix to distinguish summarization from other tasks
T5_PREFIX = "summarize: "

# CNN/DailyMail field names
ARTICLE_COL = "article"
SUMMARY_COL = "highlights"

# Default generation length bounds
MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 128


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light-touch text normalization:
      - Strip leading/trailing whitespace
      - Collapse multiple spaces/newlines into a single space
      - Remove non-printable control characters (but keep newlines for now)

    We intentionally avoid aggressive cleaning (e.g. lowercasing) so that
    ROUGE scores are computed fairly against the original references.
    """
    # Remove non-printable characters except newline/tab
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", text)
    # Collapse runs of whitespace (space, tab, newline) into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Dataset Loading ──────────────────────────────────────────────────────────

def load_cnn_dailymail(
    version: str = "3.0.0",
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Load the CNN/DailyMail dataset from HuggingFace Hub.

    Args:
        version:    Dataset version string (default "3.0.0")
        train_size: If set, subsample training split to this many examples.
                    Useful for fast iteration on Colab (e.g. 5000).
        val_size:   Subsample validation split.
        test_size:  Subsample test split.
        seed:       Random seed for reproducibility.

    Returns:
        DatasetDict with keys "train", "validation", "test".
    """
    logger.info("Loading CNN/DailyMail %s from HuggingFace Hub…", version)
    dataset = load_dataset("cnn_dailymail", version)

    # Optional subsampling — speeds up Colab runs without loss of generality
    if train_size:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(train_size))
        logger.info("Train subsampled to %d examples", train_size)
    if val_size:
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(val_size))
        logger.info("Validation subsampled to %d examples", val_size)
    if test_size:
        dataset["test"] = dataset["test"].shuffle(seed=seed).select(range(test_size))
        logger.info("Test subsampled to %d examples", test_size)

    return dataset


# ─── Tokenization ─────────────────────────────────────────────────────────────

def preprocess_function(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int = MAX_INPUT_TOKENS,
    max_target_length: int = MAX_TARGET_TOKENS,
    is_t5: bool = True,
) -> dict:
    """
    Tokenize a batch of (article, summary) pairs for seq2seq training.

    For T5: prepend the task prefix "summarize: " to each article.
    For BART: no prefix needed — BART doesn't use task prefixes.

    The labels tensor is created in the standard HuggingFace way:
      - tokenizer encodes the target summaries
      - padding tokens in labels are set to -100 so CrossEntropy ignores them

    Args:
        examples:          Batch dict with keys "article" and "highlights"
        tokenizer:         Tokenizer matching the model
        max_input_length:  Truncation length for source articles
        max_target_length: Truncation length for target summaries
        is_t5:             Whether to add "summarize: " prefix

    Returns:
        Dict with "input_ids", "attention_mask", "labels"
    """
    # Apply T5 task prefix
    prefix = T5_PREFIX if is_t5 else ""
    inputs = [prefix + clean_text(doc) for doc in examples[ARTICLE_COL]]
    targets = [clean_text(ref) for ref in examples[SUMMARY_COL]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",  # pad to max_length for uniform batch tensors
    )

    # Tokenize targets (summaries) — use tokenizer as context manager for labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )

    # Replace padding token id (-100) so loss ignores padded label positions
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = label_ids
    return model_inputs


def build_tokenized_datasets(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int = MAX_INPUT_TOKENS,
    max_target_length: int = MAX_TARGET_TOKENS,
    is_t5: bool = True,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Apply `preprocess_function` to all splits in the dataset using map().

    Uses batched=True for speed and removes original text columns
    (they're not needed by the Trainer).

    Args:
        dataset:           Raw DatasetDict
        tokenizer:         Tokenizer
        max_input_length:  Max source length
        max_target_length: Max target length
        is_t5:             Add T5 prefix?
        num_proc:          Number of CPU workers for parallel mapping

    Returns:
        Tokenized DatasetDict ready for Trainer
    """
    tokenized = dataset.map(
        lambda examples: preprocess_function(
            examples,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            is_t5=is_t5,
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=[ARTICLE_COL, SUMMARY_COL, "id"],  # drop raw text cols
        desc="Tokenizing dataset",
    )
    return tokenized
