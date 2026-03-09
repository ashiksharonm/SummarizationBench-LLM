"""
inference.py — Load fine-tuned T5-LoRA adapter and run batch inference.

This module provides:
  - load_t5_lora()  : Loads base T5-small + merges LoRA adapter weights
  - load_bart()     : Loads facebook/bart-large-cnn (pretrained baseline)
  - summarize_t5()  : Generate summary with T5-LoRA, returns (text, latency_ms)
  - summarize_bart(): Generate summary with BART, returns (text, latency_ms)
  - summarize_gemini(): Call Gemini 1.5 Pro API, returns (text, latency_ms)

Both summarize_* functions measure wall-clock inference latency so we can
report per-model speed comparisons in the evaluation script and Streamlit app.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    # Note: `pipeline` removed — transformers 5.x dropped 'summarization' and
    # 'text2text-generation' pipeline tasks. We call model.generate() directly.
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_T5_MODEL = "t5-small"  # transformers 5.x: no google/ prefix on Hub
BART_MODEL = "facebook/bart-large-cnn"

# Generation hyperparameters — kept consistent across models for fair comparison
GEN_MAX_NEW_TOKENS = 128
GEN_MIN_LENGTH = 30
GEN_NUM_BEAMS = 4           # Beam search for better quality summaries
GEN_EARLY_STOPPING = True
GEN_NO_REPEAT_NGRAM = 3     # Prevents repetitive n-grams in output


# ─── T5-LoRA loader ───────────────────────────────────────────────────────────

def load_t5_lora(
    adapter_path: str = "./fine_tuned_t5_lora",
    device: Optional[str] = None,
):
    """
    Load the base T5-small model and merge the LoRA adapter weights on top.

    PEFT's PeftModel.from_pretrained() handles the weight merging — we DON'T
    need to reinitialize the LoRA config, it's stored in adapter_config.json
    inside the adapter_path directory.

    Args:
        adapter_path: Path to the directory saved by train.py
        device:       "cuda", "cpu", or None (auto-detect)

    Returns:
        (model, tokenizer) — both moved to the selected device
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading T5-LoRA adapter from: %s  (device: %s)", adapter_path, device)

    # Step 1: Load the frozen base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_T5_MODEL)

    # Step 2: Wrap it with the saved LoRA adapter
    # PeftModel reads adapter_config.json + adapter_model.bin from adapter_path
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Step 3: Merge adapter weights into base model for faster inference
    # (optional — skip if you want to swap adapters dynamically)
    model = model.merge_and_unload()

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    logger.info("T5-LoRA model ready.")
    return model, tokenizer


# ─── BART loader ──────────────────────────────────────────────────────────────

def load_bart(device: Optional[str] = None):
    """
    Load facebook/bart-large-cnn as a pretrained summarization baseline.

    Note: transformers 5.x removed 'summarization' and 'text2text-generation'
    pipeline tasks entirely. We load via AutoModelForSeq2SeqLM and call
    model.generate() directly, exactly like T5-LoRA inference.

    Args:
        device: "cuda", "cpu", or None (auto-detect)

    Returns:
        (model, tokenizer) — both on the selected device
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading BART baseline: %s  (device: %s)", BART_MODEL, device)
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL)
    model.to(device)
    model.eval()
    logger.info("BART model ready.")
    return model, tokenizer


# ─── Individual model inference functions ─────────────────────────────────────

def summarize_t5(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    prefix: str = "summarize: ",
) -> tuple[str, float]:
    """
    Generate a summary using the fine-tuned T5-LoRA model.

    Args:
        text:      Raw input article text
        model:     Loaded T5 model (with merged LoRA weights)
        tokenizer: Matching T5 tokenizer
        device:    Device string ("cuda" or "cpu")
        prefix:    T5 task prefix

    Returns:
        (summary_text, latency_ms) — latency is wall-clock inference time
    """
    input_text = prefix + text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            min_length=GEN_MIN_LENGTH,
            num_beams=GEN_NUM_BEAMS,
            early_stopping=GEN_EARLY_STOPPING,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
        )
    latency_ms = (time.perf_counter() - start_time) * 1000

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary, round(latency_ms, 2)


def summarize_bart(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
) -> tuple[str, float]:
    """
    Generate a summary using facebook/bart-large-cnn.

    BART-large-CNN was fine-tuned on CNN/DailyMail directly, so no task
    prefix is needed. We call model.generate() identically to T5-LoRA.

    Args:
        text:      Raw input article text
        model:     Loaded BART model
        tokenizer: Matching BART tokenizer
        device:    Device string ("cuda" or "cpu")

    Returns:
        (summary_text, latency_ms)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,   # BART-large supports up to 1024 tokens
        truncation=True,
    ).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            min_length=GEN_MIN_LENGTH,
            num_beams=GEN_NUM_BEAMS,
            early_stopping=GEN_EARLY_STOPPING,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
        )
    latency_ms = (time.perf_counter() - start_time) * 1000

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary, round(latency_ms, 2)


def summarize_gemini(
    text: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-pro",
) -> tuple[str, float]:
    """
    Generate a summary using the Google Gemini 1.5 Pro API.

    The API key is read from the GEMINI_API_KEY environment variable by default.
    You can also pass it directly via the `api_key` argument.

    The prompt is designed to closely match the CNN/DailyMail summarization
    task: a brief, factual, bullet-journal-style summary.

    Args:
        text:       Raw input article text
        api_key:    Optional explicit API key (falls back to env var)
        model_name: Gemini model string

    Returns:
        (summary_text, latency_ms)

    Raises:
        ValueError: If no API key is provided/found
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        )

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not found. "
            "Set the GEMINI_API_KEY environment variable or pass api_key=."
        )

    genai.configure(api_key=key)
    gemini_model = genai.GenerativeModel(model_name)

    # Prompt mirrors the CNN/DailyMail task: concise, factual summary
    prompt = (
        "Please write a concise, factual summary of the following news article "
        "in 2–4 sentences. Focus on the key facts, who is involved, and the outcome.\n\n"
        f"Article:\n{text}\n\nSummary:"
    )

    start_time = time.perf_counter()
    response = gemini_model.generate_content(prompt)
    latency_ms = (time.perf_counter() - start_time) * 1000

    summary = response.text.strip()
    return summary, round(latency_ms, 2)
