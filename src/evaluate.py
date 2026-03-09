"""
evaluate.py — Benchmark T5-LoRA vs BART vs Gemini on 100 CNN/DailyMail test samples.

WHAT THIS SCRIPT DOES:
  1. Load 100 articles from the CNN/DailyMail test split
  2. Run each through all 3 models (T5-LoRA, BART, Gemini)
  3. Compute per-model ROUGE-1/2/L, BLEU, and mean inference latency
  4. Save results to results/metrics_comparison.json
  5. Print a clean comparison table to the terminal

USAGE:
  # Set your Gemini API key first:
  export GEMINI_API_KEY="your-key-here"

  # Run evaluation:
  python src/evaluate.py \
      --adapter_path ./fine_tuned_t5_lora \
      --num_samples 100 \
      --output_path ./results/metrics_comparison.json

NOTE ON FAIRNESS:
  - All 3 models receive the exact same 100 articles, in the same order
  - Generation hyperparameters (beam size, max tokens) are set identically
    where the API allows it
  - Gemini is a vastly bigger model — the comparison is intentional:
    we're measuring what a lightweight fine-tuned T5 can achieve vs.
    a proprietary frontier model
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_cnn_dailymail, clean_text
from inference import (
    load_t5_lora,
    load_bart,
    summarize_t5,
    summarize_bart,
    summarize_gemini,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Download NLTK tokenizer data once (needed for BLEU word tokenization)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─── ROUGE / BLEU helpers ─────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 using google-research/rouge_score.

    Returns a dict with keys "rouge1", "rouge2", "rougeL", each holding
    the mean F1 score across all prediction/reference pairs (0–1 scale).
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    return {k: round(sum(v) / len(v), 4) for k, v in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Compute corpus-level BLEU using NLTK.

    - Tokenizes by whitespace (matches common NLP paper conventions)
    - Uses SmoothingFunction method4 to handle zero-count n-grams
      (common for short hypothesis/reference pairs)
    """
    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_preds = [pred.split() for pred in predictions]
    smooth = SmoothingFunction().method4
    score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smooth)
    return round(score, 4)


# ─── Per-model evaluation loop ────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    articles: list[str],
    references: list[str],
    predict_fn,  # callable: article -> (summary, latency_ms)
) -> dict:
    """
    Run inference with `predict_fn` on all articles, collect summaries and
    latencies, then compute ROUGE + BLEU.

    Args:
        model_name:  Human-readable name for logging/output
        articles:    List of input article strings
        references:  List of ground-truth summaries (CNN/DailyMail highlights)
        predict_fn:  Function that takes an article and returns (summary, latency_ms)

    Returns:
        Dict with metrics and a sample of predictions for inspection
    """
    predictions = []
    latencies = []

    logger.info("Evaluating model: %s on %d samples…", model_name, len(articles))

    for article in tqdm(articles, desc=f"  {model_name}", unit="sample"):
        try:
            summary, latency_ms = predict_fn(article)
        except Exception as e:
            logger.warning("Inference failed for one sample (%s): %s", model_name, e)
            summary = ""
            latency_ms = 0.0
        predictions.append(summary)
        latencies.append(latency_ms)

    rouge_scores = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    avg_latency = round(sum(latencies) / len(latencies), 2)

    return {
        "model": model_name,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu,
        "avg_latency_ms": avg_latency,
        "num_samples": len(articles),
        # Save a few sample predictions for qualitative review in README
        "sample_predictions": [
            {
                "article_snippet": articles[i][:300] + "…",
                "reference": references[i],
                "prediction": predictions[i],
            }
            for i in range(min(3, len(articles)))
        ],
    }


# ─── Pretty-print table ───────────────────────────────────────────────────────

def print_results_table(results: list[dict]):
    """Print a clean terminal table summarizing all model metrics."""
    border = "─" * 85
    header = f"{'Model':<25} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BLEU':>8} {'Latency (ms)':>14}"
    print(f"\n{border}")
    print("  📊  SUMMARIZATION BENCHMARK RESULTS")
    print(border)
    print(f"  {header}")
    print(border)
    for r in results:
        print(
            f"  {r['model']:<25} "
            f"{r['rouge1']:>8.4f} "
            f"{r['rouge2']:>8.4f} "
            f"{r['rougeL']:>8.4f} "
            f"{r['bleu']:>8.4f} "
            f"{r['avg_latency_ms']:>14.1f}"
        )
    print(f"{border}\n")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate T5-LoRA, BART, and Gemini on CNN/DailyMail"
    )
    parser.add_argument(
        "--adapter_path", type=str, default="./fine_tuned_t5_lora",
        help="Path to T5-LoRA adapter directory (output of train.py)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of test samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--output_path", type=str, default="./results/metrics_comparison.json",
        help="Where to save the evaluation results JSON"
    )
    parser.add_argument(
        "--skip_gemini", action="store_true",
        help="Skip Gemini evaluation (if no API key available)"
    )
    parser.add_argument(
        "--gemini_api_key", type=str, default=None,
        help="Gemini API key (falls back to GEMINI_API_KEY env var)"
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load 100 test samples ──────────────────────────────────────────
    logger.info("Loading test data: %d samples from CNN/DailyMail", args.num_samples)
    dataset = load_cnn_dailymail(test_size=args.num_samples)
    test_data = dataset["test"]

    articles = [clean_text(row["article"]) for row in test_data]
    references = [clean_text(row["highlights"]) for row in test_data]

    logger.info("Loaded %d test articles.", len(articles))

    all_results = []

    # ── 2. Evaluate T5-LoRA ───────────────────────────────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t5_model, t5_tokenizer = load_t5_lora(args.adapter_path, device=device)

    t5_result = evaluate_model(
        model_name="T5-small (LoRA fine-tuned)",
        articles=articles,
        references=references,
        predict_fn=lambda text: summarize_t5(text, t5_model, t5_tokenizer, device=device),
    )
    all_results.append(t5_result)

    # ── 3. Evaluate BART ──────────────────────────────────────────────────
    bart_model, bart_tokenizer = load_bart(device=device)

    bart_result = evaluate_model(
        model_name="BART-large-CNN (pretrained)",
        articles=articles,
        references=references,
        predict_fn=lambda text: summarize_bart(text, bart_model, bart_tokenizer, device=device),
    )
    all_results.append(bart_result)

    # ── 4. Evaluate Gemini (optional) ─────────────────────────────────────
    if not args.skip_gemini:
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set. Skipping Gemini evaluation. "
                "Pass --skip_gemini to suppress this warning."
            )
        else:
            gemini_result = evaluate_model(
                model_name="Gemini 1.5 Pro",
                articles=articles,
                references=references,
                predict_fn=lambda text: summarize_gemini(text, api_key=api_key),
            )
            all_results.append(gemini_result)
    else:
        logger.info("Gemini evaluation skipped (--skip_gemini set).")

    # ── 5. Print results table ────────────────────────────────────────────
    print_results_table(all_results)

    # ── 6. Save to JSON ───────────────────────────────────────────────────
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    logger.info("✓ Evaluation complete. Results saved → %s", output_path)
    logger.info("\nNext step: Run `streamlit run app/app.py` to launch the dashboard.")


if __name__ == "__main__":
    main()
