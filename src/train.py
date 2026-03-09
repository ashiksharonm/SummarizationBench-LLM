"""
train.py — Fine-tune T5-small on CNN/DailyMail using LoRA (PEFT) + HuggingFace Trainer.

WHY LoRA?
---------
Full fine-tuning of even a small model like T5-small (60M params) requires
updating all parameters with their gradients, demanding ~4–6 GB of VRAM on
a standard GPU. LoRA (Low-Rank Adaptation, Hu et al. 2021) injects trainable
low-rank matrices into selected attention layers, reducing the number of
trainable parameters by >90% while matching full fine-tuning quality on many NLP tasks.

This makes the script runnable on a free Colab T4 GPU (15 GB VRAM) with
comfortable headroom, which is the target environment.

WHAT HAPPENS HERE:
  1. Load CNN/DailyMail dataset (optionally subsampled for faster iteration)
  2. Tokenize with T5 tokenizer (task prefix "summarize: " added)
  3. Wrap T5-small with a LoRA adapter using PEFT
  4. Train with HuggingFace Trainer (AdamW, cosine LR, gradient accumulation)
  5. Log per-step training loss to losses.json for plotting
  6. Save the LoRA adapter (NOT full weights) to ./fine_tuned_t5_lora/

USAGE (local or Colab):
  python src/train.py \
      --output_dir ./fine_tuned_t5_lora \
      --train_size 5000 \
      --val_size  500 \
      --num_epochs 3 \
      --batch_size 8

Run `python src/train.py --help` for all options.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)

# Import our utils from the same package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_cnn_dailymail,
    build_tokenized_datasets,
    MAX_INPUT_TOKENS,
    MAX_TARGET_TOKENS,
)

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL = "google/t5-small"

# LoRA hyperparameters — chosen to balance parameter efficiency with quality:
#   r=8     : rank of the low-rank matrices. Higher → more capacity, more params.
#   alpha=32: LoRA scaling factor. Effective LR of adapters = alpha/r × base LR.
#   dropout : prevents overfitting in adapter layers.
#   target_modules: only query ("q") and value ("v") projection matrices in
#                   T5's attention are adapted — standard practice for seq2seq.
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q", "v"]


# ─── Loss callback — logs per-step loss to a JSON file ───────────────────────

from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    """
    Custom callback that captures the training loss at every logging step
    and writes it to a JSON file at the end of training.

    This is used to plot the loss curve in the README and Streamlit app.
    """
    def __init__(self, log_path: str = "losses.json"):
        self.log_path = log_path
        self.loss_history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after every `logging_steps` steps during training."""
        if logs and "loss" in logs:
            self.loss_history.append(
                {"step": state.global_step, "loss": round(logs["loss"], 6)}
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Dump accumulated losses to disk when training finishes."""
        with open(self.log_path, "w") as f:
            json.dump(self.loss_history, f, indent=2)
        logger.info("Training loss history saved → %s", self.log_path)


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5-small on CNN/DailyMail with LoRA/PEFT"
    )
    # Dataset
    parser.add_argument(
        "--train_size", type=int, default=None,
        help="Subsample training set to N examples (default: full ~287K)"
    )
    parser.add_argument(
        "--val_size", type=int, default=None,
        help="Subsample validation set to N examples (default: full ~13K)"
    )
    # Training
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_t5_lora",
                        help="Directory to save LoRA adapter checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device training batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate for AdamW")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW regularization")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log loss every N steps")
    parser.add_argument("--loss_log_path", type=str, default="losses.json",
                        help="Path to save per-step training loss history")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 mixed precision (requires CUDA)")
    parser.add_argument("--num_proc", type=int, default=2,
                        help="CPU workers for dataset tokenization")
    return parser.parse_args()


# ─── LoRA adapter setup ───────────────────────────────────────────────────────

def build_lora_model(base_model_name: str, lora_config: LoraConfig):
    """
    Load the base T5-small model and inject LoRA adapters.

    After get_peft_model(), only the ~0.3M adapter parameters are trainable —
    the remaining ~60M base model parameters are frozen.
    """
    logger.info("Loading base model: %s", base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    # Inject LoRA into the model
    model = get_peft_model(model, lora_config)

    # Print a summary of trainable vs. total params
    model.print_trainable_parameters()

    return model


# ─── Main training routine ────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Device detection ─────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s", device.upper())
    if device == "cpu":
        logger.warning(
            "No GPU found — training will be very slow. "
            "Consider running on Google Colab (T4 GPU, free tier)."
        )

    # ── 2. Load tokenizer ──────────────────────────────────────────────────
    logger.info("Loading tokenizer for %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # ── 3. Load and tokenize dataset ──────────────────────────────────────
    raw_dataset = load_cnn_dailymail(
        train_size=args.train_size,
        val_size=args.val_size,
    )
    logger.info(
        "Dataset sizes — train: %d, val: %d",
        len(raw_dataset["train"]),
        len(raw_dataset["validation"]),
    )

    tokenized_dataset = build_tokenized_datasets(
        raw_dataset,
        tokenizer=tokenizer,
        max_input_length=MAX_INPUT_TOKENS,
        max_target_length=MAX_TARGET_TOKENS,
        is_t5=True,  # adds "summarize: " prefix
        num_proc=args.num_proc,
    )

    # ── 4. Build LoRA config and model ────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,   # seq2seq is a special PEFT task type
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",                        # don't adapt bias terms
        inference_mode=False,               # enable training
    )

    model = build_lora_model(BASE_MODEL, lora_config)

    # ── 5. Data collator ──────────────────────────────────────────────────
    # DataCollatorForSeq2Seq handles dynamic padding within each batch,
    # which is more efficient than padding to max_length statically.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,  # ignore padding in loss
        pad_to_multiple_of=8,    # aligns tensor dims for Tensor Core efficiency
    )

    # ── 6. Training arguments ─────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        # Epochs and batching
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Effective batch size = batch_size × grad_accum × n_gpus
        # Default: 8 × 4 × 1 = 32 — good balance for Colab T4

        # Optimizer
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",  # cosine decay after warmup

        # Precision — FP16 halves VRAM usage on CUDA
        fp16=args.fp16 and device == "cuda",

        # Evaluation and checkpointing
        evaluation_strategy="epoch",   # eval at end of each epoch
        save_strategy="epoch",         # save checkpoint each epoch
        load_best_model_at_end=True,   # restore best checkpoint at end
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        report_to="none",              # disable wandb/tensorboard by default

        # Generation (for eval-time ROUGE, if desired)
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_TOKENS,

        # Misc
        save_total_limit=2,            # keep only 2 most recent checkpoints
        dataloader_num_workers=0,      # safe default for Colab
    )

    # ── 7. Trainer initialization ─────────────────────────────────────────
    loss_logger = LossLoggerCallback(log_path=args.loss_log_path)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            loss_logger,
            EarlyStoppingCallback(early_stopping_patience=2),
            # Stops training if eval_loss doesn't improve for 2 epochs
        ],
    )

    # ── 8. Train! ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting LoRA fine-tuning of %s", BASE_MODEL)
    logger.info("  Epochs:          %d", args.num_epochs)
    logger.info("  Batch size:      %d (+ %d grad_accum = %d effective)",
                args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    logger.info("  Learning rate:   %.2e", args.lr)
    logger.info("  Output dir:      %s", args.output_dir)
    logger.info("=" * 60)

    train_result = trainer.train()

    # ── 9. Save LoRA adapter ──────────────────────────────────────────────
    # IMPORTANT: We only save the LoRA adapter weights (tiny, ~10 MB),
    # NOT the full fine-tuned model (~240 MB). At inference time, we load
    # the original T5-small and merge these adapter weights on top.
    logger.info("Saving LoRA adapter → %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics summary
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("✓ Training complete.")
    logger.info("  Adapter saved to: %s", args.output_dir)
    logger.info("  Loss log saved to: %s", args.loss_log_path)
    logger.info(
        "\nNext step: Run `python src/evaluate.py` to benchmark against BART and Gemini."
    )


if __name__ == "__main__":
    main()
