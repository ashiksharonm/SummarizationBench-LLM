"""Minimal LoRA fine-tuning of T5-small on CNN/DailyMail (single device, plain torch loop)."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import DATASET, T5_PREFIX, VERSION
from .models import pick_device


def train(cfg: dict, seed: int) -> dict:
    t = cfg["train"]
    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(t["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(t["base_model"])
    lc = t["lora"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=lc["r"], lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"], target_modules=lc["target_modules"], bias="none"))
    model.print_trainable_parameters()
    model.to(device).train()

    ds = load_dataset(DATASET, VERSION, split="train").shuffle(seed=seed).select(range(t["train_examples"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=0.01)
    bs = t["batch_size"]
    steps_per_epoch = len(ds) // bs
    total_steps = steps_per_epoch * t["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / total_steps))

    log, step, t0, stopped_early = [], 0, time.time(), False
    for _ in range(t["epochs"]):
        for b in range(steps_per_epoch):
            rows = ds[b * bs:(b + 1) * bs]
            enc = tok([T5_PREFIX + a for a in rows["article"]], max_length=t["max_input_tokens"],
                      truncation=True, padding=True, return_tensors="pt")
            lab = tok(text_target=rows["highlights"], max_length=t["max_target_tokens"],
                      truncation=True, padding=True, return_tensors="pt")["input_ids"]
            lab[lab == tok.pad_token_id] = -100
            batch = {k: v.to(device) for k, v in enc.items()}
            loss = model(**batch, labels=lab.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                log.append({"step": step, "loss": round(loss.item(), 4),
                            "elapsed_s": round(time.time() - t0)})
                print(log[-1], flush=True)
            if time.time() - t0 > t["time_budget_min"] * 60:
                stopped_early = True
                break
        if stopped_early:
            break
    out = Path(t["adapter_path"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    info = {"steps": step, "planned_steps": total_steps, "stopped_by_time_budget": stopped_early,
            "train_examples": t["train_examples"], "batch_size": bs, "device": device,
            "wall_clock_min": round((time.time() - t0) / 60, 1), "seed": seed, "loss_log": log}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    c = yaml.safe_load(open(a.config))
    train(c, c["seed"])
