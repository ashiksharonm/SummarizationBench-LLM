"""Model loading and single-article generation for each benchmark row."""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .data import T5_PREFIX


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def count_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


class Summarizer:
    """Wraps one model row; `summarize` returns (text, latency_seconds) for one article."""

    def __init__(self, spec: dict, device: str):
        self.spec, self.device = spec, device
        kind = spec["kind"]
        self.tok = AutoTokenizer.from_pretrained(spec["model_id"])
        if kind == "qwen":
            self.model = AutoModelForCausalLM.from_pretrained(spec["model_id"])
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(spec["model_id"])
        if kind == "t5_lora":
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, spec["adapter_path"])
            # PeftModel marks adapters non-trainable when loaded for inference; count them explicitly
            self.adapter_params = sum(
                p.numel() for n, p in self.model.named_parameters() if "lora_" in n
            )
            self.model = self.model.merge_and_unload()
        self.model.to(device).eval()
        self.kind = kind
        self.trainable, self.total = count_params(self.model)
        if kind == "t5_lora":
            # trainable during fine-tuning = LoRA params; total = base (merged) model
            self.trainable = self.adapter_params

    @torch.inference_mode()
    def summarize(self, article: str) -> tuple[str, float]:
        s, gen = self.spec, dict(self.spec.get("generate", {}))
        if self.kind == "qwen":
            msgs = [{"role": "user", "content": s["prompt"].format(article=article)}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=s["max_input_tokens"])
        else:
            prefix = T5_PREFIX if self.kind.startswith("t5") else ""
            enc = self.tok(prefix + article, return_tensors="pt", truncation=True,
                           max_length=s["max_input_tokens"])
        enc = {k: v.to(self.device) for k, v in enc.items()}
        _sync(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(**enc, **gen)
        _sync(self.device)
        dt = time.perf_counter() - t0
        if self.kind == "qwen":
            out = out[:, enc["input_ids"].shape[1]:]
        return self.tok.decode(out[0], skip_special_tokens=True).strip(), dt
