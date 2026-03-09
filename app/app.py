"""
app.py — Streamlit 3-way summarization comparison dashboard.

This app provides:
  - Text input for pasting any article
  - One-click "Summarize & Compare" button
  - 3 summary panels side by side: T5-LoRA | BART | Gemini
  - ROUGE-1/2/L, BLEU metrics for this specific input (computed live)
  - Plotly bar chart comparing model latencies
  - Sidebar with training details, dataset info, and what LoRA means

SETUP:
  1. Train T5-LoRA first: python src/train.py
  2. Set GEMINI_API_KEY in environment or .streamlit/secrets.toml
  3. Run: streamlit run app/app.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SummarizationBench-LLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .model-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        height: 100%;
    }
    .model-name {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .t5-color  { color: #7c6af7; }
    .bart-color { color: #f76a6a; }
    .gem-color  { color: #6af7a0; }
    .summary-text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #ccc;
    }
    .latency-badge {
        display: inline-block;
        font-size: 0.72rem;
        background: #2a2a3e;
        border: 1px solid #444;
        border-radius: 20px;
        padding: 2px 10px;
        margin-top: 0.7rem;
        color: #aaa;
    }
    .metric-box {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        text-align: center;
    }
    .lora-explainer {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #7c6af7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1.5rem;
        font-size: 0.85rem;
        color: #bbb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 SummarizationBench-LLM")
    st.markdown("---")

    st.markdown("### 📚 Dataset")
    st.markdown(
        "**CNN / DailyMail** (v3.0.0)  \n"
        "~287K training articles  \n"
        "Evaluation on 100 held-out test samples"
    )

    st.markdown("### ⚙️ Training Config")
    st.markdown(
        "**Base model:** `google/t5-small` (60M params)  \n"
        "**Method:** LoRA (PEFT)  \n"
        "**LoRA rank (r):** 8  \n"
        "**Target layers:** `q`, `v` (attention)  \n"
        "**Epochs:** 3  \n"
        "**Batch size:** 8 (×4 grad_accum = 32)  \n"
        "**Optimizer:** AdamW + cosine decay  "
    )

    st.markdown("### 🤖 Models Compared")
    st.markdown(
        "1. 🟣 **T5-small (LoRA)** — *Fine-tuned*  \n"
        "2. 🔴 **BART-large-CNN** — *Pretrained baseline*  \n"
        "3. 🟢 **Gemini 1.5 Pro** — *Generative frontier*"
    )

    # Load saved benchmark results if available
    st.markdown("---")
    st.markdown("### 📊 Benchmark Results")
    results_path = Path(__file__).resolve().parent.parent / "results" / "metrics_comparison.json"
    if results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
        df_saved = pd.DataFrame([
            {
                "Model": r["model"],
                "ROUGE-L": r["rougeL"],
                "BLEU": r["bleu"],
                "Latency (ms)": r["avg_latency_ms"],
            }
            for r in saved["results"]
        ])
        st.dataframe(df_saved, hide_index=True, use_container_width=True)
    else:
        st.info("Run `evaluate.py` first to see benchmark results here.")

    st.markdown("---")
    st.caption("Built by Ashik Sharon · [GitHub](https://github.com/ashiksharonm/SummarizationBench-LLM)")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 SummarizationBench-LLM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fine-tuned T5-small (LoRA) · BART-large · Gemini 1.5 Pro — live 3-way comparison</div>',
    unsafe_allow_html=True,
)


# ─── Model loading (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading T5-LoRA adapter…")
def get_t5_model():
    from inference import load_t5_lora
    adapter_path = str(Path(__file__).resolve().parent.parent / "fine_tuned_t5_lora")
    if not Path(adapter_path).exists():
        return None, None
    return load_t5_lora(adapter_path)


@st.cache_resource(show_spinner="Loading BART model…")
def get_bart_model():
    from inference import load_bart
    return load_bart()


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def compute_live_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE scores for a single prediction/reference pair."""
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    result = scorer.score(reference, prediction)
    return {k: round(v.fmeasure, 4) for k, v in result.items()}


def compute_live_bleu(prediction: str, reference: str) -> float:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download("punkt", quiet=True)
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    smooth = SmoothingFunction().method4
    return round(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth), 4)


# ─── Main Input Area ──────────────────────────────────────────────────────────
st.markdown("### 📝 Input Article")

default_text = (
    "Scientists at MIT have developed a new biodegradable plastic that decomposes "
    "within 30 days in soil. The material, made from corn starch and a new polymer "
    "compound, retains the same durability as conventional plastic during use. "
    "The team believes it could replace single-use plastics in packaging within five years. "
    "The research was published Monday in Nature Materials."
)

article_input = st.text_area(
    label="Paste any news article here:",
    value=default_text,
    height=200,
    label_visibility="collapsed",
    placeholder="Paste a news article to summarize…",
)

reference_input = st.text_input(
    "Reference summary (optional, for live ROUGE/BLEU):",
    placeholder="Paste the reference/gold summary here for live metric calculation…",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_button = st.button("⚡ Summarize & Compare", type="primary", use_container_width=True)
with col_clear:
    st.markdown("")  # spacer


# ─── Inference and display ────────────────────────────────────────────────────

if run_button and article_input.strip():

    gemini_api_key = (
        os.environ.get("GEMINI_API_KEY")
        or st.secrets.get("GEMINI_API_KEY", None)
    )

    with st.spinner("Running inference across all 3 models…"):

        results = {}

        # — T5-LoRA —
        t5_model, t5_tokenizer = get_t5_model()
        if t5_model is None:
            results["t5"] = {
                "summary": "⚠️ T5-LoRA adapter not found. Run `train.py` first.",
                "latency": 0,
            }
        else:
            import torch
            from inference import summarize_t5
            device = "cuda" if torch.cuda.is_available() else "cpu"
            t5_summary, t5_latency = summarize_t5(article_input, t5_model, t5_tokenizer, device=device)
            results["t5"] = {"summary": t5_summary, "latency": t5_latency}

        # — BART —
        try:
            bart_pipeline = get_bart_model()
            from inference import summarize_bart
            bart_summary, bart_latency = summarize_bart(article_input, bart_pipeline)
            results["bart"] = {"summary": bart_summary, "latency": bart_latency}
        except Exception as e:
            results["bart"] = {"summary": f"⚠️ BART error: {e}", "latency": 0}

        # — Gemini —
        if gemini_api_key:
            try:
                from inference import summarize_gemini
                gem_summary, gem_latency = summarize_gemini(article_input, api_key=gemini_api_key)
                results["gemini"] = {"summary": gem_summary, "latency": gem_latency}
            except Exception as e:
                results["gemini"] = {"summary": f"⚠️ Gemini error: {e}", "latency": 0}
        else:
            results["gemini"] = {
                "summary": "⚠️ GEMINI_API_KEY not set. Add it to your environment or .streamlit/secrets.toml.",
                "latency": 0,
            }

    # ── Summary cards ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Summaries")

    col1, col2, col3 = st.columns(3)
    cards = [
        ("T5-small (LoRA fine-tuned)", "🟣", "t5-color", "t5"),
        ("BART-large-CNN", "🔴", "bart-color", "bart"),
        ("Gemini 1.5 Pro", "🟢", "gem-color", "gemini"),
    ]
    for col, (model_name, emoji, css_class, key) in zip([col1, col2, col3], cards):
        with col:
            r = results[key]
            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name {css_class}">{emoji} {model_name}</div>
                    <div class="summary-text">{r['summary']}</div>
                    <div class="latency-badge">⏱ {r['latency']:.1f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Live metrics table (if reference provided) ────────────────────────
    if reference_input.strip():
        st.markdown("---")
        st.markdown("### 📊 Live Metrics (vs. your reference)")

        metric_rows = []
        for label, key in [("T5-LoRA", "t5"), ("BART", "bart"), ("Gemini", "gemini")]:
            if results[key]["latency"] > 0:  # skip errored models
                rouge = compute_live_rouge(results[key]["summary"], reference_input)
                bleu = compute_live_bleu(results[key]["summary"], reference_input)
                metric_rows.append({
                    "Model": label,
                    "ROUGE-1": rouge["rouge1"],
                    "ROUGE-2": rouge["rouge2"],
                    "ROUGE-L": rouge["rougeL"],
                    "BLEU": bleu,
                    "Latency (ms)": results[key]["latency"],
                })

        if metric_rows:
            df = pd.DataFrame(metric_rows)
            st.dataframe(df.style.highlight_max(
                subset=["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"], color="#2a4a2a"
            ), hide_index=True, use_container_width=True)

    # ── Latency bar chart ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⏱ Model Latency Comparison")

    model_labels = ["T5-LoRA", "BART-large", "Gemini 1.5 Pro"]
    latency_vals = [results["t5"]["latency"], results["bart"]["latency"], results["gemini"]["latency"]]
    colors = ["#7c6af7", "#f76a6a", "#6af7a0"]

    fig = go.Figure(
        go.Bar(
            x=model_labels,
            y=latency_vals,
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.1)", width=1),
            ),
            text=[f"{v:.0f} ms" for v in latency_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="Inference Latency (ms)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        margin=dict(t=20, b=20),
        height=380,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig, use_container_width=True)

    # ── LoRA explainer ────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="lora-explainer">
        <strong>💡 What is LoRA fine-tuning?</strong><br>
        LoRA (Low-Rank Adaptation) freezes the original model weights and injects small,
        trainable low-rank matrices into the attention layers. Instead of updating all
        ~60M parameters of T5-small, only ~0.3M adapter params are trained — making
        fine-tuning feasible on a free Colab T4 GPU. At inference time, the adapter
        weights are merged into the base model with zero latency overhead.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif run_button:
    st.warning("Please paste some article text before clicking Summarize.")
