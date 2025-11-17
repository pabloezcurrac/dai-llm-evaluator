# -*- coding: utf-8 -*-
"""model_app

Streamlit app for ML Q&A evaluation:
- Automatic evaluator: ROUGE-L + keyword coverage
- LLM-based evaluator: Hugging Face local model (e.g., Qwen2.5-7B-Instruct)

Requires:
  - streamlit
  - evaluate, rouge-score
  - transformers, torch
  - bitsandbytes (optional; for 4-bit/8-bit on GPU)
"""

import json
from pathlib import Path
from collections import Counter
import random
import textwrap
import re

import streamlit as st
import evaluate
import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Try to import BitsAndBytes; if unavailable (CPU env), we will load without quantization
try:
    from transformers import BitsAndBytesConfig  # type: ignore
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


# ===============================
# 1. DATA LOADING
# ===============================

@st.cache_data
def load_qa_data(path: str = "Q&A_db_practice.json"):
    data_path = Path(path)
    if not data_path.exists():
        st.error(f"Could not find {path}. Please make sure the file is in the same folder as model_app.py.")
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    return qa_items


qa_items = load_qa_data()

if not qa_items:
    st.stop()


# ===============================
# 2. AUTOMATIC EVALUATOR
# ===============================

# Initialize ROUGE metric
rouge = evaluate.load("rouge")


def compute_rouge_l(reference: str, prediction: str) -> float:
    """
    Compute ROUGE-L F1 score between reference and prediction.
    Returns a value between 0 and 1.
    """
    results = rouge.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=True
    )
    return results["rougeL"]


def extract_keywords(text: str, min_len: int = 4, top_k: int = 20):
    """
    Extract simple keyword candidates: lowercase tokens with at least min_len chars,
    ranked by frequency (very naive but sufficient for this assignment).
    """
    tokens = [t.strip(".,;:!?()[]\"'").lower() for t in text.split()]
    tokens = [t for t in tokens if len(t) >= min_len]
    freq = Counter(tokens)
    most_common = [w for w, _ in freq.most_common(top_k)]
    return most_common


def keyword_coverage(reference: str, prediction: str, top_k: int = 20):
    """
    Compute coverage of important reference keywords in the student's prediction.
    Returns:
        coverage_ratio (0-1),
        present_keywords,
        missing_keywords
    """
    ref_keywords = extract_keywords(reference, top_k=top_k)
    pred_tokens = set(
        t.strip(".,;:!?()[]\"'").lower() for t in prediction.split()
    )

    present = [w for w in ref_keywords if w in pred_tokens]
    missing = [w for w in ref_keywords if w not in pred_tokens]

    coverage = len(present) / max(1, len(ref_keywords))
    return coverage, present, missing


def evaluate_answer(reference: str, student_answer: str) -> dict:
    """
    Automatic evaluator:
    - Computes ROUGE-L
    - Computes keyword coverage
    - Produces numeric score (0-100)
    - Produces textual explanation
    """
    # 1) ROUGE-L (0-1)
    rouge_l = compute_rouge_l(reference, student_answer)

    # 2) Keyword coverage (0-1)
    coverage, present_kw, missing_kw = keyword_coverage(reference, student_answer)

    # 3) Combine (simple linear combination)
    alpha = 0.5  # weight for ROUGE-L
    beta = 0.5   # weight for keyword coverage

    combined = alpha * rouge_l + beta * coverage
    score_0_100 = round(combined * 100, 1)

    # 4) Explanation
    explanation_parts = []
    explanation_parts.append(
        f"ROUGE-L similarity: {rouge_l:.3f} (0–1 scale). "
        f"This reflects overlap in phrasing and sentence structure."
    )
    explanation_parts.append(
        f"Keyword coverage: {coverage:.3f} (0–1 scale). "
        f"This reflects how many core terms from the reference you mentioned."
    )

    if missing_kw:
        missing_str = ", ".join(missing_kw[:8])
        explanation_parts.append(
            f"Some important concepts not explicitly mentioned: {missing_str}."
        )
    if present_kw:
        present_str = ", ".join(present_kw[:8])
        explanation_parts.append(
            f"Key concepts you did mention: {present_str}."
        )

    explanation = " ".join(explanation_parts)

    return {
        "score": score_0_100,
        "rouge_l": rouge_l,
        "coverage": coverage,
        "present_keywords": present_kw,
        "missing_keywords": missing_kw,
        "explanation": explanation,
    }


# ===============================
# 3. LLM-BASED EVALUATOR (Hugging Face)
# ===============================

# Recommended model: multilingual, instruction-tuned, fits 1x 16GB with 4-bit
HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # Alternatives: "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3.1-8B-Instruct"
USE_4BIT = True        # If GPU + bitsandbytes available, 4-bit for lower VRAM
MAX_NEW_TOKENS = 384
TEMPERATURE = 0.1
TOP_P = 0.9

SYSTEM_INSTRUCTIONS = """
You are a rigorous and fair university-level Machine Learning professor.

Compare the STUDENT_ANSWER against the REFERENCE_ANSWER.
Return STRICT JSON ONLY with the following fields:
{
  "score": <integer 0-100>,
  "analysis": "<short (3-6 lines) explanation with strengths and weaknesses>",
  "missing_points": ["point1","point2"]
}

Scoring policy:
- 90-100: Excellent, nearly identical to reference.
- 70-89: Good, covers most key ideas with minor gaps.
- 40-69: Partial, some correct ideas but incomplete.
- 0-39: Poor, largely incorrect or missing key ideas.

Do not output anything outside the JSON object.
"""

def _maybe_set_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback: add a pad token
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

@st.cache_resource(show_spinner=True)
def load_hf_evaluator(model_id: str = HF_MODEL_ID, use_4bit: bool = USE_4BIT):
    """
    Loads the HF model (quantized if possible) and caches it in Streamlit.
    - If GPU + bitsandbytes is available: 4-bit/8-bit quantized.
    - Otherwise: standard FP16/BF16 (GPU) or FP32 (CPU).
    """
    # Choose dtype
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16  # bf16 is stable on recent GPUs
    else:
        compute_dtype = torch.float32

    quantization_config = None
    if torch.cuda.is_available() and _HAS_BNB:
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    _maybe_set_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=compute_dtype if torch.cuda.is_available() else None,
        quantization_config=quantization_config
    )
    # If we added a pad token dynamically, resize embeddings
    if tokenizer.pad_token_id is not None and model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

tokenizer_hf, model_hf = load_hf_evaluator()

def build_llm_prompt(question: str, reference_answer: str, student_answer: str) -> str:
    return f"""
[SYSTEM]
{SYSTEM_INSTRUCTIONS.strip()}

[QUESTION]
{question}

[REFERENCE_ANSWER]
{reference_answer}

[STUDENT_ANSWER]
{student_answer}

Return STRICT JSON only.
""".strip()

@torch.inference_mode()
def generate_hf(prompt: str) -> str:
    """
    Generic generate function for instruction models.
    """
    inputs = tokenizer_hf(prompt, return_tensors="pt")
    inputs = {k: v.to(model_hf.device) for k, v in inputs.items()}
    output_ids = model_hf.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer_hf.pad_token_id,
        eos_token_id=getattr(tokenizer_hf, "eos_token_id", None),
    )
    text = tokenizer_hf.decode(output_ids[0], skip_special_tokens=True)
    # Try to strip the prompt prefix if the model echoed it
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

def evaluate_answer_llm(question: str, reference: str, student_answer: str) -> dict:
    """
    LLM judge using a Hugging Face model.
    Returns: {"score": float, "analysis": str, "missing_points": list}
    """
    prompt = build_llm_prompt(question, reference, student_answer)
    raw = generate_hf(prompt)

    # Extract the first JSON object found
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            score = float(data.get("score", 0))
            analysis = str(data.get("analysis", "")).strip()
            missing_points = data.get("missing_points", [])
            if not isinstance(missing_points, list):
                missing_points = []
            # Clamp score
            score = max(0.0, min(100.0, score))
            return {"score": score, "analysis": analysis, "missing_points": missing_points}
        except Exception:
            pass

    # Robust fallback
    return {
        "score": 0.0,
        "analysis": "Could not parse strict JSON from the LLM output. Try lowering temperature, increasing MAX_NEW_TOKENS, or refining the prompt.",
        "missing_points": [],
    }


# ===============================
# 4. QUESTION SAMPLING & STATE
# ===============================

def sample_question(qa_list):
    return random.choice(qa_list)


if "current_qa" not in st.session_state:
    st.session_state.current_qa = None

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts


# ===============================
# 5. STREAMLIT UI
# ===============================

st.set_page_config(
    page_title="ML Q&A Evaluator",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Concept Q&A Evaluator")
st.write(
    "This prototype asks you questions about Machine Learning concepts, "
    "collects your answer, and evaluates it automatically (ROUGE + keyword coverage). "
    "It also provides a Hugging Face LLM-based evaluation."
)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 New random question"):
        st.session_state.current_qa = sample_question(qa_items)
        # Clear previous answer/result for clarity
        if "last_answer" in st.session_state:
            del st.session_state["last_answer"]
        if "last_eval_auto" in st.session_state:
            del st.session_state["last_eval_auto"]
        if "last_eval_llm" in st.session_state:
            del st.session_state["last_eval_llm"]

    if st.button("🧼 Reset session"):
        st.session_state.current_qa = None
        st.session_state.history = []
        for key in ("last_answer", "last_eval_auto", "last_eval_llm"):
            if key in st.session_state:
                del st.session_state[key]

    st.markdown("---")
    st.subheader("Session stats")
    st.write(f"Questions answered: **{len(st.session_state.history)}**")
    if st.session_state.history:
        avg_score = sum(h["evaluation"]["automatic"]["score"] for h in st.session_state.history) / len(st.session_state.history)
        st.write(f"Average automatic score: **{avg_score:.1f} / 100**")

    st.markdown("---")
    st.caption("Prototype for Assignment 11.00 – LLM Evaluator (Streamlit UI).")


# If there is no current question, sample one
if st.session_state.current_qa is None:
    st.session_state.current_qa = sample_question(qa_items)

current_qa = st.session_state.current_qa
question = current_qa["question"]
reference_answer = current_qa["answer"]

st.subheader("Current Question")
st.write(question)

st.markdown("### Your Answer")
default_text = st.session_state.get("last_answer", "")
student_answer = st.text_area(
    "Type your answer here:",
    value=default_text,
    height=160,
    placeholder="Write your explanation in your own words...",
)

col1, col2 = st.columns([1, 2])

with col1:
    submit_clicked = st.button("✅ Submit answer")

with col2:
    show_reference = st.checkbox("Show reference answer after evaluation")


if submit_clicked and student_answer.strip():
    # Store last answer
    st.session_state.last_answer = student_answer

    # 1) Automatic evaluation
    eval_auto = evaluate_answer(reference_answer, student_answer)

    # 2) LLM-based evaluation (Hugging Face judge)
    eval_llm = evaluate_answer_llm(question, reference_answer, student_answer)

    # Store last evals in session
    st.session_state.last_eval_auto = eval_auto
    st.session_state.last_eval_llm = eval_llm

    # Add to history
    st.session_state.history.append(
        {
            "question": question,
            "reference_answer": reference_answer,
            "student_answer": student_answer,
            "evaluation": {
                "automatic": eval_auto,
                "llm": eval_llm,
            },
        }
    )

# Display results if available
if "last_eval_auto" in st.session_state:
    eval_auto = st.session_state.last_eval_auto
    eval_llm = st.session_state.last_eval_llm

    st.markdown("## Evaluation Results")

    # Automatic evaluation
    st.markdown("### 🔍 Automatic Evaluation (ROUGE + Keywords)")
    st.write(f"**Score (0–100):** `{eval_auto['score']}`")
    st.write(textwrap.fill(eval_auto["explanation"], width=100))

    # LLM-based evaluation
    st.markdown("### 🧠 LLM-based Evaluation (Hugging Face Judge)")
    st.write(f"**LLM Score (0–100):** `{eval_llm['score']}`")
    st.write(textwrap.fill(eval_llm["analysis"], width=100))
    if eval_llm.get("missing_points"):
        st.write("**Key missing points (according to the LLM):**")
        st.write("- " + "\n- ".join(eval_llm["missing_points"]))

    # Reference answer (optional)
    if show_reference:
        st.markdown("### 📘 Reference Answer")
        st.write(textwrap.fill(reference_answer, width=100))
