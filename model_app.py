# -*- coding: utf-8 -*-
"""model_app

Streamlit app for ML Q&A evaluation:
- Automatic evaluator: ROUGE-L + keyword coverage
- LLM-based evaluator: Hugging Face (Local FLAN-T5 or Hosted Inference API)

Before running:
  1) pip install streamlit evaluate rouge-score pandas transformers huggingface_hub
  2) (Optional) Add a secrets file for hosted mode:
       In Streamlit: Settings -> Secrets -> add:
         HF_TOKEN = "hf_xxx..."
  3) streamlit run model_app.py
"""

import json
from pathlib import Path
from collections import Counter
import random
import textwrap
import re
import os

import streamlit as st
import evaluate
import pandas as pd  # optional (kept for parity with prior code)

# For local Transformers and for HF Hosted Inference API
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from huggingface_hub import InferenceClient


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
    ranked by frequency (naive but fine for the assignment).
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
    - ROUGE-L
    - Keyword coverage
    - Combined numeric score (0-100)
    - Text explanation
    """
    rouge_l = compute_rouge_l(reference, student_answer)
    coverage, present_kw, missing_kw = keyword_coverage(reference, student_answer)

    alpha = 0.5  # weight for ROUGE-L
    beta = 0.5   # weight for keyword coverage

    combined = alpha * rouge_l + beta * coverage
    score_0_100 = round(combined * 100, 1)

    explanation_parts = []
    explanation_parts.append(
        f"ROUGE-L similarity: {rouge_l:.3f} (0–1 scale). Reflects overlap in phrasing/structure."
    )
    explanation_parts.append(
        f"Keyword coverage: {coverage:.3f} (0–1 scale). Reflects how many core terms you mentioned."
    )

    if missing_kw:
        missing_str = ", ".join(missing_kw[:8])
        explanation_parts.append(f"Missing important concepts: {missing_str}.")
    if present_kw:
        present_str = ", ".join(present_kw[:8])
        explanation_parts.append(f"Covered concepts: {present_str}.")

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

"""
Two modes:
  A) Local CPU model (no API key):
     - google/flan-t5-base  (fast & light; good enough for JSON judging)
  B) Hosted HF Inference API (requires HF_TOKEN in Streamlit secrets):
     - default: meta-llama/Meta-Llama-3.1-8B-Instruct (stronger judge)
"""

LOCAL_MODEL_ID = "google/flan-t5-base"
HOSTED_MODEL_ID_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # you can change in the UI

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
""".strip()


def build_llm_prompt(question: str, reference_answer: str, student_answer: str) -> str:
    # A simple generic prompt that works for both seq2seq and chat models
    return f"""
[SYSTEM]
{SYSTEM_INSTRUCTIONS}

[QUESTION]
{question}

[REFERENCE_ANSWER]
{reference_answer}

[STUDENT_ANSWER]
{student_answer}

Return STRICT JSON only.
""".strip()


# ---------- Mode A: Local (FLAN-T5) ----------
@st.cache_resource(show_spinner=True)
def load_local_judge(model_id: str = LOCAL_MODEL_ID):
    """
    Loads a local seq2seq model for judging (FLAN-T5-base) on CPU or MPS if available.
    Works on Macs without CUDA/bitsandbytes.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Device handling:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)
    return tokenizer, model, device


def generate_local_json_judgement(tokenizer, model, device, prompt: str,
                                  max_new_tokens: int = 384, temperature: float = 0.0) -> str:
    """
    Uses the local FLAN-T5 to generate a JSON judgement string.
    """
    # FLAN-T5 doesn't use temperature directly; emulate with do_sample if > 0
    do_sample = temperature > 0.0
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=0.9 if do_sample else None,
            temperature=temperature if do_sample else None,
            num_beams=None if do_sample else 4
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ---------- Mode B: Hosted (HF Inference API) ----------
def get_hf_client(api_token: str | None):
    if not api_token:
        return None
    return InferenceClient(token=api_token)


def generate_hosted_json_judgement(client: InferenceClient, model_id: str, prompt: str,
                                   max_new_tokens: int = 384, temperature: float = 0.1, top_p: float = 0.9) -> str:
    """
    Calls Hugging Face Inference API text generation endpoint.
    """
    # For chat/instruct models, plain prompt works; we enforce JSON via instructions.
    resp = client.text_generation(
        model=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
        return_full_text=False,
    )
    # `resp` is a string here
    return resp.strip()


def parse_first_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def evaluate_answer_llm(question: str, reference: str, student_answer: str,
                        mode: str,
                        hosted_model_id: str,
                        temperature: float,
                        max_new_tokens: int) -> dict:
    """
    LLM judge using Hugging Face (Local FLAN-T5 or Hosted Inference API).
    Returns: {"score": float, "analysis": str, "missing_points": list}
    """
    prompt = build_llm_prompt(question, reference, student_answer)

    if mode == "Hosted (HF Inference API)":
        hf_token = st.secrets.get("HF_TOKEN", None)
        client = get_hf_client(hf_token)
        if client is None:
            return {
                "score": 0.0,
                "analysis": "HF Inference API not configured. Add HF_TOKEN in Streamlit secrets or use Local mode.",
                "missing_points": [],
            }
        raw = generate_hosted_json_judgement(
            client=client,
            model_id=hosted_model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9
        )
    else:
        # Local mode (FLAN-T5-base)
        tokenizer, model, device = load_local_judge(LOCAL_MODEL_ID)
        raw = generate_local_json_judgement(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    data = parse_first_json(raw)
    if data:
        try:
            score = float(data.get("score", 0))
            analysis = str(data.get("analysis", "")).strip()
            missing_points = data.get("missing_points", [])
            if not isinstance(missing_points, list):
                missing_points = []
            score = max(0.0, min(100.0, score))
            return {"score": score, "analysis": analysis, "missing_points": missing_points}
        except Exception:
            pass

    # Fallback if no strict JSON
    return {
        "score": 0.0,
        "analysis": (
            "Could not parse strict JSON from the model output. "
            "Try lowering temperature, increasing max_new_tokens, or refining the prompt."
        ),
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
    page_title="ML Q&A Evaluator (Hugging Face)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Concept Q&A Evaluator (Hugging Face)")
st.write(
    "This prototype asks you questions about Machine Learning concepts, "
    "collects your answer, and evaluates it automatically (ROUGE + keyword coverage). "
    "It also provides an LLM-based evaluation (Hugging Face: Local FLAN-T5 or Hosted API)."
)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 New random question"):
        st.session_state.current_qa = sample_question(qa_items)
        for key in ("last_answer", "last_eval_auto", "last_eval_llm"):
            if key in st.session_state:
                del st.session_state[key]

    if st.button("🧼 Reset session"):
        st.session_state.current_qa = None
        st.session_state.history = []
        for key in ("last_answer", "last_eval_auto", "last_eval_llm"):
            if key in st.session_state:
                del st.session_state[key]

    st.markdown("---")
    st.subheader("LLM Judge Settings (Hugging Face)")
    judge_mode = st.selectbox(
        "Select judge mode",
        options=["Local (FLAN-T5 base)", "Hosted (HF Inference API)"],
        index=0,
        help="Use local CPU model (no key) or Hosted HF (requires HF_TOKEN in secrets)."
    )

    hosted_model_id = st.text_input(
        "Hosted model id (HF Inference API)",
        value=HOSTED_MODEL_ID_DEFAULT,
        help="Used only in Hosted mode. Example: meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_new_tokens = st.slider("Max new tokens", min_value=64, max_value=1024, value=384, step=64)

    st.markdown("---")
    st.subheader("Session stats")
    st.write(f"Questions answered: **{len(st.session_state.history)}**")
    if st.session_state.history:
        avg_score = sum(h["evaluation"]["automatic"]["score"] for h in st.session_state.history) / len(st.session_state.history)
        st.write(f"Average automatic score: **{avg_score:.1f} / 100**")

    st.markdown("---")
    st.caption("Prototype for Assignment 11.00 – LLM Evaluator (Streamlit + Hugging Face).")


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

    # 2) LLM-based evaluation (Hugging Face)
    eval_llm = evaluate_answer_llm(
        question=question,
        reference=reference_answer,
        student_answer=student_answer,
        mode=judge_mode,
        hosted_model_id=hosted_model_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

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
