# -*- coding: utf-8 -*-
"""model_app (HF Hosted - Mistral locked)

Streamlit app for ML Q&A evaluation:
- Automatic evaluator: ROUGE-L + keyword coverage
- LLM judge: Hugging Face Inference API (Hosted)
  Model: mistralai/Mistral-7B-Instruct-v0.3 (open access)

Setup:
  1) pip install streamlit evaluate rouge-score pandas huggingface_hub
  2) Add Streamlit secret: HF_TOKEN = "hf_xxx..."
  3) streamlit run model_app.py
"""

import json
from pathlib import Path
from collections import Counter
import random
import textwrap
import re

import streamlit as st
import evaluate
import pandas as pd  # optional for future tables

from huggingface_hub import InferenceClient


# ===============================
# 1. DATA LOADING
# ===============================

@st.cache_data
def load_qa_data(path: str = "Q&A_db_practice.json"):
    data_path = Path(path)
    if not data_path.exists():
        st.error(f"Could not find {path}. Place Q&A_db_practice.json next to model_app.py.")
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
    ROUGE-L F1 between reference and prediction (0..1).
    """
    results = rouge.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=True
    )
    return results["rougeL"]

def extract_keywords(text: str, min_len: int = 4, top_k: int = 20):
    """
    Naive keyword candidates by frequency.
    """
    tokens = [t.strip(".,;:!?()[]\"'").lower() for t in text.split()]
    tokens = [t for t in tokens if len(t) >= min_len]
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_k)]

def keyword_coverage(reference: str, prediction: str, top_k: int = 20):
    """
    Coverage of reference keywords in prediction.
    Returns: (coverage_ratio, present_keywords, missing_keywords)
    """
    ref_keywords = extract_keywords(reference, top_k=top_k)
    pred_tokens = set(t.strip(".,;:!?()[]\"'").lower() for t in prediction.split())
    present = [w for w in ref_keywords if w in pred_tokens]
    missing = [w for w in ref_keywords if w not in pred_tokens]
    coverage = len(present) / max(1, len(ref_keywords))
    return coverage, present, missing

def evaluate_answer(reference: str, student_answer: str) -> dict:
    """
    Automatic evaluator:
    - ROUGE-L + keyword coverage
    - Combined score (0..100)
    - Plain-English explanation
    """
    rouge_l = compute_rouge_l(reference, student_answer)
    coverage, present_kw, missing_kw = keyword_coverage(reference, student_answer)

    alpha, beta = 0.5, 0.5
    combined = alpha * rouge_l + beta * coverage
    score_0_100 = round(combined * 100, 1)

    parts = [
        f"ROUGE-L similarity: {rouge_l:.3f} (0–1).",
        f"Keyword coverage: {coverage:.3f} (0–1)."
    ]
    if missing_kw:
        parts.append("Missing: " + ", ".join(missing_kw[:8]) + ".")
    if present_kw:
        parts.append("Covered: " + ", ".join(present_kw[:8]) + ".")
    explanation = " ".join(parts)

    return {
        "score": score_0_100,
        "rouge_l": rouge_l,
        "coverage": coverage,
        "present_keywords": present_kw,
        "missing_keywords": missing_kw,
        "explanation": explanation,
    }


# ===============================
# 3. LLM-BASED EVALUATOR (HF Hosted - Mistral locked)
# ===============================

DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # open-access, no license gating

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
    """
    Single-string, instruction-style prompt that works well for hosted instruct models.
    """
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

@st.cache_resource(show_spinner=True)
def get_hf_client() -> InferenceClient | None:
    """
    Returns an authenticated HF InferenceClient using the HF_TOKEN in Streamlit secrets.
    """
    token = st.secrets.get("HF_TOKEN", None)
    if not token:
        return None
    return InferenceClient(token=token)

def generate_hosted_json_judgement(
    client: InferenceClient,
    model_id: str,
    prompt: str,
    max_new_tokens: int = 384,
    temperature: float = 0.1,
    top_p: float = 0.9
) -> str:
    """
    Calls Hugging Face Inference API text generation endpoint with friendly errors.
    """
    try:
        return client.text_generation(
            model=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            return_full_text=False,
        ).strip()
    except ValueError as e:
        # Most common cause: model not available for your token or not mapped to serverless task
        st.error(
            "HF Inference API raised a ValueError. "
            "This usually means your token has no access to the selected model or the model "
            "is not available for serverless text-generation.\n\n"
            f"**Model:** `{model_id}`\n\n"
            "Try again later or verify model availability."
        )
        raise
    except Exception as e:
        st.error(f"HF Inference API call failed: {e}")
        raise

def parse_first_json(text: str) -> dict:
    """
    Extract the first JSON object found in a string.
    """
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def evaluate_answer_llm(
    question: str,
    reference: str,
    student_answer: str,
    temperature: float,
    max_new_tokens: int
) -> dict:
    """
    LLM judge using Hugging Face Inference API (Mistral-7B-Instruct).
    Returns: {"score": float, "analysis": str, "missing_points": list}
    """
    client = get_hf_client()
    if client is None:
        return {
            "score": 0.0,
            "analysis": "HF Inference API not configured. Add HF_TOKEN in Streamlit secrets.",
            "missing_points": [],
        }

    prompt = build_llm_prompt(question, reference, student_answer)
    raw = generate_hosted_json_judgement(
        client=client,
        model_id=DEFAULT_MODEL_ID,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9
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
            "Lower temperature, raise max_new_tokens, or refine the prompt."
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
    st.session_state.history = []


# ===============================
# 5. STREAMLIT UI
# ===============================

st.set_page_config(
    page_title="ML Q&A Evaluator (HF Hosted - Mistral)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Concept Q&A Evaluator (Hugging Face Hosted - Mistral 7B Instruct)")
st.write(
    "This app asks ML questions, collects your answer, and evaluates it automatically (ROUGE + keywords). "
    "It also uses a **hosted** Hugging Face model (Mistral-7B-Instruct) as a JSON-only judge."
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
    st.subheader("HF Judge Settings (fixed model)")
    st.markdown("**Using Hosted Model:** `mistralai/Mistral-7B-Instruct-v0.3` ✅ (open-access)")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 384, 64)

    st.markdown("---")
    st.subheader("Session stats")
    st.write(f"Questions answered: **{len(st.session_state.history)}**")
    if st.session_state.history:
        avg_score = sum(h["evaluation"]["automatic"]["score"] for h in st.session_state.history) / len(st.session_state.history)
        st.write(f"Average automatic score: **{avg_score:.1f} / 100**")
    st.markdown("---")
    st.caption("Assignment 11.00 – LLM Evaluator (Streamlit + HF Inference API, Mistral 7B).")

# Initialize question
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
    st.session_state.last_answer = student_answer

    # Automatic evaluation
    eval_auto = evaluate_answer(reference_answer, student_answer)

    # Hosted HF judge (Mistral)
    eval_llm = evaluate_answer_llm(
        question=question,
        reference=reference_answer,
        student_answer=student_answer,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    st.session_state.last_eval_auto = eval_auto
    st.session_state.last_eval_llm = eval_llm

    st.session_state.history.append(
        {
            "question": question,
            "reference_answer": reference_answer,
            "student_answer": student_answer,
            "evaluation": {"automatic": eval_auto, "llm": eval_llm},
        }
    )

# Results
if "last_eval_auto" in st.session_state:
    eval_auto = st.session_state.last_eval_auto
    eval_llm = st.session_state.last_eval_llm

    st.markdown("## Evaluation Results")
    st.markdown("### 🔍 Automatic Evaluation (ROUGE + Keywords)")
    st.write(f"**Score (0–100):** `{eval_auto['score']}`")
    st.write(textwrap.fill(eval_auto["explanation"], width=100))

    st.markdown("### 🧠 LLM-based Evaluation (HF Hosted Judge - Mistral)")
    st.write(f"**LLM Score (0–100):** `{eval_llm['score']}`")
    st.write(textwrap.fill(eval_llm["analysis"], width=100))
    if eval_llm.get("missing_points"):
        st.write("**Key missing points (according to the LLM):**")
        st.write("- " + "\n- ".join(eval_llm["missing_points"]))

    if show_reference:
        st.markdown("### 📘 Reference Answer")
        st.write(textwrap.fill(reference_answer, width=100))
