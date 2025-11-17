# -*- coding: utf-8 -*-
"""model_app

Streamlit app for ML Q&A evaluation:
- Automatic evaluator: ROUGE-L + keyword coverage
- LLM-based evaluator: Ollama local model (Llama 2 chat)

Before running:
  1) pip install streamlit evaluate rouge-score pandas requests
  2) Start Ollama and pull model:  ollama pull llama2:7b-chat
  3) streamlit run model_app.py
"""

import json
from pathlib import Path
from collections import Counter
import random
import textwrap
import re
import requests

import streamlit as st
import evaluate
import pandas as pd  # optional (kept for parity with prior code)


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
# 3. LLM-BASED EVALUATOR (Ollama - Llama 2)
# ===============================

# Use an Ollama-served Llama 2 chat model
# Pull first:  ollama pull llama2:7b-chat
DEFAULT_OLLAMA_MODEL = "llama2:7b-chat"

SYSTEM_PROMPT = """
You are a rigorous and fair university-level Machine Learning professor.

Compare the STUDENT_ANSWER to the REFERENCE_ANSWER and return STRICT JSON ONLY:
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

Do not include any text outside the JSON object.
""".strip()


def build_llm_prompt(question: str, reference_answer: str, student_answer: str) -> str:
    return f"""
[QUESTION]
{question}

[REFERENCE_ANSWER]
{reference_answer}

[STUDENT_ANSWER]
{student_answer}

Return STRICT JSON only.
""".strip()


@st.cache_resource(show_spinner=False)
def get_ollama_base_url() -> str:
    # Change here if your Ollama host is different
    return "http://localhost:11434"


def _ollama_chat(model: str, system: str, user: str, temperature: float, num_predict: int) -> str:
    """
    Calls Ollama /api/chat with system+user for Llama 2 chat.
    Returns assistant content as a string.
    """
    base = get_ollama_base_url().rstrip("/")
    url = f"{base}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        # Return a valid JSON object so the app doesn't break
        return f'{{"score": 0, "analysis": "Error calling Ollama: {e}", "missing_points": []}}'


def evaluate_answer_llm(question: str, reference: str, student_answer: str,
                        model_name: str, temperature: float, num_predict: int) -> dict:
    """
    LLM judge via Ollama (Llama 2 chat).
    Returns: {"score": float, "analysis": str, "missing_points": list}
    """
    user_prompt = build_llm_prompt(question, reference, student_answer)
    raw = _ollama_chat(
        model=model_name,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=temperature,
        num_predict=num_predict
    )

    # Extract first JSON object
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            score = float(data.get("score", 0))
            analysis = str(data.get("analysis", "")).strip()
            missing_points = data.get("missing_points", [])
            if not isinstance(missing_points, list):
                missing_points = []
            score = max(0.0, min(100.0, score))
            return {"score": score, "analysis": analysis, "missing_points": missing_points}
        except Exception:
            pass

    # Fallback if model didn’t produce strict JSON
    return {
        "score": 0.0,
        "analysis": (
            "Could not parse strict JSON from the Ollama output. "
            "Try lowering temperature, increasing num_predict, or refining the prompt."
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
    page_title="ML Q&A Evaluator (Ollama - Llama 2)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Concept Q&A Evaluator (Ollama - Llama 2)")
st.write(
    "This prototype asks you questions about Machine Learning concepts, "
    "collects your answer, and evaluates it automatically (ROUGE + keyword coverage). "
    "It also provides an Ollama LLM-based evaluation using Llama 2."
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
    st.subheader("Ollama Judge Settings")
    ollama_model = st.text_input(
        "Model name (must be pulled in Ollama)",
        value=DEFAULT_OLLAMA_MODEL,
        help="Examples: llama2:7b-chat, llama2:13b-chat"
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    num_predict = st.slider("Max new tokens (num_predict)", min_value=64, max_value=1024, value=384, step=64)

    st.markdown("---")
    st.subheader("Session stats")
    st.write(f"Questions answered: **{len(st.session_state.history)}**")
    if st.session_state.history:
        avg_score = sum(h["evaluation"]["automatic"]["score"] for h in st.session_state.history) / len(st.session_state.history)
        st.write(f"Average automatic score: **{avg_score:.1f} / 100**")

    st.markdown("---")
    st.caption("Prototype for Assignment 11.00 – LLM Evaluator (Streamlit + Ollama Llama 2).")


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

    # 2) LLM-based evaluation (Ollama - Llama 2)
    eval_llm = evaluate_answer_llm(
        question=question,
        reference=reference_answer,
        student_answer=student_answer,
        model_name=ollama_model,
        temperature=temperature,
        num_predict=num_predict
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

    # LLM-based evaluation (Ollama - Llama 2)
    st.markdown("### 🧠 LLM-based Evaluation (Ollama Judge - Llama 2)")
    st.write(f"**LLM Score (0–100):** `{eval_llm['score']}`")
    st.write(textwrap.fill(eval_llm["analysis"], width=100))
    if eval_llm.get("missing_points"):
        st.write("**Key missing points (according to the LLM):**")
        st.write("- " + "\n- ".join(eval_llm["missing_points"]))

    # Reference answer (optional)
    if show_reference:
        st.markdown("### 📘 Reference Answer")
        st.write(textwrap.fill(reference_answer, width=100))
