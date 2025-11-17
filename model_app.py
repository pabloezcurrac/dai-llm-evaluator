import streamlit as st
import json
from pathlib import Path
from collections import Counter
import random
import textwrap
from models import rouge  # Or your specific model class

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
    most_common = [w for w in freq.most_common(top_k)]
    return most_common


def keyword_coverage(reference: str, prediction: str, top_k: int = 20):
    """
    Compute coverage of important reference keywords in the student's prediction.
    Returns a value between 0 and 1.
    """
    ref_keywords = extract_keywords(reference, top_k=top_k)
    pred_tokens = set(
        [t.strip(".,;:!?()[]\"'").lower() for t in prediction.split()]
    )

    present = [w for w in ref_keywords if w in pred_tokens]
    missing = [w for w in ref_keywords if w not in pred_tokens]

    coverage = len(present) / max(1, len(ref_keywords))
    return coverage

def evaluate_answer(reference: str, student_answer: str) -> dict:
    """
    Main automatic evaluator:
    - Computes ROUGE-L
    - Computes keyword coverage
    - Produces numeric score (0-100)
    - Produces textual explanation
    """
    # 1) ROUGE-L (0-1)
    rouge_l = compute_rouge_l(reference, student_answer)

    # 2) Keyword coverage (0-1)
    coverage, present_kw, missing_kw = keyword_coverage(reference, student_answer)

    # 3) Combine the two into a score (simple linear combination)
    alpha = 0.5  # weight for ROUGE-L
    beta = 0.5   # weight for keyword coverage

    combined = alpha * rouge_l + beta * coverage
    score = combined

    return score


if st.session_state.current_qa is None:
    st.session_state.current_qa = sample_question(qa_items)
    # Clear previous answer/result for clarity
    if "last_answer" in st.session_state:
        del st.session_state["last_answer"]
    if "last_eval_auto" in st.session_state:
        del st.session_state["last_eval_auto"]
    if "last_eval_llm" in st.session_state:
        del st.session_state["last_eval_llm"]
    st.session_state.last_answer = student_answer
    st.session_state.history = []
    if "last_eval_auto" in st.session_state:
        del st.session_state["last_eval_auto"]
    if "last_eval_llm" in st.session_state:
        del st.session_state["last_eval_llm"]
    if "last_answer" in st.session_state:
        del st.session_state["last_answer"]

    st.session_state.last_answer = student_answer
    st.session_state.history = []


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

    # 2) LLM-based evaluation (stub, but structured)
    eval_llm = evaluate_answer_llm(question, reference_answer, student_answer)

    # Store last evals in session
    st.session_state.last_eval_auto = eval_auto
    st.session_state.last_eval_llm = eval_llm

    # Add to history
    st.session_state.history.append({
        "question": question,
        "reference_answer": reference_answer,
        "student_answer": student_answer,
        "evaluation": {
            "automatic": eval_auto,
            "llm": eval_llm,
        },
    })

# Display results if available
if "last_eval_auto" in st.session_state:
    eval_auto = st.session_state.last_eval_auto
    eval_llm = st.session_state.last_eval_llm

    st.markdown("## Evaluation Results")

    # Automatic evaluation
    st.markdown("### 🔍 Automatic Evaluation (ROUGE + Keywords)")
    st.write(f"**Score (0–100):** `{eval_auto['score']}`")
    st.write(f"**LLM Score (0–100):** `{eval_llm['score']}`")

    # Reference answer (optional)
    if show_reference:
        st.markdown("### 📘 Reference Answer")
        st.write(textwrap.fill(reference_answer, width=100))
