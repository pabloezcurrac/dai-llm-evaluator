"""model_app – Final Version (with OpenAI GPT-3.5 for LLM Evaluation)"""

import os
import json
from pathlib import Path
from collections import Counter
import random
import textwrap
import re

import streamlit as st
import evaluate
from transformers import pipeline  # still used for automatic ROUGE section
import openai


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
# 2. AUTOMATIC EVALUATOR (ROUGE + Keywords)
# ===============================

rouge = evaluate.load("rouge")


def compute_rouge_l(reference: str, prediction: str) -> float:
    results = rouge.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=True
    )
    return results["rougeL"]


def extract_keywords(text: str, min_len: int = 4, top_k: int = 20):
    tokens = [t.strip(".,;:!?()[]\"'").lower() for t in text.split()]
    tokens = [t for t in tokens if len(t) >= min_len]
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_k)]


def keyword_coverage(reference: str, prediction: str, top_k: int = 20):
    ref_keywords = extract_keywords(reference, top_k=top_k)
    pred_tokens = set(
        t.strip(".,;:!?()[]\"'").lower() for t in prediction.split()
    )
    present = [w for w in ref_keywords if w in pred_tokens]
    missing = [w for w in ref_keywords if w not in pred_tokens]
    coverage = len(present) / max(1, len(ref_keywords))
    return coverage, present, missing


def evaluate_answer(reference: str, student_answer: str) -> dict:
    rouge_l = compute_rouge_l(reference, student_answer)
    coverage, present_kw, missing_kw = keyword_coverage(reference, student_answer)
    combined = 0.5 * rouge_l + 0.5 * coverage
    score_0_100 = round(combined * 100, 1)

    explanation_parts = [
        f"ROUGE-L similarity: {rouge_l:.3f} (0–1 scale). This reflects overlap in phrasing and sentence structure.",
        f"Keyword coverage: {coverage:.3f} (0–1 scale). This reflects how many core terms from the reference you mentioned."
    ]
    if missing_kw:
        explanation_parts.append(f"Missing: {', '.join(missing_kw[:8])}.")
    if present_kw:
        explanation_parts.append(f"Included: {', '.join(present_kw[:8])}.")
    return {
        "score": score_0_100,
        "explanation": " ".join(explanation_parts),
    }


# ===============================
# 3. LLM-BASED EVALUATOR (GPT-3.5 Turbo)
# ===============================

# ===============================
# 3. LLM-BASED EVALUATOR (Hugging Face – Chat Model, no API keys)
# ===============================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # <— small, instruction-tuned, CPU-friendly

LLM_SYSTEM_PROMPT = (
    "You are a professor grading a student's short answer about a Machine Learning concept. "
    "Compare the student's answer to the reference answer, explain briefly what is correct or missing, "
    "and give a score from 0 to 100. Respond ONLY in this exact format:\n\n"
    "Score: <number between 0 and 100>\n"
    "Feedback: <1-3 sentences>"
)

@st.cache_resource
def load_local_llm():
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="auto",          # CPU or GPU automatically
        torch_dtype=torch.float32,  # safe on CPU; will use half on GPU if available
        trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl

tokenizer, llm_model = load_local_llm()


def build_llm_messages(question: str, reference_answer: str, student_answer: str):
    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Question: {question}\n\nReference Answer: {reference_answer}\n\nStudent Answer: {student_answer}"}
    ]


def call_llm_evaluator(question: str, reference_answer: str, student_answer: str) -> dict:
    """Use the chat template for a real instruct model and parse the result robustly."""
    messages = build_llm_messages(question, reference_answer, student_answer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
        # Take only the new text
        gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    st.sidebar.text_area("LLM raw output (debug)", gen_text, height=140)

    # Parse "Score:" and "Feedback:" robustly
    score = 75.0
    feedback = gen_text
    m_score = re.search(r"Score\s*[:\-]\s*(\d{1,3})", gen_text, flags=re.IGNORECASE)
    m_fb = re.search(r"(Feedback|Explanation)\s*[:\-]\s*(.*)", gen_text, flags=re.IGNORECASE | re.DOTALL)

    if m_score:
        try:
            score = float(m_score.group(1))
        except:
            pass
    score = max(0.0, min(100.0, score))

    if m_fb:
        feedback = m_fb.group(2).strip()

    return {"score": score, "analysis": feedback}


def evaluate_answer_llm(question: str, reference: str, student_answer: str) -> dict:
    return call_llm_evaluator(question, reference, student_answer)


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
    page_title="ML Q&A Evaluator",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Concept Q&A Evaluator")
st.write(
    "This prototype asks you questions about Machine Learning concepts, "
    "collects your answer, and evaluates it automatically (ROUGE + keyword coverage). "
    "It also includes an LLM-based evaluation using OpenAI GPT-3.5-Turbo."
)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 New random question"):
        st.session_state.current_qa = sample_question(qa_items)
        for key in ["last_answer", "last_eval_auto", "last_eval_llm"]:
            if key in st.session_state:
                del st.session_state[key]

    if st.button("🧼 Reset session"):
        st.session_state.current_qa = None
        st.session_state.history = []
        for key in ["last_answer", "last_eval_auto", "last_eval_llm"]:
            if key in st.session_state:
                del st.session_state[key]

    st.markdown("---")
    st.subheader("Session stats")
    st.write(f"Questions answered: *{len(st.session_state.history)}*")
    if st.session_state.history:
        avg_score = sum(h["evaluation"]["automatic"]["score"] for h in st.session_state.history) / len(st.session_state.history)
        st.write(f"Average automatic score: *{avg_score:.1f} / 100*")
    st.markdown("---")
    st.caption("Prototype for Assignment 11.00 – LLM Evaluator (Streamlit UI).")


if st.session_state.current_qa is None:
    st.session_state.current_qa = sample_question(qa_items)

current_qa = st.session_state.current_qa
question = current_qa["question"]
reference_answer = current_qa["answer"]

st.subheader("Current Question")
st.write(question)

st.markdown("### Your Answer")
student_answer = st.text_area(
    "Type your answer here:",
    value=st.session_state.get("last_answer", ""),
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
    eval_auto = evaluate_answer(reference_answer, student_answer)
    eval_llm = evaluate_answer_llm(question, reference_answer, student_answer)

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

if "last_eval_auto" in st.session_state:
    eval_auto = st.session_state.last_eval_auto
    eval_llm = st.session_state.last_eval_llm

    st.markdown("## Evaluation Results")

    st.markdown("### 🔍 Automatic Evaluation (ROUGE + Keywords)")
    st.write(f"*Score (0–100):* {eval_auto['score']}")
    st.write(textwrap.fill(eval_auto["explanation"], width=100))

    st.markdown("### 🧠 LLM-based Evaluation (Hugging Face – Qwen2.5 Instruct)")
    st.write(f"*LLM Score (0–100):* {eval_llm['score']}")
    st.write(textwrap.fill(eval_llm["analysis"], width=100))

    if show_reference:
        st.markdown("### 📘 Reference Answer")
        st.write(textwrap.fill(reference_answer, width=100))
