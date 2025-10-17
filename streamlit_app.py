import re
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
QUESTIONS_CSV = APP_DIR / "questions.csv"

# ---------- Helpers ----------
def base_id(qid: str) -> str:
    """Turn Q1a -> Q1, Q23c -> Q23."""
    m = re.match(r"^([A-Za-z]*\d+)", qid)
    return m.group(1) if m else qid

def load_groups(df: pd.DataFrame):
    """Group rows into base question blocks Q1, Q2, ..."""
    df = df.copy()
    df["base"] = df["qid"].apply(base_id)
    order = df.drop_duplicates("base")["base"].tolist()
    groups = {b: g for b, g in df.groupby("base")}
    return order, groups

def score_input(label: str, key: str):
    opts = ["—", 1, 2, 3, 4, 5]
    return st.radio(label, opts, index=0, horizontal=True, key=key)

def all_scored(score_dict):
    return all(v in [1, 2, 3, 4, 5] for v in score_dict.values())

def append_rows_csv(path: Path, rows: list[dict]):
    path_exists = path.exists()
    df_new = pd.DataFrame(rows)
    if path_exists:
        df_old = pd.read_csv(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)

# ---------- UI ----------
st.set_page_config(page_title="LLM Answer Comparison", layout="wide")

st.sidebar.header("Study Info")
name = st.sidebar.text_input("Name (required)", "")
st.sidebar.caption(
    "Your ratings are saved locally to a per-person CSV (results_<Name>.csv) next to the app."
)

if not QUESTIONS_CSV.exists():
    st.error(f"Missing questions file: {QUESTIONS_CSV.name}")
    st.stop()

df_q = pd.read_csv(QUESTIONS_CSV)
if df_q.empty:
    st.error("questions.csv is empty.")
    st.stop()

order, groups = load_groups(df_q)
st.sidebar.write(f"Questions loaded: {len(order)}")

# Resume logic: find answered base_ids for this name
results_path = APP_DIR / f"results_{name.strip()}.csv" if name.strip() else None
answered_bases = set()
if results_path and results_path.exists():
    try:
        df_prev = pd.read_csv(results_path)
        if "base_id" in df_prev.columns:
            answered_bases = set(df_prev["base_id"].unique().tolist())
    except Exception:
        pass

# init index
if "idx" not in st.session_state:
    # start at first unanswered base
    start_idx = 0
    for i, b in enumerate(order):
        if b not in answered_bases:
            start_idx = i
            break
    st.session_state.idx = start_idx

# guard for name
if not name.strip():
    st.title("LLM Answer Evaluation (3 answers per question)")
    st.info("Enter your **Name** in the left sidebar to begin.")
    st.stop()

# progress
st.title("LLM Answer Evaluation")
st.progress(st.session_state.idx / len(order))
st.caption(f"Progress: {st.session_state.idx} / {len(order)}")

# current block
base = order[st.session_state.idx]
block = groups[base].reset_index(drop=True)

question_text = block.loc[0, "question"]
st.subheader(f"Question {st.session_state.idx + 1} of {len(order)}")
st.markdown(f"**QID group:** `{base}`")
st.write(f"**Question:** {question_text}")

# Show three answers (A/B/C) with independent scoring
variant_rows = []
cols = st.columns([1,1,1])

criteria = ["Accuracy", "Completeness", "Usefulness", "Style/Tone"]

for j in range(len(block)):
    qid = block.loc[j, "qid"]
    ans = block.loc[j, "model_answer"]
    with cols[j % 3]:
        st.markdown(f"### Answer {qid[-1].upper()}")
        st.write(ans)
        scores = {}
        for crit in criteria:
            key = f"{base}_{qid}_{crit}"
            scores[crit] = score_input(crit, key)
        comment = st.text_area("Optional comment", key=f"c_{base}_{qid}", height=70)

        variant_rows.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "name": name.strip(),
                "base_id": base,
                "qid": qid,
                "question": question_text,
                "answer_variant": qid[-1].upper(),
                "accuracy": scores["Accuracy"],
                "completeness": scores["Completeness"],
                "usefulness": scores["Usefulness"],
                "style": scores["Style/Tone"],
                "comment": comment.strip(),
            }
        )

# Save & Next
left, mid, right = st.columns([1,1,1])
with mid:
    if st.button("💾 Save & Next", use_container_width=True):
        # validate all radios filled (no '—')
        flat_scores = [
            r[k] for r in variant_rows for k in ["accuracy","completeness","usefulness","style"]
        ]
        if any(v == "—" for v in flat_scores):
            st.warning("Please score all criteria for all three answers before continuing.")
        else:
            if results_path is None:
                st.error("No results file path (name missing).")
                st.stop()
            append_rows_csv(results_path, variant_rows)
            st.success(f"Saved to {results_path.name}")
            # advance
            if st.session_state.idx < len(order) - 1:
                st.session_state.idx += 1
                st.rerun()
            else:
                st.balloons()
                st.success("All questions completed! You can close the app or send us your CSV.")
                st.stop()

# Download backup (optional)
st.download_button(
    "⬇️ Download my results CSV",
    data=(results_path.read_bytes() if (results_path and results_path.exists()) else b""),
    file_name=(results_path.name if results_path else "results.csv"),
    disabled=not (results_path and results_path.exists()),
)
