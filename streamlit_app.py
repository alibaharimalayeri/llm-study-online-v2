import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, WorksheetNotFound

APP_DIR = Path(__file__).parent
QUESTIONS_CSV = APP_DIR / "questions.csv"

# ====== CONFIG ======
RESULTS_SHEET_NAME = "results_v2"  # tab name in your spreadsheet
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Columns written to Google Sheets (order matters)
HEADER = [
    "ts_iso",
    "participant",
    "base_id",
    "qid",
    "question",
    "answer_variant",
    "accuracy",
    "completeness",
    "usefulness",
    "style_tone",
    "comment",
]

def norm(s: str) -> str:
    """Normalize strings for comparison (strip + lowercase)."""
    return (s or "").strip().lower()

# ====== GOOGLE SHEETS HELPERS ======
@st.cache_resource(show_spinner=False)
def get_ws():
    """Authorize and return the results worksheet (create if missing)."""
    if "SHEET_ID" not in st.secrets or "gcp_service_account" not in st.secrets:
        st.error("Missing secrets: set SHEET_ID and [gcp_service_account] in Streamlit secrets.")
        st.stop()

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sh = client.open_by_key(st.secrets["SHEET_ID"])
    try:
        ws = sh.worksheet(RESULTS_SHEET_NAME)
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=RESULTS_SHEET_NAME, rows=1000, cols=len(HEADER))
        ws.append_row(HEADER, value_input_option="RAW")

    # Ensure header is correct
    existing_header = ws.row_values(1)
    if existing_header != HEADER:
        if not existing_header:
            ws.update("1:1", [HEADER])
        else:
            ws.delete_rows(1)
            ws.insert_row(HEADER, 1)
    return ws

def append_rows_ws(ws, rows: List[Dict[str, object]], max_retries: int = 5):
    """Append rows to Google Sheet with simple 429 retry/backoff."""
    values = [[r.get(col, "") for col in HEADER] for r in rows]
    delay = 1.0
    for _ in range(max_retries):
        try:
            ws.append_rows(values, value_input_option="RAW")
            return
        except APIError as e:
            msg = str(e)
            if "429" in msg or "Quota exceeded" in msg or any(code in msg for code in ["500", "502", "503", "504"]):
                time.sleep(delay)
                delay = min(delay * 2, 10)
                continue
            raise

@st.cache_data(show_spinner=False, ttl=30)
def get_answered_bases_for_participant(participant_raw: str) -> set:
    """
    Return base_ids already answered by this participant (cached briefly).
    Uses header-aware read to avoid partial / ragged rows.
    """
    ws = get_ws()
    # Get ALL values once (one API call), then build a DataFrame with the header row.
    rows = ws.get_all_values()  # includes header at rows[0]
    if not rows or len(rows[0]) == 0:
        return set()

    header = rows[0]
    # If the header is not exactly what we expect, try to map columns by name.
    # (This makes us tolerant to column order changes.)
    try:
        df = pd.DataFrame(rows[1:], columns=header)
    except ValueError:
        # Fallback to strict assumption
        return set()

    # Normalize column names we rely on
    cols = {c.lower().strip(): c for c in df.columns}
    if "participant" not in cols or "base_id" not in cols:
        return set()

    participant_col = cols["participant"]
    base_col = cols["base_id"]

    # Filter by normalized participant
    want = norm(participant_raw)
    mask = df[participant_col].astype(str).str.strip().str.lower() == want
    base_vals = df.loc[mask, base_col].dropna().astype(str).str.strip()
    return set(base_vals.tolist())

# ====== LOCAL HELPERS ======
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

# ====== UI ======
st.set_page_config(page_title="LLM Answer Comparison (v2)", layout="wide")

st.sidebar.header("Study Info")
name_input = st.sidebar.text_input("Name (required)", "")
st.sidebar.caption(
    "Your ratings are saved to a secure Google Sheet. "
    "Resume anytime by entering the same name."
)

# Load questions
if not QUESTIONS_CSV.exists():
    st.error(f"Missing questions file: {QUESTIONS_CSV.name}")
    st.stop()

df_q = pd.read_csv(QUESTIONS_CSV)
if df_q.empty:
    st.error("questions.csv is empty.")
    st.stop()

order, groups = load_groups(df_q)
st.sidebar.write(f"Questions loaded: {len(order)}")

# Guard for name
if not name_input.strip():
    st.title("LLM Answer Evaluation (3 answers per question)")
    st.info("Enter your **Name** in the left sidebar to begin.")
    st.stop()

participant = name_input.strip()              # display value
participant_norm = norm(participant)          # for comparisons

# Resume: which base_ids already answered?
answered_bases = get_answered_bases_for_participant(participant_norm)

# Initialize current index at first unanswered (based on normalized matches)
if "idx" not in st.session_state:
    start_idx = 0
    for i, b in enumerate(order):
        if b not in answered_bases:
            start_idx = i
            break
    st.session_state.idx = start_idx

# Progress
st.title("LLM Answer Evaluation")
st.progress(st.session_state.idx / len(order))
st.caption(f"Progress: {st.session_state.idx} / {len(order)}")

# Current block
base = order[st.session_state.idx]
block = groups[base].reset_index(drop=True)
question_text = block.loc[0, "question"]

st.subheader(f"Question {st.session_state.idx + 1} of {len(order)}")
st.markdown(f"**QID group:** `{base}`")
st.write(f"**Question:** {question_text}")

# Show up to three answers side-by-side
criteria = ["Accuracy", "Completeness", "Usefulness", "Style/Tone"]
cols = st.columns([1, 1, 1])

variant_rows: List[Dict[str, object]] = []
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
                "ts_iso": datetime.now(timezone.utc).isoformat(),
                "participant": participant,  # stored as typed (nice for reading)
                "base_id": base,
                "qid": qid,
                "question": question_text,
                "answer_variant": qid[-1].upper(),
                "accuracy": scores["Accuracy"],
                "completeness": scores["Completeness"],
                "usefulness": scores["Usefulness"],
                "style_tone": scores["Style/Tone"],
                "comment": (comment or "").strip(),
            }
        )

# Save & Next
left, mid, right = st.columns([1, 1, 1])
with mid:
    if st.button("💾 Save & Next", use_container_width=True):
        # validate all radios filled (no '—')
        flat_scores = [
            r[k] for r in variant_rows for k in ["accuracy", "completeness", "usefulness", "style_tone"]
        ]
        if any(v == "—" for v in flat_scores):
            st.warning("Please score all criteria for all answers before continuing.")
            st.stop()

        ws = get_ws()
        try:
            append_rows_ws(ws, variant_rows)
        except Exception as e:
            st.error(f"Saving to Google Sheet failed: {e}")
            st.stop()

        # clear progress cache so next question is computed fresh
        get_answered_bases_for_participant.clear()

        st.success("Saved!")
        # advance
        if st.session_state.idx < len(order) - 1:
            st.session_state.idx += 1
            st.rerun()
        else:
            st.balloons()
            st.success("All questions completed! Thank you.")
            st.stop()

# Debug aide
with st.expander("Resume debug (safe to ignore)"):
    st.write("Answered groups detected for you:", sorted(list(answered_bases)))
