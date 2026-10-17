import html
import json
import os
import re
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

MODEL_NAME = "gemini-3.8-flash"
MAX_CV_CHARS = 12_000
BATCH_SIZE = 5
MAX_WORKERS = 5
MAX_RETRIES = 3
MIN_API_REQUEST_INTERVAL = 12.2
RETRYABLE_MARKERS = (
    "503",
    "429",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "HIGH DEMAND",
    "TRY AGAIN",
    "OVERLOADED",
)
_API_REQUEST_LOCK = threading.Lock()
_NEXT_API_REQUEST_AT = 0.0

def load_local_env() -> None:
    env_file = Path(__file__).with_name(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)


load_local_env()

st.set_page_config(
    page_title="Candidate Ranker",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"],
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"] { display: none !important; }
      .block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem; }
      header[data-testid="stHeader"] { background: transparent; }
      .hero {
        background: linear-gradient(135deg, #152238 0%, #1c3d5a 55%, #0f766e 140%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 8px;
      }
      .hero h1 {
        color: #f8fafc; font-size: 2rem; margin: 0 0 6px 0; letter-spacing: -0.03em;
      }
      .hero p { color: #cbd5e1; margin: 0; font-size: 1.02rem; }
      .section-label {
        font-size: .78rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: #94a3b8; margin: 4px 0 8px;
      }
      .stTextArea textarea {
        min-height: 240px;
        border-radius: 14px !important;
        font-size: 0.98rem !important;
        line-height: 1.5 !important;
      }
      .stButton > button {
        height: 3.1rem; border-radius: 12px; font-weight: 700;
        font-size: 1.05rem; letter-spacing: .01em;
      }
      .hire-card {
        background: linear-gradient(180deg, #064e3b 0%, #052e24 100%);
        border: 1px solid #10b981;
        border-radius: 16px; padding: 22px 26px; margin: 8px 0 22px;
      }
      .hire-kicker {
        color: #6ee7b7; font-size: .75rem; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; margin: 0 0 8px;
      }
      .hire-name { color: #ecfdf5; font-size: 1.55rem; font-weight: 800; margin: 0 0 10px; }
      .hire-why { color: #d1fae5; font-size: 1.05rem; line-height: 1.6; margin: 0; }
      .candidate-card {
        background: #111827; border: 1px solid #243044; border-radius: 16px;
        padding: 18px 20px; margin: 0 0 16px;
      }
      .card-head {
        display: flex; align-items: center; justify-content: space-between;
        gap: 12px; margin-bottom: 14px;
      }
      .rank-pill {
        display: inline-flex; align-items: center; justify-content: center;
        width: 34px; height: 34px; border-radius: 999px;
        background: #1e293b; color: #93c5fd; font-weight: 800; margin-right: 10px;
      }
      .cand-name { color: #f1f5f9; font-size: 1.2rem; font-weight: 700; }
      .score {
        font-size: 1.35rem; font-weight: 800; white-space: nowrap;
      }
      .score.high { color: #34d399; }
      .score.mid { color: #fbbf24; }
      .score.low { color: #f87171; }
      .box-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
      }
      @media (max-width: 800px) {
        .box-grid { grid-template-columns: 1fr; }
      }
      .info-box {
        border-radius: 12px; padding: 14px 16px; min-height: 120px;
      }
      .info-box h4 {
        margin: 0 0 10px; font-size: .78rem; letter-spacing: .08em;
        text-transform: uppercase;
      }
      .info-box ul { margin: 0; padding-left: 1.1rem; }
      .info-box li { margin: 0 0 8px; line-height: 1.5; font-size: .98rem; }
      .info-box p { margin: 0; line-height: 1.55; font-size: .98rem; }
      .strengths { background: #052e1a; border: 1px solid #166534; }
      .strengths h4, .strengths li, .strengths p { color: #bbf7d0; }
      .weaknesses { background: #3f1410; border: 1px solid #9a3412; }
      .weaknesses h4, .weaknesses li, .weaknesses p { color: #fed7aa; }
      .file-chip {
        display: inline-block; background: #1e293b; color: #cbd5e1;
        border-radius: 999px; padding: 4px 10px; margin: 0 6px 6px 0; font-size: .82rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def cv_char_limit(count: int) -> int:
    if count >= 30:
        return 3_200
    if count >= 15:
        return 5_000
    return MAX_CV_CHARS


def extract_text(uploaded_file, limit: int = MAX_CV_CHARS) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        raw = uploaded_file.read()
        text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"Could not extract text from {uploaded_file.name}")
    return text[:limit]


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def is_retryable(exc: Exception) -> bool:
    text = str(exc).upper()
    return any(marker in text for marker in RETRYABLE_MARKERS)


def retry_after_seconds(exc: Exception) -> float:
    text = str(exc)
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]([\d.]+)s['\"]", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"retry in\s+([\d.]+)s", text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def wait_for_api_slot() -> None:
    global _NEXT_API_REQUEST_AT
    with _API_REQUEST_LOCK:
        now = time.monotonic()
        request_at = max(now, _NEXT_API_REQUEST_AT)
        _NEXT_API_REQUEST_AT = request_at + MIN_API_REQUEST_INTERVAL
        delay = request_at - now
    if delay > 0:
        time.sleep(delay)


def as_items(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"\n|;|(?:^|\s)[-•]\s+", text)
    if len(parts) == 1 and "," in text and len(text) < 280:
        parts = text.split(",")
    items = [re.sub(r"^[-•\s]+", "", part).strip() for part in parts]
    return [item for item in items if item]


def score_class(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 60:
        return "mid"
    return "low"


def items_html(items: list[str], empty: str) -> str:
    if not items:
        return f"<p>{html.escape(empty)}</p>"
    lis = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<ul>{lis}</ul>"


def build_candidate_prompt(job_description: str, candidates: list[dict]) -> str:
    candidate_blocks = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_blocks.append(
            f"--- CANDIDATE {index} ---\n"
            f"Source file: {candidate['filename']}\n"
            f"CV text:\n{candidate['text']}"
        )
    cv_bundle = "\n\n".join(candidate_blocks)
    return f"""You are an expert technical recruiter and hiring manager.

Evaluate each candidate against the job description. Use a consistent score:
90-100 exceptional fit, 75-89 strong fit, 60-74 partial fit, below 60 weak fit.

JOB DESCRIPTION:
{job_description[:3500]}

 CANDIDATE CVS, IN ORDER:
{cv_bundle}

Respond strictly with valid JSON and no markdown, prose, or code fences. Return a JSON array with exactly one object per candidate, in the same order as the CVs above. Each object must have exactly these keys:
[
    {{
        "Name": "Candidate full name, or a clean version of the source filename if not stated",
        "Match_Score": 0,
        "Key_Strengths": ["Concrete strength supported by CV evidence"],
        "Missing_Requirements": ["Concrete job requirement that is missing or not evidenced"],
        "Top_Recommendation_Status": "Not Recommended"
    }}
]

Rules:
- Match_Score must be an integer from 0 to 100.
- Key_Strengths and Missing_Requirements must be arrays of concise strings.
- Set Top_Recommendation_Status to "Recommended" only when the candidate is an exceptional fit; otherwise use "Not Recommended". The app will determine the final top-ranked candidate across all CVs.
- Be specific and do not invent experience or qualifications.
- Include every candidate exactly once and preserve their order.
"""


def generate_json(
    client,
    prompt: str,
    expected_count: int,
    before_request=wait_for_api_slot,
) -> list:
    last_error = None
    delay = 0.8
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            before_request()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(response.text)
            if (
                not isinstance(data, list)
                or len(data) != expected_count
                or any(not isinstance(row, dict) for row in data)
            ):
                raise ValueError(
                    f"Gemini returned JSON that did not contain exactly {expected_count} candidate objects."
                )
            return data
        except Exception as exc:
            last_error = exc
            if not is_retryable(exc) or attempt == MAX_RETRIES:
                raise
            time.sleep(max(delay, retry_after_seconds(exc)))
            delay = min(delay * 2, 4)
    raise last_error


def rank_candidates(
    api_key: str,
    job_description: str,
    candidates: list[dict],
    progress=None,
    status=None,
) -> list[dict]:
    total = len(candidates)
    batches = list(batched(candidates, BATCH_SIZE))
    leaderboard = []
    finished = 0
    workers = min(MAX_WORKERS, len(batches))

    if status is not None:
        status.info(f"Ranking {total} CVs...")
    if progress is not None:
        progress.progress(0, text="Starting ranking...")

    def run_batch(batch: list[dict]) -> list[dict]:
        client = genai.Client(api_key=api_key)
        parsed_rows = generate_json(
            client,
            build_candidate_prompt(job_description, batch),
            expected_count=len(batch),
        )
        results = []
        for candidate, parsed in zip(batch, parsed_rows):
            score = max(0, min(100, int(parsed.get("Match_Score", 0))))
            results.append(
                {
                    "Name": str(parsed.get("Name") or os.path.splitext(candidate["filename"])[0]),
                    "Match_Score": score,
                    "Key_Strengths": as_items(parsed.get("Key_Strengths")),
                    "Missing_Requirements": as_items(parsed.get("Missing_Requirements")),
                    "Top_Recommendation_Status": str(
                        parsed.get("Top_Recommendation_Status") or "Not Recommended"
                    ),
                }
            )
        return results

    def record_result(row: dict) -> None:
        nonlocal finished
        leaderboard.append(row)
        finished += 1
        if progress is not None:
            progress.progress(
                finished / total,
                text=f"Processed {finished} of {total} CVs",
            )
        if status is not None:
            status.info(f"Processed {finished} of {total} CVs")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                for row in future.result():
                    record_result(row)
            except Exception as exc:
                for candidate in batch:
                    candidate_name = os.path.splitext(candidate["filename"])[0]
                    st.warning(f"{candidate_name}: Failed to Analyze ({exc})")
                    record_result(
                        {
                            "Name": candidate_name,
                            "Match_Score": 0,
                            "Key_Strengths": [],
                            "Missing_Requirements": ["Failed to Analyze"],
                            "Top_Recommendation_Status": "Failed to Analyze",
                        }
                    )

    leaderboard.sort(key=lambda row: row["Match_Score"], reverse=True)
    if not leaderboard:
        raise ValueError("No candidates could be ranked.")

    best = next((row for row in leaderboard if row["Match_Score"] > 0), None)
    if best is not None:
        best["Top_Recommendation_Status"] = "Top Recommendation"
        for row in leaderboard:
            if row is not best and row["Top_Recommendation_Status"] != "Failed to Analyze":
                row["Top_Recommendation_Status"] = "Not Recommended"
    if status is not None:
        status.empty()
    if progress is not None:
        progress.empty()
    return leaderboard


def get_api_key() -> str:
    try:
        secret = str(st.secrets.get("GEMINI_API_KEY", "") or "").strip()
    except Exception:
        secret = ""
    return (
        secret
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()


st.markdown(
    """
    <div class="hero">
      <h1>Candidate Ranker</h1>
      <p>Paste the role requirements, drop in CVs, and get a ranked hire recommendation with full strengths and weaknesses.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns((1.35, 1), gap="large")
with left:
    st.markdown('<div class="section-label">Job description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job Description",
        height=280,
        placeholder="Paste the full job description, must-have skills, and nice-to-haves...",
        label_visibility="collapsed",
    )
with right:
    st.markdown('<div class="section-label">Candidate CVs</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload candidate CVs (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        preview = uploaded_files[:6]
        chips = " ".join(
            f'<span class="file-chip">{html.escape(item.name)}</span>'
            for item in preview
        )
        extra = (
            f' <span class="file-chip">+{len(uploaded_files) - 6} more</span>'
            if len(uploaded_files) > 6
            else ""
        )
        st.markdown(
            f"<div>{len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''} ready<br/>{chips}{extra}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("PDF or TXT. You can select several files at once.")

rank_clicked = st.button("Rank Candidates", type="primary", use_container_width=True)

if rank_clicked:
    api_key = get_api_key()
    errors = []
    if not api_key:
        errors.append("No Gemini API key found in .streamlit/secrets.toml.")
    if not job_description.strip():
        errors.append("Paste a job description first.")
    if not uploaded_files:
        errors.append("Upload at least one PDF or TXT CV.")

    if errors:
        for message in errors:
            st.error(message)
    else:
        st.session_state.pop("ranking_results", None)
        st.session_state.pop("ranking_dataframe", None)
        st.session_state.pop("top_recommendation", None)
        candidates = []
        parse_errors = []
        text_limit = cv_char_limit(len(uploaded_files))
        for uploaded in uploaded_files:
            try:
                candidates.append(
                    {
                        "filename": uploaded.name,
                        "text": extract_text(uploaded, text_limit),
                    }
                )
            except Exception as exc:
                candidate_name = os.path.splitext(uploaded.name)[0]
                parse_errors.append(
                    {
                        "Name": candidate_name,
                        "Match_Score": 0,
                        "Key_Strengths": [],
                        "Missing_Requirements": [f"Failed to Analyze: {exc}"],
                        "Top_Recommendation_Status": "Failed to Analyze",
                    }
                )

        if not candidates:
            results = parse_errors
            st.error("No readable CVs were found.")
        else:
            progress = st.progress(0, text="Starting ranking...")
            status = st.empty()
            try:
                result = rank_candidates(
                    api_key,
                    job_description.strip(),
                    candidates,
                    progress=progress,
                    status=status,
                )
                results = result + parse_errors
            except Exception as exc:
                progress.empty()
                status.empty()
                results = parse_errors
                st.error(
                    "Ranking failed after retries. Gemini was overloaded or unreachable. "
                    f"Wait a minute and try again. Details: {exc}"
                )

        results.sort(key=lambda row: row["Match_Score"], reverse=True)
        top_candidate = next(
            (row for row in results if row["Top_Recommendation_Status"] == "Top Recommendation"),
            None,
        )
        if top_candidate:
            strengths = as_items(top_candidate.get("Key_Strengths"))
            rationale = " ".join(strengths[:2])
            recommendation_text = f"{top_candidate['Name']} ({top_candidate['Match_Score']}%). {rationale}"
        else:
            recommendation_text = "No candidate could be recommended because no CV was successfully analyzed."

        st.session_state["ranking_results"] = results
        st.session_state["ranking_dataframe"] = pd.DataFrame(
            results,
            columns=[
                "Name",
                "Match_Score",
                "Key_Strengths",
                "Missing_Requirements",
                "Top_Recommendation_Status",
            ],
        )
        st.session_state["top_recommendation"] = recommendation_text

if "ranking_results" in st.session_state:
    st.markdown('<div class="section-label">Top recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hire-card"><p class="hire-kicker">Top candidate</p>'
        f'<p class="hire-why">{html.escape(st.session_state["top_recommendation"])}</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Leaderboard</div>', unsafe_allow_html=True)
    ranking_dataframe = st.session_state["ranking_dataframe"]
    st.dataframe(ranking_dataframe, use_container_width=True, hide_index=True)
    st.download_button(
        "Download leaderboard as CSV",
        data=ranking_dataframe.to_csv(index=False).encode("utf-8"),
        file_name="candidate_leaderboard.csv",
        mime="text/csv",
    )
