import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

MODEL_NAME = "gemini-3.8-flash"
MAX_CV_CHARS = 12_000
BATCH_SIZE = 8
MAX_WORKERS = 5
MAX_RETRIES = 3
RETRYABLE_MARKERS = (
    "503",
    "429",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "HIGH DEMAND",
    "TRY AGAIN",
    "OVERLOADED",
)

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


def build_batch_prompt(job_description: str, candidates: list[dict]) -> str:
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        blocks.append(
            f"--- CANDIDATE {index} ---\n"
            f"Source file: {candidate['filename']}\n"
            f"CV text:\n{candidate['text']}\n"
        )
    cv_bundle = "\n".join(blocks)
    return f"""You are an expert technical recruiter and hiring manager.

Score each candidate against the Job Description independently, using the same scale every time:
90-100 exceptional fit, 75-89 strong fit, 60-74 partial fit, below 60 weak fit.

JOB DESCRIPTION:
{job_description[:3500]}

CANDIDATE CVs:
{cv_bundle}

Return JSON only:
{{
  "leaderboard": [
    {{
      "name": "Candidate full name (infer from the CV; if missing, use a clean version of the filename)",
      "source_file": "exact source filename",
      "match_score": 0,
      "key_strengths": ["Concrete strength with evidence from the CV"],
      "weaknesses": ["Concrete gap, missing requirement, or risk versus the job description"]
    }}
  ]
}}

Rules:
- Include every candidate in this batch exactly once.
- match_score is an integer from 0 to 100.
- key_strengths and weaknesses must each contain exactly 3 short but complete phrases.
- Weaknesses should cover missing requirements and role risks.
- Be specific. Do not invent job history that is not in the CV.
- Keep the JSON compact.
"""


def generate_json(client, prompt: str) -> dict:
    last_error = None
    delay = 0.8
    for attempt in range(1, MAX_RETRIES + 1):
        try:
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
            if not isinstance(data, dict):
                raise ValueError("Gemini returned JSON that was not an object.")
            return data
        except Exception as exc:
            last_error = exc
            if not is_retryable(exc) or attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 4)
    raise last_error


def attach_source_files(rows: list[dict], candidates: list[dict]) -> list[dict]:
    unused = list(candidates)
    attached = []
    for row in rows:
        source = str(row.get("source_file") or "").strip().lower()
        match = None
        if source:
            match = next(
                (item for item in unused if item["filename"].lower() == source),
                None,
            )
        if match is None and unused:
            match = unused[0]
        if match is not None:
            unused.remove(match)
            row["source_file"] = match["filename"]
        attached.append(row)
    for leftover in unused:
        attached.append(
            {
                "name": os.path.splitext(leftover["filename"])[0],
                "source_file": leftover["filename"],
                "match_score": 0,
                "key_strengths": ["Ranking did not return details for this CV."],
                "weaknesses": ["Re-run ranking if this candidate should be scored."],
            }
        )
    return attached


def score_batch(client, job_description: str, candidates: list[dict]) -> list[dict]:
    data = generate_json(client, build_batch_prompt(job_description, candidates))
    rows = data.get("leaderboard")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Gemini returned an empty leaderboard for a batch.")
    return attach_source_files(rows, candidates)


def score_batch_resilient(client, job_description: str, candidates: list[dict]) -> list[dict]:
    try:
        return score_batch(client, job_description, candidates)
    except Exception as exc:
        if len(candidates) == 1 or not is_retryable(exc):
            raise
        mid = max(1, len(candidates) // 2)
        return score_batch_resilient(client, job_description, candidates[:mid]) + (
            score_batch_resilient(client, job_description, candidates[mid:])
        )


def fallback_justification(best: dict) -> str:
    strengths = as_items(best.get("key_strengths"))
    if len(strengths) >= 2:
        return f"{strengths[0]} {strengths[1]}"
    if strengths:
        return f"{best.get('name', 'This candidate')} is the strongest match. {strengths[0]}"
    return (
        f"{best.get('name', 'This candidate')} has the highest match score against the job description. "
        "Review the strengths and weaknesses below before making a final hire."
    )


def rank_candidates(
    api_key: str,
    job_description: str,
    candidates: list[dict],
    progress=None,
    status=None,
) -> dict:
    batches = list(batched(candidates, BATCH_SIZE))
    total = max(len(batches), 1)
    leaderboard = []
    failed = []
    finished = 0
    workers = min(MAX_WORKERS, total)

    if status is not None:
        status.info(f"Ranking {len(candidates)} CVs in {total} parallel batches...")
    if progress is not None:
        progress.progress(0, text="Starting parallel ranking...")

    def run_batch(batch: list[dict]) -> list[dict]:
        client = genai.Client(api_key=api_key)
        return score_batch_resilient(client, job_description, batch)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            finished += 1
            try:
                leaderboard.extend(future.result())
            except Exception as exc:
                failed.extend(item["filename"] for item in batch)
                for item in batch:
                    leaderboard.append(
                        {
                            "name": os.path.splitext(item["filename"])[0],
                            "source_file": item["filename"],
                            "match_score": 0,
                            "key_strengths": ["Could not score this CV because Gemini was overloaded."],
                            "weaknesses": [f"Retry this file later. Last error: {exc}"],
                        }
                    )
            if progress is not None:
                progress.progress(
                    finished / total,
                    text=f"Scored {finished} of {total} batches",
                )
            if status is not None:
                status.info(f"Scored {finished} of {total} batches")

    leaderboard.sort(key=lambda row: int(row.get("match_score", 0) or 0), reverse=True)
    if not leaderboard:
        raise ValueError("No candidates could be ranked.")

    best = next((row for row in leaderboard if int(row.get("match_score", 0) or 0) > 0), leaderboard[0])
    recommendation = {
        "name": str(best.get("name") or "Unknown candidate"),
        "justification": fallback_justification(best),
    }
    if status is not None:
        if failed:
            status.warning(
                "Ranked most CVs, but Gemini skipped some during high demand: "
                + ", ".join(failed)
            )
        else:
            status.empty()
    if progress is not None:
        progress.empty()
    return {"top_recommendation": recommendation, "leaderboard": leaderboard}


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
                parse_errors.append(f"{uploaded.name}: {exc}")

        if parse_errors:
            for message in parse_errors:
                st.warning(message)

        if not candidates:
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
                st.session_state["ranking_result"] = result
            except Exception as exc:
                progress.empty()
                status.empty()
                st.session_state.pop("ranking_result", None)
                st.error(
                    "Ranking failed after retries. Gemini was overloaded or unreachable. "
                    f"Wait a minute and try again. Details: {exc}"
                )

result = st.session_state.get("ranking_result")
if result:
    recommendation = result.get("top_recommendation", {})
    name = html.escape(str(recommendation.get("name", "Unknown candidate")))
    justification = html.escape(str(recommendation.get("justification", "")))

    st.markdown('<div class="section-label">Top recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hire-card">
          <p class="hire-kicker">Hire this candidate</p>
          <p class="hire-name">{name}</p>
          <p class="hire-why">{justification}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Leaderboard</div>', unsafe_allow_html=True)
    for index, row in enumerate(result.get("leaderboard", []), start=1):
        score = max(0, min(100, int(row.get("match_score", 0))))
        strengths = as_items(row.get("key_strengths") or row.get("Key Strengths"))
        weaknesses = as_items(
            row.get("weaknesses")
            or row.get("missing_requirements")
            or row.get("Missing Requirements")
        )
        person = html.escape(str(row.get("name", "Unknown")))
        st.markdown(
            f"""
            <div class="candidate-card">
              <div class="card-head">
                <div>
                  <span class="rank-pill">{index}</span>
                  <span class="cand-name">{person}</span>
                </div>
                <div class="score {score_class(score)}">{score}%</div>
              </div>
              <div class="box-grid">
                <div class="info-box strengths">
                  <h4>Strengths</h4>
                  {items_html(strengths, "No strengths noted.")}
                </div>
                <div class="info-box weaknesses">
                  <h4>Weaknesses</h4>
                  {items_html(weaknesses, "No weaknesses noted.")}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
