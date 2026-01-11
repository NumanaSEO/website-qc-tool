import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import re
import difflib
import unicodedata
import io
import csv
import glob
import json
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import Counter

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(page_title="Content QC Agent", page_icon="🧪", layout="wide")

# Persist results across reruns (dropdown changes, etc.)
if "qc_df" not in st.session_state:
    st.session_state.qc_df = None
if "qc_problem_rows" not in st.session_state:
    st.session_state.qc_problem_rows = None
if "qc_ran" not in st.session_state:
    st.session_state.qc_ran = False

# ======================================================
# REQUESTS SESSION (RETRIES)
# ======================================================
def build_requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = build_requests_session()

# ======================================================
# SMALL UTILS
# ======================================================
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)

def first_existing_col(rows, *candidates):
    if not rows:
        return None
    headers = set(rows[0].keys())
    for c in candidates:
        if c in headers:
            return c
    return None

# ======================================================
# AUTH
# ======================================================
def get_creds(uploaded_key=None):
    creds_info = None

    if "gcp_service_account" in st.secrets:
        creds_info = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_info and isinstance(creds_info["private_key"], str):
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

    elif uploaded_key is not None:
        try:
            creds_info = json.loads(uploaded_key.getvalue().decode("utf-8"))
            if "private_key" in creds_info and isinstance(creds_info["private_key"], str):
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        except Exception:
            creds_info = None

    else:
        for f in glob.glob("*.json"):
            if "service_account" in f:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        creds_info = json.load(fh)
                    if "private_key" in creds_info and isinstance(creds_info["private_key"], str):
                        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
                    break
                except Exception:
                    pass

    if not creds_info:
        return None

    return service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=[
            "https://www.googleapis.com/auth/documents.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    )

def creds_json_for_cache(uploaded_key=None):
    """
    Used only for cache scoping; avoids cross-user cache bleed.
    """
    try:
        if "gcp_service_account" in st.secrets:
            cj = dict(st.secrets["gcp_service_account"])
            if "private_key" in cj and isinstance(cj["private_key"], str):
                cj["private_key"] = cj["private_key"].replace("\\n", "\n")
            return json.dumps(cj, sort_keys=True)

        if uploaded_key is not None:
            cj = json.loads(uploaded_key.getvalue().decode("utf-8"))
            if "private_key" in cj and isinstance(cj["private_key"], str):
                cj["private_key"] = cj["private_key"].replace("\\n", "\n")
            return json.dumps(cj, sort_keys=True)
    except Exception:
        return None

    return None

# ======================================================
# GOOGLE DOC EXTRACTION (STRUCTURE SAFE, TABLE SAFE)
# ======================================================
def read_doc_elements(elements):
    text = ""
    for e in elements or []:
        if "paragraph" in e:
            for pe in e["paragraph"].get("elements", []):
                text += pe.get("textRun", {}).get("content", "")
        elif "table" in e:
            for row in e["table"].get("tableRows", []):
                row_cells = []
                for cell in row.get("tableCells", []):
                    row_cells.append(read_doc_elements(cell.get("content", [])))
                text += " | ".join([c.strip() for c in row_cells if c.strip()]) + "\n"
    return text

@st.cache_data(show_spinner=False, ttl=3600)
def get_doc_text_cached(doc_url: str, creds_json: str) -> str:
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=[
            "https://www.googleapis.com/auth/documents.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    )
    service = build("docs", "v1", credentials=creds, cache_discovery=False)

    match = re.search(r"/d/([a-zA-Z0-9-_]+)", doc_url or "")
    if not match:
        return ""
    doc_id = match.group(1)

    try:
        doc = service.documents().get(documentId=doc_id).execute()
        return read_doc_elements(doc.get("body", {}).get("content", []))
    except HttpError as e:
        raise RuntimeError(f"Google Docs API error: {e}") from e

def get_doc_text(creds, doc_url: str) -> str:
    service = build("docs", "v1", credentials=creds, cache_discovery=False)
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", doc_url or "")
    if not match:
        return ""
    doc_id = match.group(1)
    doc = service.documents().get(documentId=doc_id).execute()
    return read_doc_elements(doc.get("body", {}).get("content", []))

# ======================================================
# WEB EXTRACTION (OXYGEN: REQUIRED .page-content-area)
# ======================================================
def extract_structured_text(soup: BeautifulSoup, content_tag) -> str:
    """
    Adds minimal separators for readability. Does NOT remove content.
    Keeps tables + accordion text if present in DOM.
    """
    # Tables: delimit cells and rows
    for table in content_tag.find_all("table"):
        for tr in table.find_all("tr"):
            tr.append(soup.new_string("\n"))
        for cell in table.find_all(["th", "td"]):
            cell.append(soup.new_string(" | "))

    # Add newlines after block-like elements (helps readability; not used for scoring after whitespace collapse)
    block_tags = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div", "section", "article"]
    for tag in content_tag.find_all(block_tags):
        tag.append(soup.new_string("\n"))

    for br in content_tag.find_all("br"):
        br.replace_with("\n")

    text = content_tag.get_text(separator=" ", strip=False)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def fetch_web_text_uncached(url: str) -> str:
    headers = {"User-Agent": "QC-Bot/1.0"}
    resp = SESSION.get(url, headers=headers, timeout=25)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} fetching {url}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # remove things that are never content
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    content = soup.find(class_="page-content-area")
    if not content:
        raise RuntimeError("Could not find required .page-content-area container")

    return extract_structured_text(soup, content)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_web_text_cached(url: str) -> str:
    return fetch_web_text_uncached(url)

def get_web_text(url: str, use_cache: bool) -> str:
    return fetch_web_text_cached(url) if use_cache else fetch_web_text_uncached(url)

# ======================================================
# QC NORMALIZATION + TOKENIZATION
# ======================================================
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def qc_normalize(text: str, normalize_smart_punct: bool = True) -> str:
    """
    Layout-insensitive, punctuation-aware normalization.
    Collapses whitespace but keeps punctuation tokens.
    """
    t = text or ""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u200b", "").replace("\ufeff", "")  # zero-width/BOM
    t = t.replace("\xa0", " ")  # NBSP

    if normalize_smart_punct:
        t = (
            t.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'")
             .replace("—", "-").replace("–", "-")
             .replace("…", "...")
        )

    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")

# ======================================================
# NOISE REMOVAL (PREVENT FORMAT/UI FALSE POSITIVES)
# ======================================================
NOISE_PATTERNS = [
    r"^https?$",
    r"^www$",
    r"^https?://",
    r"\.(jpg|jpeg|png|gif|webp|svg|mp4|mov|avi|m4v|pdf)$",
    r"^wpdrpurl$",
    r"^wp$",
    r"^wordpress$",
]

# These are OPTIONAL; add/remove based on what your pages commonly inject.
NOISE_PHRASES = [
    "schedule an appointment",
    "book an appointment",
    "hear from",
]

STOP_TOKENS = {
    ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", '"', "'",
}

def remove_noise_phrases(text: str) -> str:
    t = text or ""
    tl = t.lower()
    for p in NOISE_PHRASES:
        tl = tl.replace(p, " ")
    return tl

def filter_tokens(tokens: list[str], ignore_pipes: bool = True) -> list[str]:
    if not tokens:
        return []

    out = []
    for t in tokens:
        tl = t.lower()

        # Ignore the literal pipe token introduced for table readability
        if ignore_pipes and t == "|":
            continue

        # Drop url-ish/media-ish tokens
        if any(re.search(p, tl) for p in NOISE_PATTERNS):
            continue

        out.append(t)

    return out

# ======================================================
# SCORING
# ======================================================
def token_sequence_similarity(doc_tokens: list[str], web_tokens: list[str]) -> float:
    """
    Sequence alignment score (order-sensitive). Useful for diagnostics,
    but NOT a good single source of truth for web vs doc.
    """
    if not doc_tokens and not web_tokens:
        return 100.0
    if not doc_tokens or not web_tokens:
        return 0.0
    sm = difflib.SequenceMatcher(a=doc_tokens, b=web_tokens, autojunk=False)
    return round(sm.ratio() * 100, 2)

def doc_coverage_score(doc_tokens: list[str], web_tokens: list[str]) -> float:
    """
    Primary QC score: percent of DOC tokens found on WEB (counts matter).
    Robust to formatting/reordering and repeated UI blocks.

    Ignores pure punctuation tokens to reduce formatting noise.
    """
    if not doc_tokens and not web_tokens:
        return 100.0
    if not doc_tokens:
        return 100.0
    if not web_tokens:
        return 0.0

    d = [t.lower() for t in doc_tokens if t not in STOP_TOKENS]
    w = [t.lower() for t in web_tokens if t not in STOP_TOKENS]

    if not d and not w:
        return 100.0
    if not d:
        return 100.0
    if not w:
        return 0.0

    dc = Counter(d)
    wc = Counter(w)

    matched = 0
    total = sum(dc.values())
    for tok, cnt in dc.items():
        matched += min(cnt, wc.get(tok, 0))

    return round((matched / max(total, 1)) * 100, 2)

def token_diff_stats(doc_tokens: list[str], web_tokens: list[str]) -> dict:
    sm = difflib.SequenceMatcher(a=doc_tokens, b=web_tokens, autojunk=False)
    rep = ins = dele = eq = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            eq += (i2 - i1)
        elif tag == "replace":
            rep += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            ins += (j2 - j1)
        elif tag == "delete":
            dele += (i2 - i1)
    return {"equal": eq, "replace": rep, "insert": ins, "delete": dele}

# ======================================================
# DIFF RENDERING (FOR HUMAN REVIEW)
# ======================================================
def tokens_to_lines(tokens: list[str], per_line: int = 14) -> list[str]:
    return [" ".join(tokens[i:i + per_line]) for i in range(0, len(tokens), per_line)]

def create_token_diff_html(doc_tokens: list[str], web_tokens: list[str]) -> str:
    d = difflib.HtmlDiff(wrapcolumn=120)
    return d.make_file(
        tokens_to_lines(doc_tokens),
        tokens_to_lines(web_tokens),
        "Doc (Tokens)",
        "Web (Tokens)",
        context=True,
        numlines=2,
    )

# ======================================================
# UI
# ======================================================
st.title("Content QC Agent")

with st.sidebar:
    uploaded_key = None
    if "gcp_service_account" not in st.secrets:
        uploaded_key = st.file_uploader("Service Account JSON", type="json")

    st.markdown("### QC Settings")
    sensitivity = st.slider("Coverage Threshold (%)", 70, 100, 98)
    normalize_smart_punct = st.toggle("Ignore Smart Quotes/Dashes", value=True)
    ignore_pipes = st.toggle("Ignore Table Pipes (|)", value=True)
    strip_noise_phrases = st.toggle("Strip Common CTA Phrases", value=True)
    use_cache = st.toggle("Cache Fetches (Faster Re-Runs)", value=True)

    st.markdown("### Notes")
    st.caption(
        "Coverage is the primary pass/fail (robust to formatting/reordering). "
        "Sequence % is shown for diagnostics but not used to fail a page."
    )

creds = get_creds(uploaded_key)
if not creds:
    st.error("Google credentials required")
    st.stop()

creds_cache_key = creds_json_for_cache(uploaded_key) if use_cache else None

csv_file = st.file_uploader("Upload QC CSV", type="csv")
run_clicked = st.button("Run QC")

if run_clicked:
    raw = csv_file.getvalue().decode("utf-8-sig") if csv_file else ""
    rows = list(csv.DictReader(io.StringIO(raw))) if raw else []

    if not rows:
        st.warning("Upload a CSV first (or CSV appears empty).")
        st.session_state.qc_df = None
        st.session_state.qc_problem_rows = None
        st.session_state.qc_ran = False
    else:
        url_col = first_existing_col(rows, "URL", "Url", "url")
        doc_col = first_existing_col(rows, "google_doc_url", "Google Doc URL", "Doc URL", "doc_url")
        page_col = first_existing_col(rows, "Page Title", "Page", "Title", "page_title")

        if not url_col or not doc_col:
            st.error("CSV must include columns for URL and google_doc_url (header names can vary).")
            st.session_state.qc_df = None
            st.session_state.qc_problem_rows = None
            st.session_state.qc_ran = False
        else:
            results = []
            total = len(rows)
            progress = st.progress(0)

            for i, r in enumerate(rows, start=1):
                url = safe_str(r.get(url_col)).strip()
                doc_url = safe_str(r.get(doc_col)).strip()
                page_title = safe_str(r.get(page_col)).strip() if page_col else ""
                display_page = page_title or url or f"Row {i}"

                try:
                    if not url:
                        raise ValueError("Missing URL")
                    if not doc_url:
                        raise ValueError("Missing google_doc_url")

                    # Fetch texts
                    if use_cache and creds_cache_key:
                        doc_text = get_doc_text_cached(doc_url, creds_cache_key)
                    else:
                        doc_text = get_doc_text(creds, doc_url)

                    web_text = get_web_text(url, use_cache=use_cache)

                    # QC normalization
                    doc_qc = qc_normalize(doc_text, normalize_smart_punct=normalize_smart_punct)
                    web_qc = qc_normalize(web_text, normalize_smart_punct=normalize_smart_punct)

                    # Optional: strip common CTA phrases (helps reduce false mismatches)
                    if strip_noise_phrases:
                        doc_qc = remove_noise_phrases(doc_qc)
                        web_qc = remove_noise_phrases(web_qc)

                    # Tokenize + filter
                    doc_tokens = filter_tokens(tokenize(doc_qc), ignore_pipes=ignore_pipes)
                    web_tokens = filter_tokens(tokenize(web_qc), ignore_pipes=ignore_pipes)

                    # Primary score: coverage
                    coverage_pct = doc_coverage_score(doc_tokens, web_tokens)

                    # Secondary diagnostic: sequence
                    seq_pct = token_sequence_similarity(doc_tokens, web_tokens)

                    stats = token_diff_stats(doc_tokens, web_tokens)

                    warning = ""
                    if len(doc_tokens) > 0 and len(web_tokens) < 0.5 * len(doc_tokens):
                        warning = (
                            f"Web text much shorter than doc (web={len(web_tokens)} tokens, doc={len(doc_tokens)}). "
                            "Possible missing DOM content (JS-injected accordion) or extraction issue."
                        )
                    elif coverage_pct >= sensitivity and seq_pct < sensitivity:
                        warning = (
                            "High coverage but low sequence alignment. Likely formatting/reordering/UI block differences "
                            "(content probably copied correctly)."
                        )

                    status = "MATCH" if coverage_pct >= sensitivity else "MISMATCH"

                    # Only generate diff when coverage fails (otherwise diff can be misleading)
                    diff_html = create_token_diff_html(doc_tokens, web_tokens) if status != "MATCH" else ""

                    results.append(
                        {
                            "Page": display_page,
                            "URL": url,
                            "Doc URL": doc_url,
                            "Status": status,
                            "Coverage %": coverage_pct,
                            "Sequence %": seq_pct,
                            "Replaced": stats["replace"],
                            "Inserted": stats["insert"],
                            "Deleted": stats["delete"],
                            "Warning": warning,
                            "Error": "",
                            "Diff": diff_html,
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "Page": display_page,
                            "URL": url,
                            "Doc URL": doc_url,
                            "Status": "ERROR",
                            "Coverage %": 0.0,
                            "Sequence %": 0.0,
                            "Replaced": 0,
                            "Inserted": 0,
                            "Deleted": 0,
                            "Warning": "",
                            "Error": repr(e),
                            "Diff": "",
                        }
                    )

                progress.progress(i / total)

            df = pd.DataFrame(results)
            st.session_state.qc_df = df
            st.session_state.qc_problem_rows = df[df["Status"] != "MATCH"].reset_index(drop=True)
            st.session_state.qc_ran = True

# ======================================================
# DISPLAY (renders on every rerun if results exist)
# ======================================================
if st.session_state.qc_df is not None:
    df = st.session_state.qc_df
    st.dataframe(df.drop(columns=["Diff"]), width="stretch")

    problem_rows = st.session_state.qc_problem_rows

    if problem_rows is not None and not problem_rows.empty:
        st.download_button(
            "Download Review CSV",
            problem_rows.drop(columns=["Diff"]).to_csv(index=False).encode("utf-8"),
            "qc_review.csv",
        )

        labels = problem_rows["Page"].fillna("").astype(str).str.strip()
        labels = labels.where(labels != "", other=problem_rows["URL"].fillna("").astype(str).str.strip())
        labels = labels.where(labels != "", other=("Row " + (problem_rows.index + 1).astype(str)))

        sel_i = st.selectbox(
            "Inspect page",
            options=list(problem_rows.index),
            key="inspect_selectbox",
            format_func=lambda idx: (
                f"{labels.iloc[idx]} — {problem_rows.loc[idx, 'Status']} — "
                f"Coverage {problem_rows.loc[idx, 'Coverage %']}% / Seq {problem_rows.loc[idx, 'Sequence %']}%"
            ),
        )

        err_msg = safe_str(problem_rows.loc[sel_i, "Error"]).strip()
        warn_msg =_
