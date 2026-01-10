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

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="Content QC Agent",
    page_icon="🧪",
    layout="wide",
)

# ======================================================
# HELPERS
# ======================================================
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)

def build_requests_session() -> requests.Session:
    """
    Retry transient failures; makes scraping more reliable on Streamlit Cloud.
    """
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
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
# AUTH
# ======================================================
def get_creds(uploaded_key=None):
    creds_info = None

    # Preferred: Streamlit secrets
    if "gcp_service_account" in st.secrets:
        creds_info = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_info and isinstance(creds_info["private_key"], str):
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

    # Optional: user-uploaded JSON
    elif uploaded_key is not None:
        try:
            creds_info = json.loads(uploaded_key.getvalue().decode("utf-8"))
        except Exception:
            creds_info = None

    # Fallback: local file (useful for local dev)
    else:
        for f in glob.glob("*.json"):
            if "service_account" in f:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        creds_info = json.load(fh)
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

# ======================================================
# GOOGLE DOC EXTRACTION (STRUCTURE SAFE)
# ======================================================
def read_doc_elements(elements):
    text = ""
    for e in elements or []:
        if "paragraph" in e:
            for pe in e["paragraph"].get("elements", []):
                text += pe.get("textRun", {}).get("content", "")
        elif "table" in e:
            for row in e["table"].get("tableRows", []):
                for cell in row.get("tableCells", []):
                    text += read_doc_elements(cell.get("content", [])) + " "
                text += "\n"
    return text

@st.cache_data(show_spinner=False, ttl=3600)
def get_doc_text_cached(doc_url: str, creds_json: str) -> str:
    """
    Cache doc fetches. creds_json is used only to scope the cache.
    (We do not display it; it just prevents cross-user cache bleed on shared infra.)
    """
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
        # Return empty string; caller will see row-level error handling
        raise RuntimeError(f"Google Docs API error: {e}") from e

def get_doc_text(creds, doc_url: str) -> str:
    """
    Non-cached fetch. If you want caching, we call get_doc_text_cached using the secrets JSON.
    """
    service = build("docs", "v1", credentials=creds, cache_discovery=False)
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", doc_url or "")
    if not match:
        return ""
    doc_id = match.group(1)
    doc = service.documents().get(documentId=doc_id).execute()
    return read_doc_elements(doc.get("body", {}).get("content", []))

# ======================================================
# WEB EXTRACTION (CONTENT-FIRST)
# ======================================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_web_text_cached(url: str) -> str:
    """
    Cache web fetches to reduce load + speed up re-runs.
    """
    headers = {
        "User-Agent": "QC-Bot/1.0 (+https://example.com)",  # doesn't need to resolve; just be stable
        "Accept": "text/html,application/xhtml+xml",
    }
    resp = SESSION.get(url, headers=headers, timeout=25)
    # Raise on hard failures so you can record a meaningful Error per row
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} fetching {url}")
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Try your preferred main container first
    content = soup.find(class_="page-content-area")
    text = content.get_text("\n") if content else soup.get_text("\n")
    return text

def get_web_text(url: str) -> str:
    return get_web_text_cached(url)

# ======================================================
# NORMALIZATION
# ======================================================
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def deweight_ui_text(text: str) -> str:
    """
    Tries to remove short UI-ish lines that can distort similarity.
    """
    lines = (text or "").splitlines()
    keep = []
    for l in lines:
        ll = l.strip()
        if not ll:
            continue
        if len(ll.split()) < 5:
            continue
        if ll.lower().startswith(("book", "schedule", "call us", "request", "contact")):
            continue
        keep.append(ll)
    return "\n".join(keep)

# ======================================================
# SECTION SPLITTING (CORE vs FAQ)
# ======================================================
def split_into_sections(text: str):
    lines = (text or "").splitlines()
    core = []
    faq = []

    buffer = []
    question_count = 0

    for line in lines:
        clean = line.strip()
        if not clean:
            continue

        if clean.endswith("?") or re.match(r"^(what|how|why|can|does|do)\b", clean.lower()):
            question_count += 1
            buffer.append(clean)
        else:
            if buffer:
                if question_count >= 2:
                    faq.extend(buffer)
                else:
                    core.extend(buffer)
            buffer = [clean]
            question_count = 0

    if buffer:
        if question_count >= 2:
            faq.extend(buffer)
        else:
            core.extend(buffer)

    return " ".join(core), " ".join(faq)

# ======================================================
# CHUNK-BASED SIMILARITY
# ======================================================
def chunk_similarity(doc: str, web: str, size=40, threshold=0.65) -> float:
    doc = doc or ""
    web = web or ""
    words = doc.split()
    if not words:
        return 0.0

    chunks = [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
    matches = 0

    # For speed: comparing chunk to *entire* web is expensive; but we keep your method.
    # If you later want: sliding windows or token hashing.
    for c in chunks:
        if difflib.SequenceMatcher(None, c, web).ratio() >= threshold:
            matches += 1

    return round((matches / max(len(chunks), 1)) * 100, 2)

def create_diff(doc: str, web: str) -> str:
    d = difflib.HtmlDiff(wrapcolumn=80)
    return d.make_file(
        safe_str(doc).splitlines(),
        safe_str(web).splitlines(),
        "Doc",
        "Web",
    )

# ======================================================
# UI
# ======================================================
st.title("Content QC Agent")

with st.sidebar:
    uploaded_key = None
    if "gcp_service_account" not in st.secrets:
        uploaded_key = st.file_uploader("Service Account JSON", type="json")

    sensitivity = st.slider("Core Content Threshold (%)", 70, 100, 90)
    use_cache = st.toggle("Cache Fetches (Faster Re-Runs)", value=True)

creds = get_creds(uploaded_key)
if not creds:
    st.error("Google credentials required")
    st.stop()

# If caching doc fetches, we need a stable JSON for cache scoping.
# Prefer secrets; if not present, use uploaded key contents.
creds_json_for_cache = None
if use_cache:
    if "gcp_service_account" in st.secrets:
        cj = dict(st.secrets["gcp_service_account"])
        if "private_key" in cj and isinstance(cj["private_key"], str):
            cj["private_key"] = cj["private_key"].replace("\\n", "\n")
        creds_json_for_cache = json.dumps(cj, sort_keys=True)
    elif uploaded_key is not None:
        try:
            creds_json_for_cache = uploaded_key.getvalue().decode("utf-8")
        except Exception:
            creds_json_for_cache = None

csv_file = st.file_uploader("Upload QC CSV", type="csv")

def get_required_col(rows, *candidates):
    """
    Try multiple possible header names.
    """
    if not rows:
        return None
    headers = set(rows[0].keys())
    for c in candidates:
        if c in headers:
            return c
    return None

if csv_file and st.button("Run QC"):
    raw = csv_file.getvalue().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw)))

    if not rows:
        st.warning("CSV appears empty.")
        st.stop()

    url_col = get_required_col(rows, "URL", "Url", "url")
    doc_col = get_required_col(rows, "google_doc_url", "Google Doc URL", "Doc URL", "doc_url")
    page_col = get_required_col(rows, "Page Title", "Page", "Title", "page_title")

    if not url_col or not doc_col:
        st.error(
            "CSV is missing required columns. Need at least: "
            "'URL' and 'google_doc_url' (header names can vary)."
        )
        st.stop()

    results = []

    progress = st.progress(0)
    total = len(rows)

    for i, r in enumerate(rows, start=1):
        url = safe_str(r.get(url_col)).strip()
        doc_url = safe_str(r.get(doc_col)).strip()
        page_title = safe_str(r.get(page_col)).strip() if page_col else ""

        # Build a fallback label if Page Title is missing (prevents later selection issues)
        display_page = page_title or url or f"Row {i}"

        try:
            if not url:
                raise ValueError("Missing URL")
            if not doc_url:
                raise ValueError("Missing google_doc_url")

            # Fetch & normalize
            if use_cache and creds_json_for_cache:
                doc_text = get_doc_text_cached(doc_url, creds_json_for_cache)
            else:
                doc_text = get_doc_text(creds, doc_url)

            web_text = get_web_text(url)

            doc_raw = normalize(doc_text)
            web_raw = normalize(deweight_ui_text(web_text))

            doc_core, doc_faq = split_into_sections(doc_raw)
            web_core, web_faq = split_into_sections(web_raw)

            core_score = chunk_similarity(doc_core, web_core)
            faq_score = chunk_similarity(doc_faq, web_faq) if doc_faq else 100.0

            final_score = round((core_score * 0.75) + (faq_score * 0.25), 2)

            if core_score >= sensitivity and faq_score >= 60:
                status = "MATCH"
            elif core_score >= sensitivity:
                status = "STRUCTURAL OK / CONTENT DRIFT"
            else:
                status = "MISMATCH"

            results.append(
                {
                    "Page": display_page,
                    "URL": url,
                    "Doc URL": doc_url,
                    "Status": status,
                    "Core %": core_score,
                    "FAQ %": faq_score,
                    "Final %": final_score,
                    "Diff": create_diff(doc_raw, web_raw) if status != "MATCH" else "",
                    "Error": "",
                }
            )

        except Exception as e:
            results.append(
                {
                    "Page": display_page,
                    "URL": url,
                    "Doc URL": doc_url,
                    "Status": "ERROR",
                    "Core %": 0.0,
                    "FAQ %": 0.0,
                    "Final %": 0.0,
                    "Diff": "",
                    "Error": repr(e),
                }
            )

        progress.progress(i / total)

    df = pd.DataFrame(results)

    # Show table (future-proof Streamlit param)
    display_df = df.drop(columns=["Diff"])
    st.dataframe(display_df, width="stretch")

    # Focus on anything not MATCH
    problem_rows = df[df["Status"] != "MATCH"].copy()

    if not problem_rows.empty:
        st.download_button(
            "Download Review CSV",
            problem_rows.drop(columns=["Diff"]).to_csv(index=False).encode("utf-8"),
            "qc_review.csv",
        )

        # FUTURE-PROOF SELECTION:
        # Select by row index (stable), not by Page title matching (brittle).
        pr = problem_rows.reset_index(drop=True)

        labels = pr["Page"].fillna("").astype(str).str.strip()
        # If somehow blank, fall back to URL or a row label
        url_fallback = pr["URL"].fillna("").astype(str).str.strip()
        labels = labels.where(labels != "", other=url_fallback)
        labels = labels.where(labels != "", other=("Row " + (pr.index + 1).astype(str)))

        sel_i = st.selectbox(
            "Inspect page",
            options=list(pr.index),
            format_func=lambda idx: f"{labels.iloc[idx]}  —  {pr.loc[idx, 'Status']}",
        )

        diff_html = pr.loc[sel_i, "Diff"]
        err_msg = pr.loc[sel_i, "Error"]

        if err_msg:
            st.error(f"Row Error: {err_msg}")

        if isinstance(diff_html, str) and diff_html.strip():
            components.html(diff_html, height=600, scrolling=True)
        else:
            st.info("No diff available for this row (MATCH rows don’t generate diffs, and ERROR rows may fail before diffing).")
    else:
        st.success("All rows are MATCH ✅")
