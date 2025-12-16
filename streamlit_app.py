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

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="Content QC Agent",
    page_icon="🧪",
    layout="wide"
)

# ======================================================
# AUTH
# ======================================================
def get_creds(uploaded_key=None):
    creds_info = None

    if "gcp_service_account" in st.secrets:
        creds_info = dict(st.secrets["gcp_service_account"])
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

    elif uploaded_key:
        creds_info = json.loads(uploaded_key.getvalue().decode("utf-8"))

    else:
        for f in glob.glob("*.json"):
            if "service_account" in f:
                with open(f) as fh:
                    creds_info = json.load(fh)
                    break

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
    for e in elements:
        if "paragraph" in e:
            for pe in e["paragraph"].get("elements", []):
                text += pe.get("textRun", {}).get("content", "")
        elif "table" in e:
            for row in e["table"].get("tableRows", []):
                for cell in row.get("tableCells", []):
                    text += read_doc_elements(cell.get("content", [])) + " "
                text += "\n"
    return text

def get_doc_text(creds, doc_url):
    service = build("docs", "v1", credentials=creds)
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", doc_url)
    if not match:
        return ""
    doc_id = match.group(1)
    doc = service.documents().get(documentId=doc_id).execute()
    return read_doc_elements(doc["body"]["content"])

# ======================================================
# WEB EXTRACTION (CONTENT-FIRST)
# ======================================================
def get_web_text(url):
    headers = {"User-Agent": "QC-Bot"}
    html = requests.get(url, headers=headers, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    content = soup.find(class_="page-content-area")
    text = content.get_text("\n") if content else soup.get_text("\n")
    return text

# ======================================================
# NORMALIZATION
# ======================================================
def normalize(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def deweight_ui_text(text):
    lines = text.splitlines()
    keep = []
    for l in lines:
        if len(l.split()) < 5:
            continue
        if l.lower().startswith(("book", "schedule", "call us", "request", "contact")):
            continue
        keep.append(l)
    return " ".join(keep)

# ======================================================
# SECTION SPLITTING (CORE vs FAQ)
# ======================================================
def split_into_sections(text):
    lines = text.splitlines()
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
            if question_count >= 2:
                faq.extend(buffer)
            else:
                core.extend(buffer)
            buffer = [clean]
            question_count = 0

    if question_count >= 2:
        faq.extend(buffer)
    else:
        core.extend(buffer)

    return " ".join(core), " ".join(faq)

# ======================================================
# CHUNK-BASED SIMILARITY
# ======================================================
def chunk_similarity(doc, web, size=40, threshold=0.65):
    words = doc.split()
    chunks = [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

    matches = 0
    for c in chunks:
        if difflib.SequenceMatcher(None, c, web).ratio() >= threshold:
            matches += 1

    return round((matches / max(len(chunks), 1)) * 100, 2)

def create_diff(doc, web):
    d = difflib.HtmlDiff(wrapcolumn=80)
    return d.make_file(doc.splitlines(), web.splitlines(), "Doc", "Web")

# ======================================================
# UI
# ======================================================
st.title("Content QC Agent")

with st.sidebar:
    uploaded_key = None
    if "gcp_service_account" not in st.secrets:
        uploaded_key = st.file_uploader("Service Account JSON", type="json")

    sensitivity = st.slider("Core Content Threshold (%)", 70, 100, 90)

creds = get_creds(uploaded_key)
if not creds:
    st.error("Google credentials required")
    st.stop()

csv_file = st.file_uploader("Upload QC CSV", type="csv")

if csv_file and st.button("Run QC"):
    rows = list(csv.DictReader(io.StringIO(csv_file.getvalue().decode("utf-8-sig"))))
    results = []

    for r in rows:
        url = r.get("URL")
        doc_url = r.get("google_doc_url")

        try:
            doc_raw = normalize(get_doc_text(creds, doc_url))
            web_raw = normalize(deweight_ui_text(get_web_text(url)))

            doc_core, doc_faq = split_into_sections(doc_raw)
            web_core, web_faq = split_into_sections(web_raw)

            core_score = chunk_similarity(doc_core, web_core)
            faq_score = chunk_similarity(doc_faq, web_faq) if doc_faq else 100

            final_score = round((core_score * 0.75) + (faq_score * 0.25), 2)

            if core_score >= sensitivity and faq_score >= 60:
                status = "MATCH"
            elif core_score >= sensitivity:
                status = "STRUCTURAL OK / CONTENT DRIFT"
            else:
                status = "MISMATCH"

            results.append({
                "Page": r.get("Page Title"),
                "Status": status,
                "Core %": core_score,
                "FAQ %": faq_score,
                "Final %": final_score,
                "Diff": create_diff(doc_raw, web_raw) if status != "MATCH" else None
            })

        except Exception as e:
            results.append({
                "Page": r.get("Page Title"),
                "Status": "ERROR",
                "Core %": 0,
                "FAQ %": 0,
                "Final %": 0,
                "Diff": None
            })

    df = pd.DataFrame(results)

    st.dataframe(
        df.drop(columns=["Diff"]),
        use_container_width=True
    )

    problem_rows = df[df.Status != "MATCH"]
    if not problem_rows.empty:
        st.download_button(
            "Download Review CSV",
            problem_rows.drop(columns=["Diff"]).to_csv(index=False).encode(),
            "qc_review.csv"
        )

        sel = st.selectbox("Inspect page", problem_rows.Page)
        diff = problem_rows[problem_rows.Page == sel].iloc[0]["Diff"]
        if diff:
            components.html(diff, height=600, scrolling=True)
