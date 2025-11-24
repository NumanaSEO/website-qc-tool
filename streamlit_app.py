import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import re
import difflib
import textwrap
import unicodedata
import zipfile
import io
import csv
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build

# --- CONFIGURATION ---
st.set_page_config(page_title="QC Tool", page_icon="✅", layout="wide")

# Initialize Session State to hold results
if 'qc_results' not in st.session_state:
    st.session_state['qc_results'] = []

# --- AUTHENTICATION ---
def get_service(json_file):
    try:
        import json
        file_content = json_file.getvalue().decode("utf-8")
        creds_dict = json.loads(file_content)
        
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/documents.readonly"]
        )
        return build("docs", "v1", credentials=creds)
    except Exception as e:
        st.error(f"❌ Auth Error: {e}")
        return None

# --- SCRAPERS ---
def get_doc_text(service, doc_url):
    try:
        if "/d/" in doc_url:
            doc_id = doc_url.split("/d/")[1].split("/")[0]
        else:
            doc_id = doc_url
        
        doc = service.documents().get(documentId=doc_id).execute()
        text = ""
        for value in doc.get('body').get('content'):
            if 'paragraph' in value:
                for elem in value.get('paragraph').get('elements'):
                    text += elem.get('textRun', {}).get('content', '')
        return text
    except Exception as e:
        return f"Error: {e}"

def get_web_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript", "aside", "header", "head", "title", "meta"]):
            tag.decompose()

        junk_selectors = [
            # Standard / Cookie
            "#cookie-law-info-bar", ".cli-modal", "#cookie-law-info-again", 
            ".moove-gdpr-cookie-compliance", "#moove_gdpr_cookie_modal",
            
            # Standard Theme Structures
            ".site-header", "#masthead", ".main-navigation", ".top-bar",
            
            # Breadcrumbs (Added Rank Math)
            ".breadcrumbs", ".breadcrumb", "#breadcrumbs", ".yoast-breadcrumbs",
            ".rank-math-breadcrumb", # <--- Added for Rank Math
            
            # Accessibility & Hidden
            ".screen-reader-text", ".sr-only", ".visually-hidden", ".elementor-screen-only",
            ".hidden-desktop", ".hidden-mobile", ".hide-on-desktop", ".hide-on-mobile",
            
            # Oxygen Builder Specifics
            ".oxy-nav-menu-hamburger-wrap", # Mobile menu often hidden but present
            ".oxy-header-container",        # Common Oxygen Header wrapper
            ".ct-section-inner-wrap > .oxy-nav-menu" # Oxygen menus
        ]
        for selector in junk_selectors:
            for tag in soup.select(selector):
                tag.decompose()
            
        text = soup.get_text(separator='\n')
        return text.strip()
    except Exception as e:
        return f"Error: {e}"

def clean_web_content(web_text, doc_text):
    if not doc_text or not web_text:
        return web_text
    
    FOOTER_MARKERS = [
        "2025 All Rights Reserved", "Schedule a Consultation", 
        "Updates", "Testosterone Test", "Our Locations", 
        "Powered by GDPR", "Contact Us", "Subscribe"
    ]
    for marker in FOOTER_MARKERS:
        idx = web_text.find(marker)
        if idx != -1:
            if idx > len(web_text) * 0.3: 
                web_text = web_text[:idx]

    NOISE_PHRASES = [
        "book an appointment", "location", "woodbury, mn", 
        "plymouth, mn", "eagan, mn", "patient portal",
        "call us", "request an appointment",
        "name", "email", "phone", "message", "send", "subscribe",
        "contact us", "close", "square",
        "privacy overview", "strictly necessary cookies", "accept", 
        "close gdpr cookie settings", "gdpr cookie compliance", 
        "we are using cookies to give you the best experience on our website.",
        "name *", "email *", "phone *", "message *", "required",
        "skip to content", "open menu", "close menu"
    ]
    
    clean_lines = []
    for line in web_text.splitlines():
        line_stripped = line.strip()
        if not line_stripped: continue
        if line_stripped.lower() in NOISE_PHRASES: continue
        clean_lines.append(line_stripped)
    
    web_text = "\n".join(clean_lines)

    doc_words = re.findall(r'\S+', doc_text)
    if len(doc_words) < 20: return web_text

    def smart_escape(word):
        escaped = re.escape(word)
        for char in ['.', ',', '!', '?', ';', ':']:
            if word.endswith(char):
                target = re.escape(char)
                escaped = escaped.replace(target, r"\s*" + target)
        return escaped

    first_doc_line = next((line for line in doc_text.splitlines() if line.strip()), None)
    if first_doc_line:
        title_anchor = re.escape(first_doc_line.strip())
        match_title = re.search(title_anchor, web_text)
        if match_title:
             web_text = web_text[match_title.start():]

    end_words = doc_words[-10:]
    end_pattern = r"\s+".join([smart_escape(w) for w in end_words])
    match_end = re.search(end_pattern, web_text)
    
    if match_end:
        web_text = web_text[:match_end.end()]
    else:
        end_words_short = doc_words[-5:]
        end_pattern_short = r"\s+".join([smart_escape(w) for w in end_words_short])
        match_end_short = re.search(end_pattern_short, web_text)
        if match_end_short:
            web_text = web_text[:match_end_short.end()]
            
    return web_text

def normalize_punctuation(text):
    if not text: return ""
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        '“': '"', '”': '"', "‘": "'", "’": "'",
        '–': '-', '—': '-', '…': '...', '\xa0': ' ', '\u200b': ''
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()

def normalize_flow_text(text):
    if not text: return ""
    text = normalize_punctuation(text)
    words = text.split()
    stream = " ".join(words)
    return textwrap.fill(stream, width=80)

def create_html_diff(doc_text, web_text):
    doc_lines = doc_text.splitlines()
    web_lines = web_text.splitlines()
    differ = difflib.HtmlDiff(wrapcolumn=80)
    return differ.make_file(doc_lines, web_lines, fromdesc="Google Doc", todesc="Website")

# --- UI LOGIC ---
st.title("🚀 Website Quality Control Tool")

st.markdown("""
1. Upload your **Service Account JSON Key**.
2. Upload your **Checklist CSV** (Columns: `Page Title`, `URL`, `google_doc_url`).
3. Click **Run QC**.
""")

col1, col2 = st.columns(2)
with col1:
    key_file = st.file_uploader("🔑 Upload JSON Key", type=["json"])
with col2:
    csv_file = st.file_uploader("📂 Upload Checklist CSV", type=["csv"])

if st.button("Run QC", type="primary"):
    if not key_file or not csv_file:
        st.error("Please upload both files first!")
    else:
        service = get_service(key_file)
        
        if service:
            # Read CSV
            stringio = io.StringIO(csv_file.getvalue().decode("utf-8-sig"))
            reader = csv.DictReader(stringio)
            rows = list(reader)
            
            # Clear previous results
            st.session_state['qc_results'] = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in enumerate(rows):
                title = row.get('Page Title', 'Unknown')
                web_url = row.get('URL', '').strip()
                doc_url = row.get('google_doc_url', '').strip()
                
                status_text.text(f"Processing: {title}")
                progress_bar.progress((i + 1) / len(rows))
                
                if not web_url or not doc_url:
                    continue

                # Process
                doc_text = get_doc_text(service, doc_url)
                web_text = get_web_text(web_url)
                
                error_msg = None
                if "Error:" in doc_text: error_msg = doc_text
                if "Error:" in web_text: error_msg = web_text

                if error_msg:
                    st.session_state['qc_results'].append({
                        "Page Title": title, "Status": "ERROR", 
                        "Similarity": 0.0, "Notes": error_msg,
                        "html_diff": None
                    })
                    continue

                clean_web = clean_web_content(web_text, doc_text)
                doc_norm = normalize_flow_text(doc_text)
                web_norm = normalize_flow_text(clean_web)

                seq = difflib.SequenceMatcher(None, doc_norm, web_norm)
                similarity = seq.ratio() * 100
                status = "MATCH" if similarity > 95 else "MISMATCH"
                
                html_diff = None
                if status == "MISMATCH":
                    html_diff = create_html_diff(doc_norm, web_norm)

                st.session_state['qc_results'].append({
                    "Page Title": title, "Status": status, 
                    "Similarity": round(similarity, 2), "Notes": "OK" if status == "MATCH" else "Review Needed",
                    "html_diff": html_diff
                })

            status_text.text("✅ QC Complete!")

# --- DISPLAY RESULTS ---
if st.session_state['qc_results']:
    
    # 1. Summary Table
    df = pd.DataFrame(st.session_state['qc_results'])
    
    # Color coding helper
    def highlight_status(val):
        color = 'green' if val == 'MATCH' else 'red'
        return f'color: {color}; font-weight: bold'

    st.subheader("📊 Summary")
    st.dataframe(
        df[["Page Title", "Status", "Similarity", "Notes"]].style.applymap(highlight_status, subset=['Status']),
        use_container_width=True
    )
    
    # 2. Bulk Download
    csv_buffer = df[["Page Title", "Status", "Similarity", "Notes"]].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results (CSV)", csv_buffer, "QC_Results.csv", "text/csv")

    # 3. Detailed Inspector
    st.divider()
    st.subheader("🔍 Detailed Inspector")
    
    mismatches = [r for r in st.session_state['qc_results'] if r['Status'] == 'MISMATCH' or r['Status'] == 'ERROR']
    
    if not mismatches:
        st.success("🎉 Everything matches! No detailed inspection needed.")
    else:
        page_titles = [r['Page Title'] for r in mismatches]
        selected_page_title = st.selectbox("Select a page to inspect:", page_titles)
        
        # Find the data for selected page
        selected_data = next((item for item in mismatches if item["Page Title"] == selected_page_title), None)
        
        if selected_data:
            st.info(f"Similarity Score: {selected_data['Similarity']}%")
            if selected_data['html_diff']:
                # Render the HTML Diff
                components.html(selected_data['html_diff'], height=600, scrolling=True)
            else:
                st.warning("No visual report available (likely an error scraping the page).")