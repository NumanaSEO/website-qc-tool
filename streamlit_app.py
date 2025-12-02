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
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build

# --- CONFIGURATION ---
st.set_page_config(page_title="Content QC & Link Agent", page_icon="🏔️", layout="wide")

if 'qc_results' not in st.session_state:
    st.session_state['qc_results'] = []

# --- AUTHENTICATION ---
def get_creds(uploaded_key=None):
    creds_info = None
    if "gcp_service_account" in st.secrets:
        try:
            creds_info = dict(st.secrets["gcp_service_account"])
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        except Exception: pass

    if not creds_info and uploaded_key:
        try:
            creds_info = json.loads(uploaded_key.getvalue().decode("utf-8"))
        except Exception: pass

    if not creds_info:
        for k in glob.glob("*.json"):
            if "service_account" in k or "qc" in k:
                try:
                    with open(k, "r") as f:
                        creds_info = json.load(f)
                        break
                except: continue

    if creds_info:
        return service_account.Credentials.from_service_account_info(
            creds_info, 
            scopes=["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/documents.readonly", "https://www.googleapis.com/auth/drive.readonly"]
        )
    return None

# --- IMPROVED TEXT EXTRACTORS ---

def read_structural_elements(elements):
    """Recursively extracts text from Google Doc structures (Tables, Lists, Paragraphs)."""
    text = ""
    for value in elements:
        # 1. Handle Paragraphs & Lists
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                content = elem.get('textRun', {}).get('content', '')
                text += content
        
        # 2. Handle Tables (The Fix for Issue #1)
        elif 'table' in value:
            table = value.get('table')
            for row in table.get('tableRows', []):
                for cell in row.get('tableCells', []):
                    # Recursive call to get content inside cell
                    text += read_structural_elements(cell.get('content'))
                    text += " " # Add space between cells so words don't merge
                text += "\n" # Newline for rows
        
        # 3. Handle Section Breaks (optional, good for spacing)
        elif 'sectionBreak' in value:
            text += "\n"
            
    return text

def get_doc_text(creds, doc_url):
    try:
        service = build("docs", "v1", credentials=creds)
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url)
        if not match: return "Error: Invalid Doc URL"
        doc_id = match.group(1)
        
        doc = service.documents().get(documentId=doc_id).execute()
        # Use the new recursive function
        raw_text = read_structural_elements(doc.get('body').get('content'))
        return raw_text
    except Exception as e:
        return f"Error: {e}"

def get_web_text_clean(url):
    """Extracts text ONLY from 'page-content-area' if it exists."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (QC-Bot)'}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]): 
            tag.decompose()
        
        # Try Developer Class first
        content_area = soup.find(class_="page-content-area")
        
        if content_area:
            # Get text with separators to prevent words merging
            text = content_area.get_text(separator=' ')
        else:
            # Fallback cleaning
            oxy_junk = ["header", "footer", ".ct-header", ".ct-footer", ".oxy-header-container", ".oxy-nav-menu", ".ct-mobile-menu-icon", "#masthead", ".site-footer", ".screen-reader-text", ".visually-hidden", "#cookie-law-info-bar", ".moove-gdpr-cookie-compliance"]
            for selector in oxy_junk:
                for tag in soup.select(selector): tag.decompose()
            text = soup.get_text(separator=' ')

        return text.strip()
    except Exception as e:
        return f"Error: {e}"

# --- IMPROVED LINK EXTRACTORS ---
def get_doc_comments(creds, doc_url, ignored_authors=[]):
    try:
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url)
        if not match: return "Error: Invalid URL"
        doc_id = match.group(1)
        service = build('drive', 'v3', credentials=creds)
        
        results = service.comments().list(
            fileId=doc_id, 
            fields="comments(content, quotedFileContent, author(displayName))"
        ).execute()
        
        links = []
        for c in results.get('comments', []):
            if 'quotedFileContent' in c:
                author_name = c.get('author', {}).get('displayName', '')
                if author_name in ignored_authors: continue
                
                links.append({
                    'anchor': c['quotedFileContent']['value'].strip(), 
                    'instruction': c['content'].strip()
                })
        return links
    except Exception as e:
        return f"Error: {str(e)}"

def check_oxygen_link(html_content, anchor_text):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Define scope
    content_area = soup.find(class_="page-content-area")
    search_scope = content_area if content_area else soup
    if not content_area:
        for rubbish in soup.select('.ct-header, .ct-footer, header, footer, .oxy-nav-menu'): 
            rubbish.decompose()

    # Normalize for search
    clean_anchor = " ".join(anchor_text.split()).lower()
    
    # Find all text nodes containing the anchor text
    # We use a regex to find the text even if it spans elements slightly
    import re
    # Escape regex characters in anchor text
    safe_anchor = re.escape(clean_anchor)
    # Allow for some whitespace variation in the regex
    flexible_regex = safe_anchor.replace(r"\ ", r"\s+")
    
    # Search the HTML string directly to find if it exists at all first
    if clean_anchor not in " ".join(search_scope.get_text(separator=" ").lower().split()):
         return None, "Text not found on page body."

    # Now look for the link
    # Find specific text node
    target = search_scope.find(string=lambda t: t and clean_anchor in " ".join(t.split()).lower())
    
    if target:
        # Check parents for <a>
        curr = target.parent
        steps = 0
        while curr and steps < 12:
            if curr.name == 'a' and curr.has_attr('href'):
                # STRICT CHECK: Does the link cover the whole anchor?
                link_text = " ".join(curr.get_text(separator=" ").split()).lower()
                if clean_anchor in link_text:
                    return curr['href'], "Found (Verified)"
                else:
                    return None, f"Text found, but link only covers part of it: '{link_text}'"
            curr = curr.parent
            steps += 1
            
        return None, "Text found, but no Link tag detected."
    
    # Fallback: Deep Search
    return None, "Text found in raw HTML but element could not be isolated."

def verify_with_gemini(anchor, instruction, link, creds, region, model_name):
    if not link: return "FAIL", "Link missing"
    try:
        vertexai.init(project=creds.project_id, location=region, credentials=creds)
        model = GenerativeModel(model_name)
        prompt = f"""
        QA Task:
        Text: "{anchor}"
        Instruction: "{instruction}"
        Link found: "{link}"
        Does the Link satisfy the Instruction? 
        Return JSON ONLY: {{ "status": "PASS" or "FAIL", "reason": "short explanation" }}
        """
        response = model.generate_content(prompt, generation_config=GenerationConfig(response_mime_type="application/json"))
        data = json.loads(response.text)
        return data.get('status', 'FAIL'), data.get('reason', 'AI Error')
    except Exception as e:
        return "ERROR", str(e)

# --- UTILS ---
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text or "")
    
    # 1. Clean Quotes
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    
    # 2. Fix the "Space before Comma/Period" issue (Issue #2 Fix)
    # Replaces "word , word" with "word, word"
    text = re.sub(r'\s+([,.;:?!])', r'\1', text)
    
    # 3. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def create_diff(doc, web):
    d = difflib.HtmlDiff(wrapcolumn=80)
    return d.make_file(doc.splitlines(), web.splitlines(), fromdesc="Doc", todesc="Web")

# --- UI START ---
st.title("Content QC & Link Agent (v2)")

with st.sidebar:
    st.header("Settings")
    if "gcp_service_account" in st.secrets:
        st.success("✅ Authenticated")
        uploaded_key = None
    else:
        uploaded_key = st.file_uploader("Service Account JSON", type="json")
    
    st.divider()
    st.subheader("🔧 QC Config")
    sensitivity = st.slider("Strictness %", 80, 100, 95)
    
    use_staging = st.checkbox("Override Domain")
    staging_domain = st.text_input("New Domain") if use_staging else ""

    st.divider()
    st.subheader("🤖 AI Config")
    ai_region = st.selectbox("Region", ["us-central1", "us-west1", "us-east1"], index=0)
    ai_model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.0-flash-001"], index=0)

    st.divider()
    ignored_input = st.text_input("Ignore Comments By:", placeholder="Name 1, Name 2")
    ignored_authors = [x.strip() for x in ignored_input.split(',')] if ignored_input else []

creds = get_creds(uploaded_key)
if not creds:
    st.error("Please configure secrets.")
    st.stop()

tab1, tab2 = st.tabs(["📝 Text Comparison", "🔗 Link Auditor"])

# --- TAB 1: TEXT QC ---
with tab1:
    st.subheader("Bulk Text Comparison")
    csv_file_text = st.file_uploader("Upload Checklist CSV", type="csv", key="csv_text")
    if st.button("Run Text QC"):
        if csv_file_text:
            stringio = io.StringIO(csv_file_text.getvalue().decode("utf-8-sig"))
            rows = list(csv.DictReader(stringio))
            st.session_state['qc_results'] = []
            bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in enumerate(rows):
                url, doc_url = row.get('URL', ''), row.get('google_doc_url', '')
                if use_staging and staging_domain:
                     from urllib.parse import urlparse
                     path = urlparse(url).path
                     url = f"https://{staging_domain}{path}"

                status_text.text(f"Checking: {row.get('Page Title')}...")
                doc_txt = get_doc_text(creds, doc_url)
                web_txt = get_web_text_clean(url)
                
                doc_norm = normalize_text(doc_txt)
                web_norm = normalize_text(web_txt)
                
                seq = difflib.SequenceMatcher(None, doc_norm, web_norm)
                sim = round(seq.ratio() * 100, 2)
                
                status = "MATCH" if sim >= sensitivity else "MISMATCH"
                if "Error" in doc_txt or "Error" in web_txt: status = "ERROR"
                
                st.session_state['qc_results'].append({
                    "Title": row.get('Page Title'), "Status": status, "Score": sim,
                    "Diff": create_diff(doc_norm, web_norm) if status == "MISMATCH" else None
                })
                bar.progress((i+1)/len(rows))
            status_text.text("Done!")
            
    if st.session_state['qc_results']:
        df = pd.DataFrame(st.session_state['qc_results'])
        st.dataframe(df[["Title", "Status", "Score"]].style.applymap(
            lambda x: f'background-color: {"#d4edda" if x == "MATCH" else "#f8d7da" if x == "MISMATCH" else "#fff3cd"}', 
            subset=['Status']
        ), use_container_width=True)
        
        mismatch_df = df[(df['Status'] == 'MISMATCH') | (df['Status'] == 'ERROR')]
        if not mismatch_df.empty:
            st.download_button("🚨 Download Fixes", mismatch_df.drop(columns=['Diff']).to_csv(index=False).encode('utf-8'), "qc_fixes.csv", "text/csv")

        sel = st.selectbox("Inspect Mismatch", df[df['Status']=="MISMATCH"]['Title'].unique())
        if sel:
            d = next(item for item in st.session_state['qc_results'] if item["Title"] == sel)
            components.html(d['Diff'], height=600, scrolling=True)

# --- TAB 2: LINK AUDITOR ---
with tab2:
    st.subheader("Link Functionality & Intent Audit")
    mode = st.radio("Choose Input Method:", ["Bulk CSV Upload", "Single Page Check"], horizontal=True)
    links_to_process = [] 

    if mode == "Single Page Check":
        col_l1, col_l2 = st.columns(2)
        with col_l1: l_doc = st.text_input("Google Doc URL")
        with col_l2: l_url = st.text_input("Live Page URL")
        if l_doc and l_url: links_to_process.append({"Page Title": "Single Check", "URL": l_url, "google_doc_url": l_doc})
    else:
        csv_file_links = st.file_uploader("Upload Checklist CSV", type="csv", key="csv_links")
        if csv_file_links:
            stringio = io.StringIO(csv_file_links.getvalue().decode("utf-8-sig"))
            links_to_process = list(csv.DictReader(stringio))

    if st.button("Run Link Audit"):
        if not links_to_process:
            st.error("Please provide a CSV or Single URL.")
        else:
            final_report = []
            bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in enumerate(links_to_process):
                page_title = row.get('Page Title', 'Unknown')
                live_url, doc_url = row.get('URL', ''), row.get('google_doc_url', '')
                if use_staging and staging_domain:
                    from urllib.parse import urlparse
                    path = urlparse(live_url).path
                    live_url = f"https://{staging_domain}{path}"
                
                status_text.text(f"Scanning: {page_title}...")
                comments = get_doc_comments(creds, doc_url, ignored_authors=ignored_authors)
                
                if isinstance(comments, str) and comments.startswith("Error"):
                    final_report.append({"Page Title": page_title, "Status": "ERROR", "Reason": comments})
                elif not comments:
                    final_report.append({"Page Title": page_title, "Status": "INFO", "Reason": "No relevant instructions found"})
                else:
                    try:
                        resp = requests.get(live_url, headers={'User-Agent': 'QC-Bot'}, timeout=30)
                        html_content = resp.text
                        for item in comments:
                            link_href, status_msg = check_oxygen_link(html_content, item['anchor'])
                            status, reason = "MISSING", status_msg
                            
                            if link_href:
                                status, reason = verify_with_gemini(item['anchor'], item['instruction'], link_href, creds, ai_region, ai_model)
                            
                            final_report.append({
                                "Page Title": page_title, 
                                "Status": status, 
                                "Anchor Text": item['anchor'],
                                "Instruction": item['instruction'], 
                                "Found Link": link_href, 
                                "Reason": reason
                            })
                    except Exception as e:
                        final_report.append({"Page Title": page_title, "Status": "ERROR", "Reason": f"Website Error: {str(e)}"})

                bar.progress((i+1)/len(links_to_process))
            
            status_text.text("Audit Complete!")
            if final_report:
                res_df = pd.DataFrame(final_report)
                cols = ["Page Title", "Status", "Anchor Text", "Instruction", "Found Link", "Reason"]
                res_df = res_df[[c for c in cols if c in res_df.columns]]
                st.dataframe(res_df.style.applymap(lambda x: 'color:red;font-weight:bold' if x=='FAIL' else 'color:green;font-weight:bold' if x=='PASS' else 'color:orange', subset=['Status']), use_container_width=True)
                st.download_button("📥 Download Link Report CSV", res_df.to_csv(index=False).encode('utf-8'), "link_audit_report.csv", "text/csv")
