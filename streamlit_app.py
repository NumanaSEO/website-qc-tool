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
from requests.auth import HTTPBasicAuth

# --- CONFIGURATION ---
st.set_page_config(page_title="QC & Link Agent", page_icon="🏔️", layout="wide")

if 'qc_results' not in st.session_state:
    st.session_state['qc_results'] = []

# --- AUTHENTICATION ---
def get_creds(uploaded_key=None):
    creds_info = None
    # 1. Check Streamlit Secrets
    if "gcp_service_account" in st.secrets:
        try:
            creds_info = dict(st.secrets["gcp_service_account"])
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        except Exception:
            pass

    # 2. Check Upload
    if not creds_info and uploaded_key:
        try:
            creds_info = json.loads(uploaded_key.getvalue().decode("utf-8"))
        except Exception:
            pass

    # 3. Check Local File
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
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    return None

# --- TEXT EXTRACTORS (QC TOOL) ---
def get_doc_text(creds, doc_url):
    try:
        service = build("docs", "v1", credentials=creds)
        doc_id = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url).group(1)
        doc = service.documents().get(documentId=doc_id).execute()
        text = ""
        for value in doc.get('body').get('content'):
            if 'paragraph' in value:
                for elem in value.get('paragraph').get('elements'):
                    text += elem.get('textRun', {}).get('content', '')
        return text
    except Exception as e:
        return f"Error: {e}"

def get_web_text_clean(url, auth=None):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (QC-Bot)'}
        resp = requests.get(url, headers=headers, auth=auth, timeout=20)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Cleanup
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        # Oxygen & WP Cleanup
        oxy_junk = [
            "header", "footer", ".ct-header", ".ct-footer",
            ".oxy-header-container", ".oxy-nav-menu",
            ".ct-mobile-menu-icon", "#masthead", ".site-footer",
            ".screen-reader-text", ".visually-hidden",
            "#cookie-law-info-bar", ".moove-gdpr-cookie-compliance"
        ]
        for selector in oxy_junk:
            for tag in soup.select(selector):
                tag.decompose()
            
        text = soup.get_text(separator='\n')
        return text.strip()
    except Exception as e:
        return f"Error: {e}"

# --- LINK EXTRACTORS (LINK AGENT) ---
def get_doc_comments(creds, doc_url):
    try:
        doc_id = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url).group(1)
        service = build('drive', 'v3', credentials=creds)
        
        results = service.comments().list(
            fileId=doc_id, 
            fields="comments(content, quotedFileContent, author)"
        ).execute()
        
        links_to_check = []
        for c in results.get('comments', []):
            if 'quotedFileContent' in c:
                links_to_check.append({
                    'anchor': c['quotedFileContent']['value'].strip(),
                    'instruction': c['content'].strip(),
                })
        return links_to_check
    except Exception as e:
        return []

def check_oxygen_link(html_content, anchor_text):
    soup = BeautifulSoup(html_content, 'html.parser')
    for rubbish in soup.select('.ct-header, .ct-footer, header, footer'):
        rubbish.decompose()

    pattern = re.compile(re.escape(anchor_text), re.IGNORECASE)
    target = soup.find(string=pattern)
    
    if target:
        curr = target.parent
        steps = 0
        while curr and steps < 8:
            if curr.name == 'a' and curr.has_attr('href'):
                return curr['href'], "Found"
            curr = curr.parent
            steps += 1
        return None, "Text found, but no Link Wrapper detected."
    else:
        return None, "Anchor text not found on page."

def verify_with_gemini(anchor, instruction, link, creds):
    """
    Uses Google Vertex AI (Gemini 1.5 Flash) to check intent.
    Reuses the existing Service Account credentials.
    """
    if not link: return "FAIL", "Link missing"
    
    try:
        # Initialize Vertex AI with the same credentials
        vertexai.init(project=creds.project_id, credentials=creds)
        
        model = GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        You are a Website QA Bot.
        
        1. Text on page: "{anchor}"
        2. Content Writer's Instruction: "{instruction}"
        3. Actual Link found in code: "{link}"
        
        Task: Does the 'Actual Link' satisfy the 'Instruction'? 
        For example:
        - Instruction: "Link to contact" -> Link: "/contact-us" (PASS)
        - Instruction: "Link to TRT" -> Link: "/services/hormones" (FAIL? Depends on context, use best judgment)
        
        Return JSON ONLY: {{ "status": "PASS" or "FAIL", "reason": "short explanation" }}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(response_mime_type="application/json")
        )
        
        data = json.loads(response.text)
        return data.get('status', 'FAIL'), data.get('reason', 'AI Error')
        
    except Exception as e:
        return "ERROR", str(e)

# --- UTILS ---
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_diff(doc, web):
    d = difflib.HtmlDiff(wrapcolumn=80)
    return d.make_file(doc.splitlines(), web.splitlines(), fromdesc="Doc", todesc="Web")

# --- UI START ---
st.title("🏔️ Everest Content QC & Link Agent")

# Sidebar Auth
with st.sidebar:
    st.header("Settings")
    uploaded_key = st.file_uploader("Service Account JSON", type="json")
    
    st.info("Using Google Vertex AI (Gemini) for Link Checking")
    
    st.divider()
    use_staging = st.checkbox("Staging Mode")
    staging_domain = st.text_input("Staging Domain", "")
    staging_user = st.text_input("User", "")
    staging_pass = st.text_input("Pass", type="password")

creds = get_creds(uploaded_key)
auth = HTTPBasicAuth(staging_user, staging_pass) if staging_user else None

if not creds:
    st.error("Please upload Google Service Account JSON to start.")
    st.stop()

# TABS
tab1, tab2 = st.tabs(["📝 Text Comparison", "🔗 Link Auditor"])

# --- TAB 1: TEXT QC ---
with tab1:
    st.subheader("Bulk Text Comparison")
    csv_file = st.file_uploader("Upload Checklist CSV", type="csv", key="csv1")
    
    if st.button("Run Text QC"):
        if csv_file:
            stringio = io.StringIO(csv_file.getvalue().decode("utf-8-sig"))
            rows = list(csv.DictReader(stringio))
            
            st.session_state['qc_results'] = []
            bar = st.progress(0)
            
            for i, row in enumerate(rows):
                url = row.get('URL', '')
                if use_staging and staging_domain:
                     from urllib.parse import urlparse
                     path = urlparse(url).path
                     url = f"https://{staging_domain}{path}"

                doc_txt = get_doc_text(creds, row.get('google_doc_url', ''))
                web_txt = get_web_text_clean(url, auth)
                
                doc_norm = normalize_text(doc_txt)
                web_norm = normalize_text(web_txt)
                
                seq = difflib.SequenceMatcher(None, doc_norm, web_norm)
                sim = round(seq.ratio() * 100, 2)
                status = "MATCH" if sim > 95 else "MISMATCH"
                
                st.session_state['qc_results'].append({
                    "Title": row.get('Page Title'),
                    "Status": status,
                    "Score": sim,
                    "Diff": create_diff(doc_norm, web_norm) if status == "MISMATCH" else None
                })
                bar.progress((i+1)/len(rows))
            
    if st.session_state['qc_results']:
        df = pd.DataFrame(st.session_state['qc_results'])
        st.dataframe(df[["Title", "Status", "Score"]])
        
        sel = st.selectbox("Inspect Mismatch", df[df['Status']=="MISMATCH"]['Title'].unique())
        if sel:
            d = next(item for item in st.session_state['qc_results'] if item["Title"] == sel)
            components.html(d['Diff'], height=600, scrolling=True)

# --- TAB 2: LINK AUDIT ---
with tab2:
    st.subheader("Link Functionality & Intent Audit (Gemini Powered)")
    
    l_doc = st.text_input("Google Doc URL")
    l_url = st.text_input("Live Page URL")
    
    if st.button("Run Link Audit"):
        if not l_doc or not l_url:
            st.error("URLs required.")
        else:
            with st.spinner("Fetching comments..."):
                comments = get_doc_comments(creds, l_doc)
            
            if not comments:
                st.warning("No highlighted comments found in Doc.")
            else:
                st.info(f"Checking {len(comments)} links on {l_url}...")
                
                resp = requests.get(l_url, auth=auth, headers={'User-Agent': 'QC-Bot'})
                html_content = resp.text
                
                results = []
                bar = st.progress(0)
                
                for i, item in enumerate(comments):
                    # 1. Technical Check
                    link_href, status_msg = check_oxygen_link(html_content, item['anchor'])
                    
                    # 2. AI Check (Gemini)
                    status, reason = "MISSING", status_msg
                    if link_href:
                        status, reason = verify_with_gemini(item['anchor'], item['instruction'], link_href, creds)
                    
                    results.append({
                        "Anchor": item['anchor'],
                        "Instruction": item['instruction'],
                        "Found Link": link_href,
                        "Status": status,
                        "Reason": reason
                    })
                    bar.progress((i+1)/len(comments))
                
                res_df = pd.DataFrame(results)
                
                st.dataframe(res_df.style.applymap(lambda x: 'color:red' if x == 'FAIL' else 'color:green', subset=['Status']), use_container_width=True)
                
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "link_audit.csv", "text/csv")
