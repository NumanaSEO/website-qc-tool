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
            scopes=["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/documents.readonly", "https://www.googleapis.com/auth/drive.readonly"]
        )
    return None

# --- TEXT EXTRACTORS ---
def get_doc_text(creds, doc_url):
    try:
        service = build("docs", "v1", credentials=creds)
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url)
        if not match: return "Error: Invalid Doc URL"
        doc_id = match.group(1)
        
        doc = service.documents().get(documentId=doc_id).execute()
        text = ""
        for value in doc.get('body').get('content'):
            if 'paragraph' in value:
                for elem in value.get('paragraph').get('elements'):
                    text += elem.get('textRun', {}).get('content', '')
        return text
    except Exception as e:
        return f"Error: {e}"

def get_web_text_clean(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (QC-Bot)'}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

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

# --- LINK EXTRACTORS ---
def get_doc_comments(creds, doc_url):
    try:
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url)
        if not match: return []
        doc_id = match.group(1)
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
    for rubbish in soup.select('.ct-header, .ct-footer, header, footer, .oxy-nav-menu'):
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
    if not link: return "FAIL", "Link missing"
    try:
        vertexai.init(project=creds.project_id, credentials=creds)
        model = GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        You are a Website QA Bot.
        1. Text on page: "{anchor}"
        2. Content Writer's Instruction: "{instruction}"
        3. Actual Link found in code: "{link}"
        Task: Does the 'Actual Link' satisfy the 'Instruction'? 
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
st.title("Content QC & Link Agent")

with st.sidebar:
    st.header("Settings")
    if "gcp_service_account" in st.secrets:
        st.success("✅ Authenticated via Secrets")
        uploaded_key = None
    else:
        uploaded_key = st.file_uploader("Service Account JSON", type="json")
    
    st.divider()
    use_staging = st.checkbox("Override Domain (Optional)")
    staging_domain = ""
    if use_staging:
        st.caption("Useful if CSV has Live URLs but testing Dev.")
        staging_domain = st.text_input("New Domain", placeholder="e.g. wordpress-123.cloudwaysapps.com")

creds = get_creds(uploaded_key)
if not creds:
    st.error("Please configure secrets.toml or upload a JSON key.")
    st.stop()

# TABS
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
                url = row.get('URL', '')
                doc_url = row.get('google_doc_url', '')
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
                status = "MATCH" if sim > 95 else "MISMATCH"
                if "Error" in doc_txt or "Error" in web_txt: status = "ERROR"
                
                st.session_state['qc_results'].append({
                    "Title": row.get('Page Title'),
                    "Status": status,
                    "Score": sim,
                    "Diff": create_diff(doc_norm, web_norm) if status == "MISMATCH" else None
                })
                bar.progress((i+1)/len(rows))
            status_text.text("Done!")
            
    if st.session_state['qc_results']:
        df = pd.DataFrame(st.session_state['qc_results'])
        def color_status(val):
            color = '#d4edda' if val == 'MATCH' else '#f8d7da' if val == 'MISMATCH' else '#fff3cd'
            return f'background-color: {color}'
        st.dataframe(df[["Title", "Status", "Score"]].style.applymap(color_status, subset=['Status']), use_container_width=True)
        sel = st.selectbox("Inspect Mismatch", df[df['Status']=="MISMATCH"]['Title'].unique())
        if sel:
            d = next(item for item in st.session_state['qc_results'] if item["Title"] == sel)
            components.html(d['Diff'], height=600, scrolling=True)

# --- TAB 2: LINK AUDITOR ---
with tab2:
    st.subheader("Link Functionality & Intent Audit")
    
    # Mode Selection
    mode = st.radio("Choose Input Method:", ["Bulk CSV Upload", "Single Page Check"], horizontal=True)
    
    # Logic Container
    links_to_process = [] # list of dicts: {title, doc_url, live_url}

    if mode == "Single Page Check":
        col_l1, col_l2 = st.columns(2)
        with col_l1: l_doc = st.text_input("Google Doc URL")
        with col_l2: l_url = st.text_input("Live Page URL")
        if l_doc and l_url:
            links_to_process.append({"Page Title": "Single Check", "URL": l_url, "google_doc_url": l_doc})

    else: # Bulk CSV
        csv_file_links = st.file_uploader("Upload Checklist CSV", type="csv", key="csv_links")
        if csv_file_links:
            stringio = io.StringIO(csv_file_links.getvalue().decode("utf-8-sig"))
            links_to_process = list(csv.DictReader(stringio))

    if st.button("Run Link Audit"):
        if not links_to_process:
            st.error("Please provide a CSV or Single URL to process.")
        else:
            final_report = []
            bar = st.progress(0)
            status_text = st.empty()
            
            # Iterate through pages
            for i, row in enumerate(links_to_process):
                page_title = row.get('Page Title', 'Unknown Page')
                live_url = row.get('URL', '')
                doc_url = row.get('google_doc_url', '')

                if use_staging and staging_domain:
                    from urllib.parse import urlparse
                    path = urlparse(live_url).path
                    live_url = f"https://{staging_domain}{path}"
                
                status_text.text(f"Scanning Page: {page_title}...")
                
                # 1. Get Comments
                comments = get_doc_comments(creds, doc_url)
                
                if not comments:
                    # Log that we checked but found nothing
                    final_report.append({
                        "Page Title": page_title,
                        "Status": "INFO",
                        "Anchor Text": "-",
                        "Found Link": "-",
                        "Reason": "No highlighted comments found in Doc"
                    })
                else:
                    try:
                        # 2. Scrape Page ONCE
                        resp = requests.get(live_url, headers={'User-Agent': 'QC-Bot'})
                        html_content = resp.text
                        
                        # 3. Check every link on this page
                        for item in comments:
                            link_href, status_msg = check_oxygen_link(html_content, item['anchor'])
                            
                            status, reason = "MISSING", status_msg
                            if link_href:
                                status, reason = verify_with_gemini(item['anchor'], item['instruction'], link_href, creds)
                            
                            final_report.append({
                                "Page Title": page_title,
                                "Status": status,
                                "Anchor Text": item['anchor'],
                                "Instruction": item['instruction'],
                                "Found Link": link_href,
                                "Reason": reason
                            })
                    except Exception as e:
                        final_report.append({
                            "Page Title": page_title,
                            "Status": "ERROR",
                            "Reason": f"Could not load website: {str(e)}"
                        })

                bar.progress((i+1)/len(links_to_process))
            
            status_text.text("Link Audit Complete!")
            
            # Display Results
            if final_report:
                res_df = pd.DataFrame(final_report)
                
                # Reorder columns nicely
                cols = ["Page Title", "Status", "Anchor Text", "Instruction", "Found Link", "Reason"]
                # Filter to ensure columns exist (in case of error rows)
                existing_cols = [c for c in cols if c in res_df.columns]
                res_df = res_df[existing_cols]

                def color_link_row(val):
                    color = '#f8d7da' if val == 'FAIL' else '#d4edda' if val == 'PASS' else 'white'
                    return f'background-color: {color}'

                st.dataframe(res_df.style.applymap(lambda x: 'color:red; font-weight:bold' if x == 'FAIL' else 'color:green; font-weight:bold' if x == 'PASS' else 'color:orange', subset=['Status']), use_container_width=True)
                
                csv_dl = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Link Report CSV", csv_dl, "link_audit_report.csv", "text/csv")
