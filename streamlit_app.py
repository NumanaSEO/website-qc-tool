"""
Content QC Agent — Numana Digital
=================================

Compares approved copy in a Google Doc against the words published on a live page.

Scope: words only. Structural/SEO checks (title, meta, H1, alt text, links) are
deliberately excluded — those belong to the SEO QC pass that runs separately.

Output is a punch list, not a similarity score:
    MISSING  — text in the doc that is not on the page
    ALTERED  — text on the page that was changed from the doc
    EXTRA    — text on the page that is not in the doc
    MOVED    — text present in both but in a different position

Extraction contract: page content MUST be inside `.page-content-area`.
A page without it is reported as a DEV DEFECT, not a content mismatch.

Auth: service account, configured in Streamlit secrets only. The content team
never handles credentials.
"""

from __future__ import annotations

import csv
import difflib
import io
import re
import unicodedata
from collections import Counter
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ======================================================================
# CONFIG
# ======================================================================

CONTENT_SELECTOR = ".page-content-area"

# Extraction tiers. Tier 0 is the contract with the dev team. Anything below it
# still produces a usable QC pass, but the page is reported as a build defect.
FALLBACK_SELECTORS = [
    "main",
    "article",
    ".entry-content",
    ".site-content",
    "#content",
]

# Chrome to strip when running on a fallback selector, where the container is
# not guaranteed to exclude template furniture.
CHROME_SELECTOR = (
    'nav, footer, header, aside, form, '
    '[role="dialog"], [role="navigation"], [role="banner"], [role="contentinfo"], '
    '[aria-hidden="true"], .menu, .navbar, .breadcrumb, '
    '[class*="cookie"], [id*="cookie"], [class*="gdpr"], [id*="gdpr"], '
    '[class*="modal"], [class*="popup"], [class*="offcanvas"]'
)

# A fallback extraction yielding less than this is treated as a failed read
# rather than a page whose content is genuinely missing. Kept low on purpose:
# thin pages (locations, thank-you, short service pages) are legitimate, and a
# false "unreadable" is worse than letting a near-empty page through as CLEAN.
MIN_FALLBACK_CHARS = 120

# --- Network pacing -------------------------------------------------------
# Not exposed in the UI on purpose: these are infrastructure settings, not QC
# settings, and the content team has no basis for choosing a value.
#
# Cloudways/Imunify360 Enhanced DOS Protection on the staging server allows 250
# requests per 30-second window. At 1.5s spacing with 2 workers this run peaks
# around 20 requests per 30s — roughly an eighth of the limit. Deliberately
# conservative: a 40-page run takes about a minute either way, and other rules
# (or other tenants sharing the same egress IP) can trigger blocks that the DoS
# threshold alone does not explain.
#
# Adjust here if a client's host is stricter. Raising REQUEST_PACE_SECONDS is
# always the safe direction.
REQUEST_PACE_SECONDS = 1.5
MAX_WORKERS = 2

# --- Reporting -----------------------------------------------------------
# How similar a doc block and a page block must be before an edit is reported
# as one ALTERED finding with a word-level diff, rather than as a separate
# MISSING and EXTRA pair.
#
# Not exposed in the UI: this changes how a difference is DESCRIBED, never
# whether it is DETECTED. Verified across 0.50-0.95 — identical differences
# surface at every value. Above ~0.90 even a one-word edit splits into two
# findings the reader has to pair by eye, which is strictly worse. There is no
# setting that catches more, so there is nothing for the content team to tune.
SIMILARITY_FLOOR = 0.75

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Numana-QC/2.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Stripped from BOTH sides before comparison. These are template-injected
# strings that appear on the page but never in the copy doc.
NOISE_PHRASES = [
    "schedule an appointment",
    "book an appointment",
    "online bill pay",
]

# Blocks shorter than this are ignored entirely (stray characters, icons).
MIN_BLOCK_CHARS = 3

st.set_page_config(page_title="Content QC", page_icon="🧪", layout="wide")


# ======================================================================
# HTTP SESSION
# ======================================================================

def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


_local = threading.local()


def get_session() -> requests.Session:
    """One Session per worker thread. A shared Session across threads mutates
    shared state and can surface as connection errors."""
    sess = getattr(_local, "session", None)
    if sess is None:
        sess = _build_session()
        _local.session = sess
    return sess


# ======================================================================
# ERRORS — every message must tell the content team what to do next
# ======================================================================

# ======================================================================
# PACING + CIRCUIT BREAKER
# ======================================================================
#
# Hosts behind a WAF (Cloudways/Imunify360 among them) count requests per IP
# in short windows and firewall the source when a burst crosses the limit.
# A 40-page QC run is exactly that shape of traffic. Two guards:
#   _pace()  spaces request starts globally, across all worker threads
#   GUARD    stops the run after repeated connection failures, so a blocked
#            tool does not keep knocking and deepen the block

_pace_lock = threading.Lock()
_pace_state = {"last": 0.0, "interval": 0.0}


def set_pace(seconds: float) -> None:
    _pace_state["interval"] = max(0.0, float(seconds))
    _pace_state["last"] = 0.0


def _pace() -> None:
    """Block until at least `interval` has passed since the last request start.
    The sleep happens under the lock on purpose — that serialises pacing across
    every worker so concurrency never defeats the rate limit."""
    interval = _pace_state["interval"]
    if interval <= 0:
        return
    with _pace_lock:
        now = time.monotonic()
        wait = _pace_state["last"] + interval - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _pace_state["last"] = now


class RunGuard:
    """Trips after N consecutive connection-level failures."""

    def __init__(self, limit: int = 5):
        self._lock = threading.Lock()
        self.limit = limit
        self.consecutive = 0
        self.tripped = False

    def reset(self, limit: int = 5) -> None:
        with self._lock:
            self.limit = limit
            self.consecutive = 0
            self.tripped = False

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive += 1
            if self.consecutive >= self.limit:
                self.tripped = True

    def record_success(self) -> None:
        with self._lock:
            self.consecutive = 0

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self.tripped


GUARD = RunGuard()


class QCError(Exception):
    """Base for errors surfaced to the content team in plain language."""

    kind = "error"


class ContainerMissing(QCError):
    kind = "dev_defect"

    def __init__(self, url: str):
        super().__init__(
            f"Page is missing the {CONTENT_SELECTOR} container and no readable content "
            "block could be found either. This is a build issue, not a content issue — "
            "send this URL to the dev team."
        )
        self.url = url


class PageUnreadable(QCError):
    def __init__(self, message: str, connection_level: bool = False):
        super().__init__(message)
        self.connection_level = connection_level


class RunAborted(QCError):
    pass


class DocUnreadable(QCError):
    pass


# ======================================================================
# AUTH — secrets only
# ======================================================================

def get_credentials() -> Optional[service_account.Credentials]:
    try:
        if "gcp_service_account" not in st.secrets:
            return None
        info = dict(st.secrets["gcp_service_account"])
    except Exception:
        # No secrets file at all, or a malformed one. Both mean "not configured".
        return None
    pk = info.get("private_key")
    if isinstance(pk, str):
        info["private_key"] = pk.replace("\\n", "\n")
    try:
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        return None


def _docs_service():
    """New service per call — googleapiclient http objects are not thread-safe."""
    return build("docs", "v1", credentials=get_credentials(), cache_discovery=False)


# ======================================================================
# BLOCK MODEL
# ======================================================================

@dataclass
class Block:
    text: str
    norm: str
    kind: str          # heading | paragraph | list_item | quote
    level: int = 0     # heading depth, 0 for non-headings
    section: str = ""  # nearest preceding heading


def qc_normalize(text: str, smart_punct: bool = True) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    if smart_punct:
        for a, b in (
            ("\u201c", '"'), ("\u201d", '"'),
            ("\u2018", "'"), ("\u2019", "'"),
            ("\u2014", "-"), ("\u2013", "-"),
            ("\u2026", "..."),
        ):
            t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def strip_noise(text: str, phrases: list[str]) -> str:
    t = text or ""
    for p in phrases:
        t = re.sub(re.escape(p), " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def finalize_blocks(raw: list[tuple[str, str, int]], smart_punct: bool,
                    strip_ctas: bool) -> list[Block]:
    """raw: list of (text, kind, level). Attaches section context, drops noise."""
    blocks: list[Block] = []
    current_section = ""
    for text, kind, level in raw:
        clean = text.strip()
        if strip_ctas:
            clean = strip_noise(clean, NOISE_PHRASES)
        norm = qc_normalize(clean, smart_punct)
        if len(norm) < MIN_BLOCK_CHARS:
            continue
        if not re.search(r"\w", norm):
            continue
        if kind == "heading":
            current_section = clean
            blocks.append(Block(clean, norm, kind, level, current_section))
        else:
            blocks.append(Block(clean, norm, kind, level, current_section))
    return blocks


# ======================================================================
# GOOGLE DOC → BLOCKS
# ======================================================================

def _doc_id(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9\-_]+)", url or "")
    if not m:
        raise DocUnreadable(
            "That doesn't look like a Google Doc link. Check the doc column in your CSV."
        )
    return m.group(1)


def _walk_doc_elements(elements) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for el in elements or []:
        if "paragraph" in el:
            para = el["paragraph"]
            text = "".join(
                pe.get("textRun", {}).get("content", "")
                for pe in para.get("elements", [])
            )
            if not text.strip():
                continue
            style = para.get("paragraphStyle", {}).get("namedStyleType", "NORMAL_TEXT")
            if style.startswith("HEADING_"):
                out.append((text, "heading", int(style.split("_")[1])))
            elif style in ("TITLE", "SUBTITLE"):
                out.append((text, "heading", 1))
            elif "bullet" in para:
                out.append((text, "list_item", 0))
            else:
                out.append((text, "paragraph", 0))
        elif "table" in el:
            for row in el["table"].get("tableRows", []):
                for cell in row.get("tableCells", []):
                    out.extend(_walk_doc_elements(cell.get("content", [])))
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_doc_raw(doc_url: str) -> list[tuple[str, str, int]]:
    """Cached on doc URL only — credentials never enter the cache key."""
    doc_id = _doc_id(doc_url)
    try:
        doc = _docs_service().documents().get(documentId=doc_id).execute()
    except HttpError as e:
        status = getattr(e.resp, "status", None)
        if status in (403, 404):
            raise DocUnreadable(
                "The QC tool can't open this doc. It needs to be in the shared "
                "content drive — send the link to Jen."
            ) from e
        raise DocUnreadable(f"Google Docs couldn't return this file (error {status}).") from e
    return _walk_doc_elements(doc.get("body", {}).get("content", []))


# ======================================================================
# WEB PAGE → BLOCKS
# ======================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_page_raw(url: str) -> tuple[list[tuple[str, str, int]], str]:
    _pace()
    try:
        resp = get_session().get(url, headers=REQUEST_HEADERS, timeout=(10, 30), allow_redirects=True)
    except requests.exceptions.SSLError as e:
        raise PageUnreadable(f"SSL/certificate problem reaching this page. ({e.__class__.__name__})", True) from e
    except requests.exceptions.ConnectTimeout as e:
        raise PageUnreadable(
            "Connection timed out — the site didn't respond. If every page says this, "
            "the site's security is blocking this tool. Send to Jen.", True
        ) from e
    except requests.exceptions.ReadTimeout as e:
        raise PageUnreadable("The host connected but sent no response within 30s.", True) from e
    except requests.exceptions.TooManyRedirects as e:
        raise PageUnreadable("This URL redirects in a loop.") from e
    except requests.exceptions.ConnectionError as e:
        detail = str(e.__cause__ or e)
        if "Name or service not known" in detail or "nodename nor servname" in detail:
            raise PageUnreadable(
                "DNS lookup failed — this hostname does not resolve. "
                "The staging site may have moved or been torn down.", True
            ) from e
        if "Connection refused" in detail:
            raise PageUnreadable("Connection refused by the host.", True) from e
        if "reset by peer" in detail or "RemoteDisconnected" in detail:
            raise PageUnreadable(
                "The host dropped the connection — the site's security is blocking this "
                "tool. Send this URL to Jen.", True
            ) from e
        raise PageUnreadable(f"Could not connect. {detail[:180]}", True) from e
    except requests.RequestException as e:
        raise PageUnreadable(f"Request failed: {e.__class__.__name__}. {str(e)[:180]}", True) from e

    if resp.status_code == 404:
        raise PageUnreadable("Page not found (404). Check the URL in your CSV.")
    if resp.status_code in (401, 403):
        raise PageUnreadable(
            "This page is password-protected. The QC tool can't read it."
        )
    if resp.status_code >= 400:
        raise PageUnreadable(f"The site returned an error ({resp.status_code}) for this page.")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    container = soup.select_one(CONTENT_SELECTOR)
    mode = "contract"

    if container is None:
        # Contract broken. Recover so the content team isn't blocked, but the
        # caller reports this page as a build defect regardless of the outcome.
        # Take the richest candidate, not the first — a thin <main> wrapper
        # should not beat a fuller .entry-content.
        best, best_len, best_sel = None, 0, ""
        for sel in FALLBACK_SELECTORS:
            candidate = soup.select_one(sel)
            if candidate is None:
                continue
            for junk in candidate.select(CHROME_SELECTOR):
                junk.decompose()
            size = len(candidate.get_text(" ", strip=True))
            if size > best_len:
                best, best_len, best_sel = candidate, size, sel
        if best is not None and best_len >= MIN_FALLBACK_CHARS:
            container, mode = best, f"fallback:{best_sel}"

    if container is None:
        body = soup.body
        if body is not None:
            for junk in body.select(CHROME_SELECTOR):
                junk.decompose()
            if len(body.get_text(" ", strip=True)) >= MIN_FALLBACK_CHARS:
                container, mode = body, "fallback:body"

    if container is None:
        raise ContainerMissing(url)

    if mode == "contract":
        # Defensive: strip template chrome if it ever lands inside the container.
        for junk in container.select('nav, footer, [role="dialog"], [aria-hidden="true"]'):
            junk.decompose()

    raw: list[tuple[str, str, int]] = []
    emitted: set[int] = set()
    for el in container.find_all(BLOCK_TAGS):
        if any(id(p) in emitted for p in el.parents):
            continue
        emitted.add(id(el))
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        name = el.name
        if name in HEADING_TAGS:
            raw.append((text, "heading", int(name[1])))
        elif name == "li":
            raw.append((text, "list_item", 0))
        elif name == "blockquote":
            raw.append((text, "quote", 0))
        else:
            raw.append((text, "paragraph", 0))
    return raw, mode


# ======================================================================
# ALIGNMENT → FINDINGS
# ======================================================================

@dataclass
class Finding:
    kind: str          # MISSING | ALTERED | EXTRA | MOVED
    section: str
    doc_text: str = ""
    web_text: str = ""
    detail: str = ""


def word_level_change(doc_text: str, web_text: str, max_parts: int = 4) -> str:
    a, b = doc_text.split(), web_text.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        was = " ".join(a[i1:i2])
        now = " ".join(b[j1:j2])
        if tag == "replace":
            parts.append(f'"{was}" → "{now}"')
        elif tag == "delete":
            parts.append(f'removed "{was}"')
        elif tag == "insert":
            parts.append(f'added "{now}"')
    if not parts:
        return "whitespace or punctuation only"
    if len(parts) > max_parts:
        return "; ".join(parts[:max_parts]) + f" (+{len(parts) - max_parts} more)"
    return "; ".join(parts)


def align(doc_blocks: list[Block], web_blocks: list[Block],
          similarity_floor: float) -> list[Finding]:
    a = [b.norm for b in doc_blocks]
    b = [x.norm for x in web_blocks]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    findings: list[Finding] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if tag == "delete":
            for di in range(i1, i2):
                findings.append(Finding("MISSING", doc_blocks[di].section,
                                        doc_text=doc_blocks[di].text))
            continue

        if tag == "insert":
            for wj in range(j1, j2):
                findings.append(Finding("EXTRA", web_blocks[wj].section,
                                        web_text=web_blocks[wj].text))
            continue

        # replace — pair doc blocks to their closest web counterpart
        used: set[int] = set()
        for di in range(i1, i2):
            best_j, best_r = None, 0.0
            for wj in range(j1, j2):
                if wj in used:
                    continue
                r = difflib.SequenceMatcher(a=a[di], b=b[wj], autojunk=False).ratio()
                if r > best_r:
                    best_r, best_j = r, wj
            if best_j is not None and best_r >= similarity_floor:
                used.add(best_j)
                findings.append(Finding(
                    "ALTERED",
                    doc_blocks[di].section,
                    doc_text=doc_blocks[di].text,
                    web_text=web_blocks[best_j].text,
                    detail=word_level_change(doc_blocks[di].norm, web_blocks[best_j].norm),
                ))
            else:
                findings.append(Finding("MISSING", doc_blocks[di].section,
                                        doc_text=doc_blocks[di].text))
        for wj in range(j1, j2):
            if wj not in used:
                findings.append(Finding("EXTRA", web_blocks[wj].section,
                                        web_text=web_blocks[wj].text))

    return reclassify_moved(findings)


def reclassify_moved(findings: list[Finding]) -> list[Finding]:
    """A MISSING block whose text also appears as EXTRA was relocated, not lost."""
    missing = {i: qc_normalize(f.doc_text) for i, f in enumerate(findings) if f.kind == "MISSING"}
    extra = {i: qc_normalize(f.web_text) for i, f in enumerate(findings) if f.kind == "EXTRA"}

    drop: set[int] = set()
    for mi, mtext in missing.items():
        for ei, etext in extra.items():
            if ei in drop:
                continue
            if mtext == etext:
                doc_sec = findings[mi].section
                web_sec = findings[ei].section
                if doc_sec == web_sec:
                    detail = "same text, different position on the page"
                else:
                    detail = (f"in the doc under '{doc_sec or '—'}', "
                              f"on the page under '{web_sec or '—'}'")
                findings[mi] = Finding(
                    "MOVED", doc_sec,
                    doc_text=findings[mi].doc_text,
                    web_text=findings[ei].web_text,
                    detail=detail,
                )
                drop.add(ei)
                break
    return [f for i, f in enumerate(findings) if i not in drop]


# ======================================================================
# PER-ROW PIPELINE
# ======================================================================

@dataclass
class PageResult:
    page: str
    url: str
    doc_url: str
    status: str                       # CLEAN | REVIEW | DEV DEFECT | ERROR
    message: str = ""
    findings: list[Finding] = field(default_factory=list)
    doc_block_count: int = 0
    web_block_count: int = 0
    extraction: str = "contract"

    @property
    def container_missing(self) -> bool:
        return self.extraction != "contract"

    @property
    def counts(self) -> Counter:
        return Counter(f.kind for f in self.findings)


def qc_one(row: dict, cols: dict, settings: dict) -> PageResult:
    url = str(row.get(cols["url"], "") or "").strip()
    doc_url = str(row.get(cols["doc"], "") or "").strip()
    page = str(row.get(cols["page"], "") or "").strip() if cols["page"] else ""
    label = page or url or "(unnamed row)"

    try:
        if GUARD.is_tripped:
            raise RunAborted(
                "Run stopped early. Several pages in a row refused the connection, "
                "which usually means the host is blocking this tool. Remaining pages "
                "were skipped so the block isn't made worse."
            )
        if not url:
            raise PageUnreadable("No URL in this row.")
        if not doc_url:
            raise DocUnreadable("No Google Doc link in this row.")

        doc_raw = fetch_doc_raw(doc_url)
        web_raw, extraction = fetch_page_raw(url)
        GUARD.record_success()

        doc_blocks = finalize_blocks(doc_raw, settings["smart_punct"], settings["strip_ctas"])
        web_blocks = finalize_blocks(web_raw, settings["smart_punct"], settings["strip_ctas"])

        findings = align(doc_blocks, web_blocks, settings["similarity_floor"])

        if settings["ignore_extra"]:
            findings = [f for f in findings if f.kind != "EXTRA"]

        return PageResult(
            page=label, url=url, doc_url=doc_url,
            status="CLEAN" if not findings else "REVIEW",
            findings=findings,
            doc_block_count=len(doc_blocks),
            web_block_count=len(web_blocks),
            extraction=extraction,
        )

    except ContainerMissing as e:
        GUARD.record_success()  # we reached the page; the container is the problem
        return PageResult(label, url, doc_url, "DEV DEFECT", str(e))
    except QCError as e:
        if isinstance(e, PageUnreadable) and e.connection_level:
            GUARD.record_failure()
        return PageResult(label, url, doc_url, "ERROR", str(e))
    except Exception as e:  # noqa: BLE001 — never let one row kill the run
        return PageResult(label, url, doc_url, "ERROR",
                          f"Unexpected problem reading this row. Send to Jen. ({type(e).__name__})")


# ======================================================================
# UI
# ======================================================================

st.title("Content QC")
st.caption(
    "Compares the words in the approved Google Doc against the words on the live page. "
    "Formatting, SEO tags and links are not checked — those are the SEO QC pass."
)

if "results" not in st.session_state:
    st.session_state.results = None

with st.sidebar:
    st.markdown("### Settings")
    smart_punct = st.toggle("Ignore smart quotes and dashes", value=True)
    strip_ctas = st.toggle("Ignore template CTA phrases", value=True)
    ignore_extra = st.toggle("Hide EXTRA findings", value=False,
                             help="Turn on if template blocks inside the container "
                                  "are creating noise.")
    st.divider()
    st.caption(f"Content is read from `{CONTENT_SELECTOR}`.")

if get_credentials() is None:
    st.error("This tool isn't configured yet. Contact Jen — the Google credentials are missing.")
    st.stop()

st.markdown("**Upload a CSV** with a column for the page URL and a column for the Google Doc link.")

if "test_result" not in st.session_state:
    st.session_state.test_result = None


def _run_single_test(raw_url: str) -> tuple[str, str]:
    url = (raw_url or "").strip()
    if not url:
        return ("error", "Enter a URL first.")

    had_scheme = url.lower().startswith(("http://", "https://"))
    attempts = [url] if had_scheme else ["https://" + url, "http://" + url]

    set_pace(0.0)
    last_error = "Could not reach that URL."
    for attempt in attempts:
        try:
            raw, mode = fetch_page_raw(attempt)
            words = sum(len(t.split()) for t, _, _ in raw)
            if mode == "contract":
                return ("success",
                        f"Reachable. Found `{CONTENT_SELECTOR}` with {len(raw)} blocks / {words} words.")
            return ("warning",
                    f"Reachable, but `{CONTENT_SELECTOR}` is missing. Recovered {len(raw)} blocks / "
                    f"{words} words using `{mode.split(':', 1)[1]}`. This is a build issue — "
                    "send the URL to the dev team.")
        except ContainerMissing as e:
            return ("error", str(e))
        except QCError as e:
            last_error = str(e)
            continue
        except Exception as e:  # noqa: BLE001
            return ("error", f"Unexpected problem. Send to Jen. ({type(e).__name__})")
    return ("error", last_error)


with st.expander("Test one page first (recommended if the last run failed)",
                 expanded=st.session_state.test_result is not None):
    st.caption(
        "Checks a single URL without running the whole list. Use this after a failed run — "
        "it tells you whether the site is reachable using one request instead of forty."
    )
    test_url = st.text_input("Page URL", key="test_url", placeholder="https://example.com/a-page/")
    if st.button("Test this page"):
        st.session_state.test_result = _run_single_test(test_url)
        st.rerun()

    result = st.session_state.test_result
    if result:
        kind, message = result
        {"success": st.success, "warning": st.warning, "error": st.error}[kind](message)

csv_file = st.file_uploader("QC CSV", type="csv", label_visibility="collapsed")

if st.button("Run QC", type="primary", disabled=csv_file is None):
    raw = csv_file.getvalue().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw)))

    if not rows:
        st.warning("That CSV is empty.")
        st.stop()

    headers = set(rows[0].keys())

    def pick(*names):
        for n in names:
            if n in headers:
                return n
        return None

    cols = {
        "url": pick("URL", "Url", "url", "Page URL", "page_url"),
        "doc": pick("google_doc_url", "Google Doc URL", "Doc URL", "doc_url", "Doc"),
        "page": pick("Page Title", "Page", "Title", "page_title"),
    }

    if not cols["url"] or not cols["doc"]:
        st.error(
            "The CSV needs a column for the page URL and a column for the Google Doc link. "
            f"Found these columns: {', '.join(sorted(headers))}"
        )
        st.stop()

    settings = {
        "similarity_floor": SIMILARITY_FLOOR,
        "smart_punct": smart_punct,
        "strip_ctas": strip_ctas,
        "ignore_extra": ignore_extra,
    }

    set_pace(REQUEST_PACE_SECONDS)
    GUARD.reset(limit=5)

    est = int(len(rows) * REQUEST_PACE_SECONDS) + 5
    st.caption(
        f"Checking {len(rows)} pages at a deliberately slow pace so the site's bot "
        f"protection isn't triggered. This takes about {est // 60}m {est % 60}s."
    )

    progress = st.progress(0.0, text="Reading pages and docs…")
    results: list[PageResult] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(qc_one, r, cols, settings) for r in rows]
        for i, fut in enumerate(futures, start=1):
            results.append(fut.result())
            progress.progress(i / len(futures), text=f"Checked {i} of {len(futures)} pages")
    progress.empty()
    st.session_state.results = results


# ----------------------------------------------------------------------
# DISPLAY
# ----------------------------------------------------------------------

results: Optional[list[PageResult]] = st.session_state.results

if not results:
    st.info("Upload a CSV and click Run QC.")
    st.stop()

clean = [r for r in results if r.status == "CLEAN"]
review = [r for r in results if r.status == "REVIEW"]
dev = [r for r in results if r.status == "DEV DEFECT"]
errors = [r for r in results if r.status == "ERROR"]
fallback = [r for r in results if r.status in ("CLEAN", "REVIEW") and r.container_missing]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean", len(clean))
c2.metric("Need review", len(review))
c3.metric("Build defects", len(dev) + len(fallback))
c4.metric("Couldn't read", len(errors))

if fallback:
    st.warning(
        f"**{len(fallback)} page(s) are missing the `{CONTENT_SELECTOR}` container.** "
        "They were checked anyway using a fallback content block, so the results below "
        "are usable — MISSING and ALTERED findings are reliable. EXTRA findings on these "
        "pages may just be template text (menus, footers) and should be treated with "
        "suspicion.\n\n"
        "This is a build issue, not a content issue. Send this list to the dev team."
    )
    st.code("\n".join(f"{r.url}   [{r.extraction}]" for r in fallback), language=None)

if dev:
    st.error(
        f"**{len(dev)} page(s) could not be read at all.** No `{CONTENT_SELECTOR}` container "
        "and no usable content block. These pages were not checked — send to the dev team."
    )
    st.code("\n".join(r.url for r in dev), language=None)

if errors:
    with st.expander(f"{len(errors)} row(s) couldn't be read"):
        for r in errors:
            st.markdown(f"- **{r.page}** — {r.message}")

# Summary table
summary = pd.DataFrame([
    {
        "Page": r.page,
        "Status": r.status,
        "Extraction": "Contract" if r.extraction == "contract" else "FALLBACK",
        "Missing": r.counts.get("MISSING", 0),
        "Altered": r.counts.get("ALTERED", 0),
        "Extra": r.counts.get("EXTRA", 0),
        "Moved": r.counts.get("MOVED", 0),
        "URL": r.url,
    }
    for r in results
])
st.dataframe(summary, width="stretch", hide_index=True)

# Full findings export
export_rows = [
    {
        "Page": r.page,
        "URL": r.url,
        "Section": f.section,
        "Issue": f.kind,
        "What changed": f.detail,
        "Doc text": f.doc_text,
        "Page text": f.web_text,
    }
    for r in results for f in r.findings
]
if export_rows:
    st.download_button(
        "Download punch list (CSV)",
        pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
        "content_qc_punch_list.csv",
        mime="text/csv",
    )

if not review:
    if not dev and not errors:
        st.success("Every page matches its doc.")
    st.stop()

st.divider()
st.subheader("Punch list")

labels = {i: f"{r.page} — {len(r.findings)} item(s)" for i, r in enumerate(results) if r.status == "REVIEW"}
sel = st.selectbox("Page", options=list(labels), format_func=lambda i: labels[i])
chosen = results[sel]

st.caption(f"{chosen.url}  ·  doc has {chosen.doc_block_count} blocks, page has {chosen.web_block_count}")

ICONS = {"MISSING": "🔴", "ALTERED": "🟠", "EXTRA": "🔵", "MOVED": "🟣"}
ORDER = {"MISSING": 0, "ALTERED": 1, "MOVED": 2, "EXTRA": 3}

for f in sorted(chosen.findings, key=lambda x: (ORDER[x.kind], x.section)):
    section = f.section or "—"
    with st.container(border=True):
        st.markdown(f"{ICONS[f.kind]} **{f.kind}** · under *{section}*")
        if f.kind == "MISSING":
            st.markdown("In the doc, not on the page:")
            st.markdown(f"> {f.doc_text}")
        elif f.kind == "EXTRA":
            st.markdown("On the page, not in the doc:")
            st.markdown(f"> {f.web_text}")
        elif f.kind == "MOVED":
            st.markdown(f"Same text, different place — {f.detail}")
            st.markdown(f"> {f.doc_text}")
        else:
            st.markdown(f"**{f.detail}**")
            st.markdown(f"Doc: > {f.doc_text}")
            st.markdown(f"Page: > {f.web_text}")
