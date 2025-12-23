# app.py — Fast Mode Lite (Grouped + Public-Friendly + OCR Fallback)
# Adds OCR for scanned/image-only PDFs via Google Vision + PyMuPDF rendering

import io
import re
import base64
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

import streamlit as st
import requests
from pypdf import PdfReader
import fitz  # PyMuPDF

# ============================
# Policy timeline (context only)
# ============================
POLICY_TIMELINE = [
    ("2020-03-18", "VA Circular 26-20-8 — Foreclosure moratorium guidance",
     "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_8.pdf"),
    ("2020-03-27", "CARES Act §4022 — Forbearance for federally backed mortgages",
     "https://www.congress.gov/116/plaws/publ136/PLAW-116publ136.pdf"),
    ("2020-04-08", "VA Circular 26-20-12 — CARES Act forbearance guidance for VA loans",
     "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_12.pdf"),
    ("2020-06-17", "VA Circular 26-20-22 — Extends foreclosure moratorium",
     "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_22.pdf"),
]
CARES_EFFECTIVE = dt.date(2020, 3, 27)

# ============================
# Models
# ============================
@dataclass
class EvidenceRef:
    doc_name: str
    page_number: int
    excerpt: str

@dataclass
class Finding:
    rule_id: str
    severity: int
    confidence: float
    title: str
    what_we_saw: str
    why_it_matters: str
    evidence: List[EvidenceRef]
    questions: List[str]
    policy_context: Optional[str] = None

# ============================
# Helpers
# ============================
def normalize_excerpt(text: str, limit: int = 140) -> str:
    t = " ".join((text or "").split())
    return t[:limit] + ("…" if len(t) > limit else "")

def impact_label(sev: int) -> str:
    return "High" if sev >= 4 else "Medium" if sev == 3 else "Low"

def sure_label(c: float) -> str:
    return "Strong" if c >= 0.75 else "Moderate" if c >= 0.60 else "Needs more evidence"

def badge(sev: int, conf: float) -> str:
    return f"Impact: {impact_label(sev)} • How sure: {sure_label(conf)}"

def dedupe_evidence(evidence: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for ev in evidence:
        key = (ev.get("doc_name"), ev.get("page_number"), (ev.get("excerpt") or "")[:80])
        if key not in seen:
            seen.add(key)
            out.append(ev)
    return out

def locator_phrase(excerpt: str, max_words: int = 6) -> str:
    text = " ".join((excerpt or "").split())
    text = re.sub(r"[“”\"'`]", "", text)
    words = text.split()

    stop = {"the","and","or","to","of","a","in","for","with","your","you","is","are","on","this","that","as","be","by","it"}
    filtered = [w for w in words if w.lower() not in stop]

    pick = filtered if len(filtered) >= max_words else words
    return " ".join(pick[:max_words]) if pick else ""

def policy_context_text(start: dt.date, end: dt.date) -> str:
    if end >= CARES_EFFECTIVE:
        return (
            "CARES Act §4022 and VA COVID circulars generally treated forbearance as a protected status. "
            "If late fees or delinquency coding appear during an active forbearance window, the servicer "
            "should be able to explain it with a forbearance plan record and transaction ledger."
        )
    return (
        "Part of this window predates the CARES Act effective date (3/27/2020). "
        "COVID-related servicing actions during early 2020 may require additional clarification."
    )

def build_letter(borrower_name: str, loan_number: str, property_addr: str,
                 start: dt.date, end: dt.date, questions: List[str]) -> str:
    end_txt = end.isoformat() if end else "[end date if applicable]"
    bullets = "\n".join([f"- {q}" for q in questions])

    return f"""Subject: Request for clarification regarding COVID forbearance handling
Loan Number: {loan_number or "[loan number]"}
Property Address: {property_addr or "[property address]"}
Date: {dt.date.today().isoformat()}

Dear Servicing Team,

I am writing to request clarification regarding the handling of my COVID-19 forbearance on the above-referenced VA-backed mortgage.

Based on my records, I elected COVID forbearance beginning on or about {start.isoformat()} (ending on {end_txt}, if applicable).  However, my records indicate the loan may not have been treated as protected during that period.

Please provide the following to reconcile the servicing record:

{bullets}

Thank you for your attention.  I look forward to your response.

Sincerely,
{borrower_name or "[Borrower Name]"}
"""

# ============================
# OCR (Google Vision)
# ============================
def google_vision_ocr_image_bytes(img_bytes: bytes, api_key: str) -> str:
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    content_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": content_b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        return data["responses"][0]["fullTextAnnotation"]["text"] or ""
    except Exception:
        return ""

def render_pdf_page_to_png(pdf_bytes: bytes, page_index0: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def extract_pages_text_with_ocr(pdf_bytes: bytes, doc_name: str, use_ocr: bool, ocr_page_cap: int) -> List[str]:
    """
    1) Try normal PDF text extraction first.
    2) If a page is empty/very short AND OCR enabled, OCR that page (up to cap).
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    ocr_used = 0
    api_key = st.secrets.get("GOOGLE_VISION_API_KEY", "")

    for idx, p in enumerate(reader.pages):
        t = (p.extract_text() or "").strip()
        if use_ocr and len(t) < 20 and api_key and ocr_used < ocr_page_cap:
            try:
                png = render_pdf_page_to_png(pdf_bytes, idx, zoom=2.0)
                t2 = google_vision_ocr_image_bytes(png, api_key).strip()
                if t2:
                    t = t2
                ocr_used += 1
            except Exception:
                # keep whatever we had
                pass
        pages_text.append(t)

    return pages_text

# ============================
# Fast Mode Lite rules (heuristics)
# ============================
def rule_C01(doc_name: str, pages: List[str], start: dt.date, end: dt.date) -> Optional[Finding]:
    covid_hits = []
    delin_hits = []
    for i, t in enumerate(pages, start=1):
        low = (t or "").lower()
        if any(k in low for k in ["covid", "forbear", "cares"]):
            covid_hits.append((i, t))
        if any(k in low for k in ["late fee", "delinquen", "past due"]):
            delin_hits.append((i, t))

    if not covid_hits or not delin_hits:
        return None

    evidence = [
        EvidenceRef(doc_name, covid_hits[0][0], normalize_excerpt(covid_hits[0][1])),
        EvidenceRef(doc_name, delin_hits[0][0], normalize_excerpt(delin_hits[0][1])),
    ]

    questions = [
        "Provide the complete COVID forbearance plan record and system notes (start/end dates).",
        "Explain any late fees, delinquency coding, or 'past due' status during forbearance.",
        "Provide a transaction-level payment ledger for the same period (principal/interest/escrow/suspense).",
        "Provide records showing when/how the VA was notified of forbearance start/end and any loss-mitigation actions.",
    ]

    return Finding(
        rule_id="C-01",
        severity=4,
        confidence=0.70 if any("forbear" in (t or "").lower() for _, t in covid_hits) else 0.65,
        title="COVID relief may not have protected the loan like it should",
        what_we_saw="We found COVID/forbearance language and also delinquency/late-fee language in your documents.",
        why_it_matters="If forbearance was active, delinquency/fees during that period can cause compounding harm and downstream actions.",
        evidence=evidence,
        questions=questions,
        policy_context=policy_context_text(start, end),
    )

def rule_suspense(doc_name: str, pages: List[str]) -> Optional[Finding]:
    for i, t in enumerate(pages, start=1):
        if "suspense" in (t or "").lower():
            return Finding(
                rule_id="C-03",
                severity=3,
                confidence=0.65,
                title="Payments may have been routed to suspense",
                what_we_saw="We found references to payments being held in suspense.",
                why_it_matters="Suspense can create phantom delinquency and fee cascades if not reconciled.",
                evidence=[EvidenceRef(doc_name, i, normalize_excerpt(t))],
                questions=[
                    "Provide a transaction-level ledger showing how each payment was applied (P/I/escrow/suspense).",
                    "Explain why payments entered suspense and when/if they were cleared.",
                ],
            )
    return None

# ============================
# Analysis + Grouping
# ============================
def analyze(docs: List[Tuple[str, bytes]], start: dt.date, end: dt.date, use_ocr: bool, ocr_page_cap: int) -> List[Dict]:
    findings = []
    for name, data in docs:
        pages = extract_pages_text_with_ocr(data, name, use_ocr=use_ocr, ocr_page_cap=ocr_page_cap)

        f1 = rule_C01(name, pages, start, end)
        if f1:
            findings.append(asdict(f1))

        f2 = rule_suspense(name, pages)
        if f2:
            findings.append(asdict(f2))

    findings.sort(key=lambda f: (f["severity"], f["confidence"]), reverse=True)
    return findings

def group_findings(findings: List[Dict]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for f in findings:
        key = f["rule_id"]
        if key not in grouped:
            grouped[key] = {
                "rule_id": f["rule_id"],
                "title": f["title"],
                "severity": f["severity"],
                "confidence": f["confidence"],
                "what_we_saw": f["what_we_saw"],
                "why_it_matters": f["why_it_matters"],
                "policy_context": f.get("policy_context"),
                "questions": f["questions"],
                "evidence": [],
                "sources": set(),
            }

        grouped[key]["evidence"].extend(f.get("evidence", []))
        for ev in f.get("evidence", []):
            grouped[key]["sources"].add(ev.get("doc_name"))

        grouped[key]["severity"] = max(grouped[key]["severity"], f["severity"])
        grouped[key]["confidence"] = max(grouped[key]["confidence"], f["confidence"])

    out = []
    for _, g in grouped.items():
        g["evidence"] = dedupe_evidence(g["evidence"])
        g["sources_count"] = len(g["sources"])
        g.pop("sources", None)
        out.append(g)

    out.sort(key=lambda g: (g["severity"], g["confidence"]), reverse=True)
    return out

# ============================
# UI
# ============================
st.set_page_config(page_title="Fast Mode Lite — VA Mortgage Checker", layout="wide")
st.title("Fast Mode Lite — VA Mortgage Checker")
st.caption("Now with OCR fallback for scanned PDFs.  Not legal advice.")

st.subheader("Your COVID forbearance window (used for comparisons)")
start = st.date_input("Forbearance start", value=dt.date(2020, 1, 1))
end = st.date_input("Forbearance end", value=dt.date(2025, 7, 31))

with st.expander("Policy context (CARES Act + VA circulars)"):
    for d, t, u in POLICY_TIMELINE:
        st.markdown(f"- **{d}** — {t}  \n  {u}")

# OCR controls
st.subheader("OCR settings (for scanned PDFs)")
use_ocr = st.toggle("Use OCR when a page has no readable text", value=True)

# Cap OCR pages per document to control cost
ocr_page_cap = st.slider("Max OCR pages per document", min_value=5, max_value=50, value=25, step=5)
if use_ocr and not st.secrets.get("GOOGLE_VISION_API_KEY", ""):
    st.warning("OCR is ON but no GOOGLE_VISION_API_KEY is set in Streamlit Secrets.  Add it in Manage app → Settings → Secrets.")

uploads = st.file_uploader("Upload VA mortgage documents (PDF)", type=["pdf"], accept_multiple_files=True)

with st.expander("Optional: Fill for letter download"):
    borrower_name = st.text_input("Borrower name", value="")
    loan_number = st.text_input("Loan number", value="")
    property_addr = st.text_input("Property address", value="")

if st.button("Analyze"):
    if not uploads:
        st.error("Please upload at least one PDF.")
        st.stop()

    docs = [(u.name, u.getvalue()) for u in uploads]

    with st.spinner("Analyzing… (OCR may take a little longer)"):
        raw = analyze(docs, start, end, use_ocr=use_ocr, ocr_page_cap=ocr_page_cap)
        grouped = group_findings(raw)

    st.subheader("Executive summary")
    if not grouped:
        st.success("No major issues detected by Fast Mode Lite in the text we could read.")
        st.write("If you expected issues, increase OCR pages or upload the payment ledger/servicer history PDFs.")
    else:
        top = grouped[0]
        st.warning(f"Likely issue: {top['title']}")
        st.write("What this means for a borrower:")
        st.write("- Your documents contain patterns worth clarifying with your servicer.")
        st.write("- The tool gives you exact locator cues and a ready-to-send letter.")

    st.subheader("Issues found (grouped)")
    if not grouped:
        st.info("No issues found.")
    else:
        for i, g in enumerate(grouped):
            st.markdown(f"### {g['title']}")
            st.caption(f"{badge(g['severity'], g['confidence'])}  •  Seen in {g['sources_count']} document(s)")

            st.write(g["what_we_saw"])
            if g.get("policy_context"):
                st.write(f"**Policy context:** {g['policy_context']}")
            st.write(f"**Why it matters:** {g['why_it_matters']}")

            st.write("**Where to look (exact locator cues)**")
            for ev in g["evidence"][:4]:
                doc = ev.get("doc_name", "")
                page = ev.get("page_number", "")
                excerpt = ev.get("excerpt", "")
                quote = normalize_excerpt(excerpt, 140)
                search = locator_phrase(excerpt)

                st.write(f"- **{doc} — Page {page}**")
                if search:
                    st.write(f"  **Search:** `{search}`")
                st.write(f"  **Quote:** “{quote}”")

            st.write("**Questions to ask (copy/paste)**")
            qtxt = "\n".join([f"- {q}" for q in g["questions"]])
            st.code(qtxt, language="markdown")

            st.button(
                "Copy questions",
                key=f"copy_{i}",
                on_click=lambda text=qtxt: st.session_state.update({"copied": text}),
                help="Copy this list into a message to your servicer or notes.",
            )

            letter = build_letter(borrower_name, loan_number, property_addr, start, end, g["questions"])
            st.download_button(
                "Download clarification letter (TXT)",
                data=letter.encode("utf-8"),
                file_name="clarification_letter.txt",
                mime="text/plain",
                key=f"dl_{i}",
            )

            st.divider()

    st.subheader("How this helps (plain terms)")
    st.write("1) It groups your paperwork into a small number of issues.")
    st.write("2) It tells you exactly where to look (file, page, and a search phrase).")
    st.write("3) It gives you a ready-to-send letter and questions.")
