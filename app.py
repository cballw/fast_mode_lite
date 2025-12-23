# app.py — Fast Mode Lite (Scorecard + Loan Modification Review + OCR Fallback)
# Borrower-friendly: PASS / POSSIBLE ISSUE / UNKNOWN
# Compares evidence in docs against CARES Act + VA COVID circular policy context (contextual, not legal advice)

import io
import re
import base64
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from pypdf import PdfReader
import fitz  # PyMuPDF (for rendering scanned PDF pages to images)

# ============================
# Policy context (high-level references for borrower research)
# ============================
POLICY_CONTEXT = [
    ("2020-03-27", "CARES Act §4022 (Forbearance for federally-backed mortgages upon request/attestation)",
     "https://www.congress.gov/116/plaws/publ136/PLAW-116publ136.pdf"),
    ("2020-03-18", "VA Circular 26-20-8 (COVID foreclosure moratorium guidance)",
     "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_8.pdf"),
    ("2020-04-08", "VA Circular 26-20-12 (CARES Act / COVID forbearance guidance for VA loans)",
     "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_12.pdf"),
    ("2020-06-17", "VA Circular 26-20-22 (Extends foreclosure moratorium)",
     "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_22.pdf"),
    ("Index", "VA circulars index (official list; expand policy timeline here as needed)",
     "https://www.benefits.va.gov/homeloans/resources_circulars.asp"),
]
CARES_EFFECTIVE = dt.date(2020, 3, 27)

# ============================
# Models
# ============================
@dataclass
class EvidenceRef:
    doc_name: str
    page_number: int
    quote: str
    search_phrase: str

@dataclass
class ScoreItem:
    id: str
    label: str
    status: str  # PASS | POSSIBLE ISSUE | UNKNOWN
    why: str
    policy: str
    evidence: List[EvidenceRef]
    request_next: List[str]

# ============================
# Text + OCR utilities
# ============================
def normalize_space(s: str) -> str:
    return " ".join((s or "").split())

def short_quote(s: str, limit: int = 180) -> str:
    t = normalize_space(s)
    return t[:limit] + ("…" if len(t) > limit else "")

def locator_phrase(excerpt: str, max_words: int = 7) -> str:
    text = normalize_space(excerpt)
    text = re.sub(r"[“”\"'`]", "", text)
    words = text.split()
    stop = {"the","and","or","to","of","a","in","for","with","your","you","is","are","on","this","that","as","be","by","it"}
    filtered = [w for w in words if w.lower() not in stop]
    pick = filtered if len(filtered) >= max_words else words
    return " ".join(pick[:max_words]) if pick else ""

def render_pdf_page_to_png(pdf_bytes: bytes, page_index0: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

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

def extract_pages_text_with_ocr(pdf_bytes: bytes, use_ocr: bool, ocr_page_cap: int) -> List[str]:
    """
    Extract selectable text first. If a page is empty/very short and OCR is enabled, OCR that page (up to cap).
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    ocr_used = 0
    api_key = st.secrets.get("GOOGLE_VISION_API_KEY", "")

    for idx, page in enumerate(reader.pages):
        t = (page.extract_text() or "").strip()
        if use_ocr and len(t) < 20 and api_key and ocr_used < ocr_page_cap:
            try:
                png = render_pdf_page_to_png(pdf_bytes, idx, zoom=2.0)
                ocr_text = google_vision_ocr_image_bytes(png, api_key).strip()
                if ocr_text:
                    t = ocr_text
                ocr_used += 1
            except Exception:
                pass
        pages_text.append(t)
    return pages_text

# ============================
# Evidence extraction helpers
# ============================
def find_hits(pages: List[str], patterns: List[str], max_hits: int = 3) -> List[Tuple[int, str]]:
    """
    Return list of (page_number, snippet) for pages that match any pattern.
    """
    hits = []
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    for i, t in enumerate(pages, start=1):
        if not t:
            continue
        if any(r.search(t) for r in regs):
            # take a focused snippet around first match (best-effort)
            low = t.lower()
            idx = min([low.find(p.lower().strip("\\b")) for p in patterns if low.find(p.lower().strip("\\b")) != -1] or [0])
            snippet = t[max(0, idx-120): idx+260]
            hits.append((i, short_quote(snippet)))
            if len(hits) >= max_hits:
                break
    return hits

def evidence_from_hits(doc_name: str, hits: List[Tuple[int, str]]) -> List[EvidenceRef]:
    ev = []
    for page, snippet in hits:
        ev.append(EvidenceRef(
            doc_name=doc_name,
            page_number=page,
            quote=short_quote(snippet, 180),
            search_phrase=locator_phrase(snippet, 7),
        ))
    return ev

# ============================
# Document classification (lightweight)
# ============================
def classify_doc(pages: List[str]) -> str:
    blob = "\n".join(pages[:5]).lower()
    if any(k in blob for k in ["loan modification", "modification agreement", "capitalized", "arrears", "effective date of modification"]):
        return "MODIFICATION"
    if "escrow" in blob and "analysis" in blob:
        return "ESCROW"
    if any(k in blob for k in ["payment coupon", "amount due", "billing statement", "late charge", "past due"]):
        return "BILLING"
    return "OTHER"

# ============================
# Scorecard logic (PASS / POSSIBLE ISSUE / UNKNOWN)
# ============================
def build_policy_blurb(forb_start: dt.date, forb_end: dt.date) -> str:
    if forb_end >= CARES_EFFECTIVE:
        return ("CARES Act §4022 and VA COVID circular guidance (e.g., 26-20-12) created/clarified COVID forbearance options "
                "and expected protected servicing treatment during active forbearance.  This tool checks whether your documents "
                "show those protections in practice (without making legal conclusions).")
    return ("Your window begins before CARES Act effective date (3/27/2020).  The tool still checks COVID-era handling, "
            "but early-2020 policy context may differ and may require additional clarification.")

def scorecard(docs_text: List[Tuple[str, str, List[str]]], forb_start: dt.date, forb_end: dt.date) -> List[ScoreItem]:
    """
    docs_text entries: (doc_name, doc_type, pages_text_list)
    """
    # Aggregate evidence across docs
    forb_evidence = []
    delin_evidence = []
    suspense_evidence = []
    mod_evidence = []
    mod_terms_evidence = []
    va_coord_evidence = []

    for doc_name, doc_type, pages in docs_text:
        # Forbearance evidence
        forb_hits = find_hits(pages, [r"\bforbear", r"\bcovid", r"\bcares act\b", r"\bpayment pause\b"], max_hits=2)
        forb_evidence += evidence_from_hits(doc_name, forb_hits)

        # Delinquency/fees evidence
        delin_hits = find_hits(pages, [r"\blate fee\b", r"\blate charge\b", r"\bpast due\b", r"\bdelinquen"], max_hits=2)
        delin_evidence += evidence_from_hits(doc_name, delin_hits)

        # Suspense evidence
        susp_hits = find_hits(pages, [r"\bsuspense\b"], max_hits=2)
        suspense_evidence += evidence_from_hits(doc_name, susp_hits)

        # Modification evidence (presence)
        if doc_type == "MODIFICATION":
            mh = find_hits(pages, [r"\bloan modification\b", r"\bmodification agreement\b", r"\beffective date\b"], max_hits=2)
            mod_evidence += evidence_from_hits(doc_name, mh)

            # Modification key terms (arrears/capitalization/term/rate/payment)
            terms = find_hits(pages, [
                r"\barrears\b", r"\bcapitaliz", r"\bprincipal balance\b",
                r"\binterest rate\b", r"\bmaturity\b", r"\bterm\b", r"\bpayment\b"
            ], max_hits=3)
            mod_terms_evidence += evidence_from_hits(doc_name, terms)

        # VA coordination hints
        va_hits = find_hits(pages, [r"\bVA\b", r"\bVALERI\b", r"\bLoan Guaranty\b", r"\bveterans affairs\b"], max_hits=2)
        va_coord_evidence += evidence_from_hits(doc_name, va_hits)

    # Deduplicate evidence lightly (by doc+page+phrase)
    def dedupe(ev: List[EvidenceRef]) -> List[EvidenceRef]:
        seen = set()
        out = []
        for e in ev:
            key = (e.doc_name, e.page_number, e.search_phrase)
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    forb_evidence = dedupe(forb_evidence)
    delin_evidence = dedupe(delin_evidence)
    suspense_evidence = dedupe(suspense_evidence)
    mod_evidence = dedupe(mod_evidence)
    mod_terms_evidence = dedupe(mod_terms_evidence)
    va_coord_evidence = dedupe(va_coord_evidence)

    policy_note = build_policy_blurb(forb_start, forb_end)

    items: List[ScoreItem] = []

    # 1) Forbearance documented
    if forb_evidence:
        status = "PASS"
        why = "Your documents include COVID/forbearance-related language."
        req = []
        ev = forb_evidence[:4]
    else:
        status = "UNKNOWN"
        why = "We did not find a clear forbearance plan/approval record in the uploaded text."
        req = [
            "Request the complete COVID forbearance plan record (start/end dates) and system notes.",
            "Request any forbearance approval/confirmation letters or portal messages.",
        ]
        ev = []
    items.append(ScoreItem(
        id="S1",
        label="Forbearance is documented in the record",
        status=status,
        why=why,
        policy=policy_note,
        evidence=ev,
        request_next=req
    ))

    # 2) Protected treatment during forbearance (fees/delinquency)
    if delin_evidence and forb_evidence:
        status = "POSSIBLE ISSUE"
        why = "We found delinquency/late-fee language as well as COVID/forbearance language. This combination should be explained with a ledger and forbearance plan details."
        req = [
            "Ask the servicer to explain any late fees/delinquency coding during the forbearance window.",
            "Request a transaction-level ledger for the same period (P/I/escrow/suspense).",
        ]
        ev = (delin_evidence[:2] + forb_evidence[:2])[:4]
    elif delin_evidence and not forb_evidence:
        status = "UNKNOWN"
        why = "We found delinquency/late-fee language, but did not find a clear forbearance record in the uploaded text."
        req = [
            "Request the complete forbearance plan record to confirm whether the loan should have been protected at that time.",
            "Request the full transaction ledger covering the suspected forbearance months.",
        ]
        ev = delin_evidence[:3]
    else:
        status = "UNKNOWN"
        why = "We did not find clear delinquency/late-fee language in the uploaded text. This does not prove it didn’t occur (scanned PDFs may hide it)."
        req = [
            "If you suspect fees/delinquency occurred, upload monthly statements or the servicing ledger for that time period.",
        ]
        ev = []
    items.append(ScoreItem(
        id="S2",
        label="Protected treatment during forbearance (no late fees / delinquency coding)",
        status=status,
        why=why,
        policy="CARES Act/VA COVID circular context generally expected protected treatment during active forbearance; exceptions should be documented and explainable.",
        evidence=ev,
        request_next=req
    ))

    # 3) Payment application / suspense
    if suspense_evidence:
        status = "POSSIBLE ISSUE"
        why = "We found 'suspense' references. Suspense can indicate payments were held or applied in a non-standard way."
        req = [
            "Request a transaction-level ledger showing how every payment was applied (principal/interest/escrow/suspense).",
            "Ask for a suspense reconciliation showing when/why it was created and when it was cleared.",
        ]
        ev = suspense_evidence[:4]
    else:
        status = "UNKNOWN"
        why = "We did not find suspense language. If your PDF is scanned, we may miss it; the ledger is the best source of truth."
        req = [
            "If you have it, upload the payment history / transaction ledger. That is the best document for payment-application checks.",
        ]
        ev = []
    items.append(ScoreItem(
        id="S3",
        label="Payments were applied correctly (no suspense trap / misapplication)",
        status=status,
        why=why,
        policy="Servicing records should show clear payment application. During COVID relief, misapplication can create phantom delinquency.",
        evidence=ev,
        request_next=req
    ))

    # 4) Loan modification is documented (if uploaded)
    if mod_evidence:
        status = "PASS"
        why = "We detected a loan modification document and key modification language."
        req = []
        ev = mod_evidence[:4]
    else:
        status = "UNKNOWN"
        why = "We did not detect a clear loan modification agreement in the uploaded text."
        req = [
            "If you had a modification, upload the Modification Agreement or Trial Plan documents (usually titled 'Loan Modification Agreement').",
        ]
        ev = []
    items.append(ScoreItem(
        id="S4",
        label="Loan modification is documented (if applicable)",
        status=status,
        why=why,
        policy="Post-forbearance outcomes often include repayment plans or modifications; documentation should clearly state effective date and new terms.",
        evidence=ev,
        request_next=req
    ))

    # 5) Modification transparency: arrears/capitalization/term/rate/payment clearly stated
    if mod_evidence and mod_terms_evidence:
        status = "PASS"
        why = "The modification document appears to include term details like arrears/capitalization/rate/payment/term indicators (at least partially)."
        req = []
        ev = mod_terms_evidence[:4]
    elif mod_evidence and not mod_terms_evidence:
        status = "POSSIBLE ISSUE"
        why = "A modification document was detected, but we did not find clear language explaining arrears/capitalization/term/rate/payment changes in the text we read."
        req = [
            "Request the arrears reconciliation used to build the modification (what amounts were added and why).",
            "Request the comparison or disclosure that explains how the modification changes total cost and term.",
        ]
        ev = mod_evidence[:3]
    else:
        status = "UNKNOWN"
        why = "No modification transparency check could be completed without a modification agreement document."
        req = []
        ev = []
    items.append(ScoreItem(
        id="S5",
        label="Modification terms are transparent (arrears/capitalization/term/rate/payment)",
        status=status,
        why=why,
        policy="For borrowers doing research: modifications should clearly explain what was added (arrears/escrow/fees), whether amounts were capitalized, and how terms changed.",
        evidence=ev,
        request_next=req
    ))

    # 6) VA coordination evidence present
    if va_coord_evidence:
        status = "PASS"
        why = "We found VA-related references (VA/Loan Guaranty/VALERI wording) in the uploaded text."
        req = []
        ev = va_coord_evidence[:4]
    else:
        status = "UNKNOWN"
        why = "We did not find clear VA coordination references in the text we read. This does not prove VA wasn’t notified; it means the documents you uploaded don’t show it."
        req = [
            "Request any VA case identifiers or confirmation of VA notifications (dates/methods) tied to forbearance and loss mitigation.",
            "If you have FOIA records, Deep Mode can analyze VA internal timelines more directly.",
        ]
        ev = []
    items.append(ScoreItem(
        id="S6",
        label="Evidence that VA was coordinated/notified (where applicable)",
        status=status,
        why=why,
        policy="VA circular guidance and VA servicing systems (VALERI) often reflect key milestones; missing references can indicate a recordkeeping gap.",
        evidence=ev,
        request_next=req
    ))

    return items

# ============================
# Letter generator (uses UNKNOWN/POSSIBLE ISSUE requests)
# ============================
def build_clarification_letter(borrower_name: str, loan_number: str, property_addr: str,
                              forb_start: dt.date, forb_end: dt.date, items: List[ScoreItem]) -> str:
    # Pull request items from anything not PASS
    reqs = []
    for it in items:
        if it.status != "PASS":
            reqs.extend(it.request_next)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for r in reqs:
        if r and r not in seen:
            seen.add(r)
            deduped.append(r)

    bullets = "\n".join([f"- {r}" for r in deduped]) if deduped else "- Please provide the complete servicing file and forbearance records for my loan."

    return f"""Subject: Request for clarification and records regarding COVID forbearance and loss mitigation
Loan Number: {loan_number or "[loan number]"}
Property Address: {property_addr or "[property address]"}
Date: {dt.date.today().isoformat()}

Dear Servicing Team,

I am requesting clarification and records regarding the handling of my COVID-19 forbearance and related loss-mitigation activity on my VA-backed mortgage.

My COVID forbearance window is approximately {forb_start.isoformat()} through {forb_end.isoformat()}.

Based on my document review, I need the following to reconcile the servicing record:

{bullets}

Thank you for your attention. I look forward to your response.

Sincerely,
{borrower_name or "[Borrower Name]"}
"""

# ============================
# UI
# ============================
st.set_page_config(page_title="Fast Mode Lite — VA COVID Relief Compliance Checker", layout="wide")
st.title("VA COVID Relief Compliance Checker (Fast Mode Lite)")
st.caption("Borrower-friendly scorecard based on your documents.  Not legal advice.")

st.subheader("Your COVID forbearance window (used for the scorecard)")
forb_start = st.date_input("Forbearance start", value=dt.date(2020, 1, 1))
forb_end = st.date_input("Forbearance end", value=dt.date(2025, 7, 31))

with st.expander("Policy references (for your research)"):
    for d, t, u in POLICY_CONTEXT:
        st.markdown(f"- **{d}** — {t}  \n  {u}")

st.subheader("OCR settings (for scanned PDFs)")
use_ocr = st.toggle("Use OCR when a page has no readable text", value=True)
ocr_page_cap = st.slider("Max OCR pages per document (controls cost)", 5, 50, 25, 5)
if use_ocr and not st.secrets.get("GOOGLE_VISION_API_KEY", ""):
    st.warning("OCR is ON but GOOGLE_VISION_API_KEY is not set.  Add it in Manage app → Settings → Secrets.")

uploads = st.file_uploader("Upload mortgage documents (PDF)", type=["pdf"], accept_multiple_files=True)

with st.expander("Optional: for letter download"):
    borrower_name = st.text_input("Borrower name", value="")
    loan_number = st.text_input("Loan number", value="")
    property_addr = st.text_input("Property address", value="")

if st.button("Run compliance scorecard"):
    if not uploads:
        st.error("Please upload at least one PDF.")
        st.stop()

    docs_text: List[Tuple[str, str, List[str]]] = []

    with st.spinner("Reading documents (OCR may take a bit)…"):
        for u in uploads:
            pdf_bytes = u.getvalue()
            pages = extract_pages_text_with_ocr(pdf_bytes, use_ocr=use_ocr, ocr_page_cap=ocr_page_cap)
            doc_type = classify_doc(pages)
            docs_text.append((u.name, doc_type, pages))

    items = scorecard(docs_text, forb_start, forb_end)

    # Executive summary (very plain)
    st.subheader("Executive summary")
    possible = [it for it in items if it.status == "POSSIBLE ISSUE"]
    unknown = [it for it in items if it.status == "UNKNOWN"]

    if possible:
        st.warning("Result: Needs clarification")
        st.write("We found patterns in your documents that commonly indicate COVID relief may not have been applied or documented correctly.")
    elif unknown:
        st.info("Result: Inconclusive with the documents we could read")
        st.write("We didn’t find enough evidence to confirm key COVID relief details. The scorecard tells you what to request next.")
    else:
        st.success("Result: Looks consistent based on the documents we could read")
        st.write("No major issues were detected in the text we could read. This does not prove everything was perfect, but it’s a good sign.")

    # Scorecard display
    st.subheader("COVID Relief Compliance Scorecard")
    for it in items:
        status_icon = "✅" if it.status == "PASS" else "⚠️" if it.status == "POSSIBLE ISSUE" else "❓"
        st.markdown(f"### {status_icon} {it.label} — **{it.status}**")
        st.write(it.why)
        st.write(f"**Policy context:** {it.policy}")

        if it.evidence:
            st.write("**Where to look (exact locator cues)**")
            for ev in it.evidence[:4]:
                st.write(f"- **{ev.doc_name} — Page {ev.page_number}**")
                if ev.search_phrase:
                    st.write(f"  **Search:** `{ev.search_phrase}`")
                st.write(f"  **Quote:** “{ev.quote}”")

        if it.request_next:
            st.write("**What to request next**")
            req_txt = "\n".join([f"- {r}" for r in it.request_next])
            st.code(req_txt, language="markdown")

        st.divider()

    # Action Pack
    st.subheader("Action Pack")
    letter = build_clarification_letter(borrower_name, loan_number, property_addr, forb_start, forb_end, items)

    st.download_button(
        "Download clarification letter (TXT)",
        data=letter.encode("utf-8"),
        file_name="covid_relief_clarification_letter.txt",
        mime="text/plain",
    )

    st.write("**How this helps the public (plain terms):**")
    st.write("1) It turns paperwork into a simple PASS / POSSIBLE ISSUE / UNKNOWN checklist.")
    st.write("2) It shows exactly where in your documents the evidence appears (file, page, and search phrase).")
    st.write("3) It tells you what record to request when something is missing.")
