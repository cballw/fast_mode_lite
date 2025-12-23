# app.py — Fast Mode Lite (Minimal + a little explanation)
# VA Mortgage Checker with CARES Act + VA Circular policy context panel
# Note: This Lite version reads selectable PDF text (scanned PDFs may need OCR upgrade next).

import io
import re
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader

# ----------------------------
# Policy timeline (curated, borrower-facing context)
# ----------------------------
POLICY_TIMELINE = [
    {
        "date": "2020-03-18",
        "title": "VA Circular 26-20-8 — Foreclosure moratorium guidance (COVID)",
        "url": "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_8.pdf",
    },
    {
        "date": "2020-03-27",
        "title": "CARES Act §4022 — Forbearance for federally backed mortgages (upon request/attestation)",
        "url": "https://www.congress.gov/116/plaws/publ136/PLAW-116publ136.pdf",
    },
    {
        "date": "2020-04-08",
        "title": "VA Circular 26-20-12 — CARES Act forbearance guidance for VA loans",
        "url": "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_12.pdf",
    },
    {
        "date": "2020-06-17",
        "title": "VA Circular 26-20-22 — Extends foreclosure moratorium",
        "url": "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_22.pdf",
    },
    {
        "date": "Index",
        "title": "VA circulars index (official list; use to expand policy timeline later)",
        "url": "https://www.benefits.va.gov/homeloans/resources_circulars.asp",
    },
]

CARES_EFFECTIVE = dt.date(2020, 3, 27)

# ----------------------------
# Models
# ----------------------------
@dataclass
class EvidenceRef:
    doc_name: str
    page_number: int  # 1-indexed
    excerpt: str

@dataclass
class Finding:
    rule_id: str
    severity: int  # 1-5
    confidence: float
    title: str
    what_we_saw: str
    why_it_matters: str
    evidence: List[EvidenceRef]
    questions: List[str]
    policy_context: Optional[str] = None  # small, borrower-facing context line

# ----------------------------
# Helpers
# ----------------------------
KEYWORDS = [
    "covid", "forbear", "cares", "suspense", "late fee", "delinquen",
    "past due", "escrow", "modification", "capitaliz", "loss mitigation", "reinstatement"
]

MONEY_RE = re.compile(r"(?<!\w)\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")

def normalize_excerpt(txt: str, max_len: int = 160) -> str:
    t = " ".join((txt or "").split())
    return t[:max_len] + ("…" if len(t) > max_len else "")

def pdf_extract_pages(file_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages

def money_values(text: str) -> List[float]:
    vals = []
    for m in MONEY_RE.finditer(text or ""):
        s = m.group(1).replace(",", "")
        try:
            vals.append(float(s))
        except:
            pass
    return vals

# Borrower-friendly labels
def impact_label(severity: int) -> str:
    if severity >= 4:
        return "High"
    if severity == 3:
        return "Medium"
    return "Low"

def sure_label(conf: float) -> str:
    if conf >= 0.75:
        return "Strong"
    if conf >= 0.60:
        return "Moderate"
    return "Needs more evidence"

def badge_text(severity: int, conf: float) -> str:
    return f"Impact: {impact_label(severity)}  •  How sure: {sure_label(conf)}"

def compact_evidence(ev) -> str:
    # ev may be an object or a dict (after asdict)
    doc = ev["doc_name"] if isinstance(ev, dict) else ev.doc_name
    page = ev["page_number"] if isinstance(ev, dict) else ev.page_number
    excerpt = ev["excerpt"] if isinstance(ev, dict) else ev.excerpt
    return f'{doc} — p. {page}: “{normalize_excerpt(excerpt)}”'

def questions_block(questions: List[str]) -> str:
    return "\n".join([f"- {q}" for q in questions])

# ----------------------------
# Fast Mode Lite Rules (heuristics)
# ----------------------------
def build_policy_context(forb_start: dt.date, forb_end: dt.date) -> Optional[str]:
    """
    Minimal, borrower-facing policy note.
    We do not declare violations; we provide context and prompt for clarification.
    """
    # If their window overlaps CARES effective date onward, policy context is relevant.
    if forb_end >= CARES_EFFECTIVE:
        return ("Policy context: CARES Act §4022 and VA COVID circular guidance generally supported forbearance "
                "as a protected status.  If late fees or delinquency coding appear during an active forbearance window, "
                "the servicer should be able to explain it with the forbearance plan record and a transaction-level ledger.")
    return ("Policy context: This portion of your forbearance window begins before the CARES Act effective date (3/27/2020).  "
            "We still flag COVID-related handling issues, but the applicable policy context may differ for early 2020.")

def rule_C01_relief_recognition_failure(doc_name: str, pages: List[str], forb_start: dt.date, forb_end: dt.date) -> Optional[Finding]:
    """
    Lite heuristic:
    - Detect forbearance/COVID language AND late fee/delinquency language in the same document set.
    (Next upgrade will associate these with page dates to make this far stronger.)
    """
    covid_pages = []
    late_pages = []

    for i, t in enumerate(pages, start=1):
        low = (t or "").lower()
        if ("forbear" in low) or ("covid" in low) or ("cares" in low):
            covid_pages.append((i, t))
        if ("late fee" in low) or ("delinquen" in low) or ("past due" in low):
            late_pages.append((i, t))

    if not covid_pages or not late_pages:
        return None

    ev = [
        EvidenceRef(doc_name, covid_pages[0][0], normalize_excerpt(covid_pages[0][1])),
        EvidenceRef(doc_name, late_pages[0][0], normalize_excerpt(late_pages[0][1])),
    ]

    return Finding(
        rule_id="C-01",
        severity=4,
        confidence=0.65,
        title="COVID relief may not have protected the loan like it should",
        what_we_saw="Your documents mention COVID/forbearance and also show late-fee or delinquency language.",
        why_it_matters="If forbearance was active, delinquency/fees during that period can cause compounding harm (fees, balance changes, downstream actions).",
        evidence=ev,
        questions=[
            "Provide the complete forbearance plan record (start/end dates) and all system notes.",
            "Explain any late fees, delinquency coding, or 'past due' status during the relief window.",
            "Provide a transaction-level ledger showing how payments were applied (principal/interest/escrow/suspense).",
            "Provide records showing when/how the VA was notified of forbearance start/end and any loss-mitigation actions.",
        ],
        policy_context=build_policy_context(forb_start, forb_end),
    )

def rule_C03_payment_misapplication(doc_name: str, pages: List[str], forb_start: dt.date, forb_end: dt.date) -> Optional[Finding]:
    suspense = []
    for i, t in enumerate(pages, start=1):
        if "suspense" in (t or "").lower():
            suspense.append((i, t))
    if not suspense:
        return None

    ev = [EvidenceRef(doc_name, suspense[0][0], normalize_excerpt(suspense[0][1]))]

    return Finding(
        rule_id="C-03",
        severity=3,
        confidence=0.60,
        title="Payments may have been routed to suspense",
        what_we_saw="We found references to 'suspense' (payments can be held or applied in a non-standard way).",
        why_it_matters="Payments in suspense can create phantom delinquency and fee cascades if not cleared correctly.",
        evidence=ev,
        questions=[
            "Provide a transaction-level payment ledger showing application to principal/interest/escrow/suspense.",
            "Explain why payments were placed into suspense and when/if they were cleared.",
        ],
        policy_context=None,
    )

def rule_ESCROW_indicator(doc_name: str, pages: List[str], forb_start: dt.date, forb_end: dt.date) -> Optional[Finding]:
    escrow_hits = []
    for i, t in enumerate(pages, start=1):
        low = (t or "").lower()
        if "escrow" in low:
            vals = money_values(t)
            escrow_hits.append((i, t, max(vals) if vals else None))
    if not escrow_hits:
        return None

    ev = [EvidenceRef(doc_name, escrow_hits[0][0], normalize_excerpt(escrow_hits[0][1]))]

    return Finding(
        rule_id="E-ESCROW",
        severity=2,
        confidence=0.50,
        title="Escrow activity detected (check for payment jumps)",
        what_we_saw="We found escrow statements or escrow analysis language in your documents.",
        why_it_matters="Escrow recalculations after relief can cause sudden payment increases and should be itemized clearly.",
        evidence=ev,
        questions=[
            "Provide escrow analysis statements and itemized advances (tax/insurance) during and after relief.",
            "Explain any shortage calculations and what notices were provided.",
        ],
        policy_context=None,
    )

def rule_C08_policy_timing_indicator(doc_name: str, pages: List[str], forb_start: dt.date, forb_end: dt.date) -> Optional[Finding]:
    """
    Policy Timing Check (Lite):
    If the borrower window overlaps CARES/VA COVID period AND we detect delinquency language,
    we show a policy-context 'needs clarification' indicator.
    NOTE: Without page dates, this stays moderate/needs-evidence.
    """
    if forb_end < CARES_EFFECTIVE:
        return None

    delinquency_pages = []
    for i, t in enumerate(pages, start=1):
        low = (t or "").lower()
        if ("delinquen" in low) or ("past due" in low) or ("late fee" in low):
            delinquency_pages.append((i, t))

    if not delinquency_pages:
        return None

    ev = [EvidenceRef(doc_name, delinquency_pages[0][0], normalize_excerpt(delinquency_pages[0][1]))]

    return Finding(
        rule_id="C-08-Lite",
        severity=3,
        confidence=0.55,
        title="Policy context mismatch indicator (needs clarification)",
        what_we_saw="Delinquency/late-fee language appears in documents during the broader COVID policy period. We need statement dates to confirm timing inside your forbearance window.",
        why_it_matters="CARES Act/VA COVID guidance generally treated forbearance as protected status. If delinquency coding occurred during active forbearance, it should be explained with records.",
        evidence=ev,
        questions=[
            "Provide statement dates for the pages showing delinquency/late fees and confirm whether those dates fall within your forbearance window.",
            "Provide the forbearance plan record and transaction ledger for the same period.",
        ],
        policy_context=build_policy_context(forb_start, forb_end),
    )

def run_fast_mode_lite(docs: List[Tuple[str, bytes]], forb_start: dt.date, forb_end: dt.date) -> Dict[str, Any]:
    findings: List[Finding] = []
    doc_summaries = []

    for doc_name, b in docs:
        pages = pdf_extract_pages(b)

        # keyword hits summary (transparency)
        hits = []
        for i, t in enumerate(pages, start=1):
            low = (t or "").lower()
            if any(k in low for k in KEYWORDS):
                hits.append({"page": i, "excerpt": normalize_excerpt(t)})
                if len(hits) >= 6:
                    break

        doc_summaries.append({"doc_name": doc_name, "pages": len(pages), "keyword_hits": hits})

        # Rules (Fast Mode Lite)
        for rule in (
            rule_C01_relief_recognition_failure,
            rule_C08_policy_timing_indicator,
            rule_C03_payment_misapplication,
            rule_ESCROW_indicator,
        ):
            f = rule(doc_name, pages, forb_start, forb_end)
            if f:
                findings.append(f)

    findings.sort(key=lambda f: (f.severity, f.confidence), reverse=True)

    return {"doc_summaries": doc_summaries, "findings": [asdict(f) for f in findings]}

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Fast Mode Lite — VA Mortgage Checker", layout="wide")
st.title("Fast Mode Lite — VA Mortgage Checker")
st.caption("Minimal, evidence-based flags + policy context.  Not legal advice.")

# Forbearance window inputs (pre-filled to your dates)
st.subheader("Your COVID forbearance window (used for comparisons)")
forb_start = st.date_input("Forbearance start", value=dt.date(2020, 1, 1))
forb_end = st.date_input("Forbearance end", value=dt.date(2025, 7, 31))

st.subheader("Policy context (Fast Mode)")
with st.expander("CARES Act + VA COVID circulars (what we compare against)"):
    for p in POLICY_TIMELINE:
        st.markdown(f"- **{p['date']}** — {p['title']}  \n  {p['url']}")

uploads = st.file_uploader("Upload VA mortgage documents (PDF)", type=["pdf"], accept_multiple_files=True)

with st.expander("What this checks (Fast Mode Lite)"):
    st.write("- COVID/forbearance + delinquency/late-fee mismatch (Lite)")
    st.write("- Policy context mismatch indicator (Lite)")
    st.write("- Suspense indicators")
    st.write("- Escrow activity indicators")
    st.write("This Lite version reads selectable text.  Scanned PDFs may need OCR (next upgrade).")

run = st.button("Analyze")

if run:
    if not uploads:
        st.error("Please upload at least one PDF.")
        st.stop()

    docs = [(u.name, u.getvalue()) for u in uploads]

    with st.spinner("Analyzing…"):
        result = run_fast_mode_lite(docs, forb_start, forb_end)

    st.success("Done.")

    st.subheader("Red Flags (simple)")
    if not result["findings"]:
        st.info("No high-signal flags found with Lite checks.  This does not mean nothing happened.")
    else:
        for f in result["findings"]:
            st.markdown(f"### {f['title']}")
            st.caption(badge_text(f["severity"], f["confidence"]))

            st.write(f["what_we_saw"])

            if f.get("policy_context"):
                st.write(f"**Policy context:** {f['policy_context']}")

            st.write(f"**Why it matters:** {f['why_it_matters']}")

            st.write("**Where to look**")
            for ev in f["evidence"][:2]:
                st.write("- " + compact_evidence(ev))

            qtxt = questions_block(f["questions"])
            st.code(qtxt, language="markdown")

            st.button(
                f"Copy questions ({f['rule_id']})",
                on_click=lambda text=qtxt: st.session_state.update({"copied": text}),
                help="Copy this list into a message to your servicer or notes.",
            )
            st.divider()

    st.subheader("Document signals (for transparency)")
    for ds in result["doc_summaries"]:
        with st.expander(f"{ds['doc_name']} — {ds['pages']} pages"):
            if ds["keyword_hits"]:
                for h in ds["keyword_hits"]:
                    st.write(f"Page {h['page']}: “{h['excerpt']}”")
            else:
                st.write("No keyword hits detected. If this is a scanned PDF, OCR is likely needed.")
