
import io
import re
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader

# ----------------------------
# Models
# ----------------------------
@dataclass
class EvidenceRef:
    doc_name: str
    page_number: int  # 1-indexed
    excerpt: str

@dataclass
class LoanEvent:
    date: Optional[str]
    type: str
    amount: Optional[float]
    details: Dict[str, Any]
    source: str  # borrower | servicer | inferred
    confidence: float
    evidence: List[EvidenceRef]

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

# ----------------------------
# Helpers
# ----------------------------
KEYWORDS = [
    "covid", "forbear", "cares", "suspense", "late fee", "delinquen",
    "escrow", "modification", "capitaliz", "loss mitigation", "reinstatement"
]

DATE_PATTERNS = [
    # mm/dd/yyyy
    re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-]((19|20)\d\d)\b"),
    # Month dd, yyyy
    re.compile(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
               r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
               r"Nov(?:ember)?|Dec(?:ember)?)\s+([0-3]?\d),\s+((19|20)\d\d)\b",
               re.IGNORECASE),
]

MONEY_RE = re.compile(r"(?<!\w)\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")

def normalize_excerpt(txt: str, max_len: int = 220) -> str:
    t = " ".join(txt.split())
    return t[:max_len] + ("…" if len(t) > max_len else "")

def pdf_extract_pages(file_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages

def find_keyword_hits(pages: List[str]) -> List[Tuple[int, str]]:
    hits = []
    for i, text in enumerate(pages, start=1):
        low = text.lower()
        if any(k in low for k in KEYWORDS):
            # capture a short excerpt around first keyword match
            idx = min([low.find(k) for k in KEYWORDS if low.find(k) != -1] or [0])
            snippet = text[max(0, idx-80): idx+200]
            hits.append((i, normalize_excerpt(snippet)))
    return hits

def find_dates(text: str) -> List[str]:
    found = []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            found.append(m.group(0))
    # de-dup, preserve order
    seen = set()
    out = []
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out

def money_values(text: str) -> List[float]:
    vals = []
    for m in MONEY_RE.finditer(text):
        s = m.group(1).replace(",", "")
        try:
            vals.append(float(s))
        except:
            pass
    return vals

# ----------------------------
# Fast Mode Lite Rules (heuristic)
# ----------------------------
def rule_C01_relief_recognition_failure(doc_name: str, pages: List[str]) -> Optional[Finding]:
    """
    Lite heuristic:
    - Evidence of forbearance/COVID exists AND
    - During same doc set, evidence of late fees / delinquency exists
    """
    covid_pages = []
    late_pages = []
    for i, t in enumerate(pages, start=1):
        low = t.lower()
        if "forbear" in low or "covid" in low or "cares" in low:
            covid_pages.append((i, t))
        if "late fee" in low or "delinquen" in low or "past due" in low:
            late_pages.append((i, t))

    if not covid_pages or not late_pages:
        return None

    ev = []
    # take 1-2 evidence refs
    cp = covid_pages[0]
    lp = late_pages[0]
    ev.append(EvidenceRef(doc_name, cp[0], normalize_excerpt(cp[1])))
    ev.append(EvidenceRef(doc_name, lp[0], normalize_excerpt(lp[1])))

    return Finding(
        rule_id="C-01",
        severity=4,
        confidence=0.65,
        title="COVID relief may not be reflected in loan behavior",
        what_we_saw="We found language suggesting COVID forbearance/relief and also found late-fee/delinquency indicators in the same document set.",
        why_it_matters="If relief existed but the loan was treated as delinquent during that window, downstream fees, balances, and loss-mitigation actions can be based on an incorrect servicing state.",
        evidence=ev,
        questions=[
            "Provide the complete forbearance plan record (start/end dates) and all system notes.",
            "Explain any late fees, delinquency coding, or 'past due' status during the relief window.",
            "Provide a transaction-level ledger showing how any payments were applied (principal/interest/escrow/suspense)."
        ]
    )

def rule_C03_payment_misapplication(doc_name: str, pages: List[str]) -> Optional[Finding]:
    suspense = []
    for i, t in enumerate(pages, start=1):
        if "suspense" in t.lower():
            suspense.append((i, t))
    if not suspense:
        return None
    ev = [EvidenceRef(doc_name, suspense[0][0], normalize_excerpt(suspense[0][1]))]
    return Finding(
        rule_id="C-03",
        severity=3,
        confidence=0.60,
        title="Payments may have been routed to suspense",
        what_we_saw="We found references to 'suspense' which can indicate payments were held or applied in a non-standard way.",
        why_it_matters="Misapplied payments can create phantom delinquency and fee cascades. A ledger-level reconciliation is often needed.",
        evidence=ev,
        questions=[
            "Provide a transaction-level payment ledger showing application to principal/interest/escrow/suspense.",
            "Explain why payments were placed into suspense and when/if they were cleared."
        ]
    )

def rule_C07_escrow_shock_indicator(doc_name: str, pages: List[str]) -> Optional[Finding]:
    """
    Lite heuristic: if 'escrow' appears with big money deltas (>=20% not computed, just highlight).
    """
    escrow_hits = []
    for i, t in enumerate(pages, start=1):
        low = t.lower()
        if "escrow" in low:
            vals = money_values(t)
            if vals:
                escrow_hits.append((i, t, max(vals)))
            else:
                escrow_hits.append((i, t, None))
    if not escrow_hits:
        return None
    ev = [EvidenceRef(doc_name, escrow_hits[0][0], normalize_excerpt(escrow_hits[0][1]))]
    return Finding(
        rule_id="E-ESCROW",
        severity=2,
        confidence=0.50,
        title="Escrow activity detected (review for shocks)",
        what_we_saw="We detected escrow-related statements. Escrow recalculations after forbearance can cause sudden payment changes.",
        why_it_matters="Escrow shortages or advances can create unexpected payment spikes and may mask servicing errors if notices are unclear.",
        evidence=ev,
        questions=[
            "Provide escrow analysis statements and itemized advances (tax/insurance) during and after relief.",
            "Explain any shortage calculations and the notices provided."
        ]
    )

def run_fast_mode_lite(docs: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    all_findings: List[Finding] = []
    doc_summaries = []

    for doc_name, b in docs:
        pages = pdf_extract_pages(b)
        hits = find_keyword_hits(pages)
        # record summary
        doc_summaries.append({
            "doc_name": doc_name,
            "pages": len(pages),
            "keyword_hits": [{"page": p, "excerpt": ex} for p, ex in hits[:10]]
        })

        # rules
        for rule in (rule_C01_relief_recognition_failure,
                     rule_C03_payment_misapplication,
                     rule_C07_escrow_shock_indicator):
            f = rule(doc_name, pages)
            if f:
                all_findings.append(f)

    # rank findings by severity then confidence
    all_findings.sort(key=lambda f: (f.severity, f.confidence), reverse=True)

    return {
        "doc_summaries": doc_summaries,
        "findings": [asdict(f) for f in all_findings]
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fast Mode Lite — VA Mortgage Checker", layout="wide")
st.title("Fast Mode Lite — VA Mortgage COVID Relief Checker")
st.caption("Upload PDFs.  Get quick, evidence-based red flags in minutes.  Not legal advice.")

uploads = st.file_uploader("Upload VA mortgage documents (PDF)", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Progress")
    st.write("1) Reading your documents")
    st.write("2) Building your loan timeline (lite)")
    st.write("3) Checking for common VA & COVID issues (lite)")
    st.write("4) Preparing results")

with col2:
    st.subheader("Run")
    run = st.button("Analyze (Fast Mode Lite)")

if run:
    if not uploads:
        st.error("Please upload at least one PDF.")
        st.stop()

    docs = [(u.name, u.getvalue()) for u in uploads]
    with st.spinner("Analyzing… (Fast Mode Lite)"):
        result = run_fast_mode_lite(docs)

    st.success("Analysis complete.")

    st.subheader("Red Flags (Ranked)")
    if not result["findings"]:
        st.info("No high-signal flags found with the Lite heuristics.  (This does not mean nothing happened.)")
    for f in result["findings"]:
        with st.expander(f'[{f["rule_id"]}] {f["title"]}  —  Severity {f["severity"]}/5  •  Confidence {int(f["confidence"]*100)}%'):
            st.write("**What we saw**")
            st.write(f["what_we_saw"])
            st.write("**Why it matters**")
            st.write(f["why_it_matters"])
            st.write("**Evidence**")
            for ev in f["evidence"]:
                st.write(f'- {ev["doc_name"]}, page {ev["page_number"]}: "{ev["excerpt"]}"')
            st.write("**What to ask next**")
            for q in f["questions"]:
                st.write(f"- {q}")

    st.subheader("Keyword Hits (per document)")
    for ds in result["doc_summaries"]:
        with st.expander(f'{ds["doc_name"]} — {ds["pages"]} pages'):
            if ds["keyword_hits"]:
                for h in ds["keyword_hits"]:
                    st.write(f'Page {h["page"]}: "{h["excerpt"]}"')
            else:
                st.write("No keyword hits detected with current keyword list.  (OCR may be needed if this is a scanned PDF.)")

    st.warning("Note: If your PDFs are scanned images with no selectable text, Lite mode may miss content.  Next step is adding OCR for scanned pages.")
