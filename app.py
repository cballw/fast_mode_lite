import io
import re
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

# ----------------------------
# Borrower-friendly labels
# ----------------------------
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

def compact_evidence(ev: EvidenceRef) -> str:
    return f'{ev.doc_name} — p. {ev.page_number}: “{normalize_excerpt(ev.excerpt)}”'

def questions_block(questions: List[str]) -> str:
    return "\n".join([f"- {q}" for q in questions])

# ----------------------------
# Fast Mode Lite Rules (heuristics)
# ----------------------------
def rule_C01_relief_recognition_failure(doc_name: str, pages: List[str]) -> Optional[Finding]:
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
        ],
    )

def rule_C03_payment_misapplication(doc_name: str, pages: List[str]) -> Optional[Finding]:
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
    )

def rule_ESCROW_indicator(doc_name: str, pages: List[str]) -> Optional[Finding]:
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
    )

def run_fast_mode_lite(docs: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    findings: List[Finding] = []
    doc_summaries = []

    for doc_name, b in docs:
        pages = pdf_extract_pages(b)

        # keyword hits summary (for debugging + transparency)
        hits = []
        for i, t in enumerate(pages, start=1):
            low = (t or "").lower()
            if any(k in low for k in KEYWORDS):
                # keep it short
                hits.append({"page": i, "excerpt": normalize_excerpt(t)})
                if len(hits) >= 6:
                    break

        doc_summaries.append({"doc_name": doc_name, "pages": len(pages), "keyword_hits": hits})

        for rule in (rule_C01_relief_recognition_failure, rule_C03_payment_misapplication, rule_ESCROW_indicator):
            f = rule(doc_name, pages)
            if f:
                findings.append(f)

    findings.sort(key=lambda f: (f.severity, f.confidence), reverse=True)

    return {"doc_summaries": doc_summaries, "findings": [asdict(f) for f in findings]}

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Fast Mode Lite — VA Mortgage Checker", layout="wide")
st.title("Fast Mode Lite — VA Mortgage Checker")
st.caption("Minimal, evidence-based flags.  Not legal advice.")

uploads = st.file_uploader("Upload VA mortgage documents (PDF)", type=["pdf"], accept_multiple_files=True)

with st.expander("What this checks (Fast Mode Lite)"):
    st.write("- COVID/forbearance + delinquency/late-fee mismatch")
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
        result = run_fast_mode_lite(docs)

    st.success("Done.")

    st.subheader("Red Flags (simple)")
    if not result["findings"]:
        st.info("No high-signal flags found with Lite checks.  This does not mean nothing happened.")
    else:
        for f in result["findings"]:
            st.markdown(f"### {f['title']}")
            st.caption(badge_text(f["severity"], f["confidence"]))

            # minimal explanation
            st.write(f["what_we_saw"])
            st.write(f"**Why it matters:** {f['why_it_matters']}")

            st.write("**Where to look**")
            for ev in f["evidence"][:2]:
                st.write("- " + compact_evidence(ev))

            # copy questions button
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
