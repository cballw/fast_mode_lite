# app.py — Fast Mode Lite (Stable)
# VA Mortgage Checker with CARES Act + VA Circular policy context
# Minimal UX, borrower-facing, Streamlit-safe

import io
import re
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Optional

import streamlit as st
from pypdf import PdfReader

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
KEYWORDS = [
    "covid", "forbear", "cares", "late fee", "delinquen",
    "past due", "suspense", "escrow", "modification"
]

def normalize_excerpt(text: str, limit: int = 160) -> str:
    t = " ".join((text or "").split())
    return t[:limit] + ("…" if len(t) > limit else "")

def extract_pages(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [(p.extract_text() or "").strip() for p in reader.pages]

def impact_label(sev: int) -> str:
    return "High" if sev >= 4 else "Medium" if sev == 3 else "Low"

def confidence_label(c: float) -> str:
    return "Strong" if c >= 0.75 else "Moderate" if c >= 0.60 else "Needs more evidence"

def badge(sev: int, conf: float) -> str:
    return f"Impact: {impact_label(sev)} • How sure: {confidence_label(conf)}"

def compact_evidence(ev) -> str:
    doc = ev["doc_name"] if isinstance(ev, dict) else ev.doc_name
    page = ev["page_number"] if isinstance(ev, dict) else ev.page_number
    excerpt = ev["excerpt"] if isinstance(ev, dict) else ev.excerpt
    return f"{doc} — p. {page}: “{normalize_excerpt(excerpt)}”"

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

# ============================
# Fast Mode Lite rules
# ============================
def rule_C01(doc_name, pages, start, end):
    covid, late = [], []
    for i, t in enumerate(pages, start=1):
        low = t.lower()
        if any(k in low for k in ["covid", "forbear", "cares"]):
            covid.append((i, t))
        if any(k in low for k in ["late fee", "delinquen", "past due"]):
            late.append((i, t))
    if not covid or not late:
        return None
    return Finding(
        rule_id="C-01",
        severity=4,
        confidence=0.65,
        title="COVID relief may not have protected the loan like it should",
        what_we_saw="Your documents mention COVID/forbearance and also show late-fee or delinquency language.",
        why_it_matters="Delinquency or fees during forbearance can cause compounding harm and downstream actions.",
        evidence=[
            EvidenceRef(doc_name, covid[0][0], normalize_excerpt(covid[0][1])),
            EvidenceRef(doc_name, late[0][0], normalize_excerpt(late[0][1])),
        ],
        questions=[
            "Provide the complete COVID forbearance plan record and system notes.",
            "Explain any late fees or delinquency coding during forbearance.",
            "Provide a transaction-level payment ledger for the same period.",
            "Provide records showing when/how VA was notified of forbearance.",
        ],
        policy_context=policy_context_text(start, end),
    )

def rule_suspense(doc_name, pages):
    for i, t in enumerate(pages, start=1):
        if "suspense" in t.lower():
            return Finding(
                rule_id="C-03",
                severity=3,
                confidence=0.60,
                title="Payments may have been routed to suspense",
                what_we_saw="We found references to payments being held in suspense.",
                why_it_matters="Suspense can create phantom delinquency and fee cascades.",
                evidence=[EvidenceRef(doc_name, i, normalize_excerpt(t))],
                questions=[
                    "Provide a transaction-level ledger showing how payments were applied.",
                    "Explain why payments entered suspense and when they were cleared.",
                ],
            )
    return None

# ============================
# Run analysis
# ============================
def analyze(docs, start, end):
    findings = []
    for name, data in docs:
        pages = extract_pages(data)
        f1 = rule_C01(name, pages, start, end)
        if f1:
            findings.append(f1)
        f2 = rule_suspense(name, pages)
        if f2:
            findings.append(f2)
    findings.sort(key=lambda f: (f.severity, f.confidence), reverse=True)
    return [asdict(f) for f in findings]

# ============================
# UI
# ============================
st.set_page_config(page_title="Fast Mode Lite — VA Mortgage Checker", layout="wide")
st.title("Fast Mode Lite — VA Mortgage Checker")
st.caption("Minimal, evidence-based flags with CARES Act & VA policy context. Not legal advice.")

st.subheader("Your COVID forbearance window")
start = st.date_input("Forbearance start", value=dt.date(2020, 1, 1))
end = st.date_input("Forbearance end", value=dt.date(2025, 7, 31))

with st.expander("Policy context we compare against"):
    for d, t, u in POLICY_TIMELINE:
        st.markdown(f"- **{d}** — {t}  \n  {u}")

uploads = st.file_uploader("Upload VA mortgage documents (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze"):
    if not uploads:
        st.error("Please upload at least one PDF.")
        st.stop()

    docs = [(u.name, u.getvalue()) for u in uploads]

    with st.spinner("Analyzing…"):
        results = analyze(docs, start, end)

    st.success("Done.")

    st.subheader("Red Flags")
    if not results:
        st.info("No high-signal flags found with Lite checks. This does not mean nothing happened.")
    else:
        for idx, f in enumerate(results):
            st.markdown(f"### {f['title']}")
            st.caption(badge(f["severity"], f["confidence"]))
            st.write(f["what_we_saw"])
            if f.get("policy_context"):
                st.write(f"**Policy context:** {f['policy_context']}")
            st.write(f"**Why it matters:** {f['why_it_matters']}")
            st.write("**Where to look**")
            for ev in f["evidence"]:
                st.write("- " + compact_evidence(ev))
            st.code("\n".join(f"- {q}" for q in f["questions"]), language="markdown")
            st.button(
                f"Copy questions",
                key=f"copy_{idx}",
                on_click=lambda text="\n".join(f"- {q}" for q in f["questions"]): st.session_state.update({"copied": text}),
            )
            st.divider()
