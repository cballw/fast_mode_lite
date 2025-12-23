# app.py — AstraSys Fast Mode Lite (Auto-baseline + Outcome-based)
# Borrower-protective research tool (NOT legal advice, NOT a compliance determination).
# Key upgrade: tool INFERs pre-mod baseline terms from uploaded docs (no user input required).

import io
import re
import base64
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from pypdf import PdfReader
import fitz  # PyMuPDF

# ============================
# Policy references (context)
# ============================
POLICY_CONTEXT = [
    ("2020-03-27", "CARES Act §4022 — Forbearance for federally-backed mortgages",
     "https://www.congress.gov/116/plaws/publ136/PLAW-116publ136.pdf"),
    ("2020-04-08", "VA Circular 26-20-12 — CARES Act/COVID forbearance guidance for VA loans",
     "https://www.benefits.va.gov/HOMELOANS/documents/circulars/26_20_12.pdf"),
    ("Index", "VA circulars index (official list)",
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
    status: str  # EVIDENCE FOUND | MISSING | CONFLICT | NEEDS CLARIFICATION
    what_we_found: str
    why_it_matters: str
    policy_context: str
    evidence: List[EvidenceRef]
    request_next: List[str]

# ============================
# Helpers
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

def dedupe_evidence(ev_list: List[EvidenceRef]) -> List[EvidenceRef]:
    seen = set()
    out = []
    for e in ev_list:
        key = (e.doc_name, e.page_number, e.search_phrase)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out

def policy_blurb(forb_start: dt.date, forb_end: dt.date) -> str:
    if forb_end >= CARES_EFFECTIVE:
        return ("COVID-era relief (CARES Act §4022 and VA COVID circular guidance) generally aimed to prevent avoidable harm during forbearance "
                "and required clear documentation of loss-mitigation outcomes. This tool flags outcome patterns (payment↑, rate↑, opaque capitalization) "
                "that warrant clarification.")
    return ("This window begins before the CARES Act effective date (3/27/2020). The tool still flags outcome patterns, but early-2020 policy context may differ.")

def parse_rate(text: str) -> Optional[float]:
    # Look for rates like 3.25% / 6.500%
    m = re.search(r"\b([0-9]{1,2}\.[0-9]{2,3})\s*%\b", text or "")
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def find_labeled_amount(text: str, label_patterns: List[str]) -> Optional[float]:
    """
    Finds first amount near any label pattern. Designed for statements/mod docs.
    """
    t = text or ""
    for lp in label_patterns:
        m = re.search(lp + r".{0,80}?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", t, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except:
                pass
    return None

# ============================
# OCR
# ============================
def render_pdf_page_to_png(pdf_bytes: bytes, page_index0: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def google_vision_ocr_image_bytes(img_bytes: bytes, api_key: str) -> str:
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    content_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {"requests": [{"image": {"content": content_b64}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        return data["responses"][0]["fullTextAnnotation"]["text"] or ""
    except Exception:
        return ""

def extract_pages_text_with_ocr(pdf_bytes: bytes, use_ocr: bool, ocr_page_cap: int) -> List[str]:
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
# Evidence extraction
# ============================
def find_hits(pages: List[str], patterns: List[str], max_hits: int = 3) -> List[Tuple[int, str]]:
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    hits = []
    for i, t in enumerate(pages, start=1):
        if not t:
            continue
        if any(r.search(t) for r in regs):
            hits.append((i, short_quote(t, 240)))
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
# Doc classification
# ============================
def classify_doc(pages: List[str]) -> str:
    blob = "\n".join(pages[:10]).lower()
    if any(k in blob for k in ["loan modification", "modification agreement", "change in terms", "effective date of this modification"]):
        return "MODIFICATION"
    if any(k in blob for k in ["payment history", "transaction history", "payment ledger"]):
        return "PAYMENT_HISTORY"
    if "escrow" in blob and "analysis" in blob:
        return "ESCROW"
    if any(k in blob for k in ["amount due", "billing statement", "late charge", "past due", "internet reprint"]):
        return "BILLING"
    return "OTHER"

# ============================
# Baseline & Mod term extraction (auto)
# ============================
def extract_terms_from_pages(pages: List[str]) -> Dict[str, Optional[float]]:
    """
    Extract terms from a document block (best effort).
    Returns: rate, pi, escrow, total
    """
    txt = "\n".join(pages)
    rate = None

    # Prefer labeled "Interest Rate"/"Note Rate" areas
    rate_candidates = []
    for m in re.finditer(r"(interest rate|note rate).{0,60}?([0-9]{1,2}\.[0-9]{2,3})\s*%", txt, re.IGNORECASE):
        try:
            rate_candidates.append(float(m.group(2)))
        except:
            pass
    if rate_candidates:
        rate = rate_candidates[0]
    else:
        rate = parse_rate(txt)

    pi = find_labeled_amount(txt, [r"(principal\s*and\s*interest|P\s*&\s*I|P/I)"])
    escrow = find_labeled_amount(txt, [r"(escrow)"])
    total = find_labeled_amount(txt, [r"(total\s+monthly\s+payment|monthly\s+payment|total\s+payment|amount\s+due)"])

    return {"rate": rate, "pi": pi, "escrow": escrow, "total": total}

def infer_baseline_from_non_mod_docs(docs_text: List[Tuple[str, str, List[str]]]) -> Tuple[Dict[str, Optional[float]], List[EvidenceRef]]:
    """
    Look across BILLING / PAYMENT_HISTORY / ESCROW docs for pre-mod baseline terms.
    We take the first strong labeled hits we can find.
    """
    baseline = {"rate": None, "pi": None, "escrow": None, "total": None}
    ev: List[EvidenceRef] = []

    # Search order: BILLING then PAYMENT_HISTORY then OTHER
    preferred_types = ["BILLING", "PAYMENT_HISTORY", "ESCROW", "OTHER"]
    for t in preferred_types:
        for doc_name, doc_type, pages in docs_text:
            if doc_type != t:
                continue

            # Look for highly specific labeled lines first
            hits = find_hits(pages, [r"principal\s*and\s*interest", r"\bP\s*&\s*I\b", r"\bescrow\b", r"total\s+monthly\s+payment", r"interest rate", r"note rate"], max_hits=2)
            if hits:
                ev.extend(evidence_from_hits(doc_name, hits))

            terms = extract_terms_from_pages(pages)

            for k in ["rate", "pi", "escrow", "total"]:
                if baseline[k] is None and terms[k] is not None:
                    baseline[k] = terms[k]

            # If we have enough baseline info, stop early
            if baseline["pi"] is not None and baseline["total"] is not None and baseline["rate"] is not None:
                return baseline, dedupe_evidence(ev)

    return baseline, dedupe_evidence(ev)

def infer_mod_terms_from_mod_docs(docs_text: List[Tuple[str, str, List[str]]]) -> Tuple[Dict[str, Optional[float]], List[EvidenceRef], List[EvidenceRef]]:
    """
    Extract modification terms from MODIFICATION docs. Also capture arrears/capitalization evidence.
    Returns: new_terms, term_evidence, cap_evidence
    """
    new_terms = {"rate": None, "pi": None, "escrow": None, "total": None}
    term_ev: List[EvidenceRef] = []
    cap_ev: List[EvidenceRef] = []
    found = False

    for doc_name, doc_type, pages in docs_text:
        if doc_type != "MODIFICATION":
            continue
        found = True
        term_hits = find_hits(pages, [r"interest rate", r"principal\s*and\s*interest", r"\bescrow\b", r"total\s+monthly\s+payment", r"monthly\s+payment", r"maturity", r"term"], max_hits=3)
        term_ev.extend(evidence_from_hits(doc_name, term_hits))

        cap_hits = find_hits(pages, [r"arrears", r"capitaliz", r"principal balance", r"deferred", r"past due amount"], max_hits=2)
        cap_ev.extend(evidence_from_hits(doc_name, cap_hits))

        terms = extract_terms_from_pages(pages)
        for k in ["rate", "pi", "escrow", "total"]:
            if new_terms[k] is None and terms[k] is not None:
                new_terms[k] = terms[k]

        # stop early if we got enough
        if new_terms["pi"] is not None and new_terms["total"] is not None and new_terms["rate"] is not None:
            break

    return new_terms, dedupe_evidence(term_ev), dedupe_evidence(cap_ev)

# ============================
# Scorecard + Outcome rules
# ============================
def build_scorecard(docs_text: List[Tuple[str, str, List[str]]], forb_start: dt.date, forb_end: dt.date) -> Tuple[List[ScoreItem], str]:
    policy_note = policy_blurb(forb_start, forb_end)

    # Evidence buckets
    forb_ev, delin_ev, suspense_ev, va_ev = [], [], [], []

    for doc_name, doc_type, pages in docs_text:
        forb_ev += evidence_from_hits(doc_name, find_hits(pages, [r"\bforbear", r"\bcovid", r"\bcares act\b", r"\bpayment pause\b"], max_hits=2))
        delin_ev += evidence_from_hits(doc_name, find_hits(pages, [r"\blate fee\b", r"\blate charge\b", r"\bpast due\b", r"\bdelinquen"], max_hits=2))
        suspense_ev += evidence_from_hits(doc_name, find_hits(pages, [r"\bsuspense\b"], max_hits=2))
        va_ev += evidence_from_hits(doc_name, find_hits(pages, [r"\bVALERI\b", r"\bLoan Guaranty\b", r"\bVeterans Affairs\b", r"\bVA\b"], max_hits=2))

    forb_ev = dedupe_evidence(forb_ev)
    delin_ev = dedupe_evidence(delin_ev)
    suspense_ev = dedupe_evidence(suspense_ev)
    va_ev = dedupe_evidence(va_ev)

    baseline, baseline_ev = infer_baseline_from_non_mod_docs(docs_text)
    new_terms, mod_terms_ev, cap_ev = infer_mod_terms_from_mod_docs(docs_text)

    # Build items
    items: List[ScoreItem] = []

    # S1: Forbearance evidence
    if forb_ev:
        items.append(ScoreItem(
            id="S1",
            label="COVID forbearance/relief is documented",
            status="EVIDENCE FOUND",
            what_we_found="We found COVID/forbearance-related language in your documents.",
            why_it_matters="Relief must be documented to evaluate protected treatment and downstream actions.",
            policy_context=policy_note,
            evidence=forb_ev[:4],
            request_next=[]
        ))
    else:
        items.append(ScoreItem(
            id="S1",
            label="COVID forbearance/relief is documented",
            status="MISSING",
            what_we_found="We did not find clear COVID forbearance documentation in the text we could read.",
            why_it_matters="Without the forbearance plan record (start/end dates), it is harder to prove protected status and reconcile servicing.",
            policy_context=policy_note,
            evidence=[],
            request_next=[
                "Request the complete COVID forbearance plan record and system notes (start/end dates).",
                "Request any forbearance approval/confirmation letters or portal messages."
            ]
        ))

    # S2: Protected treatment indicator
    if delin_ev and forb_ev:
        items.append(ScoreItem(
            id="S2",
            label="Protected treatment during forbearance (no delinquency/late fees during active relief)",
            status="CONFLICT",
            what_we_found="We found delinquency/late-fee language and forbearance/COVID language in the documents.",
            why_it_matters="In a COVID relief context, delinquency/fees during active forbearance should be explainable with the plan record and a transaction ledger.",
            policy_context="CARES/VA relief intent: avoid avoidable harm during forbearance; exceptions should be clearly documented and explainable.",
            evidence=(delin_ev[:2] + forb_ev[:2])[:4],
            request_next=[
                "Provide a transaction-level ledger for the forbearance months showing exact payment application and fee assessment.",
                "Explain any delinquency coding and late fees during active forbearance (cite dates).",
            ]
        ))
    elif delin_ev and not forb_ev:
        items.append(ScoreItem(
            id="S2",
            label="Protected treatment during forbearance (no delinquency/late fees during active relief)",
            status="NEEDS CLARIFICATION",
            what_we_found="We found delinquency/late-fee language, but did not find a clear forbearance record in the text we could read.",
            why_it_matters="We need the forbearance plan record to confirm whether the loan should have been treated as protected at those times.",
            policy_context="Protected-status evaluation requires both forbearance start/end dates and dated servicing events.",
            evidence=delin_ev[:3],
            request_next=[
                "Provide the complete forbearance plan record and system notes (start/end dates).",
                "Provide a transaction-level payment ledger covering the same months."
            ]
        ))
    else:
        items.append(ScoreItem(
            id="S2",
            label="Protected treatment during forbearance (no delinquency/late fees during active relief)",
            status="NEEDS CLARIFICATION",
            what_we_found="We did not find delinquency/late-fee language in the text we could read.",
            why_it_matters="This does not prove it didn’t occur. The servicing ledger is the best source of truth.",
            policy_context="If you suspect delinquency/fees occurred, upload monthly statements for that period or the full payment history/ledger.",
            evidence=[],
            request_next=[]
        ))

    # S3: Suspense/payment misapplication
    if suspense_ev:
        items.append(ScoreItem(
            id="S3",
            label="Payments applied correctly (no suspense trap)",
            status="CONFLICT",
            what_we_found="We found 'suspense' references in the documents.",
            why_it_matters="Suspense can indicate payments were held or applied in a non-standard way, creating phantom delinquency.",
            policy_context="During COVID relief, misapplication can create compounding harm. A transaction ledger reconciliation is critical.",
            evidence=suspense_ev[:4],
            request_next=[
                "Provide a transaction-level payment ledger showing application to principal/interest/escrow/suspense.",
                "Provide a suspense reconciliation showing when/why it was created and when it was cleared."
            ]
        ))
    else:
        items.append(ScoreItem(
            id="S3",
            label="Payments applied correctly (no suspense trap)",
            status="NEEDS CLARIFICATION",
            what_we_found="We did not detect suspense language in the text we could read.",
            why_it_matters="This does not prove there was no suspense. Payment history/ledger is the best document to confirm.",
            policy_context="Payment application checks require a transaction ledger for full confidence.",
            evidence=[],
            request_next=[]
        ))

    # S4: Baseline (pre-mod) terms inferred
    if any(v is not None for v in baseline.values()):
        items.append(ScoreItem(
            id="S4",
            label="Pre-mod baseline terms inferred from your documents",
            status="EVIDENCE FOUND",
            what_we_found=f"Inferred baseline (best effort): rate={baseline['rate']}%, P&I={baseline['pi']}, escrow={baseline['escrow']}, total={baseline['total']}.",
            why_it_matters="Outcome checks (payment↑/rate↑) are strongest when we can compare pre-mod vs post-mod terms from your own documents.",
            policy_context="If baseline terms are wrong/missing, upload a pre-mod statement that clearly shows P&I, escrow, total payment, and rate.",
            evidence=baseline_ev[:4],
            request_next=[]
        ))
    else:
        items.append(ScoreItem(
            id="S4",
            label="Pre-mod baseline terms inferred from your documents",
            status="MISSING",
            what_we_found="We could not infer baseline terms (pre-mod payment and rate) from the text we read.",
            why_it_matters="Without baseline terms, the tool cannot reliably prove payment↑ or rate↑ due to the modification.",
            policy_context="Upload a pre-mod statement or note disclosure that shows your prior rate and payment breakdown.",
            evidence=[],
            request_next=[
                "Upload a pre-mod monthly statement showing P&I, escrow, and total payment.",
                "Upload any note/closing disclosure page showing the original interest rate (or a statement that shows it)."
            ]
        ))

    # S5: Modification terms inferred
    if any(v is not None for v in new_terms.values()) or mod_terms_ev:
        items.append(ScoreItem(
            id="S5",
            label="Modification terms detected from your documents",
            status="EVIDENCE FOUND",
            what_we_found=f"Inferred modification terms (best effort): rate={new_terms['rate']}%, P&I={new_terms['pi']}, escrow={new_terms['escrow']}, total={new_terms['total']}.",
            why_it_matters="We use these terms to run outcome checks (payment↑ / rate↑) in a borrower-protective way.",
            policy_context="If any of these terms look wrong, increase OCR pages or upload the page where new terms are listed clearly.",
            evidence=(mod_terms_ev[:3] + cap_ev[:1])[:4],
            request_next=[]
        ))
    else:
        items.append(ScoreItem(
            id="S5",
            label="Modification terms detected from your documents",
            status="MISSING",
            what_we_found="We did not reliably extract modification terms (rate/payment) from the text we could read.",
            why_it_matters="Without the post-mod terms, we cannot evaluate whether the modification worsened payment or rate.",
            policy_context="Upload the signed modification agreement pages that list the new rate, P&I, escrow, total payment, and term/maturity.",
            evidence=[],
            request_next=[
                "Upload the page(s) of the modification agreement showing the new interest rate and the new payment breakdown.",
                "Increase OCR pages per document if the modification PDF is scanned."
            ]
        ))

    # S6: Outcome-based modification check (the key)
    # Borrower-protective: payment↑ or rate↑ => CONFLICT unless escrow-only clearly shown
    outcome_evidence = (mod_terms_ev + baseline_ev + cap_ev)[:6]
    baseline_rate, baseline_pi, baseline_total = baseline.get("rate"), baseline.get("pi"), baseline.get("total")
    new_rate, new_pi, new_total = new_terms.get("rate"), new_terms.get("pi"), new_terms.get("total")

    if (baseline_rate is None and baseline_pi is None and baseline_total is None) or (new_rate is None and new_pi is None and new_total is None):
        items.append(ScoreItem(
            id="S6",
            label="Modification outcome check (payment↑ / rate↑ / opaque capitalization)",
            status="NEEDS CLARIFICATION",
            what_we_found="We do not have enough extracted baseline and/or modification term data to run a reliable outcome comparison.",
            why_it_matters="A borrower-protective review treats payment↑ or rate↑ during COVID relief as a conflict unless the documents clearly show an escrow-only increase and include itemized calculations.",
            policy_context="Outcome-based rule: payment↑ and/or rate↑ during COVID loss mitigation = conflict unless explicitly justified and documented.",
            evidence=outcome_evidence,
            request_next=[
                "Provide a pre-mod and post-mod statement showing P&I vs escrow vs total payment.",
                "Provide the page listing the new interest rate in the modification agreement.",
                "Provide the arrears/capitalization reconciliation (itemized breakdown)."
            ]
        ))
    else:
        conflicts = []
        needs = []

        # Rate increase
        if baseline_rate is not None and new_rate is not None and new_rate > baseline_rate + 1e-6:
            conflicts.append(f"Interest rate increased ({baseline_rate}% → {new_rate}%).")

        # P&I increase (strongest signal)
        if baseline_pi is not None and new_pi is not None:
            if new_pi > baseline_pi + 0.01:
                conflicts.append(f"P&I increased (${baseline_pi:.2f} → ${new_pi:.2f}).")

        # Total payment increase (weaker unless we can separate P&I)
        if baseline_total is not None and new_total is not None and new_total > baseline_total + 0.01:
            # If we *do* have P&I and it did not increase, treat as escrow-driven possibility
            if baseline_pi is not None and new_pi is not None and new_pi <= baseline_pi + 0.01:
                needs.append(f"Total payment increased (${baseline_total:.2f} → ${new_total:.2f}) but P&I appears unchanged (possible escrow-driven). Needs escrow itemization.")
            elif baseline_pi is None or new_pi is None:
                needs.append(f"Total payment increased (${baseline_total:.2f} → ${new_total:.2f}) but P&I breakdown is unclear. Treat as conflict unless escrow-only is documented.")

        # Capitalization/arrears transparency
        if cap_ev:
            # If we see arrears/capitalization signals but no obvious itemization language, require reconciliation
            needs.append("Arrears/capitalization appears referenced. Needs itemized arrears/capitalization reconciliation (what was added and why).")

        if conflicts:
            status = "CONFLICT"
            found = " / ".join(conflicts)
            why = ("Borrower-protective rule: payment↑ and/or rate↑ during COVID loss mitigation is treated as a conflict unless the documents explicitly justify it "
                   "and show clear itemized calculations (including escrow-only exceptions).")
        elif needs:
            status = "NEEDS CLARIFICATION"
            found = " / ".join(needs)
            why = ("We need clearer breakdowns (P&I vs escrow) and arrears itemization to determine whether changes were allowable/justified under VA COVID relief framework.")
        else:
            status = "EVIDENCE FOUND"
            found = "No obvious payment↑ or rate↑ detected from extracted fields (limited to what we could read)."
            why = ("This does NOT mean 'compliant.' It means no obvious outcome conflict was detected from extracted terms. Full review still requires complete term pages, "
                   "arrears itemization, and dated servicing history.")

        req = []
        if status in ("CONFLICT", "NEEDS CLARIFICATION"):
            req = [
                "Provide the pre-mod vs post-mod payment comparison used at the time (P&I vs escrow).",
                "Provide the arrears/capitalization reconciliation showing what amounts were rolled in and why.",
                "Provide any option evaluation/waterfall showing what alternatives were considered and why the final option was selected.",
            ]

        items.append(ScoreItem(
            id="S6",
            label="Modification outcome check (payment↑ / rate↑ / opaque capitalization)",
            status=status,
            what_we_found=found,
            why_it_matters=why,
            policy_context="Outcome-based research rule: payment↑ and/or rate↑ during COVID relief = conflict unless escrow-only and explicitly documented with itemized records.",
            evidence=outcome_evidence,
            request_next=req
        ))

    # S7: VA coordination evidence
    if va_ev:
        items.append(ScoreItem(
            id="S7",
            label="Evidence of VA coordination/notification (where applicable)",
            status="EVIDENCE FOUND",
            what_we_found="We found VA/Loan Guaranty/VALERI-related references in the uploaded text.",
            why_it_matters="VA coordination evidence helps confirm the official timeline and whether VA processes were followed.",
            policy_context="Absence does not prove non-notification, but it is a research gap. FOIA records can provide VA internal timelines.",
            evidence=va_ev[:4],
            request_next=[]
        ))
    else:
        items.append(ScoreItem(
            id="S7",
            label="Evidence of VA coordination/notification (where applicable)",
            status="MISSING",
            what_we_found="We did not find clear VA coordination references in the text we read.",
            why_it_matters="This does not prove VA was not notified; it means your uploaded documents don’t show it, which matters for research and escalation.",
            policy_context="Borrower research: request VA case identifiers or proof of notifications; FOIA records can provide VA internal timelines.",
            evidence=[],
            request_next=[
                "Request any VA case identifiers or confirmation of VA notifications (dates/methods) tied to forbearance and loss mitigation.",
                "If you have FOIA records, Deep Mode later can analyze VA internal timelines."
            ]
        ))

    return items, policy_note

# ============================
# Clarification letter
# ============================
def build_clarification_letter(borrower_name: str, loan_number: str, property_addr: str,
                               forb_start: dt.date, forb_end: dt.date, items: List[ScoreItem]) -> str:
    reqs = []
    for it in items:
        if it.status != "EVIDENCE FOUND":
            reqs.extend(it.request_next)

    seen = set()
    deduped = []
    for r in reqs:
        if r and r not in seen:
            seen.add(r)
            deduped.append(r)

    bullets = "\n".join([f"- {r}" for r in deduped]) if deduped else "- Please provide the complete servicing file and forbearance records for my loan."

    return f"""Subject: Request for clarification and records regarding COVID forbearance and loan modification outcomes
Loan Number: {loan_number or "[loan number]"}
Property Address: {property_addr or "[property address]"}
Date: {dt.date.today().isoformat()}

Dear Servicing Team,

I am requesting clarification and records regarding the handling of my COVID-19 forbearance and related loss-mitigation activity on my VA-backed mortgage.

My COVID forbearance window is approximately {forb_start.isoformat()} through {forb_end.isoformat()}.

To reconcile the servicing record and evaluate whether outcomes were properly documented and justified, please provide:

{bullets}

Thank you for your attention. I look forward to your response.

Sincerely,
{borrower_name or "[Borrower Name]"}
"""

# ============================
# UI
# ============================
st.set_page_config(page_title="AstraSys — VA COVID Relief Outcome Checker (Fast Mode Lite)", layout="wide")
st.title("AstraSys — VA COVID Relief Outcome Checker (Fast Mode Lite)")
st.caption("This tool does NOT declare compliance.  It flags conflicts (payment↑/rate↑/opaque capitalization) and missing evidence for borrower research.")

st.subheader("Your COVID forbearance window")
forb_start = st.date_input("Forbearance start", value=dt.date(2020, 1, 1))
forb_end = st.date_input("Forbearance end", value=dt.date(2025, 7, 31))

with st.expander("Policy references (for your research)"):
    for d, t, u in POLICY_CONTEXT:
        st.markdown(f"- **{d}** — {t}  \n  {u}")

st.subheader("OCR settings (for scanned PDFs)")
use_ocr = st.toggle("Use OCR when a page has no readable text", value=True)
ocr_page_cap = st.slider("Max OCR pages per document (controls cost)", 5, 80, 30, 5)
if use_ocr and not st.secrets.get("GOOGLE_VISION_API_KEY", ""):
    st.warning("OCR is ON but GOOGLE_VISION_API_KEY is not set.  Add it in Manage app → Settings → Secrets.")

uploads = st.file_uploader("Upload mortgage documents (PDF)", type=["pdf"], accept_multiple_files=True)

with st.expander("Optional: for letter download"):
    borrower_name = st.text_input("Borrower name", value="")
    loan_number = st.text_input("Loan number", value="")
    property_addr = st.text_input("Property address", value="")

if st.button("Run outcome-based scorecard"):
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

    items, policy_note = build_scorecard(docs_text, forb_start, forb_end)

    # Executive Summary
    st.subheader("Executive summary")
    conflicts = [it for it in items if it.status == "CONFLICT"]
    missing = [it for it in items if it.status == "MISSING"]
    needs = [it for it in items if it.status == "NEEDS CLARIFICATION"]

    if conflicts:
        st.error("Result: Conflict(s) detected — likely mishandling or outcome contradictions")
        st.write("This means your documents show patterns like payment↑, rate↑, delinquency during forbearance, or suspense/opaque capitalization that require explanation with itemized records.")
    elif missing or needs:
        st.warning("Result: Incomplete evidence — needs records to evaluate outcomes")
        st.write("This means the tool cannot fully evaluate outcomes from the text we could read. The scorecard tells you exactly what to request.")
    else:
        st.info("Result: No obvious outcome conflicts detected from extracted fields (still NOT a compliance finding)")
        st.write("This does not prove compliance. It only means we didn’t detect obvious payment↑/rate↑ conflicts from extracted text.")

    # Scorecard
    st.subheader("Outcome-based scorecard")
    for it in items:
        icon = "✅" if it.status == "EVIDENCE FOUND" else "❌" if it.status == "CONFLICT" else "⚠️" if it.status == "NEEDS CLARIFICATION" else "❓"
        st.markdown(f"### {icon} {it.label} — **{it.status}**")
        st.write(f"**What we found:** {it.what_we_found}")
        st.write(f"**Why it matters:** {it.why_it_matters}")
        st.write(f"**Policy context:** {it.policy_context}")

        if it.evidence:
            st.write("**Where to look (exact locator cues)**")
            for ev in it.evidence[:5]:
                st.write(f"- **{ev.doc_name} — Page {ev.page_number}**")
                if ev.search_phrase:
                    st.write(f"  **Search:** `{ev.search_phrase}`")
                st.write(f"  **Quote:** “{ev.quote}”")

        if it.request_next:
            st.write("**What to request next**")
            st.code("\n".join([f"- {r}" for r in it.request_next]), language="markdown")

        st.divider()

    # Action pack
    st.subheader("Action Pack")
    letter = build_clarification_letter(borrower_name, loan_number, property_addr, forb_start, forb_end, items)
    st.download_button(
        "Download clarification letter (TXT)",
        data=letter.encode("utf-8"),
        file_name="va_covid_outcome_clarification_letter.txt",
        mime="text/plain",
    )
    st.write("Tip: Use the locator cues (file + page + search phrase) to verify each evidence point inside your own PDFs.")
