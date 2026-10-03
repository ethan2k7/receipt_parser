"""Receipt Parser: upload receipts (image/PDF), extract fields, summarize by vendor.

Run:  streamlit run app.py
Engines:
  - Tesseract (local, free, regex-based parsing)
  - OpenAI GPT-4o vision (set OPENAI_API_KEY; more accurate, sends images to the API)
"""
import base64
import io
import json
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Receipt Parser", layout="wide")

CATEGORY_KEYWORDS = {
    "Groceries": ["safeway", "trader joe", "whole foods", "costco", "walmart", "target", "grocery", "market"],
    "Dining": ["cafe", "coffee", "starbucks", "restaurant", "grill", "pizza", "burger", "bar", "kitchen"],
    "Transport": ["uber", "lyft", "shell", "chevron", "76", "bart", "parking", "gas"],
    "Health": ["cvs", "walgreens", "pharmacy", "clinic"],
    "Shopping": ["amazon", "best buy", "ikea", "home depot", "apple"],
}


# ---------- helpers ----------
def load_pages(uploaded):
    """Return a list of PIL images for an uploaded image or PDF."""
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".pdf"):
        from pdf2image import convert_from_bytes  # needs poppler installed

        return convert_from_bytes(data, dpi=200)
    return [Image.open(io.BytesIO(data)).convert("RGB")]


def guess_category(vendor: str) -> str:
    v = (vendor or "").lower()
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(w in v for w in words):
            return cat
    return "Other"


# Known brands: regex pattern -> canonical name. Searched across the WHOLE receipt text,
# so "survey.walmart.com" still resolves to "Walmart". Add your own stores here.
KNOWN_VENDORS = {
    r"wal[\s-]?mart": "Walmart",
    r"\btarget\b": "Target",
    r"\bcostco\b": "Costco",
    r"\bsafeway\b": "Safeway",
    r"trader\s+joe": "Trader Joe's",
    r"whole\s+foods": "Whole Foods",
    r"\bstarbucks\b": "Starbucks",
    r"\bcvs\b": "CVS",
    r"\bwalgreens\b": "Walgreens",
    r"\bamazon\b": "Amazon",
    r"home\s+depot": "Home Depot",
    r"best\s+buy": "Best Buy",
    r"\bikea\b": "IKEA",
}

# Lines that are never a vendor name (survey blurbs, URLs, addresses, phone numbers...)
JUNK_LINE = re.compile(
    r"feedback|survey|www\.|https?:|\.com|\.net|@|thank|welcome|receipt|"
    r"store\s*#|st#|tel|phone|\d{3}[-.\s]\d{3,4}[-.\s]\d{4}|manager|cashier",
    re.I,
)


def normalize_vendor(raw_vendor, text: str = "", lines=None) -> str:
    """Return a clean vendor name: known brand anywhere in text, else first non-junk line."""
    haystack = f"{raw_vendor or ''}\n{text or ''}"
    best = None  # (position, name): earliest known-brand mention wins
    for pattern, name in KNOWN_VENDORS.items():
        m = re.search(pattern, haystack, re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    if best:
        return best[1]

    candidates = list(lines) if lines else [raw_vendor or ""]
    for l in candidates:
        l = l.strip()
        if len(l) < 3 or JUNK_LINE.search(l) or sum(c.isdigit() for c in l) > len(l) / 2:
            continue
        l = re.sub(r"[#\s]*\d+$", "", l)  # strip trailing store numbers ("SAFEWAY #1234")
        return l.strip(" -*#").title() or "Unknown"
    return "Unknown"


def to_float(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


# ---------- engine 1: Tesseract ----------
def parse_with_tesseract(img: Image.Image) -> dict:
    import pytesseract

    text = pytesseract.image_to_string(img)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    vendor = normalize_vendor(None, text, lines)

    date = None
    m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})", text)
    if m:
        raw = m.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%Y-%m-%d"):
            try:
                date = datetime.strptime(raw, fmt).date().isoformat()
                break
            except ValueError:
                continue

    total = None
    for l in reversed(lines):
        if re.search(r"\btotal\b", l, re.I) and not re.search(r"sub\s*total", l, re.I):
            nums = re.findall(r"\d+[.,]\d{2}", l)
            if nums:
                total = to_float(nums[-1])
                break
    if total is None:  # fall back to largest amount on the receipt
        amounts = [to_float(n) for n in re.findall(r"\d+\.\d{2}", text)]
        total = max([a for a in amounts if a is not None], default=None)

    items = []
    for l in lines[1:]:
        m = re.match(r"(.+?)\s+\$?(\d+\.\d{2})$", l)
        if m and not re.search(r"total|tax|change|cash|visa|mastercard", m.group(1), re.I):
            items.append({"name": m.group(1), "price": float(m.group(2))})

    return {"vendor": vendor, "date": date, "total": total, "items": items, "raw_text": text}


# ---------- engine 2: OpenAI vision ----------
def parse_with_openai(img: Image.Image) -> dict:
    from openai import OpenAI

    client = OpenAI()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        "Extract this receipt as JSON with keys: vendor (string), date (YYYY-MM-DD or null), "
        "total (number), items (list of {name, price}). Return only JSON."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    out = json.loads(resp.choices[0].message.content)
    out["vendor"] = normalize_vendor(out.get("vendor"))
    out["raw_text"] = ""
    return out


# ---------- UI ----------
if "receipts" not in st.session_state:
    st.session_state.receipts = []  # list of dicts, each with an "image" key

st.title("🧾 Receipt Parser")

with st.sidebar:
    st.header("Settings")
    engine = st.radio("OCR engine", ["Tesseract (local)", "GPT-4o vision"])
    if engine.startswith("GPT") and not os.getenv("OPENAI_API_KEY"):
        st.warning("Set OPENAI_API_KEY in your environment first.")
    files = st.file_uploader("Upload receipts", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
    run = st.button("Parse uploaded receipts", type="primary", disabled=not files)
    if st.button("Clear all"):
        st.session_state.receipts = []

if run and files:
    for f in files:
        try:
            for i, page in enumerate(load_pages(f)):
                with st.spinner(f"Parsing {f.name} (page {i + 1})"):
                    parsed = parse_with_tesseract(page) if engine.startswith("Tess") else parse_with_openai(page)
                parsed["file"] = f.name
                parsed["image"] = page
                parsed["category"] = guess_category(parsed.get("vendor", ""))
                st.session_state.receipts.append(parsed)
        except Exception as e:
            st.error(f"{f.name}: {e}")

receipts = st.session_state.receipts
if not receipts:
    st.info("Upload one or more receipts in the sidebar, then click **Parse**.")
    st.stop()

# Editable table (OCR is imperfect, so let the user fix fields)
df = pd.DataFrame([{
    "file": r["file"], "vendor": r.get("vendor"), "date": r.get("date"),
    "total": r.get("total"), "category": r["category"],
} for r in receipts])

st.subheader("Receipts (editable)")
edited = st.data_editor(
    df, num_rows="fixed", use_container_width=True,
    column_config={"category": st.column_config.SelectboxColumn(
        options=list(CATEGORY_KEYWORDS) + ["Other"])},
)
edited["total"] = pd.to_numeric(edited["total"], errors="coerce")

st.subheader("Vendor summary")
summary = (
    edited.assign(vendor=edited["vendor"].fillna("Unknown").str.strip().str.title())
    .groupby(["vendor", "category"], as_index=False)
    .agg(purchases=("total", "size"), total_spent=("total", "sum"))
    .sort_values("total_spent", ascending=False)
)
c1, c2 = st.columns([3, 2])
c1.dataframe(summary, use_container_width=True, hide_index=True,
             column_config={"total_spent": st.column_config.NumberColumn(format="$%.2f")})
c2.bar_chart(summary.set_index("vendor")["total_spent"])
st.metric("Grand total", f"${edited['total'].sum():,.2f}")

st.download_button("Download CSV", edited.to_csv(index=False), "receipts.csv", "text/csv")

st.subheader("Receipt details")
for idx, r in enumerate(receipts):
    with st.expander(f"{r['file']}: {r.get('vendor')} (${r.get('total')})"):
        left, right = st.columns(2)
        left.image(r["image"], use_container_width=True)
        items = pd.DataFrame(r.get("items") or [])
        if items.empty:
            right.write("No line items detected.")
        else:
            right.dataframe(items, hide_index=True, use_container_width=True)
        if r.get("raw_text"):
            right.text_area("Raw OCR text", r["raw_text"], height=150, key=f"raw{idx}")