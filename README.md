# Receipt Parser

Local Streamlit dashboard: upload receipt images/PDFs, extract vendor, date, total and line items, then see spend and purchase counts per vendor.

## Setup

```bash
 pip install streamlit pandas pillow pytesseract pdf2image openai
# System dependencies
#   macOS:  brew install tesseract poppler
#   Ubuntu: sudo apt install tesseract-ocr poppler-utils
export OPENAI_API_KEY=sk-...   # optional, only for the GPT-4o engine
streamlit run app.py
```

## Notes
- Tesseract engine is fully local; parsing uses regex heuristics (vendor = first line, total = "Total" line).
- GPT-4o engine is more accurate on messy receipts but sends images to OpenAI.
- Table cells are editable, so you can fix OCR mistakes and the vendor summary updates.
- Vendor categories come from keyword matching in `CATEGORY_KEYWORDS`; extend it as needed.
