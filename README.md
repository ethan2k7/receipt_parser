# Receipt Parser

Local Streamlit dashboard:
- upload receipt images/PDF
- extract vendor, date, total and line items
- see spend and purchase counts per vendor
- graphs spending totals

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
- Tesseract engine is fully local; parsing uses regex heuristics to locate information from certain syntax (vendor = first line, total = "Total" line)
- GPT-4o engine is more accurate on more messy receipts but sends images to OpenAI, requiring tokens
 
- Vendor categories come from keyword matching in `CATEGORY_KEYWORDS`; can be edited to account for new companies
