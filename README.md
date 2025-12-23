# Fast Mode Lite (DIY Prototype)

## What this is
A simple Streamlit app that:
- accepts PDF uploads
- extracts selectable text
- runs Lite heuristics for COVID forbearance anomalies
- shows ranked findings with evidence (doc + page excerpts)

## Run locally
1) Install Python 3.10+
2) In a terminal:
   pip install -r requirements.txt
   streamlit run app.py

## Deploy options
- Streamlit Community Cloud (easiest)
- Render / Railway (container)
- Hugging Face Spaces (Gradio alt)

## Next upgrades
- Add OCR for scanned PDFs (Google Vision / Azure OCR / Textract)
- Better extractors for payment ledgers and escrow tables
- PDF exports (report + letter)
