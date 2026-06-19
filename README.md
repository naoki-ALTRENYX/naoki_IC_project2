# IC → A4 Converter

Automate Malaysian IC (MyKad) scanning to print-ready A4 PDF. Upload any PDF, JPG, or PNG — the tool auto-detects, deskews, and places the front and back of the IC at true ISO ID-1 card size (85.6 × 53.98 mm) with rounded corners on a white A4 page, ready to print at 100% actual size.

## Deploy on Railway

1. Push this repo to GitHub
2. Go to railway.app → New Project → Deploy from GitHub repo
3. Select this repo — Railway auto-detects Python and uses the Procfile
4. Done. Your live URL appears in the Railway dashboard.

## Local run

```
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://localhost:8000

## Usage

Upload one or more IC scans (PDF/JPG/PNG). Cards are detected automatically, paired front + back, and placed on A4 pages. Download the PDF and print at 100% / Actual Size.
