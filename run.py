# ── MUST be first — installs tesseract before Flask/OCR imports ───
from tesseract_setup import ensure_tesseract
ensure_tesseract()

# ── Now safe to import the app ────────────────────────────────────
from app import create_app
import os

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)