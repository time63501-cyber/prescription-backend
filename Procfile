web: apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1 2>/dev/null || true && gunicorn run:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
