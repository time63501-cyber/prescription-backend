# ── Base image: Python 3.11 on Debian Bullseye ───────────────────
# Using 3.11-slim (not 3.13) for maximum package compatibility
FROM python:3.11-slim

# ── Install system dependencies ───────────────────────────────────
# tesseract-ocr        : the OCR binary pytesseract wraps
# tesseract-ocr-eng    : English language data
# libgl1-mesa-glx      : required by opencv-python-headless
# libglib2.0-0         : required by opencv
# libsm6 libxext6      : required by opencv on headless servers
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Verify tesseract installed correctly (fails build if not) ─────
RUN tesseract --version

# ── Set working directory ─────────────────────────────────────────
WORKDIR /app

# ── Copy and install Python dependencies ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app source ───────────────────────────────────────────────
COPY . .

# ── Expose port ───────────────────────────────────────────────────
EXPOSE 8080

# ── Start gunicorn ────────────────────────────────────────────────
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]