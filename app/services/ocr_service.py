"""
ocr_service.py
--------------
Tested preprocessing pipeline for handwritten doctor prescriptions.

Key findings from empirical testing:
  - Shadow removal / heavy adaptive thresholding HURTS quality on this type of image
  - Best results: plain upscale (2.5x) → Otsu threshold
  - Second best:  upscale → fastNlMeansDenoising → Otsu
  - PSM 6 outperforms PSM 4 for prescription layouts
  - Multiple passes merged by confidence = best overall recall
"""

import cv2
import numpy as np
import pytesseract
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Railway/Linux: tesseract is in PATH, no path config needed.
# ── For local Windows dev only, uncomment:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCRService:

    # ──────────────────────────────────────────────
    # Preprocessing strategies  (empirically ranked)
    # ──────────────────────────────────────────────

    def _upscale(self, gray):
        return cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    def _strategy_otsu(self, bgr):
        """Upscale → Otsu — best overall (avg_conf ~80)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        up   = self._upscale(gray)
        _, thresh = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _strategy_denoise_otsu(self, bgr):
        """Upscale → fastNlMeans denoise → Otsu — good for noisy scans (avg_conf ~79)."""
        gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        up       = self._upscale(gray)
        denoised = cv2.fastNlMeansDenoising(up, h=10)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _strategy_plain(self, bgr):
        """Plain upscale, no threshold — good for printed text (avg_conf ~76)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return self._upscale(gray)

    def _strategy_sharpen(self, bgr):
        """Upscale → unsharp mask — helps with slightly blurry phone photos."""
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        up     = self._upscale(gray)
        blur   = cv2.GaussianBlur(up, (0, 0), 3)
        sharp  = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _get_strategies(self, image_path):
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Cannot read image at path: {image_path}")
        return [
            ("otsu",          self._strategy_otsu(bgr)),
            ("denoise_otsu",  self._strategy_denoise_otsu(bgr)),
            ("plain",         self._strategy_plain(bgr)),
            ("sharpen_otsu",  self._strategy_sharpen(bgr)),
        ]

    # ──────────────────────────────────────────────
    # Tesseract runner
    # ──────────────────────────────────────────────

    def _tesseract(self, image, psm):
        config = f"--oem 3 --psm {psm} -l eng"
        return pytesseract.image_to_data(
            image, config=config,
            output_type=pytesseract.Output.DICT
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def extract_tokens(self, image_path):
        """
        Run 4 preprocessing strategies × PSM modes 6 and 4.
        Returns deduplicated list of (token_str, avg_confidence) sorted by confidence desc.
        Minimum confidence threshold: 20.
        """
        strategies = self._get_strategies(image_path)
        # token_lower → list of (original_case_text, confidence)
        token_map = {}

        errors = []
        for name, processed in strategies:
            for psm in [6, 4]:
                try:
                    data = self._tesseract(processed, psm)
                    for raw, raw_conf in zip(data["text"], data["conf"]):
                        text = raw.strip()
                        if not text:
                            continue
                        # Tesseract returns -1 for block/line-level rows — skip those
                        try:
                            conf = int(raw_conf)
                        except (ValueError, TypeError):
                            continue
                        if conf < 20:
                            continue
                        token_map.setdefault(text.lower(), []).append((text, conf))
                except Exception as e:
                    errors.append(f"{name}/psm{psm}: {e}")

        if errors:
            print(f"[OCR] Non-fatal strategy errors: {errors}")

        # Deduplicate: best spelling + average confidence
        result = []
        for _key, entries in token_map.items():
            best_text = max(entries, key=lambda x: x[1])[0]
            avg_conf  = round(sum(c for _, c in entries) / len(entries), 1)
            result.append((best_text, avg_conf))

        result.sort(key=lambda x: -x[1])
        print(f"[OCR] Extracted {len(result)} unique tokens")
        return result

    def get_full_text(self, image_path):
        """
        Return (full_text_str, avg_confidence) from the best single pass.
        Uses image_to_string to preserve line structure (critical for prescription parsing).
        Confidence is computed separately via image_to_data on the same best image.
        """
        strategies = self._get_strategies(image_path)
        best_text  = ""
        best_conf  = 0.0
        best_image = None

        # First pass: find best image by average confidence
        for name, processed in strategies:
            for psm in [6, 4]:
                try:
                    data  = self._tesseract(processed, psm)
                    confs = []
                    for raw_conf in data["conf"]:
                        try:
                            c = int(raw_conf)
                        except (ValueError, TypeError):
                            continue
                        if c > 0:
                            confs.append(c)
                    avg = sum(confs) / len(confs) if confs else 0.0
                    if avg > best_conf:
                        best_conf  = avg
                        best_image = (processed, psm)
                except Exception as e:
                    print(f"[OCR] get_full_text confidence pass error ({name}/psm{psm}): {e}")

        # Second pass: extract line-preserved text from best image
        if best_image is not None:
            processed, psm = best_image
            try:
                config    = f"--oem 3 --psm {psm} -l eng"
                best_text = pytesseract.image_to_string(processed, config=config)
            except Exception as e:
                print(f"[OCR] image_to_string error: {e}")
                best_text = ""

        print(f"[OCR] Full text (avg_conf={best_conf:.1f}):\n{best_text[:300]}")
        return best_text, round(best_conf, 1)


ocr_service = OCRService()
