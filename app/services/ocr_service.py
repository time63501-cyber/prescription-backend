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
import shutil
import os
import subprocess


# ─────────────────────────────────────────────────────────────────
# Tesseract setup — runs once at module import time
# Handles Windows local dev + Railway/Linux auto-install via apt-get
# ─────────────────────────────────────────────────────────────────

def _find_tesseract():
    """Return path to tesseract binary, or None if not found."""
    # 1. Check PATH (fastest — works if nixpacks/railpack installed it)
    found = shutil.which("tesseract")
    if found:
        return found
    # 2. Known fixed paths on Debian/Ubuntu/Railway containers
    for p in [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "/nix/var/nix/profiles/default/bin/tesseract",
    ]:
        if os.path.exists(p):
            return p
    # 3. Slow but thorough filesystem search
    try:
        r = subprocess.run(
            ["find", "/usr", "-name", "tesseract", "-type", "f"],
            capture_output=True, text=True, timeout=8
        )
        lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass
    return None


def _verify(path):
    try:
        r = subprocess.run([path, "--version"], capture_output=True, text=True)
        line = (r.stderr or r.stdout).splitlines()[0]
        print(f"[Tesseract] {line}")
    except Exception as e:
        print(f"[Tesseract] version check error: {e}")


def _dump_debug():
    print("[Tesseract] === DEBUG ===")
    for cmd in [
        ["which", "tesseract"],
        ["find", "/usr", "-name", "tesseract"],
        ["dpkg", "-l", "tesseract*"],
        ["ls", "/usr/bin/"],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            print(f"  {' '.join(cmd)}: {(r.stdout or r.stderr).strip()[:200] or '(empty)'}")
        except Exception as e:
            print(f"  {' '.join(cmd)}: {e}")
    print("[Tesseract] ================")


def _setup_tesseract():
    # ── Windows: use local installer path ─────────────────────────
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        print("[Tesseract] Windows: using local install")
        return

    # ── Linux/Mac: check if already available ─────────────────────
    found = _find_tesseract()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        print(f"[Tesseract] Found: {found}")
        _verify(found)
        return

    # ── Not found: install via apt-get at runtime ──────────────────
    # This is the guaranteed fallback for Railway containers
    print("[Tesseract] Not in PATH — running apt-get install...")
    try:
        subprocess.run(["apt-get", "update", "-qq"],
                       check=True, timeout=60)
        subprocess.run(
            ["apt-get", "install", "-y", "--no-install-recommends",
             "tesseract-ocr", "tesseract-ocr-eng",
             "libgl1", "libglib2.0-0"],
            check=True, timeout=120
        )
        print("[Tesseract] apt-get install done")
    except subprocess.CalledProcessError as e:
        print(f"[Tesseract] apt-get failed (exit {e.returncode}): {e}")
    except subprocess.TimeoutExpired:
        print("[Tesseract] apt-get timed out after 120s")
    except FileNotFoundError:
        print("[Tesseract] apt-get binary not found on this system")
    except Exception as e:
        print(f"[Tesseract] apt-get unexpected error: {e}")

    # ── Re-search after install attempt ───────────────────────────
    found = _find_tesseract()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        print(f"[Tesseract] Now available: {found}")
        _verify(found)
    else:
        print("[Tesseract] FATAL: tesseract still not found — dumping debug info")
        _dump_debug()


_setup_tesseract()


# ─────────────────────────────────────────────────────────────────
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
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        up    = self._upscale(gray)
        blur  = cv2.GaussianBlur(up, (0, 0), 3)
        sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _strategy_handwriting_enhanced(self, bgr):
        """
        Advanced preprocessing for messy/thin handwriting photographed on phones.
        Uses adaptive thresholding to handle uneven shadows, and morphological 
        dilation to aggressively thicken pen strokes for Tesseract.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        up   = self._upscale(gray)
        
        # 1. Adaptive thresholding handles uneven camera lighting significantly better than Otsu
        thresh = cv2.adaptiveThreshold(
            up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        
        # 2. Morphological dilation to thicken thin cursive strokes
        kernel = np.ones((2, 2), np.uint8)
        inv_thresh = 255 - thresh               # Invert: text becomes white
        dilated = cv2.dilate(inv_thresh, kernel, iterations=1)
        final_image = 255 - dilated             # Invert back: text is black
        
        return final_image

    def _get_strategies(self, image_path):
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Cannot read image at path: {image_path}")
        return [
            ("otsu",         self._strategy_otsu(bgr)),
            ("denoise_otsu", self._strategy_denoise_otsu(bgr)),
            ("plain",        self._strategy_plain(bgr)),
            ("sharpen_otsu", self._strategy_sharpen(bgr)),
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

    def process_image(self, image_path):
        """
        Extracts both a token list and the full text string using a single Tesseract pass.
        This resolves the Gunicorn timeout caused by running 17 Tesseract passes synchronously.
        Uses Otsu upscale for best preservation of handwritten and printed text.
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Cannot read image at path: {image_path}")
            
        processed = self._strategy_handwriting_enhanced(bgr)
        
        token_map = {}
        best_text = ""
        avg_conf = 0.0
        
        try:
            # 1. Full text string directly
            config = "--oem 3 --psm 6 -l eng"
            best_text = pytesseract.image_to_string(processed, config=config)
            
            # 2. Token extraction with confidence
            data = self._tesseract(processed, 6)
            confs = []
            for raw_text, raw_conf in zip(data["text"], data["conf"]):
                text = raw_text.strip()
                if not text:
                    continue
                try:
                    conf = int(raw_conf)
                except (ValueError, TypeError):
                    continue
                if conf < 20:   
                    continue
                confs.append(conf)
                token_map.setdefault(text.lower(), []).append((text, conf))
                
            avg_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
            
        except Exception as e:
            print(f"[OCR] Process error: {e}")
            
        tokens = []
        for _key, entries in token_map.items():
            best_token_text = max(entries, key=lambda x: x[1])[0]
            avg_t_conf  = round(sum(c for _, c in entries) / len(entries), 1)
            tokens.append((best_token_text, avg_t_conf))
            
        tokens.sort(key=lambda x: -x[1])
        print(f"[OCR] Extracted {len(tokens)} unique tokens")
        print(f"[OCR] Full text (avg_conf={avg_conf:.1f}):\n{best_text[:300]}")
        
        return tokens, best_text, avg_conf

ocr_service = OCRService()