"""
tesseract_setup.py
------------------
Call ensure_tesseract() at the very top of run.py (before anything else).
This installs + locates tesseract at runtime, guaranteed regardless of
which Railway builder is used.
"""

import os
import shutil
import subprocess
import platform
import pytesseract


def ensure_tesseract():
    """Install tesseract if missing, then point pytesseract at it."""

    # ── Windows: just set the path, assume already installed locally ──
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
        print(f"[Tesseract] Already installed: {found}")
        return

    # ── Not found: install via apt-get at runtime ──────────────────
    print("[Tesseract] Not found — installing via apt-get...")
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(
            [
                "apt-get", "install", "-y", "--no-install-recommends",
                "tesseract-ocr",
                "tesseract-ocr-eng",
                "libgl1",
                "libglib2.0-0",
            ],
            check=True,
        )
        print("[Tesseract] apt-get install complete")
    except subprocess.CalledProcessError as e:
        print(f"[Tesseract] apt-get failed: {e}")
    except FileNotFoundError:
        print("[Tesseract] apt-get not available on this system")

    # ── Search again after install ─────────────────────────────────
    found = _find_tesseract()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        print(f"[Tesseract] Now available at: {found}")
        _verify(found)
    else:
        print("[Tesseract] FATAL: still not found after install attempt")
        _dump_debug()


def _find_tesseract():
    """Return path to tesseract binary or None."""
    # 1. PATH lookup
    found = shutil.which("tesseract")
    if found:
        return found

    # 2. Known fixed paths
    candidates = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 3. find /usr as last resort
    try:
        result = subprocess.run(
            ["find", "/usr", "-name", "tesseract", "-type", "f"],
            capture_output=True, text=True, timeout=8,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass

    return None


def _verify(path):
    try:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True
        )
        version_line = (result.stderr or result.stdout).splitlines()[0]
        print(f"[Tesseract] Version: {version_line}")
    except Exception as e:
        print(f"[Tesseract] Version check error: {e}")


def _dump_debug():
    """Print everything useful for diagnosing missing tesseract."""
    print("[Tesseract] === DEBUG INFO ===")
    for cmd in [
        ["which", "tesseract"],
        ["find", "/usr", "-name", "tesseract"],
        ["find", "/", "-name", "tesseract", "-maxdepth", "6"],
        ["apt-cache", "show", "tesseract-ocr"],
        ["dpkg", "-l", "tesseract*"],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            print(f"  {' '.join(cmd)}: {(r.stdout or r.stderr).strip()[:200]}")
        except Exception as e:
            print(f"  {' '.join(cmd)}: ERROR {e}")
    print("[Tesseract] =================")