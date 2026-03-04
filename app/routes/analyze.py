"""
analyze.py  —  Flask blueprint for prescription OCR endpoint
"""

import os
import time
import traceback

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.services.ocr_service    import ocr_service
from app.services.parser_service import (
    extract_medicines_from_tokens,
    extract_medicines_from_text,
    extract_patient_details,
    extract_doctor_details,
    merge_medicine_lists,
)

analyze_bp    = Blueprint("analyze", __name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT   = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def _cleanup(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def _conf_label(score):
    if score >= 75: return "High"
    if score >= 50: return "Medium"
    return "Low"


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

@analyze_bp.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "backend alive"})


@analyze_bp.route("/analyze-prescription", methods=["POST"])
def analyze_prescription():

    # ── 1. Validate ───────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (field name must be 'file')"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
        }), 400

    # ── 2. Save ───────────────────────────────────────────────────
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    t0 = time.time()
    filepath_to_clean = filepath

    try:
        # ── 3. OCR ────────────────────────────────────────────────
        tokens_with_conf        = ocr_service.extract_tokens(filepath)
        full_text, overall_conf = ocr_service.get_full_text(filepath)

        # ── 4. Parse ──────────────────────────────────────────────
        primary   = extract_medicines_from_tokens(tokens_with_conf)
        fallback  = extract_medicines_from_text(full_text, overall_conf)
        meds      = merge_medicine_lists(primary, fallback)

        # ── 5. Demographics ───────────────────────────────────────
        patient = extract_patient_details(full_text)
        doctor  = extract_doctor_details(full_text)

        # ── 6. Confidence summary ─────────────────────────────────
        med_confs    = [m["ocr_confidence"] for m in meds if m["ocr_confidence"] > 0]
        summary_conf = round(
            sum(med_confs) / len(med_confs) if med_confs else overall_conf, 1
        )

        _cleanup(filepath_to_clean)

        return jsonify({
            "patient_details":      patient,
            "doctor_details":       doctor,
            "medications":          meds,
            "special_instructions": {},
            "ocr_meta": {
                "overall_confidence_pct": summary_conf,
                "confidence_label":       _conf_label(summary_conf),
                "processing_time_sec":    round(time.time() - t0, 2),
                "tokens_extracted":       len(tokens_with_conf),
            },
        }), 200

    except Exception as exc:
        traceback.print_exc()
        _cleanup(filepath_to_clean)
        return jsonify({"error": str(exc)}), 500
