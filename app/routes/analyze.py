import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.services.gemini_service import analyze_prescription_image

analyze_bp = Blueprint("analyze", __name__)

import traceback

@analyze_bp.route("/analyze-prescription", methods=["POST"])
def analyze_prescription():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

        file.save(file_path)

        print("✅ File saved:", file_path)

        result = analyze_prescription_image(file_path)

        print("✅ AI Result received")

        return jsonify(result)

    except Exception as e:
        print("🔥 FULL ERROR TRACE:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if "file_path" in locals() and os.path.exists(file_path):
            os.remove(file_path)