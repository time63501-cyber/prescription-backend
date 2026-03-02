import google.generativeai as genai
import json
import re
from PIL import Image
from flask import current_app


def analyze_prescription_image(file_path):
    """
    Analyze prescription image using Gemini Vision model.
    Returns clean parsed JSON.
    """

    try:
        # Configure API key securely from config
        api_key = current_app.config.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in config.")

        genai.configure(api_key=api_key)

        # Use stable model supported by v1beta
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        Analyze this medical prescription and extract:

        1. Patient name and details
        2. Doctor name and details
        3. List of medications with:
           - Medicine name
           - Dosage
           - Frequency (BID, TID, QD etc.)
           - Duration
           - Timing (before/after food)
        4. Special instructions

        Return ONLY valid JSON.
        Do NOT include markdown.
        Do NOT include explanations.
        """

        print("🔍 Opening image:", file_path)

        # Safely open image
        with Image.open(file_path) as image:
            response = model.generate_content([prompt, image])

        raw_text = response.text.strip()

        print("🔎 RAW AI RESPONSE:")
        print(raw_text)

        # -------- CLEAN RESPONSE --------

        # Remove markdown formatting
        raw_text = raw_text.replace("```json", "")
        raw_text = raw_text.replace("```", "").strip()

        # Extract only JSON block
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            raw_text = match.group(0)

        # Parse JSON safely
        parsed = json.loads(raw_text)
        parsed.setdefault("medications", [])

        # -------- NORMALIZE MEDICATION DATA --------

        for med in parsed.get("medications", []):
         freq = med.get("frequency")
         if freq:
            parts = freq.split()
            # If structure like "1 tab BID"
            if len(parts) >= 3:
                  med["quantity_per_dose"] = f"{parts[0]} {parts[1]}"
                  med["frequency"] = parts[-1]  # BID / TID / QD

            # If structure like "BID"
            elif len(parts) == 1:
                  med["frequency"] = parts[0]

         # Ensure keys always exist
         med.setdefault("quantity_per_dose", None)
         med.setdefault("duration", None)
         med.setdefault("timing", None)

        # Ensure medications key always exists
        parsed.setdefault("medications", [])

        print("✅ JSON PARSED SUCCESSFULLY")

        return parsed

    except json.JSONDecodeError as e:
        print("❌ JSON PARSE ERROR:", e)
        return {
            "patient_details": {},
            "doctor_details": {},
            "medications": [],
            "error": "Invalid JSON from AI"
        }

    except Exception as e:
        print("🔥 GEMINI ERROR:", str(e))
        return {
            "patient_details": {},
            "doctor_details": {},
            "medications": [],
            "error": str(e)
        }