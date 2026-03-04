"""
medicine_db.py
--------------
Simple lookup table for medicine information.
Add entries here as new medicines are encountered.
"""

MEDICINE_INFO = {
    "Paracetamol":   "Analgesic / antipyretic — used for pain relief and fever reduction.",
    "Ibuprofen":     "NSAID — non-steroidal anti-inflammatory; pain, fever, inflammation.",
    "Amoxicillin":   "Penicillin-class antibiotic for bacterial infections.",
    "Azithromycin":  "Macrolide antibiotic for respiratory, skin and other infections.",
    "Metformin":     "Biguanide — first-line treatment for type 2 diabetes.",
    "Omeprazole":    "Proton pump inhibitor — reduces stomach acid production.",
    "Betaloc":       "Metoprolol (beta-1 blocker) — hypertension, angina, heart failure.",
    "Dorzolamidum":  "Carbonic anhydrase inhibitor — reduces intraocular pressure (glaucoma).",
    "Cimetidine":    "H2 blocker — reduces stomach acid; peptic ulcers, GERD.",
    "Oxprelol":      "Non-selective beta blocker — hypertension and certain arrhythmias.",
}


def get_info(medicine_name: str) -> str:
    """Return info string for a medicine, or an empty string if unknown."""
    return MEDICINE_INFO.get(medicine_name, "")
