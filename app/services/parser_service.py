"""
parser_service.py
-----------------
Parses OCR output into structured prescription data.

Two complementary strategies run and are merged:
  1. Token-level  — iterates token list, uses fuzzy matching for medicine names
  2. Full-text    — regex over the complete OCR string, fills gaps from strategy 1
"""

import re
from difflib import get_close_matches, SequenceMatcher


# ──────────────────────────────────────────────────────────────────
# Reference tables  (extend as needed)
# ──────────────────────────────────────────────────────────────────

KNOWN_MEDICINES = [
    "Betaloc", "Dorzolamidum", "Cimetidine", "Oxprelol",
    "Paracetamol", "Ibuprofen", "Amoxicillin", "Azithromycin",
    "Metformin", "Omeprazole",
]

# Raw OCR variant (lowercase) → canonical label
FREQ_MAP = {
    "bid": "BID", "b.i.d": "BID", "b.i.d.": "BID",
    "tid": "TID", "t.i.d": "TID", "tld":    "TID",
    "qd":  "QD",  "q.d":   "QD",  "od":     "QD",
    "bd":  "BD",
    "qid": "QID", "q.i.d": "QID",
}

_DOSAGE_RE   = re.compile(r"(\d[\d.,]*)\s*mg",             re.IGNORECASE)
_QTY_TAB_RE  = re.compile(r"([1-9IZz|l])\s*tab[s]?",        re.IGNORECASE)
_DURATION_RE = re.compile(r"(\d+)\s*(day[s]?|week[s]?|month[s]?)", re.IGNORECASE)
# Per-line regex handles OCR noise: "IO ms"→10mg, "|"→1 tab, "ID"→BID, "QV"→QD
_MED_LINE_RE = re.compile(
    r"(?P<n>[A-Za-z][A-Za-z]+)"
    r"(?:\s+(?P<dosage>[0-9IO]{1,4}[\d.,]*\s*m[gs]))?"
    r"(?:\s*[-\u2013]\s*(?P<qty>[1-9IZz|l])(?:\s*tab[s]?)?)?"
    r"(?:[\s\d.,]*(?P<freq>BID|TID|QD|OD|BD|QID|ID))?",
    re.IGNORECASE,
)
_QTY_OCRNORM  = {"I": "1", "l": "1", "|": "1", "Z": "2", "z": "2"}
_FREQ_OCR_MAP = {"ID": "BID", "QV": "QD"}

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _alpha_only(s):
    return re.sub(r"[^a-zA-Z]", "", s)

def _fuzzy_match(word, cutoff=0.70):
    """
    Return (canonical_medicine_name, similarity_score) or (None, 0).
    Strips non-alpha chars before comparing so OCR noise like 'Betoloe' still matches.
    """
    clean = _alpha_only(word)
    if len(clean) < 4:
        return None, 0.0
    med_lower = [m.lower() for m in KNOWN_MEDICINES]
    matches   = get_close_matches(clean.lower(), med_lower, n=1, cutoff=cutoff)
    if matches:
        idx   = med_lower.index(matches[0])
        score = SequenceMatcher(None, clean.lower(), matches[0]).ratio()
        return KNOWN_MEDICINES[idx], round(score, 2)
    return None, 0.0

def _normalise_freq(raw):
    return FREQ_MAP.get(raw.lower().strip(" ."), None)

def _empty_entry(name, score):
    return {
        "medicine_name":      name,
        "dosage":             None,
        "quantity_per_dose":  None,
        "frequency":          None,
        "duration":           None,
        "timing":             None,
        "fuzzy_match_score":  score,
        "ocr_confidence":     0.0,
    }

def _avg_conf(confs):
    return round(sum(confs) / len(confs), 1) if confs else 0.0


# ──────────────────────────────────────────────────────────────────
# Strategy 1 — token-level parsing
# ──────────────────────────────────────────────────────────────────

def extract_medicines_from_tokens(tokens_with_conf):
    """
    tokens_with_conf: list of (str, float) from ocr_service.extract_tokens()
    Returns list of medicine dicts.
    """
    medicines  = []
    current    = None
    conf_pool  = []

    for token, conf in tokens_with_conf:
        t = token.strip()
        if not t:
            continue

        # ── Try medicine name match ───────────────────────────
        med_name, score = _fuzzy_match(t)
        if med_name:
            # Save previous entry
            if current:
                current["ocr_confidence"] = _avg_conf(conf_pool)
                medicines.append(current)
            current   = _empty_entry(med_name, score)
            conf_pool = [conf]
            continue

        if current is None:
            continue

        conf_pool.append(conf)
        tl = t.lower()

        # ── Dosage ───────────────────────────────────────────
        if current["dosage"] is None:
            m = _DOSAGE_RE.search(t)
            if m:
                current["dosage"] = m.group(0).replace(" ", "").lower()

        # ── Quantity per dose ─────────────────────────────────
        if current["quantity_per_dose"] is None:
            m = _QTY_TAB_RE.search(t)
            if m:
                current["quantity_per_dose"] = f"{m.group(1)} tab"

        # ── Frequency ─────────────────────────────────────────
        if current["frequency"] is None:
            freq = _normalise_freq(t)
            if freq:
                current["frequency"] = freq

        # ── Duration ──────────────────────────────────────────
        if current["duration"] is None:
            m = _DURATION_RE.search(t)
            if m:
                current["duration"] = m.group(0)

    if current:
        current["ocr_confidence"] = _avg_conf(conf_pool)
        medicines.append(current)

    return medicines


# ──────────────────────────────────────────────────────────────────
# Strategy 2 — full-text regex parsing  (fills gaps)
# ──────────────────────────────────────────────────────────────────

def extract_medicines_from_text(full_text, overall_conf=0.0):
    """
    Regex sweep over the full OCR text string.
    Used to catch anything the token pass missed.
    """
    medicines = []
    seen      = set()

    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for m in _MED_LINE_RE.finditer(line):
            raw_name = m.group("n")
            med_name, score = _fuzzy_match(raw_name)
            if not med_name or med_name.lower() in seen:
                continue
            seen.add(med_name.lower())
            entry = _empty_entry(med_name, score)
            entry["ocr_confidence"] = round(overall_conf, 1)
            if m.group("dosage"):
                entry["dosage"] = m.group("dosage").replace(" ", "").lower()
            if m.group("qty"):
                q = m.group("qty")
                q = _QTY_OCRNORM.get(q, q)
                if q.isdigit():
                    entry["quantity_per_dose"] = f"{q} tab"
            if m.group("freq"):
                raw_freq = m.group("freq").upper()
                entry["frequency"] = _FREQ_OCR_MAP.get(raw_freq, raw_freq)
            # Also catch QV (Tesseract misread of QD) anywhere in the line
            if not entry["frequency"] and re.search(r'\bQV\b', line, re.IGNORECASE):
                entry["frequency"] = "QD"
            # Normalise dosage: "IO ms" → "10mg"
            if entry["dosage"]:
                entry["dosage"] = re.sub(r'\s+', '', entry["dosage"]).lower()
                entry["dosage"] = entry["dosage"].replace('io', '10').replace('ms', 'mg')
            medicines.append(entry)

    return medicines


# ──────────────────────────────────────────────────────────────────
# Merge
# ──────────────────────────────────────────────────────────────────

def merge_medicine_lists(primary, fallback):
    """
    Merge token-level (primary) and full-text-line (fallback) medicine lists.

    Rules:
      1. Deduplicate: same medicine_name → keep the highest-confidence version,
         then fill any None fields from the other.
      2. Order: use the prescription line order from `fallback` (which iterates
         lines top-to-bottom). Medicines only found in `primary` are appended at end.
      3. Low-confidence duplicates (fuzzy_match_score < 0.80 AND ocr_confidence < 40)
         are discarded when a better match already exists.
    """
    FIELDS = ("dosage", "quantity_per_dose", "frequency", "duration")

    # Build combined pool keyed by medicine name (lowercase)
    pool = {}  # name_lower → best entry

    def _add(entry):
        key = entry["medicine_name"].lower()
        if key not in pool:
            pool[key] = dict(entry)
        else:
            existing = pool[key]
            # Keep the higher-confidence entry as base
            if entry["ocr_confidence"] > existing["ocr_confidence"]:
                # Copy fields from existing that are set but missing in new entry
                for f in FIELDS:
                    if entry.get(f) is None and existing.get(f) is not None:
                        entry[f] = existing[f]
                pool[key] = dict(entry)
            else:
                # Fill gaps in existing from new entry
                for f in FIELDS:
                    if existing.get(f) is None and entry.get(f) is not None:
                        existing[f] = entry[f]

    # Add all entries — primary first (token-level), then fallback (line-level)
    for m in primary:
        _add(m)
    for m in fallback:
        _add(m)

    # Build final list in prescription line order (fallback order = line order)
    ordered = []
    seen = set()
    for fb in fallback:
        key = fb["medicine_name"].lower()
        if key in pool and key not in seen:
            ordered.append(pool[key])
            seen.add(key)

    # Append any medicines only found by token pass (not in fallback)
    for key, entry in pool.items():
        if key not in seen:
            # Discard very low confidence orphans
            if entry["fuzzy_match_score"] >= 0.80 or entry["ocr_confidence"] >= 40:
                ordered.append(entry)
                seen.add(key)

    return ordered


# ──────────────────────────────────────────────────────────────────
# Patient / doctor detail extraction
# ──────────────────────────────────────────────────────────────────

def extract_patient_details(full_text):
    details = {}
    patterns = {
        "name":    r"NAME[:\s]+([A-Za-z .'-]+?)(?:\s{2,}|AGE|$)",
        "age":     r"AGE[:\s]+(\d{1,3})",
        "address": r"ADDRESS[:\s]+([^\n]+)",
        "date":    r"DATE[:\s]+([\d/\-\.]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            details[key] = m.group(1).strip()
    return details

def extract_doctor_details(full_text):
    details = {}
    patterns = {
        "name":    r"Dr\.?\s+([A-Za-z .'-]+?)(?:\s{2,}|signature|$)",
        "license": r"LIC\s*#?\s*([\w\d]+)",
        "dea":     r"DEA\s*#?\s*([\w\d]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if key == "name":
                val = "Dr. " + val
            details[key] = val
    return details
