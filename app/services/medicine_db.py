"""
medicine_db.py
--------------
Extended lookup table for medicine information.
Contains common international and Indian medications to guide intelligent OCR parsing.
"""

MEDICINE_INFO = {
    # ── Pain, Fever, & Inflammation ──
    "Paracetamol": "Analgesic / antipyretic — used for pain relief and fever reduction.",
    "Ibuprofen": "NSAID — pain, fever, inflammation.",
    "Dolo": "Analgesic / antipyretic (Paracetamol base) — popular in India for fever.",
    "Dolo 650": "Paracetamol 650mg — popular in India for fever.",
    "Crocin": "Analgesic / antipyretic (Paracetamol brand).",
    "Calpol": "Analgesic / antipyretic (Paracetamol brand).",
    "Meftal": "Mefenamic Acid — used for menstrual pain and cramps.",
    "Zerodol": "Aceclofenac — NSAID for joint and muscle pain.",
    "Ecosprin": "Aspirin — blood thinner and mild analgesic.",
    "Aspirin": "NSAID — pain relief and anti-platelet.",

    # ── Antibiotics & Antivirals ──
    "Amoxicillin": "Penicillin-class antibiotic for bacterial infections.",
    "Azithromycin": "Macrolide antibiotic for respiratory, skin, and other infections.",
    "Azee": "Azithromycin brand — respiratory and bacterial infections.",
    "Augmentin": "Amoxicillin and Clavulanate Potassium — broad spectrum antibiotic.",
    "Clavam": "Amoxicillin and Clavulanic Acid — broad spectrum antibiotic.",
    "Flagyl": "Metronidazole — antibiotic and antiprotozoal.",
    "Monocef": "Ceftriaxone antibiotic — primarily used for serious infections.",
    "Taxim": "Cefotaxime / Cefixime antibiotic.",
    "Zifi": "Cefixime antibiotic — prescribed for ear, throat, and urinary tract infections.",
    "O2": "Ofloxacin and Ornidazole — used for gastrointestinal and dental infections.",

    # ── Antacids & Gastrointestinal ──
    "Omeprazole": "Proton pump inhibitor — reduces stomach acid production.",
    "Pantoprazole": "Proton pump inhibitor — treats GERD and ulcers.",
    "Pan": "Pantoprazole brand — treats acidity.",
    "Pan 40": "Pantoprazole 40mg — treats acidity and acid reflux.",
    "Pan D": "Pantoprazole and Domperidone — treats acidic reflux and nausea.",
    "Pantosec": "Pantoprazole brand — treats acidity and reflux.",
    "Drotin": "Drotaverine — antispasmodic for stomach / abdominal pain.",
    "Drotin M": "Drotaverine + Mefenamic Acid — antispasmodic and analgesic for pain.",
    "Electral": "ORS (Oral Rehydration Salts) — restores fluids and electrolytes.",
    "Rablet": "Rabeprazole — reduces stomach acid, treats ulcers.",
    "Eno": "Antacid — quick relief from acidity and heartburn.",
    "Gaviscon": "Antacid — treats heartburn and indigestion.",
    "Eldoper": "Loperamide — treats diarrhea.",
    "Sporlac": "Probiotic — restores gut flora.",
    "Econorm": "Probiotic (Saccharomyces boulardii) — treats infectious diarrhea.",
    "Enterogermina": "Probiotic — manages diarrhea and restores intestinal bacteria.",

    # ── Cough, Cold, & Allergies ──
    "Cetirizine": "Antihistamine — treats allergies and hay fever.",
    "Citrizin": "Antihistamine — treats allergies (often spelled Citrizin).",
    "Levocet": "Levocetirizine — non-sedating antihistamine for allergies.",
    "Allegra": "Fexofenadine — treats allergy symptoms.",
    "Sinarest": "Combination drug — relieves cold symptoms, congestion, and fever.",
    "Cheston Cold": "Anti-cold medication — relieves nasal congestion and sneezing.",
    "Wikoryl": "Anti-cold medication.",
    "Dyril": "Antihistamine / cough & cold syrup.",
    "Corex": "Cough syrup — relieves dry cough.",
    "Ascoril": "Expectorant cough syrup — helps clear mucus.",
    "Cofils": "Cough lozenges.",
    "Vicks": "Cold and cough relief.",
    "Strepsils": "Lozenges for sore throat.",
    "Montac D": "Montelukast and Desloratadine combination — prescribed for allergic rhinitis.",

    # ── Cardiovascular, Diabetes & BP ──
    "Metformin": "Biguanide — primary treatment for type 2 diabetes.",
    "Betaloc": "Metoprolol (beta-1 blocker) — hypertension, angina, heart failure.",
    "Cimetidine": "H2 blocker — reduces stomach acid.",
    "Oxprelol": "Non-selective beta blocker — hypertension.",
    "Telma": "Telmisartan — angiotensin II receptor blocker for hypertension.",
    "Amlong": "Amlodipine — calcium channel blocker for high blood pressure.",
    
    # ── Vitamins, Supplements & Steroids ──
    "Dexona": "Dexamethasone — corticosteroid for allergies, asthma, and inflammation.",
    "Thyronorm": "Thyroxine sodium — treats hypothyroidism.",
    "Supradyn": "Multivitamin supplement.",
    "Zincovit": "Multivitamin and multimineral tablet.",
    "Limcee": "Vitamin C chewable tablet.",
    "Shelcal": "Calcium and Vitamin D3 supplement.",
    "Eldervit": "Injectable/oral multivitamin (B-complex and Vitamin C).",
    
    # ── Eye & Others ──
    "Dorzolamidum": "Carbonic anhydrase inhibitor — reduces intraocular pressure (glaucoma)."
}

def get_info(medicine_name: str) -> str:
    """Return info string for a medicine, or an empty string if unknown."""
    return MEDICINE_INFO.get(medicine_name, "")
