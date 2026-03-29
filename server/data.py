"""
Healthcare Decision Support Environment — Patient & Drug Database
"""
from typing import Dict, Any, List

# ─────────────────────────────────────────────
# ALLERGY CROSS-REACTION MAP
# ─────────────────────────────────────────────
CROSS_REACTIONS: Dict[str, List[str]] = {
    "penicillin":    ["amoxicillin", "ampicillin", "piperacillin"],
    "sulfa":         ["sulfamethoxazole", "trimethoprim-sulfamethoxazole"],
    "aspirin":       ["ibuprofen", "naproxen", "celecoxib"],
    "codeine":       ["morphine", "oxycodone", "hydrocodone"],
    "latex":         [],
    "shellfish":     [],
    "cephalosporin": ["cefazolin", "cephalexin", "ceftriaxone"],
}

# ─────────────────────────────────────────────
# DRUG DATABASE
# ─────────────────────────────────────────────
DRUG_DB: Dict[str, Dict[str, Any]] = {
    "amoxicillin": {
        "allergy_class":     "penicillin",
        "standard_dose_mg":  500,
        "frequency":         "every 8 hours",
        "max_daily_mg":      3000,
        "renal_reduction":   True,
        "hepatic_reduction": False,
        "interactions":      ["methotrexate", "warfarin"],
        "contraindications": ["penicillin_allergy"],
    },
    "ibuprofen": {
        "allergy_class":     "nsaid",
        "standard_dose_mg":  400,
        "frequency":         "every 6-8 hours",
        "max_daily_mg":      2400,
        "renal_reduction":   True,
        "hepatic_reduction": False,
        "interactions":      ["warfarin", "lithium", "methotrexate"],
        "contraindications": ["aspirin_allergy", "ckd_stage3plus", "peptic_ulcer"],
    },
    "metformin": {
        "allergy_class":     None,
        "standard_dose_mg":  500,
        "frequency":         "twice daily",
        "max_daily_mg":      2000,
        "renal_reduction":   True,
        "hepatic_reduction": False,
        "interactions":      ["contrast_dye", "alcohol"],
        "contraindications": ["ckd_stage4plus", "liver_failure"],
    },
    "lisinopril": {
        "allergy_class":     "ace_inhibitor",
        "standard_dose_mg":  10,
        "frequency":         "once daily",
        "max_daily_mg":      40,
        "renal_reduction":   True,
        "hepatic_reduction": False,
        "interactions":      ["potassium_supplements", "spironolactone", "nsaids"],
        "contraindications": ["angioedema_history", "pregnancy", "hyperkalemia"],
    },
    "warfarin": {
        "allergy_class":     None,
        "standard_dose_mg":  5,
        "frequency":         "once daily",
        "max_daily_mg":      15,
        "renal_reduction":   False,
        "hepatic_reduction": True,
        "interactions":      ["aspirin", "ibuprofen", "naproxen", "amoxicillin",
                              "metronidazole", "fluconazole", "vitamin_k"],
        "contraindications": ["active_bleeding", "recent_surgery"],
    },
    "sulfamethoxazole": {
        "allergy_class":     "sulfa",
        "standard_dose_mg":  800,
        "frequency":         "twice daily",
        "max_daily_mg":      1600,
        "renal_reduction":   True,
        "hepatic_reduction": False,
        "interactions":      ["warfarin", "methotrexate", "phenytoin"],
        "contraindications": ["sulfa_allergy", "g6pd_deficiency"],
    },
    "atorvastatin": {
        "allergy_class":     "statin",
        "standard_dose_mg":  20,
        "frequency":         "once daily",
        "max_daily_mg":      80,
        "renal_reduction":   False,
        "hepatic_reduction": True,
        "interactions":      ["clarithromycin", "itraconazole", "cyclosporine"],
        "contraindications": ["active_liver_disease", "pregnancy"],
    },
    "amlodipine": {
        "allergy_class":     "calcium_channel_blocker",
        "standard_dose_mg":  5,
        "frequency":         "once daily",
        "max_daily_mg":      10,
        "renal_reduction":   False,
        "hepatic_reduction": True,
        "interactions":      ["simvastatin", "cyclosporine"],
        "contraindications": ["cardiogenic_shock"],
    },
    "acetaminophen": {
        "allergy_class":     None,
        "standard_dose_mg":  500,
        "frequency":         "every 6 hours",
        "max_daily_mg":      3000,
        "renal_reduction":   False,
        "hepatic_reduction": True,
        "interactions":      ["warfarin", "alcohol"],
        "contraindications": ["severe_liver_disease"],
    },
}

# ─────────────────────────────────────────────
# PATIENT CASES
# ─────────────────────────────────────────────
PATIENTS: List[Dict[str, Any]] = [

    # ── Task 1: allergy_check (easy) ──────────────────────────────────────
    {
        "patient_id":    "P001",
        "task":          "allergy_check",
        "difficulty":    "easy",
        "name":          "Alice M.",
        "age":           45,
        "weight_kg":     68,
        "sex":           "F",
        "allergies":     ["penicillin"],
        "conditions":    ["urinary tract infection"],
        "current_meds":  [],
        "labs":          {"GFR": 85, "ALT": 22, "AST": 18},
        "proposed_drug": "amoxicillin",
        "correct_flags": ["amoxicillin"],
        "correct_reason":"penicillin cross-reaction",
    },
    {
        "patient_id":    "P002",
        "task":          "allergy_check",
        "difficulty":    "easy",
        "name":          "Bob K.",
        "age":           60,
        "weight_kg":     80,
        "sex":           "M",
        "allergies":     ["aspirin"],
        "conditions":    ["mild arthritis"],
        "current_meds":  [],
        "labs":          {"GFR": 72, "ALT": 30, "AST": 25},
        "proposed_drug": "ibuprofen",
        "correct_flags": ["ibuprofen"],
        "correct_reason":"NSAID cross-reaction with aspirin allergy",
    },
    {
        "patient_id":    "P003",
        "task":          "allergy_check",
        "difficulty":    "easy",
        "name":          "Carol T.",
        "age":           33,
        "weight_kg":     55,
        "sex":           "F",
        "allergies":     ["sulfa"],
        "conditions":    ["bacterial infection"],
        "current_meds":  [],
        "labs":          {"GFR": 95, "ALT": 18, "AST": 15},
        "proposed_drug": "sulfamethoxazole",
        "correct_flags": ["sulfamethoxazole"],
        "correct_reason":"sulfa allergy direct conflict",
    },
    {
        "patient_id":    "P004",
        "task":          "allergy_check",
        "difficulty":    "easy",
        "name":          "David R.",
        "age":           52,
        "weight_kg":     90,
        "sex":           "M",
        "allergies":     ["latex"],
        "conditions":    ["hypertension"],
        "current_meds":  [],
        "labs":          {"GFR": 80, "ALT": 28, "AST": 22},
        "proposed_drug": "amlodipine",
        "correct_flags": [],
        "correct_reason":"latex allergy has no drug cross-reactions; amlodipine is safe",
    },

    # ── Task 2: dosing (medium) ───────────────────────────────────────────
    {
        "patient_id":        "P005",
        "task":              "dosing",
        "difficulty":        "medium",
        "name":              "Eve S.",
        "age":               70,
        "weight_kg":         58,
        "sex":               "F",
        "allergies":         [],
        "conditions":        ["hypertension", "chronic kidney disease stage 3"],
        "current_meds":      ["amlodipine 5mg"],
        "labs":              {"GFR": 35, "ALT": 25, "AST": 20},
        "drug_to_dose":      "lisinopril",
        "correct_dose_mg":   5,
        "dose_range_ok":     [2.5, 5],
        "correct_frequency": "once daily",
        "key_reason":        "reduce starting dose due to CKD stage 3 GFR 35",
    },
    {
        "patient_id":        "P006",
        "task":              "dosing",
        "difficulty":        "medium",
        "name":              "Frank L.",
        "age":               55,
        "weight_kg":         95,
        "sex":               "M",
        "allergies":         [],
        "conditions":        ["type 2 diabetes"],
        "current_meds":      [],
        "labs":              {"GFR": 78, "ALT": 32, "AST": 28},
        "drug_to_dose":      "metformin",
        "correct_dose_mg":   500,
        "dose_range_ok":     [500, 1000],
        "correct_frequency": "twice daily",
        "key_reason":        "start low titrate GFR 78 adequate metformin",
    },
    {
        "patient_id":        "P007",
        "task":              "dosing",
        "difficulty":        "medium",
        "name":              "Grace N.",
        "age":               8,
        "weight_kg":         25,
        "sex":               "F",
        "allergies":         [],
        "conditions":        ["otitis media"],
        "current_meds":      [],
        "labs":              {"GFR": 110, "ALT": 15, "AST": 12},
        "drug_to_dose":      "amoxicillin",
        "correct_dose_mg":   250,
        "dose_range_ok":     [200, 375],
        "correct_frequency": "every 8 hours",
        "key_reason":        "pediatric dosing 40mg/kg/day divided doses weight",
    },

    # ── Task 3: treatment_plan (hard) ─────────────────────────────────────
    {
        "patient_id":   "P008",
        "task":         "treatment_plan",
        "difficulty":   "hard",
        "name":         "Henry V.",
        "age":          65,
        "weight_kg":    82,
        "sex":          "M",
        "allergies":    ["penicillin"],
        "conditions":   ["type 2 diabetes", "hypertension", "hyperlipidemia"],
        "current_meds": [],
        "labs":         {"GFR": 55, "ALT": 38, "AST": 33},
        "plan_rubric": {
            "diabetes_drug":     ["metformin"],
            "hypertension_drug": ["lisinopril", "amlodipine"],
            "lipid_drug":        ["atorvastatin"],
            "avoid_drugs":       ["ibuprofen", "amoxicillin"],
            "interaction_check": True,
        },
    },
    {
        "patient_id":   "P009",
        "task":         "treatment_plan",
        "difficulty":   "hard",
        "name":         "Irene C.",
        "age":          58,
        "weight_kg":    70,
        "sex":          "F",
        "allergies":    ["aspirin", "sulfa"],
        "conditions":   ["atrial fibrillation", "osteoarthritis", "hypertension"],
        "current_meds": ["warfarin 5mg"],
        "labs":         {"GFR": 68, "ALT": 44, "AST": 40},
        "plan_rubric": {
            "anticoag_note":     True,
            "pain_drug":         ["acetaminophen"],
            "avoid_drugs":       ["ibuprofen", "naproxen", "aspirin", "sulfamethoxazole"],
            "hypertension_drug": ["amlodipine", "lisinopril"],
            "interaction_check": True,
        },
    },
]

# ─────────────────────────────────────────────
# ENCODE PATIENT → flat float list (info_state)
# ─────────────────────────────────────────────

ALL_ALLERGIES  = ["penicillin", "sulfa", "aspirin", "codeine",
                  "latex", "shellfish", "cephalosporin", "nsaid"]

ALL_CONDITIONS = ["type 2 diabetes", "hypertension", "hyperlipidemia",
                  "chronic kidney disease stage 3", "chronic kidney disease stage 4",
                  "atrial fibrillation", "osteoarthritis", "urinary tract infection",
                  "bacterial infection", "otitis media", "mild arthritis", "peptic ulcer"]


def encode_patient(patient: Dict[str, Any]) -> List[float]:
    """
    Encode a patient dict as a flat 28-value float vector for RL agents.
    [0-4]   demographics + task/difficulty encoding
    [5-12]  allergy one-hot
    [13-24] condition one-hot
    [25-27] lab values (GFR, ALT, AST normalised)
    """
    age_norm    = patient.get("age", 40) / 100.0
    weight_norm = patient.get("weight_kg", 70) / 150.0
    sex_enc     = 1.0 if patient.get("sex") == "F" else 0.0
    task_enc    = {"allergy_check": 0.0, "dosing": 0.5, "treatment_plan": 1.0}.get(patient.get("task", ""), 0.0)
    diff_enc    = {"easy": 0.0, "medium": 0.5, "hard": 1.0}.get(patient.get("difficulty", ""), 0.0)

    allergy_vec   = [1.0 if a in patient.get("allergies", []) else 0.0 for a in ALL_ALLERGIES]
    condition_vec = [1.0 if c in patient.get("conditions", []) else 0.0 for c in ALL_CONDITIONS]

    labs    = patient.get("labs", {})
    gfr_n   = labs.get("GFR", 90)  / 130.0
    alt_n   = labs.get("ALT", 30)  / 200.0
    ast_n   = labs.get("AST", 25)  / 200.0

    return ([age_norm, weight_norm, sex_enc, task_enc, diff_enc]
            + allergy_vec + condition_vec + [gfr_n, alt_n, ast_n])
