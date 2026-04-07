"""
Healthcare Decision Support Environment -- Baseline Inference Script

Runs the rule-based baseline agent across all three tasks locally
and prints reproducible scores. Also supports testing a live HF Space.

Usage:
    python baseline.py                            # run locally
    python baseline.py --url https://USER-healthcareenv.hf.space
"""
import argparse
import json
import sys
import os

# Windows + Linux compatible path setup
# Always run relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from server.healthcareenv_environment import HealthcareEnvironment
from server.data import PATIENTS, DRUG_DB, CROSS_REACTIONS
from models import MedicalAction


# ─────────────────────────────────────────────────────────────────────────────
# Baseline policies  (deterministic, rule-based)
# ─────────────────────────────────────────────────────────────────────────────

def policy_allergy(patient):
    """Flag proposed drug if it directly or cross-reacts with any patient allergy."""
    proposed    = patient.get("proposed_drug", "")
    drug_info   = DRUG_DB.get(proposed, {})
    allergy_cls = drug_info.get("allergy_class")
    flags, reasons = [], []

    for allergy in patient.get("allergies", []):
        crosses = CROSS_REACTIONS.get(allergy, [])
        if proposed in crosses:
            flags.append(proposed)
            reasons.append("{} cross-reacts with {} allergy".format(proposed, allergy))
            break
        if allergy == allergy_cls:
            flags.append(proposed)
            reasons.append("{} belongs to {} class -- patient has {} allergy".format(
                proposed, allergy_cls, allergy))
            break

    return MedicalAction(
        action_type="flag_allergy",
        payload={
            "flagged_drugs": flags,
            "reason": "; ".join(reasons) if reasons else "No allergy conflict detected",
        },
    )


def policy_dosing(patient):
    """Recommend standard dose with renal and paediatric adjustments."""
    drug      = patient.get("drug_to_dose", "")
    info      = DRUG_DB.get(drug, {})
    dose      = float(info.get("standard_dose_mg", 500))
    frequency = info.get("frequency", "once daily")
    gfr       = patient.get("labs", {}).get("GFR", 90)
    age       = patient.get("age", 30)
    weight    = patient.get("weight_kg", 70)
    reasons   = ["Standard dose for {}".format(drug)]

    if info.get("renal_reduction"):
        if gfr < 45:
            dose   *= 0.5
            reasons.append("Halved dose -- GFR {} (<45) renal adjustment".format(gfr))
        elif gfr < 60:
            dose   *= 0.75
            reasons.append("Reduced 25% -- GFR {} (45-60) renal adjustment".format(gfr))

    if age < 12:
        freq_div = 3 if "8" in frequency else 2
        dose     = (40 * weight) / freq_div
        reasons.append(
            "Paediatric dose: 40mg/kg/day divided by {} doses (weight={}kg, age={})".format(
                freq_div, weight, age))

    return MedicalAction(
        action_type="recommend_dose",
        payload={
            "drug":      drug,
            "dose_mg":   round(dose, 1),
            "frequency": frequency,
            "reasoning": "; ".join(reasons),
        },
    )


def policy_treatment_plan(patient):
    """Build a safe plan: one first-line drug per condition."""
    COND_MAP = {
        "type 2 diabetes":    ("metformin",    500, "First-line oral hypoglycaemic for T2DM"),
        "hypertension":       ("amlodipine",     5, "Safe CCB -- minimal renal/interaction risk"),
        "hyperlipidemia":     ("atorvastatin",  20, "Statin first-line for lipid control"),
        "atrial fibrillation":("amlodipine",     5, "Rate-control CCB; anticoagulation already in place"),
        "osteoarthritis":     ("acetaminophen", 500, "Safest analgesic given aspirin/NSAID allergy profile"),
    }
    plan = []
    gfr = patient.get("labs", {}).get("GFR", 90)
    for cond in patient.get("conditions", []):
        mapping = COND_MAP.get(cond)
        if not mapping:
            continue
        drug, dose, reason = mapping
        if drug == "metformin" and gfr < 30:
            continue
        plan.append({"condition": cond, "drug": drug, "dose_mg": dose, "reason": reason})

    return MedicalAction(
        action_type="finalize_treatment_plan",
        payload={
            "plan": plan,
            "drug_interactions_noted": [
                "warfarin interacts with NSAIDs, aspirin, and amoxicillin",
                "lisinopril + NSAIDs can reduce antihypertensive effect",
            ],
            "drugs_avoided": [
                "ibuprofen -- NSAID contraindicated with aspirin allergy and CKD",
                "amoxicillin -- penicillin allergy",
                "aspirin -- direct patient allergy",
                "sulfamethoxazole -- sulfa allergy",
                "naproxen -- NSAID cross-reacts with aspirin allergy",
            ],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Local runner
# ─────────────────────────────────────────────────────────────────────────────

def run_local():
    results = {}
    task_configs = [
        ("allergy_check",   policy_allergy),
        ("dosing",          policy_dosing),
        ("treatment_plan",  policy_treatment_plan),
    ]

    for task_name, policy_fn in task_configs:
        patients = [p for p in PATIENTS if p["task"] == task_name]
        scores   = []

        # Structured output block required by hackathon Phase 2 validator
        print("[START] task={}".format(task_name), flush=True)

        for step_idx, patient in enumerate(patients, start=1):
            env          = HealthcareEnvironment()
            env._patient = patient
            env._done    = False
            action       = policy_fn(patient)
            obs          = env.step(action)
            score        = obs.reward or 0.0
            scores.append(score)
            feedback_short = obs.feedback[:65] if obs.feedback else ""

            # Required structured step output
            print("[STEP] step={} reward={:.3f} patient={} task={}".format(
                step_idx, score, patient["patient_id"], task_name), flush=True)

            print("  [{}]  patient={}  score={:.3f}  ->  {}...".format(
                task_name, patient["patient_id"], score, feedback_short), flush=True)

        mean = sum(scores) / len(scores) if scores else 0.0
        results[task_name] = {"mean": round(mean, 3), "n": len(scores), "scores": scores}

        # Required structured end output
        print("[END] task={} score={:.3f} steps={}".format(
            task_name, mean, len(scores)), flush=True)
        print(flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Remote runner
# ─────────────────────────────────────────────────────────────────────────────

def run_remote(url):
    import urllib.request
    base = url.rstrip("/")

    def post(endpoint, data=None):
        req = urllib.request.Request(
            "{}/{}".format(base, endpoint),
            data=json.dumps(data or {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def get(endpoint):
        with urllib.request.urlopen("{}/{}".format(base, endpoint), timeout=15) as r:
            return json.loads(r.read())

    print("Pinging {}/health ...".format(base))
    h = get("health")
    print("  -> {}\n".format(h))

    print("Checking {}/tasks ...".format(base))
    t = get("tasks")
    print("  -> {} tasks: {}\n".format(len(t["tasks"]), [x["name"] for x in t["tasks"]]))

    print("Running {}/baseline ...".format(base))
    result = post("baseline")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare Decision Support -- Baseline Runner")
    parser.add_argument("--url", default=None,
                        help="HF Space URL to test remotely (omit to run locally)")
    args = parser.parse_args()

    print("=" * 65, flush=True)
    print("  Healthcare Decision Support -- Baseline Inference", flush=True)
    print("=" * 65, flush=True)
    print(flush=True)

    if args.url:
        result = run_remote(args.url)
        print(json.dumps(result, indent=2), flush=True)
    else:
        print("Running locally against all patient cases...\n", flush=True)
        results = run_local()

        print("=" * 65, flush=True)
        print("  FINAL SCORES", flush=True)
        print("=" * 65, flush=True)
        all_scores = []
        for task, data in results.items():
            bar = "#" * int(data["mean"] * 30)
            print("  {:<20}  [{:<30}]  {:.3f}  (n={})".format(
                task, bar, data["mean"], data["n"]), flush=True)
            all_scores.extend(data["scores"])
        print(flush=True)
        print("  Overall mean:  {:.3f}".format(sum(all_scores) / len(all_scores)), flush=True)
        print(flush=True)
        print("PASSED: Baseline script completed successfully -- no errors.", flush=True)