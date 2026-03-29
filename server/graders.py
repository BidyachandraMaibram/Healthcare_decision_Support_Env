"""
Healthcare Decision Support Environment — Graders

Three deterministic graders, one per task.
Each returns a score in [0.0, 1.0] with partial credit.
Scores VARY based on answer quality — no fixed/constant return values.
"""
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Allergy Safety Check  (easy)
#
# Scoring:
#   1.0  — correctly identifies ALL conflicts (or correctly says none)
#   0.5  — finds the direct conflict but misses cross-reactions
#   0.0  — misses the conflict entirely OR falsely flags a safe drug
# ─────────────────────────────────────────────────────────────────────────────

def grade_allergy_check(patient: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    flagged  = [d.lower().strip() for d in payload.get("flagged_drugs", [])]
    correct  = [d.lower().strip() for d in patient.get("correct_flags", [])]
    proposed = patient.get("proposed_drug", "").lower().strip()

    # Case: no conflict expected
    if not correct:
        if not flagged:
            return {"score": 1.0, "max_score": 1.0,
                    "feedback": "Correct — no allergy conflict exists for this patient."}
        return {"score": 0.0, "max_score": 1.0,
                "feedback": f"Incorrect — falsely flagged {flagged}. No conflict exists for this patient."}

    correct_set = set(correct)
    flagged_set = set(flagged)

    # Full marks: all correct conflicts flagged, no false positives
    if proposed in flagged_set and correct_set <= flagged_set and not (flagged_set - correct_set):
        return {"score": 1.0, "max_score": 1.0,
                "feedback": f"Correct — {flagged} flagged. Reason: {patient.get('correct_reason', '')}"}

    # Partial: found the direct drug but missed some cross-reactions
    if proposed in flagged_set:
        missed = list(correct_set - flagged_set)
        return {"score": 0.5, "max_score": 1.0,
                "feedback": f"Partial — flagged {proposed} but missed cross-reactions: {missed}"}

    # No marks: missed the conflict
    return {"score": 0.0, "max_score": 1.0,
            "feedback": f"Incorrect — {proposed} conflicts with patient allergies but was not flagged."}


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Dosing Recommendation  (medium)
#
# Four components, each worth 0.25:
#   [A] Dose is within the safe range for this patient
#   [B] Frequency matches expected (must be non-empty and correct)
#   [C] Renal function mentioned when GFR < 60  (free credit if not needed)
#   [D] Clinical reasoning mentions key factors
# ─────────────────────────────────────────────────────────────────────────────

def grade_dosing(patient: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    dose_mg   = float(payload.get("dose_mg", 0))
    frequency = payload.get("frequency", "").lower().strip()
    reasoning = payload.get("reasoning", "").lower().strip()

    dose_range    = patient.get("dose_range_ok", [0, 9999])
    expected_freq = patient.get("correct_frequency", "").lower().strip()
    key_reason    = patient.get("key_reason", "").lower().strip()
    gfr           = patient.get("labs", {}).get("GFR", 90)

    score = 0.0
    parts = []

    # [A] Dose in safe range
    if dose_mg > 0 and dose_range[0] <= dose_mg <= dose_range[1]:
        score += 0.25
        parts.append(f"Dose {dose_mg}mg ✓ (range {dose_range[0]}–{dose_range[1]}mg)")
    else:
        parts.append(f"Dose {dose_mg}mg ✗ (expected {dose_range[0]}–{dose_range[1]}mg)")

    # [B] Frequency — must be non-empty AND match
    if frequency and (expected_freq in frequency or frequency in expected_freq):
        score += 0.25
        parts.append(f"Frequency '{frequency}' ✓")
    else:
        parts.append(f"Frequency '{frequency or '(empty)'}' ✗ (expected '{expected_freq}')")

    # [C] Renal note when GFR < 60
    if gfr < 60:
        renal_kw = ["renal", "gfr", "kidney", "ckd", "creatinine", "clearance"]
        if reasoning and any(k in reasoning for k in renal_kw):
            score += 0.25
            parts.append(f"Renal consideration noted ✓ (GFR={gfr})")
        else:
            parts.append(f"Renal note missing ✗ (GFR={gfr} requires adjustment)")
    else:
        # Renal not needed — give credit only if reasoning is non-empty
        if reasoning:
            score += 0.25
        else:
            parts.append("Reasoning empty ✗")

    # [D] Clinical reasoning quality
    if reasoning:
        key_words = [w for w in key_reason.split() if len(w) > 4]
        matched   = sum(1 for w in key_words if w in reasoning)
        if matched >= 2:
            score += 0.25
            parts.append("Clinical reasoning ✓")
        elif matched == 1:
            score += 0.10
            parts.append("Clinical reasoning partial (mention more key factors)")
        else:
            parts.append("Clinical reasoning ✗ (key factors not mentioned)")
    else:
        parts.append("Reasoning empty ✗")

    return {"score": round(min(score, 1.0), 3), "max_score": 1.0, "feedback": " | ".join(parts)}


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Multi-Condition Treatment Plan  (hard)
#
# Five components, each worth 0.2:
#   [A] Correct drug classes present for each condition
#   [B] Dangerous / contraindicated drugs avoided
#   [C] Drug-drug interactions acknowledged
#   [D] Anticoagulant awareness (if patient is on warfarin)
#   [E] Reasoning quality — every plan item has a non-trivial explanation
# ─────────────────────────────────────────────────────────────────────────────

def grade_treatment_plan(patient: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    plan               = payload.get("plan", [])
    interactions_noted = [i.lower().strip() for i in payload.get("drug_interactions_noted", [])]
    drugs_avoided      = [d.lower().strip() for d in payload.get("drugs_avoided", [])]

    rubric         = patient.get("plan_rubric", {})
    all_plan_drugs = [item.get("drug", "").lower().strip() for item in plan]
    score = 0.0
    parts = []

    # [A] Correct drug coverage
    drug_keys = [k for k in rubric if k.endswith("_drug")]
    if drug_keys and plan:
        correct = sum(
            1 for k in drug_keys
            if any(d in all_plan_drugs for d in [x.lower() for x in rubric[k]])
        )
        score += 0.2 * (correct / len(drug_keys))
        parts.append(f"Drug coverage: {correct}/{len(drug_keys)} conditions ✓")
    elif drug_keys and not plan:
        parts.append("Drug coverage: plan is empty ✗")
    else:
        score += 0.2

    # [B] Dangerous drugs avoided
    avoid_list = [d.lower() for d in rubric.get("avoid_drugs", [])]
    if avoid_list:
        false_incl = [d for d in avoid_list if d in all_plan_drugs]
        avoided_ok  = len(avoid_list) - len(false_incl)
        score += 0.2 * (avoided_ok / len(avoid_list))
        if false_incl:
            parts.append(f"Dangerous drugs included ✗: {false_incl}")
        else:
            parts.append("No dangerous drugs in plan ✓")
    else:
        score += 0.2

    # [C] Interaction check noted
    if rubric.get("interaction_check"):
        has_interaction_note = (
            len(interactions_noted) > 0 or
            any("interaction" in item.get("reason", "").lower() for item in plan)
        )
        if has_interaction_note:
            score += 0.2
            parts.append("Drug interactions addressed ✓")
        else:
            parts.append("Drug interactions not mentioned ✗ (required for this case)")
    else:
        score += 0.2

    # [D] Anticoagulant awareness
    if rubric.get("anticoag_note"):
        on_warfarin = any("warfarin" in m.lower() for m in patient.get("current_meds", []))
        if on_warfarin:
            noted = (
                any("warfarin" in s for s in interactions_noted) or
                any("warfarin" in item.get("reason", "").lower() for item in plan)
            )
            if noted:
                score += 0.2
                parts.append("Warfarin / anticoagulation acknowledged ✓")
            else:
                parts.append("Patient is on warfarin — not mentioned ✗")
        else:
            score += 0.2
    else:
        score += 0.2

    # [E] Reasoning quality — penalise empty plan items
    if plan:
        with_reason = sum(1 for item in plan if len(item.get("reason", "").strip()) > 10)
        score += 0.2 * (with_reason / len(plan))
        parts.append(f"Reasoning: {with_reason}/{len(plan)} items explained ✓")
    else:
        parts.append("Plan is empty ✗")

    return {"score": round(min(score, 1.0), 3), "max_score": 1.0, "feedback": " | ".join(parts)}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_grader(patient: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    task = patient.get("task", "")
    if task == "allergy_check":
        return grade_allergy_check(patient, payload)
    elif task == "dosing":
        return grade_dosing(patient, payload)
    elif task == "treatment_plan":
        return grade_treatment_plan(patient, payload)
    return {"score": 0.0, "max_score": 1.0, "feedback": f"Unknown task: {task}"}
