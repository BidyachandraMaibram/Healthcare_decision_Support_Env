"""
Healthcare Decision Support Environment — FastAPI Server

Required endpoints (OpenEnv competition spec):
  POST /reset       → start new episode
  POST /step        → submit action, get graded result
  GET  /state       → episode metadata
  GET  /health      → liveness check
  GET  /tasks       → task list + action schema
  POST /grader      → run grader on current episode
  POST /baseline    → run baseline agent, return scores
"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.healthcareenv_environment import HealthcareEnvironment
from server.data import PATIENTS, DRUG_DB, CROSS_REACTIONS
from server.graders import run_grader
from models import MedicalAction

app = FastAPI(
    title="Healthcare Decision Support Environment",
    description="OpenEnv environment for AI agent evaluation on clinical decision tasks.",
    version="1.0.0",
)

_env = HealthcareEnvironment()


class ResetRequest(BaseModel):
    task_filter: Optional[str] = None

class StepRequest(BaseModel):
    action_type: str
    payload: dict = {}
    metadata: dict = {}


def _obs_dict(obs) -> dict:
    return {
        "done":          obs.done,
        "reward":        obs.reward,
        "info_state":    obs.info_state,
        "patient":       obs.patient,
        "task":          obs.task,
        "legal_actions": obs.legal_actions,
        "feedback":      obs.feedback,
        "metadata":      obs.metadata,
    }


@app.get("/health")
def health():
    return {"status": "ok", "environment": "healthcare_decision_support"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    if req.task_filter:
        _env = HealthcareEnvironment(task_filter=req.task_filter)
    obs = _env.reset()
    return _obs_dict(obs)


@app.post("/step")
def step(req: StepRequest):
    action = MedicalAction(
        action_type=req.action_type,
        payload=req.payload,
        metadata=req.metadata,
    )
    try:
        obs = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _obs_dict(obs)


@app.get("/state")
def state():
    s = _env.state
    return {
        "episode_id":   s.episode_id,
        "step_count":   s.step_count,
        "task":         s.task,
        "difficulty":   s.difficulty,
        "patient_id":   s.patient_id,
        "grader_score": s.grader_score,
    }


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "allergy_check", "difficulty": "easy",
                "description": "Identify drug-allergy conflicts given a patient allergy profile and a proposed drug.",
                "action_schema": {
                    "action_type": "flag_allergy",
                    "payload": {
                        "flagged_drugs": "list[str] — drug names conflicting with patient allergies",
                        "reason":        "str — explanation of each conflict",
                    },
                },
            },
            {
                "name": "dosing", "difficulty": "medium",
                "description": "Recommend a safe dose adjusted for patient weight, renal function, and age.",
                "action_schema": {
                    "action_type": "recommend_dose",
                    "payload": {
                        "drug":      "str",
                        "dose_mg":   "float",
                        "frequency": "str — e.g. 'twice daily'",
                        "reasoning": "str — clinical justification",
                    },
                },
            },
            {
                "name": "treatment_plan", "difficulty": "hard",
                "description": "Develop a safe multi-drug plan for a patient with 2-3 comorbidities.",
                "action_schema": {
                    "action_type": "finalize_treatment_plan",
                    "payload": {
                        "plan":                    "list[{condition, drug, dose_mg, reason}]",
                        "drug_interactions_noted": "list[str]",
                        "drugs_avoided":           "list[str]",
                    },
                },
            },
        ]
    }


@app.post("/grader")
def grader(req: StepRequest):
    patient = _env._patient
    if patient is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return run_grader(patient, req.payload)


@app.post("/baseline")
def baseline():
    scores = {}

    # Task 1 allergy_check
    t1 = []
    for patient in [p for p in PATIENTS if p["task"] == "allergy_check"]:
        env = HealthcareEnvironment()
        env._patient = patient
        env._done = False
        proposed = patient.get("proposed_drug", "")
        info = DRUG_DB.get(proposed, {})
        allergy_cls = info.get("allergy_class")
        flags = []
        for allergy in patient.get("allergies", []):
            if proposed in CROSS_REACTIONS.get(allergy, []) or allergy == allergy_cls:
                flags.append(proposed)
                break
        obs = env.step(MedicalAction(action_type="flag_allergy",
            payload={"flagged_drugs": flags, "reason": "Baseline allergy cross-reaction check"}))
        t1.append(obs.reward or 0.0)
    scores["allergy_check"] = {"mean": round(sum(t1)/len(t1), 3), "scores": t1, "n": len(t1)}

    # Task 2 dosing
    t2 = []
    for patient in [p for p in PATIENTS if p["task"] == "dosing"]:
        env = HealthcareEnvironment()
        env._patient = patient
        env._done = False
        drug = patient.get("drug_to_dose", "")
        info = DRUG_DB.get(drug, {})
        dose = float(info.get("standard_dose_mg", 500))
        gfr = patient.get("labs", {}).get("GFR", 90)
        age = patient.get("age", 30)
        weight = patient.get("weight_kg", 70)
        reasoning = f"Standard dose for {drug}. GFR={gfr}."
        if info.get("renal_reduction") and gfr < 45:
            dose *= 0.5; reasoning += " Halved for low GFR (renal adjustment)."
        elif info.get("renal_reduction") and gfr < 60:
            dose *= 0.75; reasoning += " Reduced 25% for borderline renal function."
        if age < 12:
            fd = 3 if "8" in info.get("frequency", "") else 2
            dose = (40 * weight) / fd
            reasoning += f" Pediatric: 40mg/kg/day divided by {fd} doses (weight={weight}kg)."
        obs = env.step(MedicalAction(action_type="recommend_dose",
            payload={"drug": drug, "dose_mg": round(dose, 1),
                     "frequency": info.get("frequency", "once daily"), "reasoning": reasoning}))
        t2.append(obs.reward or 0.0)
    scores["dosing"] = {"mean": round(sum(t2)/len(t2), 3), "scores": t2, "n": len(t2)}

    # Task 3 treatment_plan
    COND_MAP = {
        "type 2 diabetes":    ("metformin",    500, "First-line oral hypoglycaemic for T2DM"),
        "hypertension":       ("amlodipine",     5, "Calcium channel blocker, safe across comorbidities"),
        "hyperlipidemia":     ("atorvastatin",  20, "Statin first-line for lipid control"),
        "atrial fibrillation":("amlodipine",     5, "Rate-control support; anticoag already prescribed"),
        "osteoarthritis":     ("acetaminophen", 500, "Safest analgesic given aspirin/NSAID allergy"),
    }
    t3 = []
    for patient in [p for p in PATIENTS if p["task"] == "treatment_plan"]:
        env = HealthcareEnvironment()
        env._patient = patient
        env._done = False
        plan = [{"condition": c, "drug": m[0], "dose_mg": m[1], "reason": m[2]}
                for c in patient.get("conditions", []) if (m := COND_MAP.get(c))]
        obs = env.step(MedicalAction(action_type="finalize_treatment_plan",
            payload={
                "plan": plan,
                "drug_interactions_noted": [
                    "warfarin interacts with NSAIDs, aspirin, and amoxicillin",
                    "lisinopril + NSAIDs reduces antihypertensive effect",
                ],
                "drugs_avoided": [
                    "ibuprofen — NSAID, contraindicated with aspirin allergy and CKD",
                    "amoxicillin — penicillin allergy",
                    "aspirin — direct allergy",
                    "sulfamethoxazole — sulfa allergy",
                    "naproxen — NSAID cross-reacts with aspirin allergy",
                ],
            }))
        t3.append(obs.reward or 0.0)
    scores["treatment_plan"] = {"mean": round(sum(t3)/len(t3), 3), "scores": t3, "n": len(t3)}

    all_s = t1 + t2 + t3
    return {"baseline_scores": scores, "overall_mean": round(sum(all_s)/len(all_s), 3)}


def main():
    """Entry point for openenv server startup."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
