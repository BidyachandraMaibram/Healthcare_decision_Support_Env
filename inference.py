"""
Healthcare Decision Support Environment — LLM Inference Script

Follows the official OpenEnv submission template exactly.

MANDATORY environment variables:
    API_BASE_URL     — OpenAI-compatible API endpoint (default provided)
    MODEL_NAME       — Model identifier (default provided)
    HF_TOKEN         — Hugging Face / API key (NO default)
    LOCAL_IMAGE_NAME — Docker image name if using from_docker_image() (optional)

STDOUT FORMAT (exact):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import HealthcareEnv
from models import MedicalAction

# ─────────────────────────────────────────────────────────────────────────────
# Environment variables
# Defaults ONLY for API_BASE_URL and MODEL_NAME — NOT for HF_TOKEN
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional, for from_docker_image()

HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://maibram1-healthcare-decision-support-env.hf.space")

TASKS                   = ["allergy_check", "dosing", "treatment_plan"]
BENCHMARK               = "healthcare_decision_support"
MAX_PATIENTS_PER_TASK   = 20      # safety cap; loop breaks on first repeated patient
SUCCESS_SCORE_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Structured stdout — exact format required by validator
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "allergy_check": textwrap.dedent("""
        You are a clinical pharmacist AI. Given a patient profile with allergies and a proposed drug,
        identify ALL allergy conflicts including direct matches AND cross-reactions.

        Known cross-reactions:
        - penicillin allergy → avoid: amoxicillin, ampicillin, piperacillin
        - sulfa allergy → avoid: sulfamethoxazole, trimethoprim-sulfamethoxazole
        - aspirin allergy → avoid: ibuprofen, naproxen, celecoxib (NSAIDs)
        - cephalosporin allergy → avoid: cefazolin, cephalexin, ceftriaxone
        - codeine allergy → avoid: morphine, oxycodone, hydrocodone

        Respond ONLY with valid JSON, no markdown, no explanation:
        {
          "action_type": "flag_allergy",
          "payload": {
            "flagged_drugs": ["drug_name"],
            "reason": "brief reason including the allergy class and cross-reaction"
          }
        }
        Use empty list if no conflict: "flagged_drugs": []
    """).strip(),

    "dosing": textwrap.dedent("""
        You are a clinical pharmacist AI. Given a patient profile and a drug to dose,
        recommend a safe dose adjusted for weight, age, and renal function (GFR).

        Key dosing rules:
        - GFR < 45: halve the standard dose (renal adjustment)
        - GFR 45-60: reduce standard dose by 25%
        - Age < 12: use pediatric dosing: 40 mg/kg/day divided by frequency
        - Always explain your reasoning including renal function and age factors

        Respond ONLY with valid JSON, no markdown, no explanation:
        {
          "action_type": "recommend_dose",
          "payload": {
            "drug": "drug_name",
            "dose_mg": 500.0,
            "frequency": "twice daily",
            "reasoning": "clinical reasoning mentioning GFR, age, weight as relevant"
          }
        }
    """).strip(),

    "treatment_plan": textwrap.dedent("""
        You are a senior clinical pharmacist AI. Given a patient with multiple conditions,
        create a complete safe treatment plan avoiding drug interactions and allergy conflicts.

        Guidelines:
        - Type 2 diabetes → metformin 500mg twice daily (avoid if GFR < 30)
        - Hypertension → amlodipine 5mg once daily OR lisinopril 10mg once daily
        - Hyperlipidemia → atorvastatin 20mg once daily
        - Osteoarthritis (with aspirin/NSAID allergy) → acetaminophen 500mg (NOT ibuprofen/naproxen)
        - Always note warfarin interactions if patient is on anticoagulant
        - Avoid drugs the patient is allergic to (direct AND cross-reactions)
        - Each plan item must have a clear clinical reason (>10 words)

        Respond ONLY with valid JSON, no markdown, no explanation:
        {
          "action_type": "finalize_treatment_plan",
          "payload": {
            "plan": [
              {"condition": "condition_name", "drug": "drug_name", "dose_mg": 500, "reason": "detailed clinical reason"}
            ],
            "drug_interactions_noted": ["interaction description"],
            "drugs_avoided": ["drug_name -- reason why avoided"]
          }
        }
    """).strip(),
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM call via OpenAI client
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, task: str, patient: dict) -> dict:
    """Call the LLM and return a parsed action dict."""
    user_content = f"Patient profile:\n{json.dumps(patient, indent=2)}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.1,
            max_tokens=600,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return _fallback_action(task, patient)


def _fallback_action(task: str, patient: dict) -> dict:
    """Deterministic rule-based fallback if LLM call fails."""
    if task == "allergy_check":
        return {
            "action_type": "flag_allergy",
            "payload": {"flagged_drugs": [], "reason": "fallback: no conflict detected"},
        }
    elif task == "dosing":
        return {
            "action_type": "recommend_dose",
            "payload": {
                "drug":      patient.get("drug_to_dose", "unknown"),
                "dose_mg":   500.0,
                "frequency": "once daily",
                "reasoning": "fallback: standard dose, check renal and age adjustments",
            },
        }
    else:
        return {
            "action_type": "finalize_treatment_plan",
            "payload": {
                "plan": [],
                "drug_interactions_noted": [],
                "drugs_avoided": [],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run one task — emits [START] / [STEP]* / [END]
# ─────────────────────────────────────────────────────────────────────────────

async def run_task(client: OpenAI, base_url: str, task_name: str) -> None:
    rewards:      List[float] = []
    steps_taken:  int         = 0
    success:      bool        = False
    score:        float       = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = HealthcareEnv(base_url=base_url).sync()
    seen_patients: set = set()

    try:
        for _ in range(MAX_PATIENTS_PER_TASK):

            # Reset for this task
            try:
                result = env.reset(task_filter=task_name)
            except Exception as exc:
                steps_taken += 1
                log_step(step=steps_taken, action="reset()", reward=0.0,
                         done=True, error=str(exc)[:120])
                rewards.append(0.0)
                break

            patient    = result.observation.patient
            patient_id = patient.get("patient_id", f"unknown_{steps_taken}")

            # Stop when we've seen all patients (full cycle)
            if patient_id in seen_patients:
                break
            seen_patients.add(patient_id)
            steps_taken += 1

            # Get LLM action
            action_dict  = get_llm_action(client, task_name, patient)
            payload_str  = json.dumps(action_dict.get("payload", {}))[:80]
            action_label = f"{action_dict.get('action_type','?')}({payload_str})"

            # Submit to environment
            error_msg: Optional[str] = None
            reward = 0.0
            done   = True
            try:
                step_result = env.step(MedicalAction(
                    action_type=action_dict.get("action_type", ""),
                    payload=action_dict.get("payload", {}),
                ))
                reward = step_result.reward or 0.0
                done   = step_result.done
            except Exception as exc:
                error_msg = str(exc)[:120]

            rewards.append(reward)
            log_step(step=steps_taken, action=action_label,
                     reward=reward, done=done, error=error_msg)

        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "hf_placeholder")
    base_url = HF_SPACE_URL

    print("=" * 65, flush=True)
    print("  Healthcare Decision Support — LLM Inference", flush=True)
    print("=" * 65, flush=True)
    print(f"  API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"  MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(f"  HF Space URL : {base_url}", flush=True)
    print(flush=True)

    for task in TASKS:
        await run_task(client, base_url, task)
        print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())
