"""
Healthcare Decision Support Environment — Core Logic

Implements reset() / step() / state as required by OpenEnv spec.
Three tasks:
  Task 1 (easy)   — allergy_check        : flag drug-allergy conflicts
  Task 2 (medium) — dosing               : recommend a safe dose
  Task 3 (hard)   — treatment_plan       : multi-condition treatment plan
"""
import uuid
import random
from typing import Optional

try:
    from ..models import MedicalAction, MedicalObservation, MedicalState
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import MedicalAction, MedicalObservation, MedicalState

from .data import PATIENTS, encode_patient
from .graders import run_grader


LEGAL_ACTIONS_BY_TASK = {
    "allergy_check":  ["flag_allergy"],
    "dosing":         ["recommend_dose"],
    "treatment_plan": ["finalize_treatment_plan"],
}


class HealthcareEnvironment:
    """
    Healthcare Decision Support OpenEnv environment.

    Episode flow:
      1. reset()  → randomly select a patient case, return observation
      2. step()   → agent submits ONE action, grader scores it, episode ends
      3. state    → returns episode metadata (property)
    """

    def __init__(self, task_filter: Optional[str] = None):
        """
        Args:
            task_filter: restrict episodes to one task type.
                         Options: 'allergy_check', 'dosing', 'treatment_plan'
        """
        self._task_filter = task_filter
        self._patient     = None
        self._state       = MedicalState()
        self._done        = False

    # ─────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────

    def reset(self) -> MedicalObservation:
        """Select a random patient case and return the initial observation."""
        pool = PATIENTS
        if self._task_filter:
            pool = [p for p in PATIENTS if p["task"] == self._task_filter]
        if not pool:
            pool = PATIENTS

        self._patient = random.choice(pool)
        self._done    = False

        self._state = MedicalState(
            episode_id   = str(uuid.uuid4()),
            step_count   = 0,
            task         = self._patient["task"],
            difficulty   = self._patient["difficulty"],
            patient_id   = self._patient["patient_id"],
            grader_score = None,
        )

        return self._build_observation(
            done     = False,
            reward   = None,
            feedback = "New episode started. Review the patient profile and take an action.",
        )

    # ─────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────

    def step(self, action: MedicalAction) -> MedicalObservation:
        """
        Process the agent's action. Each episode ends after ONE action.
        Partial credit is built into the graders so reward is always in [0.0, 1.0].
        """
        if self._patient is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._state.step_count += 1

        # Validate action type
        allowed = LEGAL_ACTIONS_BY_TASK.get(self._patient["task"], [])
        if action.action_type not in allowed:
            return self._build_observation(
                done     = False,
                reward   = 0.01,
                feedback = (
                    f"Invalid action '{action.action_type}' for task "
                    f"'{self._patient['task']}'. Allowed: {allowed}"
                ),
            )

        # Run grader
        result = run_grader(self._patient, action.payload)
        score  = result["score"]

        self._done = True
        self._state.grader_score = score

        return self._build_observation(
            done     = True,
            reward   = score,
            feedback = result["feedback"],
        )

    # ─────────────────────────────────────────────
    # state  (property)
    # ─────────────────────────────────────────────

    @property
    def state(self) -> MedicalState:
        return self._state

    # ─────────────────────────────────────────────
    # Internal helper
    # ─────────────────────────────────────────────

    def _build_observation(self, done: bool, reward, feedback: str) -> MedicalObservation:
        patient = self._patient or {}
        task    = patient.get("task", "")

        # Expose only what the agent needs — no ground truth leaked
        safe_patient = {
            "patient_id":   patient.get("patient_id", ""),
            "age":          patient.get("age"),
            "weight_kg":    patient.get("weight_kg"),
            "sex":          patient.get("sex"),
            "allergies":    patient.get("allergies", []),
            "conditions":   patient.get("conditions", []),
            "current_meds": patient.get("current_meds", []),
            "labs":         patient.get("labs", {}),
        }

        # Task-specific fields
        if task == "allergy_check":
            safe_patient["proposed_drug"] = patient.get("proposed_drug", "")
        elif task == "dosing":
            safe_patient["drug_to_dose"] = patient.get("drug_to_dose", "")

        return MedicalObservation(
            done          = done,
            reward        = reward,
            info_state    = encode_patient(patient) if patient else [],
            patient       = safe_patient,
            task          = task,
            legal_actions = LEGAL_ACTIONS_BY_TASK.get(task, []),
            feedback      = feedback,
            metadata      = {
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "difficulty": patient.get("difficulty", ""),
            },
        )
