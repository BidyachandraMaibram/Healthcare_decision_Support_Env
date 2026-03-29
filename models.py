"""
Medical Decision Support Environment — Data Models
Action, Observation, and State types.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ─────────────────────────────────────────────
# Action  (what the agent sends to step())
# ─────────────────────────────────────────────

@dataclass
class MedicalAction:
    """
    One action the agent takes in a medical decision episode.

    action_type  : one of:
        "flag_allergy"           – Task 1: report a drug-allergy conflict
        "recommend_dose"         – Task 2: recommend a dose with reasoning
        "finalize_treatment_plan"– Task 3: submit a full multi-condition plan
    payload      : dict with action-specific fields (see README)
    """
    action_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Observation  (what the agent receives back)
# ─────────────────────────────────────────────

@dataclass
class MedicalObservation:
    """
    Everything the agent can see about the current patient case.

    info_state   : flat float list encoding patient data (for RL agents)
    patient      : human-readable dict with the full patient profile
    task         : which task is active ('allergy_check', 'dosing', 'treatment_plan')
    legal_actions: list of allowed action_type strings at this step
    feedback     : text feedback from the last action (empty on reset)
    done         : True when the episode has ended
    reward       : reward signal (None until episode ends or partial signal given)
    metadata     : extra info (grader details, step count, etc.)
    """
    done: bool
    reward: Optional[float]
    info_state: List[float]
    patient: Dict[str, Any]
    task: str
    legal_actions: List[str]
    feedback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# State  (episode metadata, returned by state())
# ─────────────────────────────────────────────

@dataclass
class MedicalState:
    """Episode-level metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task: str = ""
    difficulty: str = ""        # "easy", "medium", "hard"
    patient_id: str = ""
    grader_score: Optional[float] = None
