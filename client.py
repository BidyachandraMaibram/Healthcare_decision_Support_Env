"""
Healthcare Decision Support Environment — Client

Connects to a deployed HF Space (or local server) and provides
the standard OpenEnv interface: reset() / step() / state()

Usage:
    from healthcareenv import HealthcareEnv, MedicalAction

    with HealthcareEnv(base_url="https://YOUR_USERNAME-healthcareenv.hf.space").sync() as env:
        result = env.reset()
        print(result.observation.patient)

        result = env.step(MedicalAction(
            action_type="flag_allergy",
            payload={"flagged_drugs": ["amoxicillin"], "reason": "penicillin cross-reaction"}
        ))
        print(result.reward)
"""
import json
import urllib.request
import urllib.error
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from models import MedicalAction, MedicalObservation, MedicalState


# ─────────────────────────────────────────────
# Step result wrapper
# ─────────────────────────────────────────────

@dataclass
class StepResult:
    observation: MedicalObservation
    reward: Optional[float]
    done: bool


# ─────────────────────────────────────────────
# Sync wrapper (used with .sync())
# ─────────────────────────────────────────────

class SyncHealthcareEnv:
    """Synchronous wrapper around HealthcareEnv for use in notebooks and scripts."""

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, data: dict = None) -> dict:
        url = f"{self._base_url}/{endpoint}"
        body = json.dumps(data or {}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode()}")

    def _get(self, endpoint: str) -> dict:
        url = f"{self._base_url}/{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode()}")

    def _parse_observation(self, data: dict) -> MedicalObservation:
        return MedicalObservation(
            done          = data["done"],
            reward        = data.get("reward"),
            info_state    = data.get("info_state", []),
            patient       = data.get("patient", {}),
            task          = data.get("task", ""),
            legal_actions = data.get("legal_actions", []),
            feedback      = data.get("feedback", ""),
            metadata      = data.get("metadata", {}),
        )

    def reset(self, task_filter: Optional[str] = None) -> StepResult:
        """Start a new episode. Optionally filter by task."""
        payload = {}
        if task_filter:
            payload["task_filter"] = task_filter
        data = self._post("reset", payload)
        obs = self._parse_observation(data)
        return StepResult(observation=obs, reward=None, done=False)

    def step(self, action: MedicalAction) -> StepResult:
        """Submit an action and get back a graded observation."""
        data = self._post("step", {
            "action_type": action.action_type,
            "payload":     action.payload,
            "metadata":    action.metadata,
        })
        obs = self._parse_observation(data)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def state(self) -> MedicalState:
        """Get current episode metadata."""
        data = self._get("state")
        return MedicalState(
            episode_id   = data.get("episode_id"),
            step_count   = data.get("step_count", 0),
            task         = data.get("task", ""),
            difficulty   = data.get("difficulty", ""),
            patient_id   = data.get("patient_id", ""),
            grader_score = data.get("grader_score"),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ─────────────────────────────────────────────
# Main client class
# ─────────────────────────────────────────────

class HealthcareEnv:
    """
    Healthcare Decision Support Environment client.

    Connects to a deployed HF Space or local server.

    Example:
        with HealthcareEnv(base_url="https://user-healthcareenv.hf.space").sync() as env:
            result = env.reset()
            result = env.step(MedicalAction(...))
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def sync(self) -> SyncHealthcareEnv:
        """Return a synchronous client (for notebooks and scripts)."""
        return SyncHealthcareEnv(self._base_url)
