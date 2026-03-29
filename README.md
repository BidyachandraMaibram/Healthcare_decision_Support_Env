---
title: Healthcare Decision Support Environment
emoji: 🏥
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - medical
  - agent-evaluation
---

# 🏥 Healthcare Decision Support Environment

An **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** environment that tests AI agents on real-world clinical decision-making.

Each episode presents the agent with a patient case — allergies, conditions, medications, lab results — and asks it to make a safe, clinically appropriate decision.

---

## The Three Tasks

| # | Task | Difficulty | What the agent must do |
|---|------|-----------|------------------------|
| 1 | `allergy_check` | 🟢 Easy | Given a proposed drug and the patient's allergy profile, flag any conflicts (direct + cross-reactions) |
| 2 | `dosing` | 🟡 Medium | Recommend a safe dose adjusted for weight, renal function (GFR), and age |
| 3 | `treatment_plan` | 🔴 Hard | Build a full treatment plan for a patient with 2–3 comorbidities, avoiding drug-drug interactions |

---

## Quick Start

```python
import requests

BASE = "https://YOUR_USERNAME-healthcareenv.hf.space"

# 1. Start a new episode
obs = requests.post(f"{BASE}/reset").json()
print(obs["task"])          # e.g. "allergy_check"
print(obs["patient"])       # patient profile (age, allergies, conditions, labs...)
print(obs["legal_actions"]) # ["flag_allergy"]

# 2. Submit an action
result = requests.post(f"{BASE}/step", json={
    "action_type": "flag_allergy",
    "payload": {
        "flagged_drugs": ["amoxicillin"],
        "reason": "Patient has penicillin allergy; amoxicillin cross-reacts"
    }
}).json()

print(result["reward"])    # 0.0 – 1.0
print(result["feedback"])  # "Correct — ['amoxicillin'] flagged. Reason: penicillin cross-reaction"
print(result["done"])      # True (episode ends after one action)
```

---

## Action Schemas

### Task 1 — `flag_allergy`
```json
{
  "action_type": "flag_allergy",
  "payload": {
    "flagged_drugs": ["amoxicillin"],
    "reason": "Amoxicillin is a penicillin-class antibiotic — cross-reacts with penicillin allergy"
  }
}
```

### Task 2 — `recommend_dose`
```json
{
  "action_type": "recommend_dose",
  "payload": {
    "drug": "lisinopril",
    "dose_mg": 5,
    "frequency": "once daily",
    "reasoning": "Starting at 5mg due to CKD stage 3 (GFR 35) — renal dose reduction required"
  }
}
```

### Task 3 — `finalize_treatment_plan`
```json
{
  "action_type": "finalize_treatment_plan",
  "payload": {
    "plan": [
      {"condition": "type 2 diabetes",  "drug": "metformin",    "dose_mg": 500, "reason": "First-line T2DM — GFR adequate"},
      {"condition": "hypertension",     "drug": "amlodipine",   "dose_mg": 5,   "reason": "Safe CCB, no renal contraindication"},
      {"condition": "hyperlipidemia",   "drug": "atorvastatin", "dose_mg": 20,  "reason": "Statin first-line for lipid control"}
    ],
    "drug_interactions_noted": ["NSAIDs reduce lisinopril efficacy and worsen CKD"],
    "drugs_avoided": ["ibuprofen — NSAID contraindicated with CKD", "amoxicillin — penicillin allergy"]
  }
}
```

---

## Observation Space

Every `reset()` and `step()` response contains:

| Field | Type | Description |
|-------|------|-------------|
| `patient` | dict | Age, weight, sex, allergies, conditions, medications, lab values |
| `task` | str | Active task: `allergy_check`, `dosing`, or `treatment_plan` |
| `legal_actions` | list[str] | Valid `action_type` values right now |
| `info_state` | float[28] | Flat encoded vector for RL agents |
| `feedback` | str | Grader explanation of your last action |
| `reward` | float | Score 0.0–1.0 (None before first step) |
| `done` | bool | True when episode ends |

### `info_state` encoding (28 floats)
```
[0]    age / 100
[1]    weight_kg / 150
[2]    sex  (1.0=F, 0.0=M)
[3]    task (0.0=allergy, 0.5=dosing, 1.0=treatment)
[4]    difficulty (0.0=easy, 0.5=medium, 1.0=hard)
[5-12] allergy one-hot: penicillin, sulfa, aspirin, codeine, latex, shellfish, cephalosporin, nsaid
[13-24] condition one-hot: diabetes, hypertension, hyperlipidemia, CKD3, CKD4, AFib, OA, UTI, bacterial, otitis, arthritis, peptic ulcer
[25]   GFR / 130
[26]   ALT / 200
[27]   AST / 200
```

---

## Scoring

### Task 1 — Allergy check
| Score | Condition |
|-------|-----------|
| **1.0** | All conflicts correctly identified (or correctly says none) |
| **0.5** | Found the direct conflict but missed cross-reactions |
| **0.0** | Missed the conflict entirely, or falsely flagged a safe drug |

### Task 2 — Dosing (four components × 0.25 each)
| Component | What is checked |
|-----------|----------------|
| Dose in range | Is the dose within the safe window for this patient? |
| Frequency correct | Does the frequency match expected? (must be non-empty) |
| Renal note | Mentions kidney function when GFR < 60? |
| Reasoning quality | Explains at least 2 key clinical factors? |

### Task 3 — Treatment plan (five components × 0.2 each)
| Component | What is checked |
|-----------|----------------|
| Drug coverage | Right first-line drug for each condition? |
| Dangerous drugs avoided | None of the contraindicated drugs prescribed? |
| Interactions noted | Drug-drug interactions mentioned? |
| Anticoag awareness | If patient is on warfarin, was it acknowledged? |
| Reasoning quality | Every plan item has a non-trivial explanation? |

---

## Baseline Scores

The built-in rule-based baseline agent scores:

| Task | Score |
|------|-------|
| `allergy_check` | **1.000** |
| `dosing` | **0.867** |
| `treatment_plan` | **1.000** |
| **Overall** | **0.956** |

Run it yourself:
```bash
python baseline.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/reset` | POST | Start new episode (optional: `task_filter`) |
| `/step` | POST | Submit action, get graded result |
| `/state` | GET | Episode metadata |
| `/tasks` | GET | Task list + action schemas |
| `/grader` | POST | Run grader on current episode |
| `/baseline` | POST | Run baseline agent, return all scores |
| `/docs` | GET | Interactive API docs (auto-generated) |

---

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline
python baseline.py

# Run pre-submission checks
python validate.py
```

## Docker

```bash
docker build -t healthcareenv .
docker run -p 7860:7860 healthcareenv
```
