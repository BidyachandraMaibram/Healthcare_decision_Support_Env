"""
Healthcare Decision Support Environment -- Pre-Submission Validator

Runs through every item on the competition pre-submission checklist.
Works on Windows, Mac, and Linux.

Usage:
    python validate.py                            # check local files only
    python validate.py --url https://USER-healthcareenv.hf.space
"""
import os
import sys
import json
import argparse
import subprocess

# Always run from this file's directory (fixes Windows path issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
results = []


def check(name, ok, detail=""):
    status = PASS if ok else FAIL
    line   = "{}  {}".format(status, name)
    if detail:
        line += "\n         {}".format(detail[:120])
    print(line)
    results.append((name, ok))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 1. Local file checks
# ─────────────────────────────────────────────────────────────────────────────

def check_local_files():
    print("\n-- Local file structure " + "-" * 40)
    required = [
        "Dockerfile",
        "requirements.txt",
        "openenv.yaml",
        "README.md",
        "baseline.py",
        "inference.py",
        "models.py",
        "client.py",
        "pyproject.toml",
        os.path.join("server", "__init__.py"),
        os.path.join("server", "app.py"),
        os.path.join("server", "healthcareenv_environment.py"),
        os.path.join("server", "data.py"),
        os.path.join("server", "graders.py"),
    ]
    for f in required:
        check("File exists: {}".format(f), os.path.exists(f))


def check_openenv_yaml():
    print("\n-- openenv.yaml " + "-" * 47)
    try:
        with open("openenv.yaml") as fh:
            content = fh.read()
        check("openenv.yaml is readable", True)
        for field in ["name", "version", "tasks", "endpoints", "observation", "action"]:
            check("openenv.yaml has '{}' field".format(field), field in content)
    except Exception as e:
        check("openenv.yaml readable", False, str(e))


def check_dockerfile():
    print("\n-- Dockerfile " + "-" * 49)
    try:
        with open("Dockerfile") as fh:
            content = fh.read()
        check("Dockerfile exists", True)
        check("Dockerfile EXPOSEs 7860",   "7860" in content)
        check("Dockerfile has uvicorn CMD", "uvicorn" in content)
        check("Dockerfile copies files",    "COPY" in content)
    except Exception as e:
        check("Dockerfile readable", False, str(e))


def check_imports():
    print("\n-- Python imports " + "-" * 45)
    try:
        from models import MedicalAction, MedicalObservation, MedicalState
        check("models.py imports OK", True)
    except Exception as e:
        check("models.py imports", False, str(e))

    try:
        from server.healthcareenv_environment import HealthcareEnvironment
        check("server/healthcareenv_environment.py imports OK", True)
    except Exception as e:
        check("server/healthcareenv_environment.py imports", False, str(e))

    try:
        from server.graders import run_grader
        check("server/graders.py imports OK", True)
    except Exception as e:
        check("server/graders.py imports", False, str(e))

    try:
        from server.data import PATIENTS, DRUG_DB
        check("server/data.py imports OK", True)
    except Exception as e:
        check("server/data.py imports", False, str(e))


def check_environment_logic():
    print("\n-- Environment logic " + "-" * 42)
    try:
        from server.healthcareenv_environment import HealthcareEnvironment
        from models import MedicalAction
        from server.data import PATIENTS

        for task in ["allergy_check", "dosing", "treatment_plan"]:
            env = HealthcareEnvironment(task_filter=task)
            obs = env.reset()
            check("reset() works for task='{}'".format(task),
                  obs is not None and obs.task == task)
            check("reset() returns info_state (len>=20)", len(obs.info_state) >= 20)
            check("reset() returns legal_actions", len(obs.legal_actions) > 0)
    except Exception as e:
        check("Environment reset()", False, str(e))

    try:
        from server.healthcareenv_environment import HealthcareEnvironment
        from models import MedicalAction
        from server.data import PATIENTS

        env = HealthcareEnvironment()
        p   = next(p for p in PATIENTS if p["task"] == "allergy_check")
        env._patient = p
        env._done    = False
        obs = env.step(MedicalAction(action_type="flag_allergy",
                                     payload={"flagged_drugs": [], "reason": "test"}))
        check("step() returns reward in [0,1]",
              obs.reward is not None and 0.0 <= obs.reward <= 1.0,
              "reward={}".format(obs.reward))
        check("step() sets done=True", obs.done)
    except Exception as e:
        check("Environment step()", False, str(e))


def check_graders():
    print("\n-- Graders " + "-" * 52)
    try:
        from server.graders import run_grader
        from server.data import PATIENTS

        task_payloads = {
            "allergy_check":  {"flagged_drugs": [], "reason": "none"},
            "dosing":         {"drug": "metformin", "dose_mg": 500,
                               "frequency": "twice daily", "reasoning": "standard dose"},
            "treatment_plan": {"plan": [], "drug_interactions_noted": [], "drugs_avoided": []},
        }
        for task, payload in task_payloads.items():
            patient = next(p for p in PATIENTS if p["task"] == task)
            result  = run_grader(patient, payload)
            score   = result.get("score", -1)
            check("grader '{}' returns score in [0.0, 1.0]".format(task),
                  0.0 <= score <= 1.0, "score={}".format(score))
            check("grader '{}' returns feedback string".format(task),
                  isinstance(result.get("feedback"), str))
    except Exception as e:
        check("Graders", False, str(e))

    # Determinism check
    try:
        from server.graders import run_grader
        from server.data import PATIENTS
        p       = next(p for p in PATIENTS if p["task"] == "allergy_check")
        payload = {"flagged_drugs": ["amoxicillin"], "reason": "test"}
        s1 = run_grader(p, payload)["score"]
        s2 = run_grader(p, payload)["score"]
        check("Graders are deterministic (same input -> same score)", s1 == s2)
    except Exception as e:
        check("Graders deterministic", False, str(e))

    # Variance check (must NOT always return same score)
    try:
        from server.graders import run_grader
        from server.data import PATIENTS
        p      = next(p for p in PATIENTS if p["task"] == "allergy_check")
        score1 = run_grader(p, {"flagged_drugs": ["amoxicillin"], "reason": "conflict"})["score"]
        score2 = run_grader(p, {"flagged_drugs": [], "reason": "no conflict"})["score"]
        check("Graders vary with input quality (not always same score)", score1 != score2,
              "score_good={} score_bad={}".format(score1, score2))
    except Exception as e:
        check("Graders variance", False, str(e))


def check_baseline_script():
    print("\n-- baseline.py " + "-" * 48)
    check("baseline.py exists", os.path.exists("baseline.py"))
    try:
        # cwd=BASE_DIR ensures Windows runs it from the right folder
        result = subprocess.run(
            [sys.executable, "baseline.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=BASE_DIR,
            encoding="utf-8",
            errors="replace",
        )
        ok = result.returncode == 0
        err_detail = result.stderr.strip()[:120] if not ok else ""
        check("baseline.py runs without error", ok, err_detail)
        # Check output contains scores (works with both unicode and ASCII output)
        has_scores = (
            "FINAL SCORES" in result.stdout or
            "Overall" in result.stdout or
            "PASSED" in result.stdout or
            "mean" in result.stdout.lower()
        )
        check("baseline.py prints scores", has_scores)
    except subprocess.TimeoutExpired:
        check("baseline.py runs without error", False, "Timed out after 120 seconds")
        check("baseline.py prints scores", False, "Did not complete")
    except Exception as e:
        check("baseline.py runs without error", False, str(e))
        check("baseline.py prints scores", False, "Could not run")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Remote checks (against deployed HF Space)
# ─────────────────────────────────────────────────────────────────────────────

def check_remote(url):
    import urllib.request
    base = url.rstrip("/")

    def get(ep):
        with urllib.request.urlopen("{}/{}".format(base, ep), timeout=20) as r:
            return json.loads(r.read())

    def post(ep, data=None):
        req = urllib.request.Request(
            "{}/{}".format(base, ep),
            data=json.dumps(data or {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    print("\n-- Remote: {} ".format(url) + "-" * 20)

    try:
        h = get("health")
        check("GET /health returns 200", h.get("status") == "ok", str(h))
    except Exception as e:
        check("GET /health", False, str(e))

    try:
        t     = get("tasks")
        tasks = t.get("tasks", [])
        check("GET /tasks returns 3 tasks", len(tasks) == 3,
              "got {} tasks".format(len(tasks)))
        names = [x["name"] for x in tasks]
        check("GET /tasks has all 3 task names",
              set(names) == {"allergy_check", "dosing", "treatment_plan"}, str(names))
    except Exception as e:
        check("GET /tasks", False, str(e))

    try:
        obs = post("reset")
        check("POST /reset returns observation", "patient" in obs and "task" in obs)
        check("POST /reset returns legal_actions", len(obs.get("legal_actions", [])) > 0)
    except Exception as e:
        check("POST /reset", False, str(e))

    try:
        post("reset")
        obs    = post("step", {"action_type": "flag_allergy",
                               "payload": {"flagged_drugs": [], "reason": "test"}})
        reward = obs.get("reward")
        check("POST /step returns reward in [0,1]",
              reward is not None and 0.0 <= reward <= 1.0,
              "reward={}".format(reward))
        check("POST /step sets done=True", obs.get("done") is True)
    except Exception as e:
        check("POST /step", False, str(e))

    try:
        s = get("state")
        check("GET /state returns episode_id", "episode_id" in s)
        check("GET /state returns step_count", "step_count" in s)
    except Exception as e:
        check("GET /state", False, str(e))

    try:
        post("reset")
        g     = post("grader", {"action_type": "flag_allergy",
                                "payload": {"flagged_drugs": [], "reason": "test"}})
        score = g.get("score", -1)
        check("POST /grader returns score in [0,1]", 0.0 <= score <= 1.0,
              "score={}".format(score))
    except Exception as e:
        check("POST /grader", False, str(e))

    try:
        b  = post("baseline")
        bs = b.get("baseline_scores", {})
        check("POST /baseline returns all 3 task scores",
              set(bs.keys()) == {"allergy_check", "dosing", "treatment_plan"},
              str(list(bs.keys())))
        all_ok = all(0.0 <= v["mean"] <= 1.0 for v in bs.values())
        check("POST /baseline scores all in [0.0, 1.0]", all_ok)
    except Exception as e:
        check("POST /baseline", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="Deployed HF Space URL")
    args = parser.parse_args()

    print("=" * 65)
    print("  Healthcare Env -- Pre-Submission Validator")
    print("=" * 65)

    check_local_files()
    check_openenv_yaml()
    check_dockerfile()
    check_imports()
    check_environment_logic()
    check_graders()
    check_baseline_script()

    if args.url:
        check_remote(args.url)

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    failed = [(n, ok) for n, ok in results if not ok]

    print("\n" + "=" * 65)
    print("  RESULT: {}/{} checks passed".format(passed, total))
    if failed:
        print("\n  FAILED checks:")
        for name, _ in failed:
            print("    [X] {}".format(name))
        print("\n  Fix these before submitting!")
    else:
        print("  All checks passed -- ready to submit!")
    print("=" * 65)
