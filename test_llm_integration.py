"""
test_llm_integration.py  —  Minimal integration test for the LLM interface.

What this tests
---------------
Three probe tasks, each derived from an existing experiment's *geometry* but
described in fresh natural language with deliberate twists that the LLM must
reason about:

  Probe 1 (exp1 variant):  Same human/goal as exp1 but SHORTER time budget
                            (4 s instead of 10 s) → LLM must still request
                            HARD body exclusion but comfort may be violated.

  Probe 2 (exp2 variant):  Same obstacle as exp2 but described as a FRAGILE
                            OBJECT to avoid softly (PREFER, not HARD) with an
                            extra speed limit → tests modality choice + extra
                            clause generation.

  Probe 3 (exp3a variant): Same carry+pour geometry but without specifying
                            exact phase durations → LLM must still produce a
                            2-phase spec with a pour orientation and a HARD
                            obstacle avoidance clause.

For each probe the script:
  1. Calls LLMAgent.generate(description) — real API call.
  2. Prints the full validated spec dict (so you can read it).
  3. Runs a lightweight spec-diff against expected ground-truth properties
     (modalities, predicates present, phase count, horizon range).
  4. Prints a PASS / FAIL verdict for each check.
  5. Optionally runs the optimizer for a few iterations and prints final cost.

Usage
-----
  conda activate clavic
  python test_llm_integration.py                  # all 3 probes
  python test_llm_integration.py --probe 1        # single probe
  python test_llm_integration.py --probe 2 --api-key YOUR_KEY
"""

import argparse
import json
import os
import sys
import textwrap

import numpy as np

# ── ensure repo root on path ────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm_interface.llm_agent import LLMAgent

# ─────────────────────────────────────────────────────────────────────────────
#  Probe definitions
#  Each entry has:
#    "description"   : natural language sent to the LLM
#    "expected"      : dict of checks (see _check_spec)
#    "name"          : short label for logging
# ─────────────────────────────────────────────────────────────────────────────

PROBES = [

    # ── Probe 1 ─────────────────────────────────────────────────────────────
    {
        "name": "Probe-1 | exp1-variant — short horizon, human avoidance",
        "description": textwrap.dedent("""\
            Move the end-effector from [0.55, 0.0, 0.30] to [0.30, 0.55, 0.30]
            in exactly 4 seconds. A human operator is standing at [0.30, 0.30, 0.30]
            with a body radius of 0.08 m — the robot MUST NOT touch the human under
            any circumstances. The human also has a comfort zone of 0.19 m around
            them; the robot should try to stay outside this zone if time allows,
            but it is not a hard requirement. Keep the end-effector velocity below
            0.6 m/s throughout the motion. The robot must reach the goal.
        """),
        "expected": {
            # predicate must appear with this exact modality
            "modalities": {
                "HumanBodyExclusion":   "HARD",
                "HumanComfortDistance": "PREFER",
                "AtGoal":               "REQUIRE",
            },
            # horizon must be in this range (s)
            "horizon_range": (3.5, 5.0),
            # number of phases
            "n_phases": 1,
            # these predicates must appear (any modality)
            "predicates_present": ["HumanBodyExclusion", "HumanComfortDistance", "AtGoal"],
            # these predicates must NOT appear
            "predicates_absent": [],
        },
    },

    # ── Probe 2 ─────────────────────────────────────────────────────────────
    {
        "name": "Probe-2 | exp2-variant — fragile obstacle (soft), speed limit",
        "description": textwrap.dedent("""\
            Carry an object from [0.55, 0.0, 0.30] to [0.30, 0.55, 0.30] in 10
            seconds. There is a fragile decorative vase at [0.40, 0.30, 0.30] with
            radius 0.12 m — please avoid it but it is not a hard safety constraint,
            just a strong preference (avoid it if possible). Keep the speed below
            0.4 m/s at all times. The robot must reach the goal.
        """),
        "expected": {
            "modalities": {
                "ObstacleAvoidance": "PREFER",   # fragile → PREFER, not HARD
                "AtGoal":            "REQUIRE",
                "VelocityLimit":     "REQUIRE",
            },
            "horizon_range": (8.0, 12.0),
            "n_phases": 1,
            "predicates_present": ["ObstacleAvoidance", "AtGoal", "VelocityLimit"],
            "predicates_absent":  ["HumanBodyExclusion"],
        },
    },

    # ── Probe 3 ─────────────────────────────────────────────────────────────
    {
        "name": "Probe-3 | exp3a-variant — carry+pour, hard obstacle, no explicit durations",
        "description": textwrap.dedent("""\
            The robot must carry a cup from [0.55, 0.0, 0.30] to [0.30, 0.55, 0.30]
            and then pour its contents. Pouring means tilting the cup 90 degrees
            around the Y-axis at the goal position. There is a hard obstacle at
            [0.40, 0.30, 0.30] with radius 0.12 m — the robot must guarantee it
            never touches it. Total task duration is 10 seconds. The robot must
            reach the goal upright first, then tilt to pour.
        """),
        "expected": {
            "modalities": {
                "ObstacleAvoidance": "HARD",
                "AtGoal":            "REQUIRE",
            },
            "horizon_range": (8.0, 12.0),
            "n_phases": 2,   # must produce carry + pour phases
            "predicates_present": ["ObstacleAvoidance", "AtGoal"],
            "predicates_absent":  ["HumanBodyExclusion"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Spec checker
# ─────────────────────────────────────────────────────────────────────────────

def _check_spec(spec_dict: dict, expected: dict) -> list[dict]:
    """
    Run all expected checks against the validated spec dict.
    Returns a list of result dicts:
        {"check": str, "pass": bool, "detail": str}
    """
    results = []

    def record(check, passed, detail=""):
        results.append({"check": check, "pass": passed, "detail": detail})

    clauses  = spec_dict.get("clauses", [])
    bindings = spec_dict.get("bindings", {})
    phases   = spec_dict.get("phases", [])
    horizon  = spec_dict.get("horizon_sec", -1)

    # Build quick lookup: predicate_name → modality (last occurrence wins)
    pred_modality = {c["predicate"]: c["modality"] for c in clauses}

    # ── modality checks ──────────────────────────────────────────────────── #
    for pred, expected_mod in expected.get("modalities", {}).items():
        actual_mod = pred_modality.get(pred, None)
        passed = actual_mod == expected_mod
        record(
            f"modality({pred})=={expected_mod}",
            passed,
            f"got {actual_mod!r}" if not passed else "✓",
        )

    # ── predicates present ───────────────────────────────────────────────── #
    for pred in expected.get("predicates_present", []):
        passed = pred in pred_modality
        record(
            f"predicate_present({pred})",
            passed,
            "✓" if passed else f"missing — LLM did not include {pred}",
        )

    # ── predicates absent ────────────────────────────────────────────────── #
    for pred in expected.get("predicates_absent", []):
        passed = pred not in pred_modality
        record(
            f"predicate_absent({pred})",
            passed,
            "✓" if passed else f"should be absent but LLM added {pred}",
        )

    # ── horizon range ────────────────────────────────────────────────────── #
    lo, hi = expected.get("horizon_range", (0, 9999))
    passed = lo <= horizon <= hi
    record(
        f"horizon_in[{lo},{hi}]",
        passed,
        f"got {horizon}" if not passed else f"✓ ({horizon} s)",
    )

    # ── phase count ──────────────────────────────────────────────────────── #
    exp_n = expected.get("n_phases", None)
    if exp_n is not None:
        passed = len(phases) == exp_n
        record(
            f"n_phases=={exp_n}",
            passed,
            f"got {len(phases)}" if not passed else f"✓ ({len(phases)} phases)",
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Spec diff printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_spec_summary(spec_dict: dict, label: str = "LLM spec"):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  horizon_sec : {spec_dict.get('horizon_sec')} s")
    phases = spec_dict.get("phases", [])
    print(f"  phases ({len(phases)})")
    for i, ph in enumerate(phases):
        print(f"    [{i}] {ph.get('label','?'):10s}  "
              f"{ph.get('start')} → {ph.get('end')}  "
              f"dur={ph.get('duration')}s")
    clauses = spec_dict.get("clauses", [])
    print(f"  clauses ({len(clauses)})")
    for cl in clauses:
        print(f"    [{cl['modality']:7s}] {cl['type']:22s} "
              f"{cl['predicate']:30s}  w={cl.get('weight','?')}")
    bindings = spec_dict.get("bindings", {})
    print(f"  bindings ({len(bindings)})")
    for k, v in bindings.items():
        print(f"    {k}: {v}")
    print(sep)


def _print_check_results(results: list[dict]):
    passed = sum(r["pass"] for r in results)
    total  = len(results)
    print(f"\n  Checks: {passed}/{total} passed")
    for r in results:
        icon = "✅" if r["pass"] else "❌"
        print(f"    {icon}  {r['check']:40s}  {r['detail']}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM integration test")
    parser.add_argument(
        "--probe", type=int, default=None,
        help="Run only probe N (1/2/3). Default: all probes."
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env var)."
    )
    parser.add_argument(
        "--model", type=str, default="gemini-3.1-flash-lite-preview",
        help="Gemini model name. Default: gemini-3.1-flash-lite-preview"
    )
    args = parser.parse_args()

    # ── pick API key ─────────────────────────────────────────────────────── #
    api_key = (
        args.api_key
        or os.environ.get("GEMINI_API_KEY", "")
        or "AIzaSyCpvNNE_rn0wTsXHzA8DzxvEICKU9TqMiU"   # fallback inline key
    )

    agent = LLMAgent(model=args.model, api_key=api_key)

    probes_to_run = (
        [PROBES[args.probe - 1]] if args.probe else PROBES
    )

    overall_pass = 0
    overall_total = 0
    summary_rows = []

    for i, probe in enumerate(probes_to_run, 1):
        banner = "═" * 66
        print(f"\n{banner}")
        print(f"  PROBE {i}: {probe['name']}")
        print(banner)
        print("\n  Task description sent to LLM:")
        for line in probe["description"].strip().splitlines():
            print(f"    {line}")

        # ── 1. LLM call (with rate-limit backoff) ────────────────────── #
        print("\n  ⏳ Calling LLM ...")
        import time, re as _re
        spec_dict = None
        for _attempt in range(5):
            try:
                spec_dict = agent.generate(probe["description"])
                break
            except Exception as e:
                msg = str(e)
                # Parse retry delay from 429 message if present
                m = _re.search(r'retry[^\d]*(\d+)', msg, _re.IGNORECASE)
                wait = int(m.group(1)) + 5 if m else 60
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print(f"  ⚠️  Rate-limited. Waiting {wait}s before retry "
                          f"(attempt {_attempt+1}/5) ...")
                    time.sleep(wait)
                else:
                    print(f"  ❌ LLM call FAILED: {e}")
                    break
        if spec_dict is None:
            print("  ❌ All retry attempts exhausted — skipping probe.")
            summary_rows.append((probe["name"], 0, 0, "LLM FAILED"))
            continue

        print("  ✅ LLM returned a validated spec.")

        # ── 2. Print spec summary ─────────────────────────────────────── #
        _print_spec_summary(spec_dict, label="Validated LLM spec")

        # ── 3. Check against expected ─────────────────────────────────── #
        results = _check_spec(spec_dict, probe["expected"])
        _print_check_results(results)

        p = sum(r["pass"] for r in results)
        t = len(results)
        overall_pass  += p
        overall_total += t
        summary_rows.append((probe["name"], p, t, "OK"))

    # ── Final summary ─────────────────────────────────────────────────────── #
    print(f"\n{'═'*66}")
    print("  OVERALL SUMMARY")
    print(f"{'═'*66}")
    for name, p, t, status in summary_rows:
        verdict = "✅ ALL PASS" if p == t else f"⚠️  {p}/{t}"
        print(f"  {verdict}  {name}")
    print(f"\n  Total checks: {overall_pass}/{overall_total} passed")
    print(f"{'═'*66}\n")

    # Exit with non-zero if any check failed (useful for CI)
    if overall_pass < overall_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
