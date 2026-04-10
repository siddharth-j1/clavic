# CLAVIC Restart Guide

This is a practical "come back after a long time" guide for the CLAVIC workspace.

## 1) What this project is

CLAVIC implements language-conditioned robot trajectory synthesis with certified safety and impedance behavior.

Core idea:
- Natural-language task -> validated structured task spec
- Task spec -> temporal-logic objective + hard-safety wiring
- Multi-phase DMP + CGMS gains + obstacle projection -> rollout trace
- PIBB optimizer -> best trajectory and gain schedule

## 2) The architecture in one view

```text
Task description (text)
    -> LLMAgent (generate + validate + clamp)
    -> TaskSpec (clauses, phases, bindings, hard obstacles)
    -> MultiPhaseCertifiedPolicy
         Layer 1: DMP repulsive forcing
         Layer 2: Certified gains (K = Q^T Q)
         Layer 3: Hard radial projection for HARD obstacles
    -> Compiler objective (TL robustness + regularizers)
    -> PIBB optimization
    -> Outputs (csv, checkpoint, workspace/topdown/stiffness/orientation/kinematics plots)
```

## 3) Repository map (what each folder does)

- core/
  - certified_policy.py: trace object and policy plumbing
  - multi_phase_policy.py: multi-phase rollout and obstacle mode wiring
  - obstacle_projection.py: hard geometric projector
  - cgms/: DMP, orientation DMP, gain scheduling, quaternion utils
- logic/
  - predicates.py: robustness predicates
  - temporal_logic.py: eventually/always/until robustness operators
- optimization/
  - optimizer.py: PI2 and PIBB updaters
- spec/
  - taskspec.py: dataclasses (TaskSpec, Clause)
  - json_parser.py: JSON -> TaskSpec + hard obstacle extraction
  - compiler.py: objective construction and penalties
  - exp*_task.json: baseline experiment specs
- llm_interface/
  - llm_agent.py: LLM call + retry loop
  - validator.py: reject/repair rules
  - predicate_catalogue.py: allowed predicates, modalities, ranges
  - prompt_builder.py, scene_library.py: prompt and scene context
- scripts/
  - plot_exp2_velocity.py: publication-style velocity plot for exp2

## 4) How to run (quick start)

Use this from repo root:

```bash
conda activate clavic
python main_exp1.py
python main_exp1b.py
python main_exp2.py
python main_exp3a.py
python main_exp3b.py
```

LLM pipeline run:

```bash
conda activate clavic
export GEMINI_API_KEY="<your_key>"
python main_llm.py
```

LLM integration probes:

```bash
conda activate clavic
python test_llm_integration.py
```

## 5) What outputs you should expect

Typical files produced per run:
- expN_trajectory.csv or llm_trajectory.csv
- expN_checkpoint.npz or llm_checkpoint.npz
- expN_workspace.png
- expN_topdown.png
- expN_stiffness.png
- expN_orientation.png
- expN_kinematics.png
- exp2_velocity.png / exp2_velocity.svg for exp2 velocity analysis

## 6) Important behavior details (most useful reminders)

### A) Repulsive forcing vs comfort zone

These are different mechanisms:
- Repulsive forcing is geometric trajectory shaping in DMP dynamics.
- Comfort zone is a soft preference/cost concept (especially for human-aware behavior).

In code, obstacle influence is per obstacle spec:
- r_infl = radius * infl_factor
- default infl_factor in policy obstacle wiring is 2.5 (unless overridden in spec)

Obstacle modes in multi_phase_policy:
- HARD: repulsion + projector (geometric guarantee)
- SOFT: repulsion only (no hard guarantee)
- NONE: no repulsion, no projector

### B) Weight semantics (important for paper text)

Compiler behavior in practice:
- HARD and REQUIRE clauses use a fixed strong slack path (SLACK_WEIGHT = 500)
- PREFER clauses use clause.weight as soft penalty weight

So the optimizer-facing weights are mostly cost-level; HARD/REQUIRE are not tuned per-clause in the same way as PREFER.

### C) Temporal-logic robustness convention

Current semantics follow the standard quantitative form:
- always -> min robustness over interval
- eventually -> max robustness over interval

### D) CGMS stability and regularization

Stability structure comes from:
- K(t) = Q(t)^T Q(t)

Regularizers/clipping in compiler and parameter handling shape optimization behavior, but do not replace the structural CGMS parameterization.

## 7) Current optimization settings snapshot

From entry scripts:
- main_exp1.py, main_exp1b.py, main_llm.py: N_UPDATES = 70
- main_exp2.py, main_exp3a.py, main_exp3b.py: N_UPDATES = 120
- main_real_franka.py: N_UPDATES = 100

If you want faster iteration while resuming context, start with 40-70 updates and then scale up only when cost is still improving.

## 8) Suggested restart workflow (30-45 min)

1. Read project context docs:
   - instruction.txt
   - MATHEMATICAL_ARCHITECTURE.md
2. Run one deterministic baseline:
   - python main_exp1.py
3. Run one multi-phase baseline:
   - python main_exp3a.py
4. Run LLM pipeline once:
   - python main_llm.py
5. Check generated csv/png outputs to confirm environment and pipeline health.

## 9) Known paper-writing alignment points to keep consistent

When updating paper text/math notes, keep these aligned with implementation:
- Influence radius is obstacle-spec driven (radius * infl_factor), not only human comfort radius.
- HARD/REQUIRE use fixed strong slack path; PREFER uses tunable clause weight.
- Keep temporal robustness equations in min/max form for always/eventually.
- Keep CGMS claim tied to structural parameterization K = Q^T Q.

## 10) If you only remember one thing

Run this first:

```bash
conda activate clavic
python main_exp1.py
```

If that works and produces trajectory + plots, your full stack is ready.
