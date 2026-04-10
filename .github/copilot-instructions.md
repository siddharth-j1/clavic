# CLAVIC — AI Coding Agent Instructions

## Project Context
This is **S²TA** (Safe & Semantically-Aware Trajectory Adaptation): a research codebase for language-conditioned robot manipulation trajectory synthesis, combining CGMS (certified DMP + impedance) and wTLTL (weighted temporal logic) into a single framework. Paper target: IEEE conference submission.

**Two base repos integrated here:**
- [cgms_safe_via_point](https://github.com/siddharthjainsid1411/cgms_safe_via_point) — DMP trajectory + Cholesky-parameterized variable impedance, Lyapunov-stable by construction
- [wltl](https://github.com/siddharthjainsid1411/wltl) — PIBB policy search + wTLTL robustness objective

---

## Architecture: Data Flow

```
Natural Language → LLMAgent → validate_and_clamp() → TaskSpec JSON
       ↓
json_parser.py → TaskSpec(clauses, phases, horizon_sec)
       ↓
MultiPhaseCertifiedPolicy  ←  phases (DMP start/end/duration/quat)
       ↓
Compiler.compile(taskspec) → objective(trace) function
       ↓
PIBB optimizer  (optimization/optimizer.py)
       ↓
Plots (seaborn/matplotlib, 300dpi PDF/SVG)
```

**3-layer safety architecture** (clause `modality="HARD"`):
1. DMP repulsive forcing injected into ODE (smooth, continuous)
2. Post-rollout radial projection `p' = c + r*(p-c)/||p-c||` (hard guarantee)
3. Slack penalty in optimizer cost (same path as `REQUIRE`)

---

## Key Files & Responsibilities

| File | Role |
|------|------|
| `spec/taskspec.py` | `TaskSpec` + `Clause` dataclasses |
| `spec/json_parser.py` | JSON → `TaskSpec`; auto-extracts `hard_obstacle_specs` |
| `spec/compiler.py` | `Compiler.compile()` → cost function; `SLACK_WEIGHT=500`, `K_MAX=3000` |
| `core/multi_phase_policy.py` | Chains N DMP phases; `setup_hard_obstacles_from_taskspec()` wires Layers 1+2 |
| `core/certified_policy.py` | Single-phase policy + `Trace` dataclass |
| `optimization/optimizer.py` | PIBB (`M_roll` samples, importance-weighted mean update) |
| `logic/predicates.py` | Predicate evaluation functions (robustness scalars) |
| `logic/temporal_logic.py` | wTLTL robustness recursion |
| `llm_interface/llm_agent.py` | `LLMAgent.generate(task_description)` → validated dict; retry loop (MAX_RETRIES=3) |
| `llm_interface/predicate_catalogue.py` | **Single source of truth** for predicates, allowed modalities, param ranges |
| `llm_interface/validator.py` | Hard-reject + silent clamp before spec reaches optimizer |
| `llm_interface/prompt_builder.py` | Builds LLM system prompt from catalogue + scene_library + few-shot JSONs |
| `llm_interface/scene_library.py` | Pre-calibrated entity params (human, laptop, obstacle, fragile, wall) |

---

## Running Experiments
```bash
conda activate clavic
# Run any experiment directly (no background — watch full output)
python main_exp1.py    # Human avoidance + stiffness shaping
python main_exp1b.py   # Variant of exp1
python main_exp2.py    # Obstacle avoidance
python main_exp3a.py   # Multi-phase (carry + pour)
python main_exp3b.py   # Multi-phase variant
```
Outputs: `exp<N>_workspace.png`, `exp<N>_topdown.png`, `exp<N>_stiffness.png`, `exp<N>_orientation.png`, `exp<N>_kinematics.png`, `exp<N>_trajectory.csv`, `exp<N>_checkpoint.npz`.

---

## LLM Interface: How It Works

`LLMAgent.generate(task_description: str)` returns a **validated dict** (same schema as the `spec/exp*_task.json` files) ready to pass directly to `json_parser.load_taskspec_from_json()` (after writing to a temp file) or parsed in-memory.

**Flow:** system prompt (catalogue + weight rules + modality rules + few-shot examples) → LLM call at `temperature=0.2` → `_extract_json()` → `validate_and_clamp()` → retry with error feedback if invalid (up to 3 times).

**LLM is constrained to:**
- Only predicates in `CATALOGUE` (predicate_catalogue.py)
- Weights `[1.0, 20.0]` (auto-clamped; `PREFER` actually used, `HARD/REQUIRE` are fixed at 500 internally)
- Fixed modality rules: `HumanBodyExclusion→HARD`, `HumanComfortDistance→PREFER`, `VelocityLimit/AngularVelocityLimit/ZeroVelocity/OrientationLimit/HoldAtWaypoint→REQUIRE`
- Param values clamped to per-predicate `min/max` from catalogue

---

## Task JSON Schema (see `spec/exp1_task.json` for reference)
```json
{
  "horizon_sec": 10.0,
  "phases": [{"label": "carry", "start": [...], "end": [...], "duration": 10.0,
              "start_quat": [1,0,0,0], "end_quat": [1,0,0,0], "n_bfs_ori": 15}],
  "clauses": [
    {"type": "eventually", "predicate": "AtGoal", "weight": 10.0, "modality": "REQUIRE"},
    {"type": "always", "predicate": "HumanBodyExclusion", "weight": 20.0, "modality": "HARD",
     "hard_strength": 0.2, "hard_infl_factor": 3.0}
  ],
  "bindings": {
    "AtGoal.waypoint": [0.3, 0.55, 0.3],
    "HumanBodyExclusion.human_position": [0.3, 0.3, 0.3],
    "HumanBodyExclusion.body_radius": 0.08
  }
}
```

---

## Conventions & Anti-patterns

- **Never run processes in background** — always use full terminal output.
- **Optimizer iterations:** saturates at ~60–70 epochs; don't set `n_updates > 100` unless cost is still descending.
- **Plots:** use `seaborn` style + `matplotlib`, save at 300 dpi as PDF/SVG. Legends must not overlap trajectories. Labels, axis limits, and titles required for publication.
- **Reuse predicates generically** — don't create a new predicate class for every small parameter change; parameterise the existing ones.
- **HARD clause cost weight is ignored by `Compiler`** — `SLACK_WEIGHT=500` is always used. The `weight` field on HARD/REQUIRE clauses is documentation only.
- **Conda env:** always activate `clavic` before any terminal command.
- **No new `.md`/text files** unless explicitly requested.
