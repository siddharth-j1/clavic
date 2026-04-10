# How to Run and Test the LLM Integration

## Quick Start (30 seconds)

```bash
cd /home/siddharth/ssd_data/clavic
conda activate clavic
python main_llm.py
```

**That's it.** The LLM will be called, a trajectory optimized, and 5 plots saved.

---

## What Happens by Default

When you run `python main_llm.py` with no arguments:

1. **Task choice:** `TASK_CHOICE = "exp1"` (human avoidance — same geometry as `main_exp1.py`)
2. **LLM model:** `gemini-3-flash-preview`
3. **API key:** Already pasted at line 602: `GEMINI_API_KEY_INLINE = "AIzaSyCpvNNE_rn0wTsXHzA8DzxvEICKU9TqMiU"`

The script will:
- Send "Carry the mug..." task description to Gemini
- LLM generates JSON task spec (with validation + 3 retries if needed)
- Policy optimizes for 70 updates × 30 samples (~5 min)
- Produces 5 PNG plots + CSV + checkpoint file

**Outputs:**
```
llm_task_spec.json    — validated JSON (for reproducibility)
llm_trajectory.csv    — t, px, py, pz, vx, vy, vz, Kxx, Kyy, Kzz, trK, Dxx, Dyy, Dzz, trD
llm_checkpoint.npz    — best_theta + best_cost for warm-starting
llm_workspace.png     — 3D trajectory
llm_topdown.png       — top-down view
llm_stiffness.png     — per-axis K schedule
llm_orientation.png   — Euler angles (if multi-phase)
llm_kinematics.png    — position + speed
```

---

## Run Different Tasks

Edit **line 613** in `main_llm.py`:

```python
TASK_CHOICE = "exp1"      # change this
```

| Value | Task |
|---|---|
| `"exp1"` | Carry + human avoidance (10 s) |
| `"exp2"` | Carry + obstacle avoidance (11 s, 3-phase) |
| `"exp3a"` | Carry + pour (10 s, 2-phase with orientation) |
| `"custom"` | Use `CUSTOM_TASK_DESCRIPTION` (line 625) |
| `"argv"` | Read from command line: `python main_llm.py "..."` |

**Example:**
```bash
python main_llm.py "Carry ball from [0.5,0,0.3] to [0.3,0.5,0.3]. Obstacle at [0.4,0.3,0.3]."
```

---

## Change LLM Model

Edit **line 608**:

```python
GEMINI_MODEL = "gemini-3-flash-preview"   # change this
```

| Model | Speed | Cost | Quality |
|---|---|---|---|
| `"gemini-2.0-flash"` | **Fast** | Free tier | Good |
| `"gemini-1.5-pro"` | Medium | Free tier | Very good |
| `"gemini-3-flash-preview"` | **Fast** | Free tier | Good (current) |
| `"gemini-2.5-pro-preview-03-25"` | Slow | Paid | **Best** |

---

## Change API Key

The key is already set inline (**line 602**). If you want to update it:

### Option 1 — Edit the file (easiest):
```python
GEMINI_API_KEY_INLINE = "AIzaSyCpvNNE_rn0wTsXHzA8DzxvEICKU9TqMiU"   # line 602
```

### Option 2 — Use environment variable (no file edit):
```bash
export GEMINI_API_KEY="AIzaSyCpvNNE_rn0wTsXHzA8DzxvEICKU9TqMiU"
python main_llm.py
```

Priority: inline key → env var. If inline is empty `""`, the env var is used.

---

## Test the LLM in Isolation

Want to test just the LLM without running the full optimizer?

```python
from llm_interface.llm_agent import LLMAgent

agent = LLMAgent(model="gemini-2.0-flash", api_key="AIzaSyCpvNNE_rn0wTsXHzA8DzxvEICKU9TqMiU")

spec = agent.generate("Carry mug from [0.5,0,0.3] to [0.3,0.5,0.3] in 10s. Human at [0.3,0.3,0.3].")

print(spec)  # prints the validated JSON dict
```

---

## Robustness Guarantees

The LLM output is validated **before** reaching the optimizer:

| Issue | Handling |
|---|---|
| Unknown predicate (e.g., "MakeTeaCup") | Hard-reject + retry with error feedback |
| `weight > 20` (e.g., 500) | Auto-clamped to 20.0, warning printed |
| `HumanBodyExclusion` as PREFER | Forced to HARD (modality rule) |
| Missing `time_window` on `always_during` | Hard-reject + retry |
| `radius` out of bounds (e.g., 100 m) | Auto-clamped to catalogue max (e.g., 0.30 m) |
| Up to 3 LLM call attempts | Retry with error feedback if validation fails |

---

## Full Example: Run Exp2 (Obstacle)

```bash
conda activate clavic
cd /home/siddharth/ssd_data/clavic
```

Edit line 613:
```python
TASK_CHOICE = "exp2"
```

Run:
```bash
python main_llm.py
```

**Output:**
```
============================================================
  CLAVIC — Language-Conditioned Trajectory Synthesis
============================================================

Task: exp2
Description:
  Carry the mug from position [0.55, 0.0, 0.3] to position [0.05, 0.55, 0.3] ...

Calling LLM (gemini-3-flash-preview) ...
Gemini call attempt 1 / 3 ...
Spec validated successfully on attempt 1 (0 auto-fixes applied).

============================================================
  LLM-GENERATED TASK SPEC (validated + clamped)
============================================================
  horizon_sec : 11.0 s
  phases      : 3
    [0] carry     [0.55, 0.0, 0.3] → [0.2, 0.35, 0.3]  dur=5.0s
    [1] hold      [0.2, 0.35, 0.3] → [0.2, 0.35, 0.3]  dur=2.0s
    [2] continue  [0.2, 0.35, 0.3] → [0.3, 0.55, 0.3]  dur=4.0s
  clauses     : 7
    [HARD   ] always                 ObstacleAvoidance              w=20.0
    [REQUIRE] eventually_during      AtWaypoint                    w=10.0
    ...
  bindings    :
    AtWaypoint.waypoint: [0.2, 0.35, 0.3]
    ...

Validated spec saved to: llm_task_spec.json

TaskSpec parsed: 7 clauses, 1 HARD obstacles, 3 phase(s).
Policy: 3 phase(s), theta_dim=...

PIBB: 70 updates × 30 samples/update
--------------------------------------------------
  [001/070]  min=...  mean=...  best=...
  ...
  [070/070]  min=...  mean=...  best=...
--------------------------------------------------
Optimisation complete.

Checkpoint saved: llm_checkpoint.npz

============================================================
  LLM TASK DIAGNOSTICS
============================================================
  Best cost        : ...
  Max speed        : ... m/s
  HARD [obstacle   ] r=0.12m  min_clearance=+... cm  violations=0  (must be 0)
  Goal reached     : YES  (tolerance=0.040 m)
============================================================

Saved: llm_workspace.png
Saved: llm_topdown.png
Saved: llm_stiffness.png
Saved: llm_kinematics.png

All done — outputs:
  llm_task_spec.json   — validated task spec (for reproducibility)
  llm_trajectory.csv   — full trajectory + gains
  llm_checkpoint.npz   — best_theta + best_cost
  llm_workspace.png    — 3D trajectory
  llm_topdown.png      — top-down view
  llm_stiffness.png    — per-axis K schedule
  llm_kinematics.png   — position + speed
```

---

## Troubleshooting

### `ERROR: No GEMINI_API_KEY found.`
→ Paste key at line 602 or set env var:
```bash
export GEMINI_API_KEY="AIzaSyCpvNNE_rn0wTsXHzA8DzxvEICKU9TqMiU"
```

### `LLM failed to produce a valid spec after 3 attempts.`
→ Check the error message in the output. Common issues:
- Task description is too vague (e.g., "do something cool")
- Asking for a predicate that doesn't exist
- Model quota exceeded (try a different model or wait 24h)

### Plots look wrong or empty
→ Check `llm_task_spec.json` to verify the spec parsed correctly. If the positions don't match your description, the LLM may have hallucinated values — they should be auto-clamped by the validator.

### Slow optimization (>10 min for 70 updates)
→ Normal for 30 samples × 70 iterations. You can reduce to 50 updates at line 754:
```python
N_UPDATES = 50  # was 70
```

---

## Compare with Ground Truth

Use `experiment_scenes.md` to compare LLM-generated outputs against the reference specs:

```bash
head -20 experiment_scenes.md      # See Exp 1/2/3a/3b positions
diff llm_task_spec.json spec/exp1_task.json   # Compare LLM spec vs reference
```
