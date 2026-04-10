# CLAVIC — Experiment Scene Reference

> All coordinates are in metres `[x, y, z]`.  
> Workspace frame: x = forward, y = lateral, z = height.  
> All experiments use a constant nominal height z = 0.30 m.

---

## Exp 1 — Human Avoidance (Hard Body + Soft Comfort)
**Script:** `main_exp1.py` | **Spec:** `spec/exp1_task.json` | **CSV:** `exp1_trajectory.csv`

| Key Point | Position `[x, y, z]` |
|---|---|
| **Start** | `[0.55, 0.00, 0.30]` |
| **Goal** | `[0.30, 0.55, 0.30]` |
| **Human** | `[0.30, 0.30, 0.30]` |

| Constraint | Type | Value |
|---|---|---|
| Human body exclusion | HARD (DMP repulsion + projector) | radius = **0.08 m** |
| Human comfort zone | PREFER weight=15.0 | radius = **0.19 m** |
| Stiffness ramp starts at | Compiler implicit | d < 3×0.19 = **0.57 m** from human |
| Velocity limit | REQUIRE | v_max = **0.8 m/s** |
| Orientation limit | REQUIRE | max deviation = **0.15 rad** (~8.6°) from `[1,0,0,0]` |
| Angular velocity | REQUIRE | ω_max = **1.0 rad/s** |
| Horizon | — | **10 s** |

**What to look for in plots:**  
Trajectory curves around the red body-exclusion sphere. Stiffness (K_xx, K_yy, K_zz) drops inside the comfort zone. Body clearance must be > 0 at all times.

---

## Exp 1b — Human Avoidance (Reduced Comfort Weight — Soft Violation Demo)
**Script:** `main_exp1b.py` | **Spec:** `spec/exp1b_task.json` | **CSV:** `exp1b_trajectory.csv`

| Key Point | Position `[x, y, z]` |
|---|---|
| **Start** | `[0.55, 0.00, 0.30]` |
| **Goal** | `[0.30, 0.55, 0.30]` |
| **Human** | `[0.30, 0.30, 0.30]` |

| Constraint | Type | Value |
|---|---|---|
| Human body exclusion | HARD | radius = **0.08 m** |
| Human comfort zone | PREFER weight=**4.0** (vs 15.0 in Exp1) | radius = **0.19 m** |
| Velocity limit | REQUIRE | v_max = **0.8 m/s** |
| Orientation limit | REQUIRE | max = **0.15 rad** |
| Horizon | — | **2 s** (aggressive, short horizon) |

**Purpose:** Shows HARD vs SOFT contrast. Body exclusion is always respected (geometric guarantee). Comfort zone IS violated because weight=4.0 is too low to compete against reaching goal in 2 s.

---

## Exp 2 — 3-Phase: Carry → Hold at Waypoint → Continue (Hard Obstacle)
**Script:** `main_exp2.py` | **Spec:** `spec/exp2_task.json` | **CSV:** `exp2_trajectory.csv`

| Key Point | Position `[x, y, z]` |
|---|---|
| **Start** | `[0.55, 0.00, 0.30]` |
| **Waypoint** (mid-hold) | `[0.20, 0.35, 0.30]` |
| **Goal** | `[0.30, 0.55, 0.30]` |
| **Obstacle** | `[0.40, 0.30, 0.30]` |

| Phase | Label | Duration | From → To |
|---|---|---|---|
| 1 | carry | 0 – 5 s | Start → Waypoint |
| 2 | hold | 5 – 7 s | Waypoint (stationary, v≈0) |
| 3 | continue | 7 – 11 s | Waypoint → Goal |

| Constraint | Type | Value |
|---|---|---|
| Obstacle avoidance | HARD (DMP repulsion + projector) | radius = **0.12 m** |
| Reach waypoint | REQUIRE `eventually_during [0,5]` | tolerance = 0.04 m |
| Hold at waypoint | REQUIRE `always_during [5.1,7]` | v < 0.05 m/s |
| Reach goal | REQUIRE `eventually_during [7,11]` | tolerance = 0.04 m |
| Hold at goal | REQUIRE `always_during [9.5,11]` | v < 0.05 m/s |
| Velocity limit | REQUIRE | v_max = **0.8 m/s** |
| Orientation | REQUIRE | constant `[1,0,0,0]`, max = 0.15 rad |
| Horizon | — | **11 s** |

**What to look for:** Path bends left of obstacle. Velocity drops to near-zero at t≈5–7 s. Obstacle clearance > 0 always.

---

## Exp 3a — 2-Phase: Carry Upright → Pour at Goal (Hard Obstacle)
**Script:** `main_exp3a.py` | **Spec:** `spec/exp3a_task.json` | **CSV:** `exp3a_trajectory.csv`

| Key Point | Position `[x, y, z]` |
|---|---|
| **Start** | `[0.55, 0.00, 0.30]` |
| **Goal / Pour point** | `[0.30, 0.55, 0.30]` |
| **Obstacle** | `[0.40, 0.30, 0.30]` |

| Phase | Label | Duration | Orientation |
|---|---|---|---|
| 1 | carry | 0 – 7 s | upright `[1,0,0,0]` throughout |
| 2 | pour | 7 – 10 s | rotate to `[0.7071, 0, 0.7071, 0]` (90° tilt about Y) |

| Constraint | Type | Value |
|---|---|---|
| Obstacle avoidance | HARD | radius = **0.12 m**, strength=0.05, infl=2.0 |
| Reach goal (carry) | REQUIRE `eventually_during [0,7]` | tolerance = 0.04 m |
| Hold at goal (pour) | REQUIRE `always_during [7.05,10]` | v < 0.05 m/s |
| Carry orientation | REQUIRE `always_during [0,7]` | max = **0.2618 rad** (~15°) |
| Pour orientation | REQUIRE `eventually_during [7,10]` | target `[0.7071,0,0.7071,0]`, tol = 0.1745 rad |
| Velocity limit | REQUIRE | v_max = **0.8 m/s** |
| Angular velocity | REQUIRE | ω_max = **1.5 rad/s** |
| Horizon | — | **10 s** |

**What to look for:** Position holds fixed at [0.3,0.55,0.3] during pour phase. Orientation Euler plot shows ~90° pitch change during t=7–10 s.

---

## Exp 3b — 2-Phase: Carry → Hold at Human (Soft Obstacle — No Hard Guarantee)
**Script:** `main_exp3b.py` | **Spec:** `spec/exp3b_task.json` | **CSV:** `exp3b_trajectory.csv`

| Key Point | Position `[x, y, z]` |
|---|---|
| **Start** | `[0.55, 0.00, 0.30]` |
| **Goal** | `[0.30, 0.55, 0.30]` |
| **Obstacle** | `[0.40, 0.30, 0.30]` |

| Phase | Label | Duration | Notes |
|---|---|---|---|
| 1 | carry | 0 – 7 s | move to goal |
| 2 | hold | 7 – 10 s | stay at goal, v≈0 |

| Constraint | Type | Value |
|---|---|---|
| Obstacle avoidance | **PREFER weight=9.0** (SOFT — no geometric guarantee) | radius = 0.12 m |
| Reach goal | REQUIRE `eventually_during [0,7]` | tolerance = 0.04 m |
| Hold at goal | REQUIRE `always_during [7.05,10]` | v < 0.05 m/s |
| Orientation | PREFER `always_during [0,7]` | max = 0.30 rad (soft) |
| Velocity limit | REQUIRE | v_max = **0.8 m/s** |
| Horizon | — | **10 s** |

**Purpose:** Demonstrates soft-only avoidance — the trajectory may enter the obstacle zone. Compare against Exp 3a to see the effect of HARD vs PREFER on obstacle avoidance.

---

## Shared Coordinate Summary

```
Y (lateral)
^
|   [0.30, 0.55]  ← Goal (Exp1/1b/2/3a/3b)
|
|   [0.30, 0.30]  ← Human (Exp1/1b)
|   [0.40, 0.30]  ← Obstacle (Exp2/3a/3b)
|   [0.20, 0.35]  ← Waypoint (Exp2 only)
|
|                         [0.55, 0.00]  ← Start (all exps)
└─────────────────────────────────────────> X (forward)
```

---

## Replotting from CSV

All trajectories are saved as `exp<N>_trajectory.csv` with columns:

```
t, px, py, pz, vx, vy, vz, Kxx, Kyy, Kzz, trK, Dxx, Dyy, Dzz, trD
```

Use `plot_from_csv.py` to regenerate plots from saved CSVs without re-running optimisation.

---

## Quaternion Reference

| Orientation | Quaternion `[w, x, y, z]` | Meaning |
|---|---|---|
| Upright | `[1.0, 0.0, 0.0, 0.0]` | End-effector flat / neutral |
| Pour (90° Y) | `[0.7071, 0.0, 0.7071, 0.0]` | 90° tilt about Y axis |
