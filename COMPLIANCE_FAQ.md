# Answer to Your Questions

## Q1: Are we penalizing `trace(K)` or `Kxx, Kyy, Kzz`?

**Answer: We are penalizing individual `Kxx`, `Kyy`, `Kzz` (per-axis).**

```python
# compiler.py lines 140-151
kxx = float(K_all[t, 0, 0])
kyy = float(K_all[t, 1, 1])
kzz = float(K_all[t, 2, 2])

excess_x = max(0.0, (kxx - K_TARGET) / K_TARGET)
excess_y = max(0.0, (kyy - K_TARGET) / K_TARGET)
excess_z = max(0.0, (kzz - K_TARGET) / K_TARGET)

stiff_excess = excess_x + excess_y + excess_z  # Sum three axes
comply_cost += penalty_w * stiff_excess
```

✅ **Good:** This allows PI2 to independently reduce each axis's stiffness.

---

## Q2: Are we penalizing based on distance in X, Y, Z coordinates?

**Answer: NO. We use 3D Euclidean distance for all axes, not per-axis distances.**

```python
# compiler.py line 126
dists = np.linalg.norm(positions - center, axis=1)
# This computes: sqrt((x-cx)² + (y-cy)² + (z-cz)²)

# Then ONE proximity value for all axes
proximity = (radius - dists) / radius
penalty_w = base_w * clause_w * (proximity ** 2)

# All three axes get the SAME penalty_w
comply_cost += penalty_w * (excess_x + excess_y + excess_z)
```

❌ **Problem:** If the robot is close on the Z-axis but far on the X-axis, the large X-distance dominates the 3D vector, making the penalty weak for all axes.

---

## What We're Currently Doing

```
┌─────────────────────────────────────────────────────────────┐
│ COMPLIANCE PENALTY STRUCTURE                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Compute 3D distance                              │
│  ────────────────────────────────────────                 │
│  d3D = sqrt((x-cx)² + (y-cy)² + (z-cz)²)                 │
│                                                            │
│  Step 2: Compute proximity from 3D distance              │
│  ──────────────────────────────────────────              │
│  proximity = (radius - d3D) / radius                      │
│  (Same value for all X, Y, Z axes)                       │
│                                                            │
│  Step 3: Compute penalty weight                           │
│  ───────────────────────────────                         │
│  penalty_w = base_w × clause_w × proximity²              │
│  (Same weight for all axes)                              │
│                                                            │
│  Step 4: Penalize individual stiffness axes              │
│  ─────────────────────────────────────────               │
│  excess_x = max(0, (Kxx - 50) / 50)                      │
│  excess_y = max(0, (Kyy - 50) / 50)                      │
│  excess_z = max(0, (Kzz - 50) / 50)                      │
│                                                            │
│  cost = penalty_w × (excess_x + excess_y + excess_z)     │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## The Issue (Illustrated)

At closest approach:

```
Robot:  [0.1744, 0.3285, 0.0815]
Human:  [0.3000, 0.4000, 0.1100]

Distance components:
  dX = 0.1256 m  ← LARGE (X-axis dominates)
  dY = 0.0715 m
  dZ = 0.0285 m  ← VERY CLOSE on Z!

3D distance:
  d3D = sqrt(0.1256² + 0.0715² + 0.0285²) = 0.1473 m
       = 0.1473 m / 0.15 m = 98.2% of radius

Proximity:
  proximity = (0.15 - 0.1473) / 0.15 = 0.018 = 1.8%

Penalty weight:
  penalty_w = 0.5 × 3.0 × 0.018² = 0.00049 ← TOO WEAK!

Result:
  All axes get weak penalty even though Z is very close!
  ✗ Kzz doesn't get strongly penalized
  ✗ Compliance effect is invisible (0.002 / 9.5 = 0.02% of total cost)
```

---

## What Would Be Better

If we used **per-axis distance** instead:

```
Kxx: dX = 0.1256 m → proximity_x = 16.3% → penalty_w_x = 0.0396 (80× stronger)
Kyy: dY = 0.0715 m → proximity_y = 52.4% → penalty_w_y = 0.4113 (840× stronger)
Kzz: dZ = 0.0285 m → proximity_z = 81.0% → penalty_w_z = 0.9849 (2010× stronger)

Result:
  ✓ Z-axis gets STRONG penalty (2000×)
  ✓ Compliance cost becomes 1.11 / 9.5 = 11.7% of total
  ✓ PI2 actually sees and respects the penalty
  ✓ K drops toward target near human
```

---

## Summary Table

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Stiffness penalty** | Per-axis (✓) | Per-axis (✓) |
| **Distance metric** | 3D Euclidean (✗) | Per-axis (✓) |
| **Penalty strength** | 0.002 total cost | 1.11 total cost |
| **PI2 visibility** | 0.02% | 11.7% |
| **Kzz penalty** | 0.00049 | 0.9849 |
| **Result** | Weak compliance | Strong compliance |

---

## Files Documenting This

- `COMPLIANCE_PENALTY_DEEP_DIVE.md` — Detailed technical explanation
- `COMPLIANCE_CURRENT_STATE.md` — Current vs proposed comparison
- `test_compliance_structure.py` — Numerical breakdown
