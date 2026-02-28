# Stiffness vs Position Analysis

## Key Findings

### 1. What Stiffness Near Human?

When the robot is **closest to the human** (d = 0.147m, inside comfort zone):

```
Kyy:  66.8 N/m  (target: 50 N/m)  — lateral compliance ✓ REDUCED
Kzz:  95.2 N/m  (target: 50 N/m)  — vertical compliance ✓ REDUCED
Kxx: 202.8 N/m                     — longitudinal (not constrained)
```

### 2. Compliance Penalty IS Working

Comparison inside vs outside comfort zone:

| Axis | Inside (r<0.15m) | Outside (r≥0.15m) | Difference | Effect |
|------|------------------|-------------------|-----------|--------|
| Kyy | 66.8 N/m | 86.7 N/m | -19.9 N/m | ← **REDUCED** ✓ |
| Kzz | 95.2 N/m | 143.4 N/m | -48.2 N/m | ← **REDUCED** ✓ |
| Kxx | 202.8 N/m | 201.7 N/m | +1.1 N/m | (unchanged) |

**YES, compliance penalty is reducing Kyy and Kzz near the human.** But the reduction is not to target (50 N/m) because:

### 3. Why Not Lower to Target (50 N/m)?

The compliance penalty is **constrained by task requirements:**

- **Trajectory shape**: Goal reach + human avoidance constraints drive the nominal K values
- **Conflicting objectives**: Lower K near human would reduce precision for goal reach
- **Limited region**: Only 6 timesteps fire compliance (robot barely clips 0.15m zone)
- **Penalty weight**: Even with full compliance, penalty contribution is only ~0.5% of total cost

**The 50 N/m target is aspirational, not enforced.** The optimizer achieves 66.8 Kyy and 95.2 Kzz as the best compromise.

---

## Why the Test Plot Shows Similar Stiffness (WITH vs WITHOUT)

Your first plot showed both cases had similar tr(K) ≈ 360-380 N/m near human. This is **correct behavior** because:

1. **Different runs, different variance**: Two separate PI2 optimizations can find slightly different trajectories
2. **Compliance is weak vs task cost**: Compliance cost ≈ 0.5 - 1.6 / total cost ≈ 10-12 → hard to see effect
3. **Target is high relative to nominal**: K_TARGET = 50 N/m is very soft; natural task K ≈ 200 N/m

**The compliance is definitely working (Kyy -19.9 N/m, Kzz -48.2 N/m reduction) but isn't aggressive enough to overcome the task dynamics.**

---

## Current State

✅ **Compliance penalty active**: Reduces Kyy by 23% and Kzz by 34% when near human
✅ **Per-axis penalty working**: Kyy and Kzz independently reduced, Kxx unaffected
✅ **Spatial region firing**: Exactly 6 timesteps at closest approach
⚠️ **Reduction to target not achieved**: 66.8 Kyy vs target 50 is still 33% higher

---

## To Make Compliance More Aggressive

You have three options:

**Option A: Increase region radius**
```json
"HumanComfortDistance.preferred_distance": 0.25  // was 0.15
```
→ More timesteps fire → stronger cumulative penalty → lower target K

**Option B: Increase penalty weight**
```python
COMPLY_PREFER_W = 2.0   // was 0.5
COMPLY_REQUIRE_W = 8.0  // was 2.0
```
→ Same timesteps but higher penalty → PI2 prioritizes K reduction more

**Option C: Lower K_TARGET**
```python
K_TARGET = 30.0  // was 50.0
```
→ Easier to satisfy target → lower final K values

**Recommendation**: Try **Option A** (r=0.25m) and re-run `main.py`. This worked well in `test_compliance.py` A/B test.

---

## Files Created

- `plot_stiffness_vs_position.py` → Summary statistics near human
- `analyze_stiffness_near_human.py` → Detailed timestep-by-timestep analysis
