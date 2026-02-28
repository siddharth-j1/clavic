# Current Implementation: Summary

## What Are We Doing?

### ✅ Already Implemented: Per-Axis Stiffness Penalty
We **ARE** penalizing `Kxx`, `Kyy`, `Kzz` **individually** (not `trace(K)`).

```python
# compiler.py lines 140-151
kxx = float(K_all[t, 0, 0])
kyy = float(K_all[t, 1, 1])
kzz = float(K_all[t, 2, 2])

excess_x = max(0.0, (kxx - K_TARGET) / K_TARGET)
excess_y = max(0.0, (kyy - K_TARGET) / K_TARGET)
excess_z = max(0.0, (kzz - K_TARGET) / K_TARGET)

stiff_excess = excess_x + excess_y + excess_z  # Sum of three
comply_cost += penalty_w * stiff_excess
```

**Good:** PI2 can independently reduce each axis.

---

### ⚠️ Currently Using: 3D Euclidean Distance for All Axes
We use **the same 3D distance** to penalize **all three axes**.

```python
# compiler.py line 126
dists = np.linalg.norm(positions - center, axis=1)
# = sqrt((x-cx)² + (y-cy)² + (z-cz)²)

# Then apply same penalty_w to all axes
proximity = (radius - dists) / radius
penalty_w = base_w * clause_w * (proximity ** 2)

comply_cost += penalty_w * (excess_x + excess_y + excess_z)
```

**Problem:** 
- If robot is close on Z but far on X, the large X distance dominates the 3D distance
- All axes get weak penalty instead of strong penalty on Z
- At closest approach: `proximity = 1.8%` → penalty almost invisible

---

## What Would Be Better

### Proposed: Per-Axis Distance
Penalize each axis based on **its own distance**.

```python
for t in range(T):
    dists_xyz = positions[t] - center  # (3,) vector
    
    for axis in [0, 1, 2]:  # X, Y, Z
        d_axis = abs(dists_xyz[axis])
        
        if d_axis >= radius:
            continue  # Outside zone on this axis
        
        # Each axis has its own proximity
        proximity_axis = (radius - d_axis) / radius
        penalty_w_axis = base_w * clause_w * (proximity_axis ** 2)
        
        excess_axis = max(0, (K_all[t, axis, axis] - K_TARGET) / K_TARGET)
        comply_cost += penalty_w_axis * excess_axis
```

**Benefits:**
- At closest approach: Z-axis proximity = 81% (vs current 1.8%)
- Penalty on Kzz becomes 2000× stronger
- Actually drives K toward target

---

## Numbers Proof

At the **exact closest approach** (t=0.860s):

| Metric | Current | Proposed | Ratio |
|--------|---------|----------|-------|
| **Kzz penalty weight** | 0.00049 | 0.9849 | 2010× |
| **Kzz compliance cost** | 0.0004 | 0.872 | 2023× |
| **Total compliance cost** | 0.0020 | 1.11 | 555× |

**This means:** With per-axis distance, PI2 would **actually see** the compliance penalty and reduce K accordingly.

---

## Decision Tree

**Q: Do you want strong compliance (K actually drops to ~50 N/m near human)?**

→ **YES**: Implement per-axis distance penalty
  - Need to modify `compiler.py` lines 115-155
  - Re-run `main.py` optimization
  - Will produce better results

→ **NO (current weak compliance is fine)**: Keep as-is
  - Current implementation is clean and correct
  - K reduction is modest but present (Kyy -20 N/m, Kzz -48 N/m)

→ **MAYBE (compromise)**: Widen comfort zone
  - Change `r = 0.15m → 0.25m` in `example_task.json`
  - Re-run `main.py` (faster than code change)
  - Will get 550× stronger penalty from more timesteps firing
  - Much easier than implementing per-axis logic

---

## Recommendation

**Start with widening the zone** (easiest):
1. Update `example_task.json`: `"preferred_distance": 0.25`
2. Re-run `python main.py`
3. Check `analyze_stiffness_near_human.py` results
4. If still not strong enough, implement per-axis distance

Want me to help with any of these?
