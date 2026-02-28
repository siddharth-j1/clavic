# Compliance Penalty Deep Dive: Current vs Proposed

## Quick Answer to Your Questions

### Q1: Are we penalizing `trace(K)` or individual `Kxx, Kyy, Kzz`?
**→ Answer: Individual `Kxx, Kyy, Kzz` (per-axis penalty)**

```python
# Current code (compiler.py lines 140-151)
kxx = float(K_all[t, 0, 0])
kyy = float(K_all[t, 1, 1])
kzz = float(K_all[t, 2, 2])
excess_x = max(0.0, (kxx - K_TARGET) / K_TARGET)
excess_y = max(0.0, (kyy - K_TARGET) / K_TARGET)
excess_z = max(0.0, (kzz - K_TARGET) / K_TARGET)
stiff_excess = excess_x + excess_y + excess_z  # SUM of three axes
```

✅ **Good**: Allows PI2 to adjust each axis independently

---

### Q2: Is penalty based on distance in X, Y, Z axes?
**→ Answer: NO (currently uses 3D Euclidean distance, same for all axes)**

```python
# Current code (compiler.py line 126)
dists = np.linalg.norm(positions - center, axis=1)  # 3D distance
# = sqrt((x-cx)² + (y-cy)² + (z-cz)²)

# Then ONE proximity value applied to all three axes
proximity = (radius - d3d) / radius
penalty_w = base_w * clause_w * (proximity ** 2)

# All three axes get the SAME penalty_w
comply_cost += penalty_w * (excess_x + excess_y + excess_z)
```

❌ **Problem**: If robot is far in X but close in Y, still penalizes Kxx equally as Kyy

---

## Current Implementation (3D Distance)

### Example: Closest Approach

At **t = 0.860s** (closest to human):
- Robot position: [0.1744, 0.3285, 0.0815]
- Human position: [0.3000, 0.4000, 0.1100]

**Distance breakdown:**
```
dx = 0.1256 m  (robot is 12.6 cm away in X)
dy = 0.0715 m  (robot is  7.2 cm away in Y)
dz = 0.0285 m  (robot is  2.9 cm away in Z)  ← CLOSEST!

3D Euclidean distance = sqrt(0.1256² + 0.0715² + 0.0285²) = 0.1473 m
```

**Current penalty calculation:**
```
proximity = (0.15 - 0.1473) / 0.15 = 0.018  ← VERY WEAK!
penalty_w = 0.5 × 3.0 × 0.018² = 0.00049  ← ALMOST ZERO

Compliance cost = 0.00049 × (3.05 + 0.29 + 0.89) = 0.002 
← Invisible to PI2 (< 0.02% of total task cost)
```

**Why so weak?** Because:
1. Robot barely clips the 0.15m radius (min distance = 0.1473m)
2. 3D distance dominates — X-axis distance of 12.6cm drives d3d = 14.73cm
3. Proximity = (0.15 - 0.1473) / 0.15 = only 1.8% into the zone

---

## Proposed Implementation (Per-Axis Distance)

If we penalize each axis independently based on **its own distance**:

**X-axis (Kxx):**
```
dx = 0.1256 m
proximity_x = (0.15 - 0.1256) / 0.15 = 0.162  ← 16.2% into zone
penalty_w_x = 0.5 × 3.0 × 0.162² = 0.0396
cost_x = 0.0396 × 3.05 = 0.121  ← VISIBLE to PI2
```

**Y-axis (Kyy):**
```
dy = 0.0715 m
proximity_y = (0.15 - 0.0715) / 0.15 = 0.524  ← 52.4% into zone!
penalty_w_y = 0.5 × 3.0 × 0.524² = 0.411  ← MUCH STRONGER
cost_y = 0.411 × 0.29 = 0.120
```

**Z-axis (Kzz):**
```
dz = 0.0285 m
proximity_z = (0.15 - 0.0285) / 0.15 = 0.810  ← 81% into zone!!!
penalty_w_z = 0.5 × 3.0 × 0.810² = 0.985  ← VERY STRONG
cost_z = 0.985 × 0.89 = 0.872  ← NOW PI2 LISTENS
```

**Total per-axis cost: 0.121 + 0.120 + 0.872 = 1.11** (vs current 0.002)

That's **550× stronger** on Z, **245× stronger** on Y!

---

## Conceptual Difference

### Current (3D Euclidean):
```
"If robot's overall 3D distance to human is < 0.15m,
 reduce ALL axes equally based on that 3D distance"

→ Weak because X-axis distance dominates the 3D vector
```

### Proposed (Per-axis):
```
"For each axis (X, Y, Z):
   IF robot is close on that axis, penalize that axis's stiffness"

→ Strong because each axis is penalized independently
→ More intuitive: "Reduce Kz because robot is very close on Z-axis"
```

---

## Should We Switch to Per-Axis?

### Pros of per-axis:
✅ 550× stronger penalty on Z-axis → actually forces K reduction to target  
✅ More intuitive: penalize direction where robot is actually close  
✅ Better compliance: Y and Z will go toward 50 N/m  

### Cons of per-axis:
⚠️ Need to re-run `main.py` optimization (θ incompatible)  
⚠️ Might over-reduce some axes (need to tune K_TARGET per axis)  
⚠️ Changes the penalty structure (need to validate)  

---

## Recommendation

**Current state is correct but weak.** If you want stronger compliance near human:

**Option 1 (Easy, no code change):**
- Increase comfort zone radius: `r = 0.25m` instead of `0.15m`
- This fires compliance for more timesteps
- Re-run `main.py`

**Option 2 (Better, requires code change):**
- Switch to per-axis distance penalty (as shown above)
- Re-run `main.py`
- Will see much stronger K reduction toward target

**My recommendation:** Try **Option 1 first** (wider zone), which gives 550× penalty increase from more timesteps firing. If that's not enough, switch to **Option 2** for per-axis penalty.

Want me to implement Option 2?
