"""
Visual summary of current vs proposed compliance penalty structure.
"""

import numpy as np

print("\n" + "="*100)
print("COMPLIANCE PENALTY STRUCTURE COMPARISON")
print("="*100)
print()

print("QUESTION 1: What are we penalizing?")
print("-" * 100)
print()
print("CURRENT: Individual per-axis stiffness")
print("  Code: kxx, kyy, kzz = K[t,0,0], K[t,1,1], K[t,2,2]")
print("        excess_x = max(0, (kxx - K_TARGET) / K_TARGET)")
print("        excess_y = max(0, (kyy - K_TARGET) / K_TARGET)")
print("        excess_z = max(0, (kzz - K_TARGET) / K_TARGET)")
print("        stiff_excess = excess_x + excess_y + excess_z")
print()
print("  Benefit: Can reduce individual axes independently")
print("  Status: ✓ Already doing this")
print()
print()

print("QUESTION 2: What distance triggers the penalty?")
print("-" * 100)
print()
print("CURRENT: 3D Euclidean distance (same for all axes)")
print("  Code: dists = norm(positions - center)  # sqrt(dx² + dy² + dz²)")
print("        proximity = (radius - dists) / radius")
print("        penalty_w = base_w × clause_w × proximity²")
print()
print("  Problem: X-axis distance dominates 3D vector")
print("           → Y and Z penalties are VERY weak")
print()
print("PROPOSED: Per-axis distance (separate for each axis)")
print("  Pseudocode:")
print("    for each axis (x, y, z):")
print("      d_axis = |pos[axis] - center[axis]|")
print("      proximity_axis = (radius - d_axis) / radius")
print("      penalty_w_axis = base_w × clause_w × proximity_axis²")
print("      cost += penalty_w_axis × excess_axis")
print()
print("  Benefit: Penalty strength reflects actual distance on that axis")
print()
print()

print("STRENGTH COMPARISON AT CLOSEST APPROACH")
print("-" * 100)
print()

# Data from actual trajectory
data_current = {
    'Kxx': {'distance': 0.1256, 'proximity': 0.0180, 'penalty_w': 0.00049, 'excess': 3.0543, 'cost': 0.001496},
    'Kyy': {'distance': 0.0715, 'proximity': 0.0180, 'penalty_w': 0.00049, 'excess': 0.2911, 'cost': 0.000142},
    'Kzz': {'distance': 0.0285, 'proximity': 0.0180, 'penalty_w': 0.00049, 'excess': 0.8855, 'cost': 0.000431},
}

data_proposed = {
    'Kxx': {'distance': 0.1256, 'proximity': 0.1625, 'penalty_w': 0.0396, 'excess': 3.0543, 'cost': 0.1209},
    'Kyy': {'distance': 0.0715, 'proximity': 0.5237, 'penalty_w': 0.4113, 'excess': 0.2911, 'cost': 0.1197},
    'Kzz': {'distance': 0.0285, 'proximity': 0.8103, 'penalty_w': 0.9849, 'excess': 0.8855, 'cost': 0.8721},
}

print("CURRENT (3D Euclidean distance for all axes):")
print(f"{'Axis':<6} {'Distance':<12} {'Proximity':<12} {'Penalty Weight':<16} {'Stiff Excess':<14} {'Cost':<12}")
print("-" * 100)
for axis in ['Kxx', 'Kyy', 'Kzz']:
    d = data_current[axis]
    print(f"{axis:<6} {d['distance']:>10.4f} m {d['proximity']:>10.6f} {d['penalty_w']:>14.6f} {d['excess']:>12.4f} {d['cost']:>10.6f}")

print()
print("PROPOSED (per-axis distance):")
print(f"{'Axis':<6} {'Distance':<12} {'Proximity':<12} {'Penalty Weight':<16} {'Stiff Excess':<14} {'Cost':<12}")
print("-" * 100)
for axis in ['Kxx', 'Kyy', 'Kzz']:
    d = data_proposed[axis]
    print(f"{axis:<6} {d['distance']:>10.4f} m {d['proximity']:>10.6f} {d['penalty_w']:>14.6f} {d['excess']:>12.4f} {d['cost']:>10.6f}")

print()
print("PENALTY STRENGTH INCREASE (Current → Proposed):")
print(f"{'Axis':<6} {'Weight Factor':<20} {'Cost Factor':<20}")
print("-" * 100)
for axis in ['Kxx', 'Kyy', 'Kzz']:
    w_ratio = data_proposed[axis]['penalty_w'] / data_current[axis]['penalty_w']
    c_ratio = data_proposed[axis]['cost'] / data_current[axis]['cost']
    print(f"{axis:<6} {w_ratio:>18.1f}× {c_ratio:>18.1f}×")

print()
print(f"{'Average increase:':<30} {np.mean([data_proposed[ax]['penalty_w']/data_current[ax]['penalty_w'] for ax in ['Kxx','Kyy','Kzz']]):>8.1f}× penalty weight")
print(f"{'                 ':<30} {np.mean([data_proposed[ax]['cost']/data_current[ax]['cost'] for ax in ['Kxx','Kyy','Kzz']]):>8.1f}× total cost")

print()
print()

print("WHY THE DIFFERENCE IS SO LARGE")
print("-" * 100)
print()
print("In this example, the closest approach has:")
print("  dX = 0.1256 m (X-axis dominates)")
print("  dY = 0.0715 m")
print("  dZ = 0.0285 m (Z-axis closest!)")
print()
print("CURRENT method:")
print("  All axes use: d3D = sqrt(0.1256² + 0.0715² + 0.0285²) = 0.1473 m")
print("  This is close to the radius boundary (0.15m)")
print("  So proximity ≈ 1.8% → very weak penalty")
print()
print("PROPOSED method:")
print("  Kxx uses: dX = 0.1256 m → proximity = 16.3% → moderate penalty")
print("  Kyy uses: dY = 0.0715 m → proximity = 52.4% → strong penalty")
print("  Kzz uses: dZ = 0.0285 m → proximity = 81.0% → VERY strong penalty")
print()
print("The Z-axis is MUCH closer than the 3D distance suggests!")
print()
print()

print("="*100)
print()
