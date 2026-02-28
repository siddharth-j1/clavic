"""
Detailed analysis: Show stiffness values at each timestep when robot is near human.
"""

import numpy as np
from core.certified_policy import CertifiedPolicy

data = np.load('optimal_checkpoint.npz', allow_pickle=True)
theta = data['best_theta']
tau = float(data['optimal_tau'])

policy = CertifiedPolicy(tau=tau)
trace = policy.rollout(theta)

pos = trace.position
K_all = np.array(list(trace.gains['K']))
kxx = K_all[:, 0, 0]
kyy = K_all[:, 1, 1]
kzz = K_all[:, 2, 2]

human_pos = np.array([0.30, 0.40, 0.11])
dists = np.linalg.norm(pos - human_pos, axis=1)
times = trace.time

print("\n" + "="*120)
print("STIFFNESS NEAR HUMAN (closest approach)")
print("="*120)
print()

# Find all timesteps near human (within 0.20m)
near_human = dists < 0.20
near_indices = np.where(near_human)[0]

if len(near_indices) > 0:
    print(f"{'Time':<8} {'Distance':<12} {'Kxx':<10} {'Kyy':<10} {'Kzz':<10} {'tr(K)':<10} {'Status':<20}")
    print(f"{'(s)':<8} {'(m)':<12} {'(N/m)':<10} {'(N/m)':<10} {'(N/m)':<10} {'(N/m)':<10} {'':<20}")
    print("-" * 120)
    
    for idx in near_indices:
        t = times[idx]
        d = dists[idx]
        kx = kxx[idx]
        ky = kyy[idx]
        kz = kzz[idx]
        tr = kx + ky + kz
        
        # Zone detection
        if d < 0.06:
            zone = "BODY (HARD EXCLUDE)"
        elif d < 0.15:
            zone = "COMFORT (PREFER)"
        elif d < 0.20:
            zone = "APPROACH"
        else:
            zone = "FAR"
        
        print(f"{t:<8.3f} {d:<12.4f} {kx:<10.1f} {ky:<10.1f} {kz:<10.1f} {tr:<10.1f} {zone:<20}")
    
    # Summary statistics
    print()
    print("Summary within 0.20m zone:")
    print(f"  Kxx: {kxx[near_human].mean():6.1f} ± {kxx[near_human].std():5.1f} N/m")
    print(f"  Kyy: {kyy[near_human].mean():6.1f} ± {kyy[near_human].std():5.1f} N/m")
    print(f"  Kzz: {kzz[near_human].mean():6.1f} ± {kzz[near_human].std():5.1f} N/m")
    print(f"  tr(K): {(kxx[near_human]+kyy[near_human]+kzz[near_human]).mean():6.1f} ± {(kxx[near_human]+kyy[near_human]+kzz[near_human]).std():5.1f} N/m")
print()

# Compare to overall mean
overall_mean_kxx = kxx.mean()
overall_mean_kyy = kyy.mean()
overall_mean_kzz = kzz.mean()

print("Overall trajectory (all timesteps):")
print(f"  Kxx: {overall_mean_kxx:6.1f} N/m (near human: {kxx[near_human].mean():6.1f})")
print(f"  Kyy: {overall_mean_kyy:6.1f} N/m (near human: {kyy[near_human].mean():6.1f})")
print(f"  Kzz: {overall_mean_kzz:6.1f} N/m (near human: {kzz[near_human].mean():6.1f})")
print()

# Check if compliance penalty is actually working
inside_comfort = dists < 0.15
print(f"Compliance firing analysis:")
print(f"  Timesteps inside comfort (r=0.15m): {inside_comfort.sum()} out of {len(dists)}")
if inside_comfort.sum() > 0:
    print(f"    Average Kyy inside: {kyy[inside_comfort].mean():.1f} N/m")
    print(f"    Average Kyy outside: {kyy[~inside_comfort].mean():.1f} N/m")
    print(f"    → Kyy reduced by {(kyy[~inside_comfort].mean() - kyy[inside_comfort].mean()):.1f} N/m (compliance effect)")
    print(f"    Average Kzz inside: {kzz[inside_comfort].mean():.1f} N/m")
    print(f"    Average Kzz outside: {kzz[~inside_comfort].mean():.1f} N/m")
    print(f"    → Kzz reduced by {(kzz[~inside_comfort].mean() - kzz[inside_comfort].mean()):.1f} N/m (compliance effect)")
else:
    print(f"    Robot NEVER enters comfort zone → compliance penalty NEVER fires")
print()
print("="*120)
print()
