"""
Analyze stiffness (Kxx, Kyy, Kzz) vs spatial position.

Shows:
1. Stiffness near human (inside/outside comfort zone)
2. Stiffness vs distance to human 
3. Spatial distribution of stiffness

Helps answer: "What stiffness is the arm using when near the human?"
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from core.certified_policy import CertifiedPolicy

def load_checkpoint():
    """Load best theta and rollout."""
    data = np.load('optimal_checkpoint.npz', allow_pickle=True)
    theta = data['best_theta']
    tau = float(data['optimal_tau'])
    
    policy = CertifiedPolicy(tau=tau)
    trace = policy.rollout(theta)
    return trace

def main():
    trace = load_checkpoint()
    data = np.load('optimal_checkpoint.npz', allow_pickle=True)
    tau = float(data['optimal_tau'])
    
    # Extract trajectory
    pos = trace.position           # (T, 3)
    K_all = np.array(list(trace.gains['K']))  # (T, 3, 3)
    
    # Extract diagonal stiffness components
    kxx = K_all[:, 0, 0]
    kyy = K_all[:, 1, 1]
    kzz = K_all[:, 2, 2]
    tr_k = kxx + kyy + kzz
    
    # Gaussian smoothing for cleaner visualization
    kxx_smooth = gaussian_filter1d(kxx, sigma=2)
    kyy_smooth = gaussian_filter1d(kyy, sigma=2)
    kzz_smooth = gaussian_filter1d(kzz, sigma=2)
    
    # Human position and zones
    human_pos = np.array([0.30, 0.40, 0.11])
    comfort_r = 0.15
    body_r = 0.06
    
    # Distance to human at each timestep
    dists = np.linalg.norm(pos - human_pos, axis=1)
    
    print(f"\n{'='*70}")
    print(f"STIFFNESS VS POSITION ANALYSIS (τ = {tau} s)")
    print(f"{'='*70}\n")
    
    print(f"Distance to human:")
    print(f"  Min: {dists.min():.4f} m")
    print(f"  Max: {dists.max():.4f} m")
    print(f"  Mean: {dists.mean():.4f} m")
    print()
    
    print(f"Safety zones:")
    print(f"  Body exclusion: r = {body_r} m (HARD constraint)")
    print(f"  Comfort zone:   r = {comfort_r} m (PREFER constraint)")
    print()
    
    print(f"Stiffness ranges (overall):")
    print(f"  Kxx: {kxx.min():6.1f} – {kxx.max():6.1f} N/m  (mean = {kxx.mean():6.1f})")
    print(f"  Kyy: {kyy.min():6.1f} – {kyy.max():6.1f} N/m  (mean = {kyy.mean():6.1f})")
    print(f"  Kzz: {kzz.min():6.1f} – {kzz.max():6.1f} N/m  (mean = {kzz.mean():6.1f})")
    print()
    
    print(f"Stiffness ranges (smoothed):")
    print(f"  Kxx: {kxx_smooth.min():6.1f} – {kxx_smooth.max():6.1f} N/m  (mean = {kxx_smooth.mean():6.1f})")
    print(f"  Kyy: {kyy_smooth.min():6.1f} – {kyy_smooth.max():6.1f} N/m  (mean = {kyy_smooth.mean():6.1f})")
    print(f"  Kzz: {kzz_smooth.min():6.1f} – {kzz_smooth.max():6.1f} N/m  (mean = {kzz_smooth.mean():6.1f})")
    print()
    
    inside_comfort = dists < comfort_r
    inside_body = dists < body_r
    
    print(f"INSIDE comfort zone (d < {comfort_r} m):")
    print(f"  Timesteps: {inside_comfort.sum()} / {len(dists)}")
    if inside_comfort.sum() > 0:
        print(f"  Kxx: {kxx[inside_comfort].min():6.1f} – {kxx[inside_comfort].max():6.1f} N/m  (mean = {kxx[inside_comfort].mean():6.1f} ± {kxx[inside_comfort].std():5.1f})")
        print(f"  Kyy: {kyy[inside_comfort].min():6.1f} – {kyy[inside_comfort].max():6.1f} N/m  (mean = {kyy[inside_comfort].mean():6.1f} ± {kyy[inside_comfort].std():5.1f})")
        print(f"  Kzz: {kzz[inside_comfort].min():6.1f} – {kzz[inside_comfort].max():6.1f} N/m  (mean = {kzz[inside_comfort].mean():6.1f} ± {kzz[inside_comfort].std():5.1f})")
    else:
        print(f"  (Robot NEVER enters comfort zone — compliance penalty never fires)")
    print()
    
    outside_comfort = dists >= comfort_r
    print(f"OUTSIDE comfort zone (d ≥ {comfort_r} m):")
    print(f"  Timesteps: {outside_comfort.sum()} / {len(dists)}")
    print(f"  Kxx: {kxx[outside_comfort].min():6.1f} – {kxx[outside_comfort].max():6.1f} N/m  (mean = {kxx[outside_comfort].mean():6.1f} ± {kxx[outside_comfort].std():5.1f})")
    print(f"  Kyy: {kyy[outside_comfort].min():6.1f} – {kyy[outside_comfort].max():6.1f} N/m  (mean = {kyy[outside_comfort].mean():6.1f} ± {kyy[outside_comfort].std():5.1f})")
    print(f"  Kzz: {kzz[outside_comfort].min():6.1f} – {kzz[outside_comfort].max():6.1f} N/m  (mean = {kzz[outside_comfort].mean():6.1f} ± {kzz[outside_comfort].std():5.1f})")
    print()
    
    if inside_comfort.sum() > 0:
        print(f"DIFFERENCE (inside - outside):")
        diff_kxx = kxx[inside_comfort].mean() - kxx[outside_comfort].mean()
        diff_kyy = kyy[inside_comfort].mean() - kyy[outside_comfort].mean()
        diff_kzz = kzz[inside_comfort].mean() - kzz[outside_comfort].mean()
        print(f"  Kxx: {diff_kxx:+7.1f} N/m  {'← REDUCED' if diff_kxx < 0 else '← INCREASED'}")
        print(f"  Kyy: {diff_kyy:+7.1f} N/m  {'← REDUCED' if diff_kyy < 0 else '← INCREASED'}")
        print(f"  Kzz: {diff_kzz:+7.1f} N/m  {'← REDUCED' if diff_kzz < 0 else '← INCREASED'}")
    print()
    
    print(f"Target stiffness (K_TARGET): 50.0 N/m per axis")
    print(f"Distance where compliance penalty at 50% strength:")
    print(f"  Comfort zone: d = {comfort_r * 0.707:.4f} m  (proximity = 0.707)")
    print()
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
