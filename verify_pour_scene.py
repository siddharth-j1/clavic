"""
verify_pour_scene.py — Load the saved pour checkpoint and produce
a single comprehensive verification figure with PASS/FAIL annotations.

Checks verified:
  [1] Phase 1  : Robot reaches goal position within t=0-4 s
  [2] Phase 2  : Robot holds still at goal position during t=4-6 s
  [3] Phase 1-2: Mug stays UPRIGHT (tilt < 15 deg) during carry & hold
  [4] Phase 3  : Mug tilts to POUR orientation (< 10 deg error) by t=10 s
  [5] All time : Obstacle never entered (clearance > 0)
  [6] All time : Linear velocity < 0.8 m/s
  [7] All time : Angular velocity < 1.5 rad/s
  [8] All time : Stiffness K always positive definite
  [9] Phase timeline: phases clearly separated at t=4 and t=6

Output : scene2_pour_verification.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys

sys.path.insert(0, ".")

from experiment_checkpoint_warmstart import load_checkpoint
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from spec.json_parser import load_taskspec_from_json
from core.cgms.quat_utils import quat_distance, quat_normalize

# ── Constants (must match main_scene2_pour.py) ───────────────────────
START    = np.array([0.55, 0.00, 0.30])
GOAL     = np.array([0.30, 0.55, 0.30])
OBSTACLE = np.array([0.40, 0.30, 0.30])
OBS_RAD  = 0.10

Q_UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
Q_POUR    = quat_normalize(np.array([0.70710678, 0.0, 0.70710678, 0.0]))

T_CARRY_END = 4.0
T_HOLD_END  = 6.0
HORIZON     = 10.0

# Limits
POS_TOL_M      = 0.04     # m   — goal / hold position tolerance
HOLD_V_MAX     = 0.05     # m/s — max speed during hold (50 mm/s, same as zero_velocity default)
VEL_LIMIT      = 0.80     # m/s — Cartesian speed limit
OMEGA_LIMIT    = 1.50     # rad/s
UPRIGHT_DEG    = 15.0     # deg — max tilt during carry+hold
POUR_TOL_DEG   = 10.0     # deg — pour orientation tolerance

sns.set_style("whitegrid")
palette = sns.color_palette("deep")

# ── Load checkpoint & rollout ────────────────────────────────────────
print("Loading checkpoint: scene2_pour_checkpoint.npz ...")
best_theta, saved_tau, best_cost, can_use = load_checkpoint("scene2_pour_checkpoint.npz")
if best_theta is None:
    raise FileNotFoundError("scene2_pour_checkpoint.npz not found. Run main_scene2_pour.py first.")
print(f"  theta_dim={len(best_theta)}, cost={best_cost:.4f}, tau={saved_tau:.1f}s")

taskspec = load_taskspec_from_json("spec/scene2_pour_task.json")
policy   = MultiPhaseCertifiedPolicy(taskspec.phases)
trace    = policy.rollout(best_theta)

t      = trace.time
pos    = trace.position
vel    = trace.velocity
q      = trace.orientation          # (T,4)
omega  = trace.angular_velocity     # (T,3)
K_arr  = trace.gains["K"]
D_arr  = trace.gains["D"]

# ── Derived signals ──────────────────────────────────────────────────
speed      = np.linalg.norm(vel, axis=1)
omega_norm = np.linalg.norm(omega, axis=1)
d_goal     = np.linalg.norm(pos - GOAL, axis=1)
d_obs      = np.linalg.norm(pos[:, :2] - OBSTACLE[:2], axis=1)
d_upright  = np.degrees(np.array([quat_distance(q[k], Q_UPRIGHT) for k in range(len(t))]))
d_pour     = np.degrees(np.array([quat_distance(q[k], Q_POUR)    for k in range(len(t))]))
trK        = np.array([np.trace(K) for K in K_arr])
trD        = np.array([np.trace(D) for D in D_arr])
K_eig_min  = np.array([np.linalg.eigvalsh(K)[0] for K in K_arr])

# ── Masks for each phase ─────────────────────────────────────────────
m_carry = t <= T_CARRY_END
m_hold  = (t >= T_CARRY_END) & (t <= T_HOLD_END)
m_pour  = t >= T_HOLD_END

# ── Run all 9 checks ────────────────────────────────────────────────
checks = {}

# [1] Goal reached in phase 1
goal_in_p1  = np.any(d_goal[m_carry] < POS_TOL_M)
goal_t      = t[np.argmax(d_goal < POS_TOL_M)] if np.any(d_goal < POS_TOL_M) else None
checks[1]   = (goal_in_p1, f"Goal reached in phase 1 @ t={goal_t:.2f}s" if goal_in_p1
               else f"Goal NOT reached in phase 1 (min dist={d_goal[m_carry].min()*100:.1f} cm)")

# [2] Hold: near position AND low velocity during t=4-6s
d_hold_max  = d_goal[m_hold].max()
v_hold_max  = speed[m_hold].max()
hold_ok     = (d_hold_max < POS_TOL_M) and (v_hold_max < HOLD_V_MAX)
checks[2]   = (hold_ok, f"Hold OK: max dist={d_hold_max*100:.1f}cm, max speed={v_hold_max*1000:.2f}mm/s"
               if hold_ok else f"Hold FAIL: dist={d_hold_max*100:.1f}cm, speed={v_hold_max*1000:.2f}mm/s")

# [3] Upright during carry+hold (t=0-6s)
m_upright_window = t <= T_HOLD_END
max_tilt    = d_upright[m_upright_window].max()
upright_ok  = max_tilt < UPRIGHT_DEG
checks[3]   = (upright_ok, f"Max tilt during carry/hold = {max_tilt:.1f}° (limit {UPRIGHT_DEG}°)")

# [4] Pour orientation reached in phase 3
pour_in_p3   = np.any(d_pour[m_pour] < POUR_TOL_DEG)
min_pour_err = d_pour[m_pour].min()
checks[4]    = (pour_in_p3, f"Pour reached: min error = {min_pour_err:.1f}° (limit {POUR_TOL_DEG}°)"
                if pour_in_p3 else f"Pour NOT reached: min error = {min_pour_err:.1f}°")

# [5] Obstacle clearance
min_clear_m  = d_obs.min() - OBS_RAD
obs_ok       = min_clear_m > 0
checks[5]    = (obs_ok, f"Min obstacle clearance = {min_clear_m*100:.1f} cm")

# [6] Linear velocity
vel_ok       = speed.max() < VEL_LIMIT
checks[6]    = (vel_ok, f"Max speed = {speed.max():.3f} m/s (limit {VEL_LIMIT} m/s)")

# [7] Angular velocity
omega_ok     = omega_norm.max() < OMEGA_LIMIT
checks[7]    = (omega_ok, f"Max angular vel = {omega_norm.max():.3f} rad/s (limit {OMEGA_LIMIT} rad/s)")

# [8] K positive definite
k_pd_ok      = K_eig_min.min() > 0
checks[8]    = (k_pd_ok, f"K min eigenvalue = {K_eig_min.min():.5f} (must be > 0)")

# ── Print summary to terminal ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  POUR SCENE VERIFICATION   (best_cost = {best_cost:.4f})")
print(f"{'='*60}")
all_pass = True
for idx, (ok, msg) in checks.items():
    status = "PASS" if ok else "FAIL"
    print(f"  [{idx}] {status}  {msg}")
    if not ok:
        all_pass = False
print(f"{'='*60}")
print(f"  OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
print(f"{'='*60}\n")

# ── Build verification figure (3×3 subplots) ─────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle(
    f"Pour Scene — Verification Dashboard   (best cost = {best_cost:.4f}  |  "
    + ("ALL 8 CHECKS PASS" if all_pass else "SOME CHECKS FAILED") + ")",
    fontsize=15, fontweight='bold', y=1.00
)

PHASE_COLORS = {"carry": "royalblue", "hold": "seagreen", "pour": "tomato"}
PHASE_ALPHA  = 0.10

def _shade_phases(ax):
    ax.axvspan(0,            T_CARRY_END, alpha=PHASE_ALPHA, color=PHASE_COLORS["carry"], zorder=0)
    ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=PHASE_ALPHA*2, color=PHASE_COLORS["hold"],  zorder=0)
    ax.axvspan(T_HOLD_END,   HORIZON,     alpha=PHASE_ALPHA, color=PHASE_COLORS["pour"],  zorder=0)
    for t_ph in [T_CARRY_END, T_HOLD_END]:
        ax.axvline(t_ph, color='gray', linestyle=':', lw=1.2, alpha=0.7, zorder=1)
    ax.set_xlim(0, HORIZON)

def _pass_label(ax, ok):
    label = "PASS" if ok else "FAIL"
    color = "green" if ok else "red"
    ax.text(0.98, 0.97, label, transform=ax.transAxes, fontsize=12,
            fontweight='bold', color=color, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, linewidth=2.0))

# ─── [1] Distance to goal ────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t, d_goal * 100, color=palette[0], lw=2.2, label='Dist to goal')
ax.axhline(POS_TOL_M * 100, color='red', linestyle='--', lw=1.8, label=f'Tolerance {POS_TOL_M*100:.0f} cm')
ax.fill_between(t[m_carry], 0, d_goal[m_carry] * 100,
                where=d_goal[m_carry] < POS_TOL_M, alpha=0.3, color='green', label='Within tol.')
_shade_phases(ax)
ax.set_ylabel("Distance (cm)", fontsize=10, fontweight='bold')
ax.set_title("[1] Goal reached in Phase 1", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[1][0])

# ─── [2] Hold: position + speed during t=4-6s ────────────────────────
ax = axes[0, 1]
ax2 = ax.twinx()
ax.plot(t, d_goal * 100, color=palette[0], lw=2.2, label='Dist to goal (cm)')
ax2.plot(t, speed * 1000, color=palette[1], lw=2.2, linestyle='--', label='Speed (mm/s)')
ax.axhline(POS_TOL_M * 100, color='blue', linestyle='--', lw=1.5, alpha=0.7)
ax2.axhline(HOLD_V_MAX * 1000, color='orange', linestyle='--', lw=1.5, alpha=0.7)
_shade_phases(ax)
ax.set_ylabel("Dist (cm)", fontsize=10, fontweight='bold', color=palette[0])
ax2.set_ylabel("Speed (mm/s)", fontsize=10, fontweight='bold', color=palette[1])
ax.set_title("[2] Hold still at goal (t=4–6s)", fontsize=11, fontweight='bold')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[2][0])

# ─── [3] Mug tilt (upright constraint) ───────────────────────────────
ax = axes[0, 2]
ax.plot(t, d_upright, color=palette[3], lw=2.5, label='Tilt from upright')
ax.axhline(UPRIGHT_DEG, color='red', linestyle='--', lw=1.8, label=f'Limit {UPRIGHT_DEG}°')
ax.fill_between(t, d_upright, UPRIGHT_DEG,
                where=d_upright > UPRIGHT_DEG, alpha=0.4, color='red', label='Violation')
_shade_phases(ax)
ax.set_ylabel("Tilt angle (deg)", fontsize=10, fontweight='bold')
ax.set_title("[3] Mug upright during carry/hold", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[3][0])

# ─── [4] Pour orientation convergence ────────────────────────────────
ax = axes[1, 0]
ax.plot(t, d_pour, color=palette[2], lw=2.5, label='Error to pour orient.')
ax.axhline(POUR_TOL_DEG, color='red', linestyle='--', lw=1.8, label=f'Tolerance {POUR_TOL_DEG}°')
ax.fill_between(t[m_pour], 0, d_pour[m_pour],
                where=d_pour[m_pour] < POUR_TOL_DEG, alpha=0.3, color='green', label='Within tol.')
_shade_phases(ax)
ax.set_ylabel("Angle error (deg)", fontsize=10, fontweight='bold')
ax.set_title("[4] Pour orientation in Phase 3", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[4][0])

# ─── [5] Obstacle clearance ──────────────────────────────────────────
ax = axes[1, 1]
clearance_cm = (d_obs - OBS_RAD) * 100
ax.plot(t, clearance_cm, color=palette[4], lw=2.5, label='Obstacle clearance')
ax.axhline(0, color='red', linestyle='--', lw=1.8, label='Collision boundary')
ax.fill_between(t, 0, clearance_cm,
                where=clearance_cm < 0, alpha=0.5, color='red', label='Collision!')
_shade_phases(ax)
ax.set_ylabel("Clearance (cm)", fontsize=10, fontweight='bold')
ax.set_title("[5] Obstacle avoidance (always)", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[5][0])

# ─── [6] Linear velocity ─────────────────────────────────────────────
ax = axes[1, 2]
ax.plot(t, speed, color=palette[0], lw=2.2, label='EE speed')
ax.axhline(VEL_LIMIT, color='red', linestyle='--', lw=1.8, label=f'Limit {VEL_LIMIT} m/s')
ax.fill_between(t, speed, VEL_LIMIT,
                where=speed > VEL_LIMIT, alpha=0.5, color='red', label='Violation')
_shade_phases(ax)
ax.set_ylabel("Speed (m/s)", fontsize=10, fontweight='bold')
ax.set_title("[6] Linear velocity limit", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[6][0])

# ─── [7] Angular velocity ────────────────────────────────────────────
ax = axes[2, 0]
ax.plot(t, omega_norm, color='darkorchid', lw=2.2, label='||ω||')
ax.axhline(OMEGA_LIMIT, color='red', linestyle='--', lw=1.8, label=f'Limit {OMEGA_LIMIT} rad/s')
ax.fill_between(t, omega_norm, OMEGA_LIMIT,
                where=omega_norm > OMEGA_LIMIT, alpha=0.5, color='red', label='Violation')
_shade_phases(ax)
ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
ax.set_ylabel("||ω|| (rad/s)", fontsize=10, fontweight='bold')
ax.set_title("[7] Angular velocity limit", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[7][0])

# ─── [8] Stiffness PD check ──────────────────────────────────────────
ax = axes[2, 1]
ax.plot(t, K_eig_min, color='steelblue', lw=2.2, label='min eigenvalue K')
ax.plot(t, trK / 3, color='mediumpurple', lw=1.5, linestyle='--', label='tr(K)/3 (mean diag)')
ax.axhline(0, color='red', linestyle='--', lw=1.8, label='PD boundary (0)')
_shade_phases(ax)
ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
ax.set_ylabel("Eigenvalue (N/m)", fontsize=10, fontweight='bold')
ax.set_title("[8] Stiffness K always PD", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
_pass_label(ax, checks[8][0])

# ─── [9] Workspace 2D trajectory ─────────────────────────────────────
ax = axes[2, 2]
ax.plot(pos[m_carry, 0], pos[m_carry, 1], color=PHASE_COLORS["carry"], lw=3.0,
        label='Phase 1: carry', zorder=3)
ax.plot(pos[m_hold, 0],  pos[m_hold, 1],  color=PHASE_COLORS["hold"],  lw=4.0,
        label='Phase 2: hold',  zorder=4)
ax.plot(pos[m_pour, 0],  pos[m_pour, 1],  color=PHASE_COLORS["pour"],  lw=3.0,
        label='Phase 3: pour',  zorder=3)
# Obstacle
obs_circle = plt.Circle(OBSTACLE[:2], OBS_RAD, color='red', fill=True,
                        alpha=0.15, label=f'Obstacle r={OBS_RAD}m')
ax.add_patch(obs_circle)
ax.add_patch(plt.Circle(OBSTACLE[:2], OBS_RAD, color='red', fill=False,
                        linestyle='--', lw=2.0))
ax.add_patch(plt.Circle(GOAL[:2], POS_TOL_M, color='green', fill=False,
                        linestyle=':', lw=2.0, label=f'Goal tol. {POS_TOL_M*100:.0f}cm'))
ax.scatter(*START[:2], s=150, c='royalblue',  zorder=6, marker='o',
           edgecolor='navy', lw=2, label='Start')
ax.scatter(*GOAL[:2],  s=200, c='black',       zorder=6, marker='*', label='Goal')
ax.scatter(*OBSTACLE[:2], s=80, c='red',       zorder=5, marker='+', lw=3)
ax.set_xlabel("X (m)", fontsize=10, fontweight='bold')
ax.set_ylabel("Y (m)", fontsize=10, fontweight='bold')
ax.set_title("[9] 2D Workspace Trajectory", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

# ── Phase legend bar at bottom ────────────────────────────────────────
carry_p = mpatches.Patch(color=PHASE_COLORS["carry"], alpha=0.5, label='Phase 1: Carry (0–4 s)')
hold_p  = mpatches.Patch(color=PHASE_COLORS["hold"],  alpha=0.5, label='Phase 2: Hold  (4–6 s)')
pour_p  = mpatches.Patch(color=PHASE_COLORS["pour"],  alpha=0.5, label='Phase 3: Pour  (6–10 s)')
fig.legend(handles=[carry_p, hold_p, pour_p],
           loc='lower center', ncol=3, fontsize=11,
           framealpha=0.95, edgecolor='black',
           bbox_to_anchor=(0.5, -0.015))

plt.tight_layout(rect=[0, 0.03, 1, 1])
save_path = "scene2_pour_verification.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved -> {save_path}")
plt.close()
