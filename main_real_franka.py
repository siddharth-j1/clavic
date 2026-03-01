"""
Real Franka workspace — cup handover to human.

Task:
  Phase 1 (0-4 s):  Start (holding cup) -> Hold position above laptop right edge
  Phase 2 (4-6 s):  Hold still at waypoint (near-zero velocity)
  Phase 3 (6-10 s): Hold position -> Human (goal)

Key coordinates (all lifted to safe_z = 0.297 m):
  #1 Start : [0.36845, -0.07430, 0.297]
  #7 Hold  : [0.61554,  0.27877, 0.297]
  #6 Goal  : [0.30180,  0.72820, 0.297]
  Laptop exclusion — ONE circumscribed circle (diagonal radius):
        Centre: [0.33825, 0.34875, 0.297]
        Radius: 0.223 m  (half-diagonal 0.2030 m + 2 cm margin)
        Covers entire laptop footprint; hold->goal straight line clips it ->
        optimizer must arc around the back.
Planning note:
  safe_z = max(all measured Z) + 4 cm = 0.257 + 0.040 = 0.297 m.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from optimization.optimizer import PIBB          # range-normalised, stable means
from experiment_checkpoint_warmstart import save_checkpoint

from logic.predicates import (
    at_goal_pose, at_waypoint, hold_at_waypoint,
    obstacle_avoidance, velocity_limit,
    human_comfort_distance, human_body_exclusion,
)

# ── Scene constants (from real Franka measurements) ──────────────────
SAFE_Z   = 0.297          # planning plane height (m)
START    = np.array([0.36845, -0.07430, SAFE_Z])
HOLD     = np.array([0.61554,  0.27877, SAFE_Z])
GOAL     = np.array([0.30180,  0.72820, SAFE_Z])

# Laptop corners (for plotting only)
LAPTOP_CORNERS_XY = np.array([
    [0.51034, 0.23381],   # #2 right-front
    [0.51542, 0.46060],   # #3 right-back
    [0.15995, 0.46442],   # #4 left-back
    [0.16797, 0.25102],   # #5 left-front
])

# Single circumscribed circle (diagonal radius) — covers entire laptop
LAPTOP_CENTER = np.array([0.33825, 0.34875, SAFE_Z])
LAPTOP_RADIUS = 0.223     # half-diagonal 0.2030 m + 2 cm margin

# Phase time boundaries (must match JSON durations: 4+2+4 = 10 s)
T_HOLD_START      = 4.0
T_HOLD_END        = 6.0
HOLD_WINDOW_START = T_HOLD_START + 0.05   # skip single boundary-velocity point


# ── Predicate registry ───────────────────────────────────────────────
def build_predicate_registry():
    return {
        "AtGoalPose":           at_goal_pose,
        "AtWaypoint":           at_waypoint,
        "HoldAtWaypoint":       hold_at_waypoint,
        "LaptopExclusion":      obstacle_avoidance,
        "ObstacleAvoidance":    obstacle_avoidance,
        "VelocityLimit":        velocity_limit,
        "HumanComfortDistance": human_comfort_distance,
        "HumanBodyExclusion":   human_body_exclusion,
    }


# ── Diagnostics ──────────────────────────────────────────────────────
def print_diagnostics(trace, best_cost):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)

    mask_hold = (trace.time >= HOLD_WINDOW_START) & (trace.time <= T_HOLD_END)

    rho_goal = at_goal_pose(trace, GOAL, 0.05)
    goal_hit = trace.time[rho_goal > 0]

    rho_wp   = at_waypoint(trace, HOLD, 0.04)
    wp_hit   = trace.time[rho_wp > 0]

    lap_dist = np.linalg.norm(pos[:, :2] - LAPTOP_CENTER[:2], axis=1)
    clearance = np.min(lap_dist) - LAPTOP_RADIUS

    print("\n========== RESULT DIAGNOSTICS ==========")
    print(f"  Best cost       : {best_cost:.4f}")
    print(f"  Waypoint reached: {'YES at t=' + f'{wp_hit[0]:.2f}s' if len(wp_hit) else 'NO'}")
    print(f"  Goal reached    : {'YES at t=' + f'{goal_hit[0]:.2f}s' if len(goal_hit) else 'NO'}")
    print(f"  Max speed global: {np.max(speed):.4f} m/s  (limit 0.8 m/s)")
    print(f"  Max speed hold  : {np.max(speed[mask_hold]):.6f} m/s  (limit 0.01 m/s)")
    print(f"  Laptop clearance: {clearance*100:.1f} cm  ({'OK' if clearance >= 0 else 'VIOLATION'})")

    K_arr  = trace.gains["K"]
    D_arr  = trace.gains["D"]
    trK    = np.array([np.trace(K) for K in K_arr])
    trD    = np.array([np.trace(D) for D in D_arr])
    K_eig_min = np.min([np.linalg.eigvalsh(K)[0] for K in K_arr])
    D_eig_min = np.min([np.linalg.eigvalsh(D)[0] for D in D_arr])
    alpha = 0.05
    lyap_viol = sum(1 for D in D_arr
                    if np.any(np.linalg.eigvalsh(alpha * np.eye(3) - D) > 1e-9))
    zeta_arr  = trD / (2.0 * np.sqrt(trK))

    print(f"  tr(K) range     : [{np.min(trK):.1f}, {np.max(trK):.1f}] N/m")
    print(f"  tr(D) range     : [{np.min(trD):.3f}, {np.max(trD):.3f}] Ns/m")
    print(f"  K eig min       : {K_eig_min:.4f}  (must be > 0)")
    print(f"  D eig min       : {D_eig_min:.4f}  (must be > 0)")
    print(f"  Damping ratio z : [{np.min(zeta_arr):.4f}, {np.max(zeta_arr):.4f}]")
    print(f"  Lyapunov viol   : {lyap_viol}/{len(K_arr)}")
    print("=========================================\n")


# ── Plotting ─────────────────────────────────────────────────────────
def plot_results(trace, best_cost, save_path="real_franka_result.png"):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    trK   = np.array([np.trace(K) for K in K_arr])
    trD   = np.array([np.trace(D) for D in D_arr])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Real Franka — Cup Handover  |  best_cost={best_cost:.3f}",
        fontsize=13, fontweight='bold'
    )

    # ── 1. Workspace XY (top-down) ───────────────────────────────────
    ax = axes[0, 0]

    lc = np.vstack([LAPTOP_CORNERS_XY, LAPTOP_CORNERS_XY[0]])
    ax.fill(LAPTOP_CORNERS_XY[:, 0], LAPTOP_CORNERS_XY[:, 1],
            color='gray', alpha=0.25, label='Laptop surface')
    ax.plot(lc[:, 0], lc[:, 1], 'k-', linewidth=1.0, alpha=0.6)

    # Single circumscribed exclusion circle
    circ = plt.Circle(LAPTOP_CENTER[:2], LAPTOP_RADIUS,
                      color='red', fill=True, alpha=0.12,
                      label=f'Laptop excl. (r={LAPTOP_RADIUS} m)')
    ax.add_patch(circ)
    ax.add_patch(plt.Circle(LAPTOP_CENTER[:2], LAPTOP_RADIUS,
                            color='red', fill=False, linestyle='--', linewidth=1.4))

    # Waypoint (hold) tolerance circle
    ax.add_patch(plt.Circle(HOLD[:2], 0.04, color='blue', fill=False,
                            linestyle=':', linewidth=1.5, label='Hold tol (r=0.04 m)'))

    # Goal tolerance circle
    ax.add_patch(plt.Circle(GOAL[:2], 0.05, color='black', fill=False,
                            linestyle=':', linewidth=1.0, alpha=0.5,
                            label='Goal tol (r=0.05 m)'))

    # Phase-coloured trajectory
    t_p1_end = np.searchsorted(trace.time, T_HOLD_START)
    t_p2_end = np.searchsorted(trace.time, T_HOLD_END)
    ax.plot(pos[:t_p1_end+1,  0], pos[:t_p1_end+1,  1], 'b-', lw=2, label='Phase 1 (approach)')
    ax.plot(pos[t_p1_end:t_p2_end+1, 0], pos[t_p1_end:t_p2_end+1, 1],
            'g-', lw=3, label='Phase 2 (hold)')
    ax.plot(pos[t_p2_end:, 0],  pos[t_p2_end:, 1],  'r-', lw=2, label='Phase 3 (to goal)')

    ax.scatter(*START[:2], s=120, c='green', zorder=6, marker='o', label='Start (cup)')
    ax.scatter(*HOLD[:2],  s=120, c='blue',  zorder=6, marker='s', label='Hold waypoint')
    ax.scatter(*GOAL[:2],  s=120, c='black', zorder=6, marker='*', label='Goal (human)')
    ax.scatter(*LAPTOP_CENTER[:2], s=50, c='red', zorder=5, marker='+')

    for c, lbl in zip(LAPTOP_CORNERS_XY, ['#2','#3','#4','#5']):
        ax.annotate(lbl, c, fontsize=7, color='gray',
                    xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Workspace (XY view, planning plane z=0.297 m)")
    ax.legend(fontsize=7, loc='upper left')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # ── 2. Speed vs time ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(trace.time, speed, 'b-', lw=1.5)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green', label='Hold phase')
    ax.axhline(0.01, color='gray',  linestyle='--', lw=1.2, label='Hold threshold (0.01 m/s)')
    ax.axhline(0.80, color='red',   linestyle='--', lw=1.0, label='Speed limit (0.8 m/s)')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
    ax.set_title("End-Effector Speed vs Time")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 3. Distances vs time ─────────────────────────────────────────
    ax = axes[0, 2]
    d_hold = np.linalg.norm(pos - HOLD,       axis=1)
    d_goal = np.linalg.norm(pos - GOAL,       axis=1)
    d_lap  = np.linalg.norm(pos[:, :2] - LAPTOP_CENTER[:2], axis=1)
    ax.plot(trace.time, d_hold, 'b-',  lw=1.5, label='dist(hold waypoint)')
    ax.plot(trace.time, d_goal, 'k--', lw=1.5, label='dist(goal)')
    ax.plot(trace.time, d_lap,  'r:',  lw=1.5, label='dist(laptop centre)')
    ax.axhline(0.04, color='blue',  linestyle=':', alpha=0.6, label='hold tol (0.04 m)')
    ax.axhline(0.05, color='black', linestyle=':', alpha=0.4, label='goal tol (0.05 m)')
    ax.axhline(LAPTOP_RADIUS, color='red', linestyle=':', alpha=0.6,
               label=f'laptop r ({LAPTOP_RADIUS} m)')
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.1, color='green')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
    ax.set_title("Key Distances vs Time")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── 4. Stiffness tr(K) ───────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(trace.time, trK, 'm-', lw=1.5)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green', label='Hold phase')
    for ph_t, lbl in [(T_HOLD_START, 'p1->p2'), (T_HOLD_END, 'p2->p3')]:
        ax.axvline(ph_t, color='orange', linestyle='--', lw=0.8, alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("tr(K) (N/m)")
    ax.set_title("Stiffness Evolution tr(K)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 5. Damping tr(D) ─────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(trace.time, trD, 'c-', lw=1.5)
    ax.axhline(30.0, color='orange', linestyle='--', lw=1.2,
               label='Min target tr(D)=30 Ns/m')
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green', label='Hold phase')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("tr(D) (Ns/m)")
    ax.set_title("Damping Evolution tr(D)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 6. Z height profile ──────────────────────────────────────────
    ax = axes[1, 2]
    ax.plot(trace.time, pos[:, 2], 'g-', lw=1.5, label='Z (height)')
    ax.axhline(SAFE_Z, color='gray', linestyle='--', lw=1.0,
               label=f'Planning z = {SAFE_Z} m')
    ax.axhline(0.002, color='brown', linestyle=':', lw=1.0,
               label='Laptop surface z ~0.002 m')
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green', label='Hold phase')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Z height (m)")
    ax.set_title("Height Profile (Z vs Time)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved -> {save_path}")
    plt.close()

    # ── Figure 2: Per-axis velocities and stiffnesses ─────────────────
    vel    = trace.velocity                              # (T, 3)
    K_diag = np.array([np.diag(K) for K in K_arr])      # (T, 3) Kxx,Kyy,Kzz

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
    fig2.suptitle("Per-Axis Velocities and Stiffnesses", fontsize=13, fontweight='bold')
    axis_labels = ['X', 'Y', 'Z']
    vel_colors  = ['steelblue', 'darkorange', 'seagreen']
    k_colors    = ['mediumpurple', 'tomato', 'goldenrod']

    for i, (lbl, vc, kc) in enumerate(zip(axis_labels, vel_colors, k_colors)):
        ax2 = axes2[0, i]
        ax2.plot(trace.time, vel[:, i], color=vc, lw=1.5)
        ax2.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green')
        ax2.axhline(0.0, color='gray', linestyle='--', lw=0.8)
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel(f"V{lbl} (m/s)")
        ax2.set_title(f"Velocity {lbl}-axis"); ax2.grid(True, alpha=0.3)

        ax2 = axes2[1, i]
        ax2.plot(trace.time, K_diag[:, i], color=kc, lw=1.5)
        ax2.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green', label='Hold phase')
        ax2.axhline(200.0, color='gray', linestyle='--', lw=0.8, label='K0=200 N/m')
        for ph_t in [T_HOLD_START, T_HOLD_END]:
            ax2.axvline(ph_t, color='orange', linestyle='--', lw=0.8, alpha=0.7)
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel(f"K{lbl}{lbl} (N/m)")
        ax2.set_title(f"Stiffness K_{lbl}{lbl}")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    peraxis_path = save_path.replace(".png", "_peraxis.png")
    plt.savefig(peraxis_path, dpi=300, bbox_inches='tight')
    print(f"Saved -> {peraxis_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    taskspec = load_taskspec_from_json("spec/real_franka_task.json")

    assert taskspec.phases is not None, "real_franka_task.json must contain 'phases'"
    policy    = MultiPhaseCertifiedPolicy(taskspec.phases)
    theta_dim = policy.parameter_dimension()
    print(f"Multi-phase policy: {len(taskspec.phases)} phases, theta_dim = {theta_dim}")

    predicate_registry = build_predicate_registry()
    compiler     = Compiler(predicate_registry)
    objective_fn = compiler.compile(taskspec)

    # Nominal diagnostics
    trace0 = policy.rollout(np.zeros(theta_dim))
    cost0  = objective_fn(trace0)
    print(f"\nNominal (theta=0) — start: {trace0.position[0]}  "
          f"end: {trace0.position[-1]}  cost: {cost0:.4f}")

    # PIBB setup — range-normalised weights prevent mean explosion
    # when costs span orders of magnitude (obstacle penalty ~300 vs goal ~0.5)
    theta_init = np.zeros(theta_dim)
    sigma_init = policy.structured_sigma(
        sigma_traj_xy=3.0,   # trajectory XY — allows meaningful detour
        sigma_traj_z=0.5,    # z is fixed at safe_z, keep very small
        sigma_sd=2.0,        # damping weights — moderate exploration
        sigma_sk=2.0,        # stiffness weights — keep well below SK_CLIP=15
    )

    # Reduce hold-phase exploration (start==end -> nominal is already correct)
    hold_off = policy.offsets[1]
    hold_dim = policy.theta_dims[1]
    sigma_init[hold_off:hold_off + hold_dim] *= 0.05  # 20x smaller

    optimizer = PIBB(
        theta=theta_init,
        sigma=sigma_init,
        beta=8.0,     # softmax sharpness (range-normalised -> no explosion)
        decay=0.99,
    )

    N_SAMPLES = 20
    N_UPDATES = 200
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    print("\nStarting PIBB Optimization ...")
    print(f"  Task: {N_UPDATES} updates x {N_SAMPLES} samples = {N_UPDATES*N_SAMPLES} rollouts")
    for update_idx in range(N_UPDATES):
        samples = optimizer.sample(N_SAMPLES)
        costs   = np.array([
            objective_fn(policy.rollout(samples[i]))
            for i in range(N_SAMPLES)
        ])

        # Clip exploded costs before update so a single NaN/Inf sample
        # (from ODE blow-up at large SK weights) doesn't corrupt the mean.
        # Replace NaN/Inf with a large finite cost, then clip to 1e4.
        costs_safe = np.where(np.isfinite(costs), costs, 1e4)
        costs_safe = np.clip(costs_safe, 0.0, 1e4)
        optimizer.update(samples, costs_safe)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        if (update_idx + 1) % 10 == 0 or update_idx == 0:
            print(f"  Update {update_idx+1:03d} | "
                  f"Min: {costs.min():.4f} | "
                  f"Mean: {costs.mean():.4f} | "
                  f"BestSoFar: {best_cost:.4f}")

    print("Optimization Complete.\n")

    trace_final = policy.rollout(best_theta)
    save_checkpoint(best_theta, taskspec.horizon_sec, best_cost, trace_final)
    print_diagnostics(trace_final, best_cost)
    plot_results(trace_final, best_cost, save_path="real_franka_result.png")


if __name__ == "__main__":
    main()
