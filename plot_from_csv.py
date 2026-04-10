"""
plot_from_csv.py — Regenerate all Exp 1 / 1b plots from a saved trajectory CSV.

No training required. Just supply:
  --csv     path to trajectory CSV  (default: exp1_trajectory.csv)
  --scene   "1" or "1b"             (sets human/radius constants + output prefix)
  --horizon horizon in seconds      (default: auto-detect from CSV row count)

CSV format (produced by save_trajectory_csv):
  x, y, z,            — position
  dx, dy, dz,         — velocity
  k11..k33,           — 3×3 stiffness matrix (row-major)
  d11..d33            — 3×3 damping matrix   (row-major)

Usage:
  python plot_from_csv.py --csv exp1_trajectory.csv  --scene 5
  python plot_from_csv.py --csv exp1b_trajectory.csv --scene 5b
  python plot_from_csv.py --csv my_traj.csv --scene 5 --horizon 10.0 --prefix my_run
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
import matplotlib.patches as mpatches
import seaborn as sns

# ── style ──────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# ── shared geometry (same for Scene 5 and 5b) ─────────────────────────
START = np.array([0.55, 0.00, 0.30])
GOAL  = np.array([0.30, 0.55, 0.30])
HUMAN = np.array([0.30, 0.30, 0.30])

HUMAN_BODY_RAD    = 0.08
HUMAN_COMFORT_RAD = 0.19
HUMAN_RAMP_RAD    = 3.0 * HUMAN_COMFORT_RAD
K_AXIS_LIMIT      = 100.0

# ── colours ────────────────────────────────────────────────────────────
C_CARRY   = "#4C72B0"
C_BODY    = "#E74C3C"
C_COMFORT = "#F39C12"
C_START   = "#2CA02C"
C_GOAL    = "#1F77B4"
C_DASH    = "#999999"
C_KX      = "#4C72B0"
C_KY      = "#DD8452"
C_KZ      = "#55A868"


# ── load CSV ───────────────────────────────────────────────────────────
def load_csv(csv_path, horizon):
    df = pd.read_csv(csv_path)
    T  = len(df)
    t  = np.linspace(0.0, horizon, T)

    pos = df[["x","y","z"]].values
    vel = df[["dx","dy","dz"]].values

    K_arr = np.zeros((T, 3, 3))
    D_arr = np.zeros((T, 3, 3))
    cols_K = ["k11","k12","k13","k21","k22","k23","k31","k32","k33"]
    cols_D = ["d11","d12","d13","d21","d22","d23","d31","d32","d33"]
    for i in range(T):
        K_arr[i] = df[cols_K].values[i].reshape(3, 3)
        D_arr[i] = df[cols_D].values[i].reshape(3, 3)

    return t, pos, vel, K_arr, D_arr


# ── helpers ────────────────────────────────────────────────────────────
def _shade_human_zones(ax, t, d_human, vertical=True):
    """Shade comfort and body proximity time spans on a time-series axes."""
    for mask, col in [
        (d_human < HUMAN_COMFORT_RAD, C_COMFORT),
        (d_human < HUMAN_BODY_RAD,    C_BODY),
    ]:
        if not np.any(mask):
            continue
        starts = np.where(np.diff(mask.astype(int)) ==  1)[0]
        ends   = np.where(np.diff(mask.astype(int)) == -1)[0]
        if mask[0]:  starts = np.concatenate([[0], starts])
        if mask[-1]: ends   = np.concatenate([ends, [len(mask)-1]])
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], alpha=0.12, color=col, zorder=0)


# ── PLOT 1 — 3D workspace ──────────────────────────────────────────────
def plot_3d_workspace(t, pos, vel, K_arr, D_arr, label, base):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    cz = HUMAN[2]
    n_seg   = 40
    theta_c = np.linspace(0, 2*np.pi, n_seg)

    # body cylinder (solid red)
    h_body = 0.40
    z_bot  = cz - 0.10
    z_top  = z_bot + h_body
    xc = HUMAN[0] + HUMAN_BODY_RAD * np.cos(theta_c)
    yc = HUMAN[1] + HUMAN_BODY_RAD * np.sin(theta_c)
    for zb in [z_bot, z_top]:
        ax.plot(xc, yc, [zb]*n_seg, color=C_BODY, lw=1.0, alpha=0.80)
    Th, Zg = np.meshgrid(theta_c, np.linspace(z_bot, z_top, 2))
    ax.plot_surface(HUMAN[0] + HUMAN_BODY_RAD*np.cos(Th),
                    HUMAN[1] + HUMAN_BODY_RAD*np.sin(Th), Zg,
                    alpha=0.35, color=C_BODY, edgecolor="none")

    # comfort cylinder (translucent orange)
    z_cbot = cz - 0.05
    z_ctop = z_cbot + 0.20
    xcc = HUMAN[0] + HUMAN_COMFORT_RAD * np.cos(theta_c)
    ycc = HUMAN[1] + HUMAN_COMFORT_RAD * np.sin(theta_c)
    for zb in [z_cbot, z_ctop]:
        ax.plot(xcc, ycc, [zb]*n_seg, color=C_COMFORT, lw=0.8, alpha=0.50, ls="--")
    Thc, Zgc = np.meshgrid(theta_c, np.linspace(z_cbot, z_ctop, 2))
    ax.plot_surface(HUMAN[0] + HUMAN_COMFORT_RAD*np.cos(Thc),
                    HUMAN[1] + HUMAN_COMFORT_RAD*np.sin(Thc), Zgc,
                    alpha=0.07, color=C_COMFORT, edgecolor="none")

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], [START[2], GOAL[2]],
            "--", color=C_DASH, lw=1.6, alpha=0.55, label="Shortest path", zorder=2)
    ax.plot(pos[:,0], pos[:,1], pos[:,2],
            color=C_CARRY, lw=2.5, solid_capstyle="round", zorder=5, label="Trajectory")
    ax.scatter(*START, s=70, c=C_START, zorder=10, depthshade=False,
               edgecolors="black", linewidth=0.6)
    ax.scatter(*GOAL,  s=70, c=C_GOAL,  zorder=10, depthshade=False,
               marker="D", edgecolors="black", linewidth=0.6)
    ax.text(HUMAN[0], HUMAN[1]+HUMAN_BODY_RAD+0.01, HUMAN[2]+0.25,
            "Human", fontsize=8, color=C_BODY, ha="center", fontweight="bold")

    ax.set_xlabel("X — forward (m)", fontsize=9, labelpad=7)
    ax.set_ylabel("Y — lateral (m)", fontsize=9, labelpad=7)
    ax.set_zlabel("Z — height (m)",  fontsize=9, labelpad=7)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=22, azim=-50)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.28)

    extra = [
        mpatches.Patch(facecolor=C_BODY,    alpha=0.40, edgecolor=C_BODY,
                       label=f"Body exclusion — HARD ($r$={HUMAN_BODY_RAD:.2f} m)"),
        mpatches.Patch(facecolor=C_COMFORT, alpha=0.20, edgecolor=C_COMFORT,
                       label=f"Comfort zone — SOFT ($r$={HUMAN_COMFORT_RAD:.2f} m)"),
        mpatches.Patch(facecolor=C_START,   alpha=0.80, edgecolor="black", label="Start"),
        mpatches.Patch(facecolor=C_GOAL,    alpha=0.80, edgecolor="black", label="Goal"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 0.97), framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    ax.set_title(label, fontsize=9, pad=10)
    plt.tight_layout()
    out = f"{base}_workspace.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ── PLOT 2 — 2D top-down ───────────────────────────────────────────────
def plot_2d_topdown(t, pos, vel, K_arr, D_arr, label, base):
    fig, ax = plt.subplots(figsize=(7, 6.5))

    # comfort zone
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_COMFORT_RAD,
                             color=C_COMFORT, alpha=0.15, zorder=1))
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_COMFORT_RAD,
                             color=C_COMFORT, fill=False,
                             linestyle="--", linewidth=1.5, zorder=2, alpha=0.85))

    # body exclusion
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_BODY_RAD,
                             color=C_BODY, alpha=0.40, zorder=3))
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_BODY_RAD,
                             color=C_BODY, fill=False,
                             linestyle="-", linewidth=1.8, zorder=4, alpha=1.0))

    # human marker
    ax.scatter(HUMAN[0], HUMAN[1], s=60, c=C_COMFORT, edgecolors=C_BODY,
               linewidth=1.5, zorder=6)
    ax.text(HUMAN[0]+0.01, HUMAN[1]+HUMAN_COMFORT_RAD+0.02,
            "Human", fontsize=8, ha="center", color=C_BODY, fontweight="bold")

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
            "--", color=C_DASH, lw=1.5, alpha=0.50, zorder=3, label="Shortest path")
    ax.plot(pos[:,0], pos[:,1],
            color=C_CARRY, lw=2.5, solid_capstyle="round", zorder=6, label="Trajectory")

    ax.scatter(START[0], START[1], s=80, c=C_START, zorder=10,
               edgecolors="black", linewidth=0.7, label="Start")
    ax.scatter(GOAL[0],  GOAL[1],  s=75, c=C_GOAL,  zorder=10,
               marker="D", edgecolors="black", linewidth=0.7, label="Goal")

    ax.set_xlabel("$x$ (m)", fontsize=12)
    ax.set_ylabel("$y$ (m)", fontsize=12)
    ax.set_title("Top-down View — Human Avoidance", fontsize=11)
    ax.set_aspect("equal")
    margin = 0.12
    ax.set_xlim(min(START[0], GOAL[0]) - margin, max(START[0], GOAL[0]) + margin)
    ax.set_ylim(min(START[1], GOAL[1]) - margin, max(START[1], GOAL[1]) + margin)
    ax.grid(True, alpha=0.25)

    legend_extra = [
        mpatches.Patch(facecolor=C_BODY,    alpha=0.45, edgecolor=C_BODY,
                       label=f"Body excl. — HARD ($r$={HUMAN_BODY_RAD:.2f} m)"),
        mpatches.Patch(facecolor=C_COMFORT, alpha=0.20, edgecolor=C_COMFORT,
                       label=f"Comfort zone — SOFT ($r$={HUMAN_COMFORT_RAD:.2f} m)"),
    ]
    handles, labels_h = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_extra, fontsize=8.5, loc="lower right",
              framealpha=0.92, edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    out = f"{base}_topdown.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ── PLOT 3 — per-axis stiffness ────────────────────────────────────────
def plot_stiffness(t, pos, vel, K_arr, D_arr, label, base, horizon):
    Kd = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])
    d_human = np.linalg.norm(pos - HUMAN, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.2))

    # shade zones (ramp → comfort → body)
    for mask, col, alpha, lbl in [
        (d_human < HUMAN_RAMP_RAD,    C_COMFORT, 0.07,
         f"Stiffness ramp (d < {HUMAN_RAMP_RAD:.2f} m)"),
        (d_human < HUMAN_COMFORT_RAD, C_COMFORT, 0.14,
         f"Comfort zone (d < {HUMAN_COMFORT_RAD:.2f} m)"),
        (d_human < HUMAN_BODY_RAD,    C_BODY,    0.20,
         f"Body zone (d < {HUMAN_BODY_RAD:.2f} m)"),
    ]:
        if not np.any(mask):
            continue
        starts = np.where(np.diff(mask.astype(int)) ==  1)[0]
        ends   = np.where(np.diff(mask.astype(int)) == -1)[0]
        if mask[0]:  starts = np.concatenate([[0], starts])
        if mask[-1]: ends   = np.concatenate([ends, [len(mask)-1]])
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(t[s], t[e], alpha=alpha, color=col,
                       label=lbl if i == 0 else None, zorder=0)

    ax.plot(t, Kd[:,0], color=C_KX, lw=2.0, label=r"$K_{xx}$")
    ax.plot(t, Kd[:,1], color=C_KY, lw=2.0, label=r"$K_{yy}$")
    ax.plot(t, Kd[:,2], color=C_KZ, lw=2.0, label=r"$K_{zz}$")
    ax.axhline(K_AXIS_LIMIT, color="#CC4444", ls="--", lw=1.4, alpha=0.75,
               label=f"$K_{{\\rm axis}}$ target ≤ {K_AXIS_LIMIT:.0f} N/m")

    ax.set_ylim(0, Kd.max() * 1.15)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Stiffness (N/m)", fontsize=11)
    ax.set_title("Per-axis Stiffness — Compliance Near Human", fontsize=11)
    ax.set_xlim(0, horizon)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)
    plt.tight_layout()
    out = f"{base}_stiffness.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ── PLOT 4 — kinematics (position + velocity) ──────────────────────────
def plot_kinematics(t, pos, vel, K_arr, D_arr, label, base, horizon):
    speed   = np.linalg.norm(vel, axis=1)
    d_human = np.linalg.norm(pos - HUMAN, axis=1)
    d_goal  = np.linalg.norm(pos - GOAL,  axis=1)

    c_x = "#4C72B0"; c_y = "#DD8452"; c_z = "#55A868"; c_spd = "#8172B2"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axi in axes:
        _shade_human_zones(axi, t, d_human)

    ax0, ax1 = axes
    ax0.plot(t, pos[:,0], color=c_x, lw=1.8, label=r"$x(t)$")
    ax0.plot(t, pos[:,1], color=c_y, lw=1.8, label=r"$y(t)$")
    ax0.plot(t, pos[:,2], color=c_z, lw=1.8, label=r"$z(t)$")
    for val, col, ls in [
        (START[0], c_x, ":"), (START[1], c_y, ":"), (START[2], c_z, ":"),
        (GOAL[0],  c_x, "--"), (GOAL[1],  c_y, "--"), (GOAL[2],  c_z, "--"),
    ]:
        ax0.axhline(val, color=col, lw=0.6, ls=ls, alpha=0.30)

    ax0_r = ax0.twinx()
    ax0_r.plot(t, d_human, color=C_BODY,    lw=1.0, ls=":", alpha=0.60, label="d(human)")
    ax0_r.plot(t, d_goal,  color=C_GOAL,    lw=1.0, ls=":", alpha=0.60, label="d(goal)")
    ax0_r.axhline(HUMAN_BODY_RAD,    color=C_BODY,    lw=0.7, ls="-", alpha=0.40)
    ax0_r.axhline(HUMAN_COMFORT_RAD, color=C_COMFORT, lw=0.7, ls="-", alpha=0.40)
    ax0_r.set_ylabel("Distance (m)", fontsize=9, color="#777777")
    ax0_r.tick_params(labelsize=8)
    ax0_r.legend(fontsize=7.5, loc="center right", framealpha=0.85,
                 edgecolor="lightgrey", fancybox=False)

    ax0.set_ylabel("Position (m)", fontsize=11)
    ax0.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=3)

    ax1.plot(t, vel[:,0], color=c_x,  lw=1.5, alpha=0.75, label=r"$\dot x$")
    ax1.plot(t, vel[:,1], color=c_y,  lw=1.5, alpha=0.75, label=r"$\dot y$")
    ax1.plot(t, vel[:,2], color=c_z,  lw=1.5, alpha=0.75, label=r"$\dot z$")
    ax1.plot(t, speed,    color=c_spd, lw=2.2, label=r"$\|\dot p\|$")
    ax1.axhline( 0.8, color="#333333", ls=":", lw=1.2, alpha=0.65, label="$v_{max}$=0.8")
    ax1.axhline(-0.8, color="#333333", ls=":", lw=1.2, alpha=0.65)
    ax1.axhline( 0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)
    vhi = max(speed.max(), 0.85) * 1.20
    ax1.set_ylim(-vhi, vhi)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_xlim(0, horizon)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=4)

    fig.suptitle("Per-axis Position & Velocity", fontsize=12, y=1.01)
    plt.tight_layout()
    out = f"{base}_kinematics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Regenerate scene plots from CSV.")
    parser.add_argument("--csv",     default="exp1_trajectory.csv",
                        help="Path to trajectory CSV file")
    parser.add_argument("--scene",   default="5",
                        help="Scene identifier: '1' or '1b' (sets output prefix)")
    parser.add_argument("--horizon", type=float, default=None,
                        help="Trajectory horizon in seconds (auto-detected if omitted)")
    parser.add_argument("--prefix",  default=None,
                        help="Output file prefix (default: scene<scene>)")
    args = parser.parse_args()

    prefix  = args.prefix if args.prefix else f"scene{args.scene}"
    horizon = args.horizon

    print(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)
    T  = len(df)

    # auto-detect horizon from CSV filename hint or default
    if horizon is None:
        if "5b" in args.csv or args.scene == "5b":
            horizon = 4.0
        else:
            horizon = 10.0
        print(f"Auto-detected horizon: {horizon} s  (pass --horizon N to override)")

    t, pos, vel, K_arr, D_arr = load_csv(args.csv, horizon)

    d_human = np.linalg.norm(pos - HUMAN, axis=1)
    d_goal  = np.linalg.norm(pos - GOAL,  axis=1)
    speed   = np.linalg.norm(vel, axis=1)
    reached = np.any(d_goal < 0.06)

    print(f"  Rows: {T},  horizon: {horizon} s")
    print(f"  Goal reached:   {'YES' if reached else 'NO'}")
    print(f"  Body clearance: {(d_human.min()-HUMAN_BODY_RAD)*100:+.1f} cm")
    print(f"  Comfort clear:  {(d_human.min()-HUMAN_COMFORT_RAD)*100:+.1f} cm")
    print(f"  Max speed:      {speed.max():.4f} m/s")
    print(f"  Output prefix:  {prefix}")

    scene_label = (f"Scene {args.scene} — Carry to Goal with Human Avoidance "
                   f"(T={horizon:.0f} s)")

    plot_3d_workspace(t, pos, vel, K_arr, D_arr, scene_label, prefix)
    plot_2d_topdown  (t, pos, vel, K_arr, D_arr, scene_label, prefix)
    plot_stiffness   (t, pos, vel, K_arr, D_arr, scene_label, prefix, horizon)
    plot_kinematics  (t, pos, vel, K_arr, D_arr, scene_label, prefix, horizon)

    print(f"\nDone — 4 plots saved with prefix '{prefix}'.")


if __name__ == "__main__":
    main()
