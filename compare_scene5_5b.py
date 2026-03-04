"""
compare_scene5_5b.py
────────────────────
Overlay comparison of Scene 5 (T=10 s, comfort SOFT w=15)
and Scene 5b (T=2 s, comfort SOFT w=4) — two colours, one plot each.

  1. compare_topdown.png     — 2-D top-down  (both trajectories overlaid)
  2. compare_workspace.png   — 3-D workspace (both trajectories overlaid)
  3. compare_stiffness.png   — K_xx/yy/zz vs normalised time (both overlaid)
  4. compare_distance.png    — distance to human vs normalised time (both overlaid)
  5. compare_kinematics.png  — position & velocity vs normalised time (both overlaid)

Run:
    python compare_scene5_5b.py
    python compare_scene5_5b.py --csv5 scene5_trajectory.csv --csv5b scene5b_trajectory.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# ── style ────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# ── shared scene constants ─────────────────────────────────────────────────────
START = np.array([0.55, 0.00, 0.30])
GOAL  = np.array([0.30, 0.55, 0.30])
HUMAN = np.array([0.30, 0.30, 0.30])

HUMAN_BODY_RAD    = 0.08
HUMAN_COMFORT_RAD = 0.19
HUMAN_RAMP_RAD    = 3.0 * HUMAN_COMFORT_RAD
K_AXIS_LIMIT      = 100.0

HORIZON_5  = 10.0
HORIZON_5B =  2.0

# ── colours ────────────────────────────────────────────────────────────────────────
C_5       = "#4C72B0"   # blue   — Scene 5  (10 s, comfort respected)
C_5B      = "#C44E52"   # red    — Scene 5b (2 s,  comfort violated)
C_BODY    = "#E74C3C"
C_COMFORT = "#F39C12"
C_START   = "#2CA02C"
C_GOAL    = "#555555"
C_DASH    = "#AAAAAA"
C_KX      = "#4C72B0"
C_KY      = "#DD8452"
C_KZ      = "#55A868"

LABEL_5  = r"$T=10\,$s  (comfort respected)"
LABEL_5B = r"$T=2\,$s   (comfort violated)"


# ══════════════════════════════════════════════════════════════════════
#  LOAD CSV
# ══════════════════════════════════════════════════════════════════════
def load_csv(path, horizon):
    df  = pd.read_csv(path)
    T   = len(df)
    t   = np.linspace(0.0, horizon, T)
    pos = df[["x","y","z"]].values
    vel = df[["dx","dy","dz"]].values
    cols_K = ["k11","k12","k13","k21","k22","k23","k31","k32","k33"]
    cols_D = ["d11","d12","d13","d21","d22","d23","d31","d32","d33"]
    K_arr = df[cols_K].values.reshape(T, 3, 3)
    D_arr = df[cols_D].values.reshape(T, 3, 3)
    return dict(t=t, pos=pos, vel=vel, K_arr=K_arr, D_arr=D_arr,
                horizon=horizon, T=T)


# ══════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════
def _draw_human_circles(ax):
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_COMFORT_RAD,
                             color=C_COMFORT, alpha=0.13, zorder=1))
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_COMFORT_RAD,
                             color=C_COMFORT, fill=False,
                             linestyle="--", lw=1.6, zorder=2, alpha=0.85))
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_BODY_RAD,
                             color=C_BODY, alpha=0.35, zorder=3))
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_BODY_RAD,
                             color=C_BODY, fill=False,
                             linestyle="-", lw=1.8, zorder=4))
    ax.scatter(HUMAN[0], HUMAN[1], s=55, c=C_COMFORT,
               edgecolors=C_BODY, linewidth=1.5, zorder=6)
    ax.text(HUMAN[0], HUMAN[1] + HUMAN_COMFORT_RAD + 0.025,
            "Human", fontsize=9, ha="center", color=C_BODY, fontweight="bold")


def _draw_human_cylinders(ax, cz):
    n  = 40
    th = np.linspace(0, 2*np.pi, n)
    for rad, col, alpha, ls in [
        (HUMAN_COMFORT_RAD, C_COMFORT, 0.06, "--"),
        (HUMAN_BODY_RAD,    C_BODY,    0.28,  "-"),
    ]:
        xc = HUMAN[0] + rad * np.cos(th)
        yc = HUMAN[1] + rad * np.sin(th)
        z_bot = cz - 0.10 if rad == HUMAN_BODY_RAD else cz - 0.05
        z_top = z_bot + (0.40 if rad == HUMAN_BODY_RAD else 0.22)
        for zb in [z_bot, z_top]:
            ax.plot(xc, yc, [zb]*n, color=col, lw=0.9, alpha=0.75, ls=ls)
        Th, Zg = np.meshgrid(th, np.linspace(z_bot, z_top, 2))
        ax.plot_surface(HUMAN[0]+rad*np.cos(Th), HUMAN[1]+rad*np.sin(Th), Zg,
                        alpha=alpha, color=col, edgecolor="none")
    ax.text(HUMAN[0], HUMAN[1]+HUMAN_BODY_RAD+0.01, HUMAN[2]+0.28,
            "Human", fontsize=8, color=C_BODY, ha="center", fontweight="bold")


# ══════════════════════════════════════════════════════════════════════
#  PLOT 1 — 2-D top-down  (both trajectories overlaid)
# ══════════════════════════════════════════════════════════════════════
def plot_compare_topdown(d5, d5b, base="compare_topdown"):
    fig, ax = plt.subplots(figsize=(7, 6.8))

    _draw_human_circles(ax)

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
            "--", color=C_DASH, lw=1.3, alpha=0.55, zorder=3,
            label="Shortest path")

    ax.plot(d5b["pos"][:,0], d5b["pos"][:,1],
            color=C_5B, lw=2.4, solid_capstyle="round", zorder=5,
            label=LABEL_5B)
    ax.plot(d5["pos"][:,0], d5["pos"][:,1],
            color=C_5, lw=2.4, solid_capstyle="round", zorder=6,
            label=LABEL_5)

    ax.scatter(START[0], START[1], s=80, c=C_START, zorder=10,
               edgecolors="black", lw=0.8, label="Start")
    ax.scatter(GOAL[0],  GOAL[1],  s=75, c=C_GOAL,  zorder=10,
               marker="D", edgecolors="black", lw=0.8, label="Goal")

    legend_extra = [
        mpatches.Patch(fc=C_BODY,    alpha=0.45, ec=C_BODY,
                       label=f"Body excl. — HARD ($r$={HUMAN_BODY_RAD:.2f} m)"),
        mpatches.Patch(fc=C_COMFORT, alpha=0.22, ec=C_COMFORT,
                       label=f"Comfort zone — SOFT ($r$={HUMAN_COMFORT_RAD:.2f} m)"),
    ]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + legend_extra, fontsize=8.5, loc="lower right",
              framealpha=0.92, edgecolor="lightgrey", fancybox=False)

    ax.set_xlabel("$x$ (m)", fontsize=12)
    ax.set_ylabel("$y$ (m)", fontsize=12)
    ax.set_title("Top-down Trajectory — HARD body vs SOFT comfort under time pressure",
                 fontsize=10)
    ax.set_aspect("equal")
    m = 0.12
    ax.set_xlim(min(START[0],GOAL[0])-m, max(START[0],GOAL[0])+m)
    ax.set_ylim(min(START[1],GOAL[1])-m, max(START[1],GOAL[1])+m)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = f"{base}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 2 — 3-D workspace  (both trajectories overlaid)
# ══════════════════════════════════════════════════════════════════════
def plot_compare_workspace(d5, d5b, base="compare_workspace"):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    _draw_human_cylinders(ax, HUMAN[2])

    ax.plot([START[0],GOAL[0]], [START[1],GOAL[1]], [START[2],GOAL[2]],
            "--", color=C_DASH, lw=1.4, alpha=0.50, label="Shortest path")

    ax.plot(d5b["pos"][:,0], d5b["pos"][:,1], d5b["pos"][:,2],
            color=C_5B, lw=2.4, solid_capstyle="round", zorder=5,
            label=LABEL_5B)
    ax.plot(d5["pos"][:,0], d5["pos"][:,1], d5["pos"][:,2],
            color=C_5, lw=2.4, solid_capstyle="round", zorder=6,
            label=LABEL_5)

    ax.scatter(*START, s=65, c=C_START, zorder=10, depthshade=False,
               edgecolors="black", lw=0.6)
    ax.scatter(*GOAL,  s=65, c=C_GOAL,  zorder=10, depthshade=False,
               marker="D", edgecolors="black", lw=0.6)

    ax.set_xlabel("X (m)", fontsize=9, labelpad=5)
    ax.set_ylabel("Y (m)", fontsize=9, labelpad=5)
    ax.set_zlabel("Z (m)", fontsize=9, labelpad=5)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=22, azim=-50)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.25)
    ax.set_title("3-D Workspace — HARD body vs SOFT comfort under time pressure",
                 fontsize=9, pad=10)

    extra = [
        mpatches.Patch(fc=C_BODY,    alpha=0.40, ec=C_BODY,
                       label=f"Body — HARD ($r$={HUMAN_BODY_RAD:.2f} m)"),
        mpatches.Patch(fc=C_COMFORT, alpha=0.20, ec=C_COMFORT,
                       label=f"Comfort — SOFT ($r$={HUMAN_COMFORT_RAD:.2f} m)"),
        mpatches.Patch(fc=C_START,   alpha=0.80, ec="black", label="Start"),
        mpatches.Patch(fc=C_GOAL,    alpha=0.80, ec="black", label="Goal"),
    ]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h+extra, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 0.97), framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    out = f"{base}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 3 — per-axis stiffness vs normalised time  (both overlaid)
# ══════════════════════════════════════════════════════════════════════
def plot_compare_stiffness(d5, d5b, base="compare_stiffness"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axis_labels = [r"$K_{xx}$", r"$K_{yy}$", r"$K_{zz}$"]
    kcolors     = [C_KX, C_KY, C_KZ]

    Kd5  = np.array([[K[0,0],K[1,1],K[2,2]] for K in d5["K_arr"]])
    Kd5b = np.array([[K[0,0],K[1,1],K[2,2]] for K in d5b["K_arr"]])

    tau5  = d5["t"]  / HORIZON_5
    tau5b = d5b["t"] / HORIZON_5B

    d_h5  = np.linalg.norm(d5["pos"]  - HUMAN, axis=1)
    d_h5b = np.linalg.norm(d5b["pos"] - HUMAN, axis=1)

    kmax = max(Kd5.max(), Kd5b.max()) * 1.12

    for i, (ax, lbl, kcol) in enumerate(zip(axes, axis_labels, kcolors)):
        mask_5b = d_h5b < HUMAN_COMFORT_RAD
        if np.any(mask_5b):
            s0 = np.where(np.diff(mask_5b.astype(int)) ==  1)[0]
            e0 = np.where(np.diff(mask_5b.astype(int)) == -1)[0]
            if mask_5b[0]:  s0 = np.concatenate([[0], s0])
            if mask_5b[-1]: e0 = np.concatenate([e0, [len(mask_5b)-1]])
            for j, (s, e) in enumerate(zip(s0, e0)):
                ax.axvspan(tau5b[s], tau5b[e], alpha=0.13, color=C_5B,
                           label="5b in comfort zone" if j==0 and i==0 else None,
                           zorder=0)

        mask_5 = d_h5 < HUMAN_COMFORT_RAD
        if np.any(mask_5):
            s0 = np.where(np.diff(mask_5.astype(int)) ==  1)[0]
            e0 = np.where(np.diff(mask_5.astype(int)) == -1)[0]
            if mask_5[0]:  s0 = np.concatenate([[0], s0])
            if mask_5[-1]: e0 = np.concatenate([e0, [len(mask_5)-1]])
            for j, (s, e) in enumerate(zip(s0, e0)):
                ax.axvspan(tau5[s], tau5[e], alpha=0.09, color=C_5,
                           label="5 in comfort zone" if j==0 and i==0 else None,
                           zorder=0)

        ax.plot(tau5,  Kd5[:,i],  color=kcol, lw=2.0, ls="-",
                label=f"{lbl}  " + LABEL_5)
        ax.plot(tau5b, Kd5b[:,i], color=kcol, lw=2.0, ls="--",
                label=f"{lbl}  " + LABEL_5B)
        ax.axhline(K_AXIS_LIMIT, color="#CC4444", ls=":", lw=1.3, alpha=0.80,
                   label=f"Compliance target {K_AXIS_LIMIT:.0f} N/m" if i==0 else None)

        ax.set_ylim(0, kmax)
        ax.set_ylabel(f"{lbl}  (N/m)", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9,
                  edgecolor="lightgrey", fancybox=False, ncol=2)

    axes[-1].set_xlabel("Normalised time $t/T$", fontsize=11)
    axes[-1].set_xlim(0, 1)
    fig.suptitle("Per-axis Stiffness — Scene 5 vs 5b\n"
                 "(shaded: comfort zone entry; stiffness drops near human)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = f"{base}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 4 — distance to human vs normalised time  (both overlaid)
# ══════════════════════════════════════════════════════════════════════
def plot_compare_distance(d5, d5b, base="compare_distance"):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    d_h5  = np.linalg.norm(d5["pos"]  - HUMAN, axis=1)
    d_h5b = np.linalg.norm(d5b["pos"] - HUMAN, axis=1)
    tau5  = d5["t"]  / HORIZON_5
    tau5b = d5b["t"] / HORIZON_5B

    ax.plot(tau5,  d_h5,  color=C_5,  lw=2.2, label=LABEL_5)
    ax.plot(tau5b, d_h5b, color=C_5B, lw=2.2, ls="--", label=LABEL_5B)

    ax.axhline(HUMAN_BODY_RAD, color=C_BODY, lw=1.5, ls="-", alpha=0.85,
               label=f"Body radius {HUMAN_BODY_RAD:.2f} m — HARD")
    ax.axhline(HUMAN_COMFORT_RAD, color=C_COMFORT, lw=1.5, ls="--", alpha=0.85,
               label=f"Comfort radius {HUMAN_COMFORT_RAD:.2f} m — SOFT")

    ax.fill_between([0,1], 0, HUMAN_BODY_RAD,
                    color=C_BODY, alpha=0.07, zorder=0)
    ax.fill_between([0,1], HUMAN_BODY_RAD, HUMAN_COMFORT_RAD,
                    color=C_COMFORT, alpha=0.07, zorder=0)

    ax.set_xlabel("Normalised time $t/T$", fontsize=12)
    ax.set_ylabel("Distance to human (m)", fontsize=12)
    ax.set_title("Distance to Human — Scene 5 vs 5b\n"
                 r"($T=10\,$s stays outside comfort zone; $T=2\,$s enters it)",
                 fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(d_h5.max(), d_h5b.max()) * 1.08)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92,
              edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    out = f"{base}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 5 — kinematics: pos (top) + speed (bottom), both overlaid
# ══════════════════════════════════════════════════════════════════════
def plot_compare_kinematics(d5, d5b, base="compare_kinematics"):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    tau5  = d5["t"]  / HORIZON_5
    tau5b = d5b["t"] / HORIZON_5B

    d_h5  = np.linalg.norm(d5["pos"]  - HUMAN, axis=1)
    d_h5b = np.linalg.norm(d5b["pos"] - HUMAN, axis=1)

    speed5  = np.linalg.norm(d5["vel"],  axis=1)
    speed5b = np.linalg.norm(d5b["vel"], axis=1)

    c_x = "#4C72B0"; c_y = "#DD8452"; c_z = "#55A868"

    for i, (c, lbl) in enumerate(zip([c_x, c_y, c_z], ["$x$","$y$","$z$"])):
        ax0.plot(tau5,  d5["pos"][:,i],  color=c, lw=1.8, label=lbl)
        ax0.plot(tau5b, d5b["pos"][:,i], color=c, lw=1.8, ls="--", alpha=0.75)

    for val, c in zip([GOAL[0],GOAL[1],GOAL[2]], [c_x, c_y, c_z]):
        ax0.axhline(val, color=c, lw=0.5, ls=":", alpha=0.25)

    ax0r = ax0.twinx()
    ax0r.plot(tau5,  d_h5,  color=C_5,  lw=1.0, ls=":", alpha=0.55,
              label=r"$d$(human) 10 s")
    ax0r.plot(tau5b, d_h5b, color=C_5B, lw=1.0, ls=":", alpha=0.55,
              label=r"$d$(human) 2 s")
    ax0r.axhline(HUMAN_BODY_RAD,    color=C_BODY,    lw=0.8, ls="-",  alpha=0.40)
    ax0r.axhline(HUMAN_COMFORT_RAD, color=C_COMFORT, lw=0.8, ls="--", alpha=0.40)
    ax0r.set_ylabel("Distance to human (m)", fontsize=9, color="#777777")
    ax0r.tick_params(labelsize=8)
    ax0r.legend(fontsize=8, loc="center right", framealpha=0.85,
                edgecolor="lightgrey", fancybox=False)

    ax0.set_ylabel("Position (m)", fontsize=11)
    ax0.grid(True, alpha=0.25)

    proxy = [
        Line2D([0],[0], color="grey", lw=1.8, ls="-",  label=LABEL_5),
        Line2D([0],[0], color="grey", lw=1.8, ls="--", label=LABEL_5B),
    ]
    h, _ = ax0.get_legend_handles_labels()
    ax0.legend(handles=h + proxy, fontsize=8.5, loc="upper right",
               framealpha=0.9, edgecolor="lightgrey", fancybox=False, ncol=3)

    ax1.plot(tau5,  speed5,  color=C_5,  lw=2.4,
             label=r"$\|\dot{p}\|$ " + LABEL_5)
    ax1.plot(tau5b, speed5b, color=C_5B, lw=2.4, ls="--",
             label=r"$\|\dot{p}\|$ " + LABEL_5B)
    ax1.axhline(0.8, color="#333333", ls=":", lw=1.2, alpha=0.65,
                label=r"$v_{\rm max}=0.8$ m/s")
    ax1.axhline(0.0, color="#999999", ls="-", lw=0.5, alpha=0.25)

    vhi = max(speed5.max(), speed5b.max()) * 1.18
    ax1.set_ylim(0, vhi)
    ax1.set_ylabel("Speed (m/s)", fontsize=11)
    ax1.set_xlabel("Normalised time $t/T$", fontsize=12)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False)

    ax1.set_xlim(0, 1)
    fig.suptitle("Kinematics Comparison — Scene 5 vs 5b\n"
                 "(solid = $T=10$ s; dashed = $T=2$ s)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = f"{base}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv5",  default="scene5_trajectory.csv")
    parser.add_argument("--csv5b", default="scene5b_trajectory.csv")
    args = parser.parse_args()

    print(f"Loading Scene 5  : {args.csv5}  (T={HORIZON_5} s)")
    d5  = load_csv(args.csv5,  HORIZON_5)
    print(f"Loading Scene 5b : {args.csv5b}  (T={HORIZON_5B} s)")
    d5b = load_csv(args.csv5b, HORIZON_5B)

    d_h5  = np.linalg.norm(d5["pos"]  - HUMAN, axis=1)
    d_h5b = np.linalg.norm(d5b["pos"] - HUMAN, axis=1)
    print(f"\n  Scene 5  — body clear: {(d_h5.min()-HUMAN_BODY_RAD)*100:+.1f} cm  "
          f"comfort clear: {(d_h5.min()-HUMAN_COMFORT_RAD)*100:+.1f} cm")
    print(f"  Scene 5b — body clear: {(d_h5b.min()-HUMAN_BODY_RAD)*100:+.1f} cm  "
          f"comfort clear: {(d_h5b.min()-HUMAN_COMFORT_RAD)*100:+.1f} cm\n")

    plot_compare_topdown  (d5, d5b)
    plot_compare_workspace(d5, d5b)
    plot_compare_stiffness(d5, d5b)
    plot_compare_distance (d5, d5b)
    plot_compare_kinematics(d5, d5b)

    print("\nDone — 5 comparison plots saved.")


if __name__ == "__main__":
    main()
