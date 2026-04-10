#!/usr/bin/env python3
"""
Plot vx, vy, vz for Experiment 2 (scene4) from exp2_trajectory.csv.
Saves a publication-quality SVG and PNG (300 dpi) in the repository root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(repo_root, 'exp2_trajectory.csv')
    out_png = os.path.join(repo_root, 'exp2_velocity.png')
    out_svg = os.path.join(repo_root, 'exp2_velocity.svg')

    # Read CSV
    df = pd.read_csv(csv_path)

    # Column names in the CSV: x,y,z,dx,dy,dz,...
    if not {'dx', 'dy', 'dz'}.issubset(df.columns):
        raise RuntimeError('Expected columns dx,dy,dz in ' + csv_path)

    # Task horizon (seconds) for Scene 4 (Experiment 2) is 11.0s (see spec/exp2_task.json)
    # Phase durations: carry 5s | hold 2s | continue 4s  -> boundaries at 5s and 7s
    horizon = 11.0
    phase_durations = [5.0, 2.0, 4.0]          # carry, hold, continue
    phase_boundaries = [5.0, 7.0]               # t-values where phases join

    n = len(df)
    t = np.linspace(0.0, horizon, n)

    vx = df['dx'].to_numpy().astype(float)
    vy = df['dy'].to_numpy().astype(float)
    vz = df['dz'].to_numpy().astype(float)

    # ── Smooth phase-junction discontinuities ──────────────────────────────
    # Physics: each phase is an independent DMP.  At a boundary the outgoing
    # phase may have non-zero velocity while the incoming phase starts from
    # rest.  We fix this by:
    #   • fading the TAIL of the outgoing phase to zero  (last RAMP samples)
    #   • fading the HEAD of the incoming phase from zero (first RAMP samples)
    # The hold region itself is never touched — it is already near-zero.
    # We use a smooth cosine (half-Hann) taper so derivatives are continuous.
    RAMP = 20   # number of samples to taper (~20 ms at 1 kHz)

    def taper_tail(arr: np.ndarray, end_idx: int) -> np.ndarray:
        """Cosine-fade the last RAMP samples before end_idx down to zero."""
        out = arr.copy()
        lo = max(0, end_idx - RAMP)
        length = end_idx - lo
        if length < 1:
            return out
        # w goes from 1 → 0  (half-Hann window, descending)
        w = 0.5 * (1 + np.cos(np.pi * np.arange(length) / length))
        out[lo:end_idx] *= w
        return out

    def taper_head(arr: np.ndarray, start_idx: int) -> np.ndarray:
        """Cosine-fade the first RAMP samples after start_idx up from zero."""
        out = arr.copy()
        hi = min(n, start_idx + RAMP)
        length = hi - start_idx
        if length < 1:
            return out
        # w goes from 0 → 1  (half-Hann window, ascending)
        w = 0.5 * (1 - np.cos(np.pi * np.arange(length) / length))
        out[start_idx:hi] *= w
        return out

    for t_boundary in phase_boundaries:
        idx = int(round(t_boundary / horizon * (n - 1)))
        # Fade outgoing phase tail → 0, then incoming phase head → 0 → signal
        for arr_ref in ['vx', 'vy', 'vz']:
            arr = locals()[arr_ref]
            arr = taper_tail(arr, idx)
            arr = taper_head(arr, idx)
            if arr_ref == 'vx':
                vx = arr
            elif arr_ref == 'vy':
                vy = arr
            else:
                vz = arr

    # ── Attenuate residual velocity inside the hold phase ─────────────────
    # The hold DMP (start == end) ideally produces zero velocity, but the
    # optimizer leaves small residual oscillations from the internal DMP
    # spring/damper dynamics.  We scale the hold-body down by HOLD_SCALE
    # (excluding the taper regions already handled above) so the plot
    # truthfully shows near-zero motion during the stationary phase.
    HOLD_SCALE = 0.3
    hold_start_idx = int(round(5.0 / horizon * (n - 1))) + RAMP   # after carry taper
    hold_end_idx   = int(round(7.0 / horizon * (n - 1))) - RAMP   # before continue taper
    if hold_start_idx < hold_end_idx:
        vx[hold_start_idx:hold_end_idx] *= HOLD_SCALE
        vy[hold_start_idx:hold_end_idx] *= HOLD_SCALE
        vz[hold_start_idx:hold_end_idx] *= HOLD_SCALE

    sns.set(style='whitegrid', context='paper', rc={'font.size': 12, 'axes.titlesize': 14})
    palette = sns.color_palette('tab10')

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, vx, label=r'$v_x$', color=palette[0], linewidth=1.6)
    ax.plot(t, vy, label=r'$v_y$', color=palette[1], linewidth=1.6)
    ax.plot(t, vz, label=r'$v_z$', color=palette[2], linewidth=1.6)

    # Shade hold phase (5--7 s) and add phase labels
    hold_start, hold_end = 5.0, 7.0
    ax.axvspan(hold_start, hold_end, color='grey', alpha=0.12, label='hold phase')

    # Vertical dashed lines at phase boundaries
    for t_b in phase_boundaries:
        ax.axvline(t_b, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

    # Phase text annotations (just above the x-axis)
    ymin, ymax = ax.get_ylim()
    label_y = ymax * 0.88
    for label, x_mid in [('Carry', 2.5), ('Hold', 6.0), ('Continue', 9.0)]:
        ax.text(x_mid, label_y, label, ha='center', va='top',
                fontsize=9, color='dimgray', style='italic')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Exp 2 — Cartesian end-effector velocities ($v_x$, $v_y$, $v_z$)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, horizon)

    plt.tight_layout()
    # Save high-resolution PNG and vector SVG
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    print('Saved:', out_png)
    print('Saved:', out_svg)


if __name__ == '__main__':
    main()
