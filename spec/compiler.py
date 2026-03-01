# spec/compiler.py

from logic import predicates
from logic import temporal_logic
import numpy as np


class Compiler:

    def __init__(self, predicate_registry):
        self.predicate_registry = predicate_registry

    def compile(self, taskspec):

        hard_clauses = []
        soft_clauses = []

        for clause in taskspec.clauses:
            if clause.modality == "REQUIRE":
                hard_clauses.append(clause)
            else:
                soft_clauses.append(clause)

        def objective(trace):

            total_cost = 0.0

            # --- Hard clauses with slack relaxation ---
            SLACK_WEIGHT = 500.0  # λ_s (tuneable)

            for clause in hard_clauses:
                rho = self._evaluate_clause(trace, clause)
                s = max(0.0, -rho)
                total_cost += SLACK_WEIGHT * (s ** 2)

            # --- Soft clauses ---
            for clause in soft_clauses:
                rho = self._evaluate_clause(trace, clause)
                J = max(0.0, -rho)
                total_cost += clause.weight * J

            # --- Intrinsic stiffness regularizer (no JSON entry needed) ---
            # Two-layer defence:
            #   1. set_theta() clips SK/SD weights to ±SK_CLIP (Fix B) → ODE stays bounded
            #   2. This penalty acts on the RAW (pre-clip) theta stored on the trace,
            #      so PI2 sees a real cost for samples with large SK/SD weights even
            #      when the clip absorbs them.  This steers the PI2 mean away from the
            #      clip boundary honestly, preventing mean drift.
            #
            # Penalty form: λ · mean(max(0, |w| - SK_CLIP)²)
            #   - mean (not sum) → scale independent of n_bfs_slack
            #   - zero inside ±SK_CLIP, grows quadratically outside
            #   - at SK=±20 (5 over clip): penalty = 1.0 * 25 / 42 ≈ 0.6 (small, informative)
            #   - at SK=±50 (35 over clip): penalty = 1.0 * 1225 / 42 ≈ 29  (large, PI2 avoids)
            SK_CLIP      = 15.0
            STIFF_WEIGHT = 1.0
            if hasattr(trace, 'raw_sk_weights') and trace.raw_sk_weights is not None:
                w = trace.raw_sk_weights
                excess = np.maximum(0.0, np.abs(w) - SK_CLIP)
                total_cost += STIFF_WEIGHT * float(np.mean(excess**2))
            if hasattr(trace, 'raw_sd_weights') and trace.raw_sd_weights is not None:
                w = trace.raw_sd_weights
                excess = np.maximum(0.0, np.abs(w) - SK_CLIP)
                total_cost += STIFF_WEIGHT * float(np.mean(excess**2))

            # --- SK/SD weight smoothness penalty ---
            # Root cause of stiffness spikes: adjacent RBF weights in SK/SD networks
            # vary sharply, causing large |dK/dt| at transitions (measured: 20,931 N/m/s).
            # Fix: penalise the L2 difference between consecutive weight values.
            # This is CGMS-safe — it is a cost on parameters only, NOT on the ODE structure.
            # The Cholesky ODE guarantee K=QᵀQ≻0 is unaffected; PI2 just steers toward
            # smoother weight profiles.
            # Scale: SMOOTH_WEIGHT=0.1, typical |Δw|=1.0, ~(n_bfs-1)*6 terms per phase
            #        → cost ~ 0.1 * 1 = 0.1  (same order as one TL clause ≈ 0.5-2.0)
            SMOOTH_WEIGHT = 0.1
            for attr in ('raw_sk_weights', 'raw_sd_weights'):
                if hasattr(trace, attr) and getattr(trace, attr) is not None:
                    w = getattr(trace, attr)
                    # w may be 1-D (all phases concatenated) or 2-D (n_bfs × n_channels)
                    w = np.atleast_2d(w) if w.ndim == 1 else w
                    if w.shape[0] > 1:
                        diffs = w[1:] - w[:-1]
                        total_cost += SMOOTH_WEIGHT * float(np.mean(diffs**2))

            # --- Hard K ceiling penalty ---
            # The Q-ODE Q̇ = α·Q + 0.5·Q⁻ᵀ·B is exponentially growing when
            # SK weights are large; SK_CLIP=±15 still allows tr(K) to reach
            # ~20,000+ N/m over a long phase.  This penalty directly penalises
            # any trace where tr(K) exceeds K_MAX, giving PI2 a clear signal
            # to avoid solutions that would cause excessive stiffness.
            # K_MAX = 600 N/m (3 axes × 200 N/m) — 3× nominal, safely below Franka limit 3000 N/m
            K_MAX       = 600.0   # N/m (tr(K) = sum of diagonal stiffnesses)
            K_CEIL_WEIGHT = 2.0   # same weight class as damping regulariser
            if trace.gains is not None and "K" in trace.gains:
                K_arr = trace.gains["K"]
                trK_arr = np.array([np.trace(K) for K in K_arr])
                excess_k = np.maximum(0.0, trK_arr - K_MAX)
                # Normalise by K_MAX² so cost is O(1) when trK = 2·K_MAX
                total_cost += K_CEIL_WEIGHT * float(np.mean((excess_k / K_MAX)**2))

            # --- Minimum damping regularizer (hardware safety) ---
            # The optimizer tends to zero out SD weights because low damping
            # gives faster motion that satisfies TL clauses more easily.
            # This leaves D ≈ alpha*H = 0.05*I Ns/m → zeta ≈ 0.002 (500× underdamped).
            # For Franka Panda, minimum safe damping per axis ≈ 10 Ns/m (zeta ≈ 0.35).
            # We penalize tr(D) < D_MIN_TRACE with a one-sided quadratic cost.
            #   D_MIN_TRACE = 30.0 Ns/m  (10 N·s/m per axis × 3 axes)
            #   Penalty is normalised by D_MIN_TRACE² so it is O(1) at nominal (theta=0)
            #   and reaches ~0 once tr(D) ≥ D_MIN_TRACE.
            D_MIN_TRACE = 30.0   # Ns/m  (≈ zeta 0.35 at K0=200 N/m, H=I)
            DAMP_WEIGHT = 2.0    # at nominal: 2.0 * mean((29.85/30)^2) ≈ 2.0 — same order as TL clauses
            if trace.gains is not None and "D" in trace.gains:
                D_arr = trace.gains["D"]          # (T, 3, 3)
                trD_arr = np.array([np.trace(D) for D in D_arr])
                # Normalise deficit by D_MIN_TRACE so penalty is dimensionless
                deficit_frac = np.maximum(0.0, (D_MIN_TRACE - trD_arr) / D_MIN_TRACE)
                total_cost += DAMP_WEIGHT * float(np.mean(deficit_frac**2))

            return total_cost

        return objective

    def _evaluate_clause(self, trace, clause):

        predicate_fn = self.predicate_registry[clause.predicate]
        rho_trace = predicate_fn(trace, **clause.parameters)

        if clause.operator == "eventually":
            return temporal_logic.eventually(rho_trace)

        elif clause.operator == "always":
            return temporal_logic.always(rho_trace)

        elif clause.operator == "always_during":
            t_start = clause.time_window[0]
            t_end   = clause.time_window[1]
            return temporal_logic.always_during(rho_trace, trace.time, t_start, t_end)

        elif clause.operator == "eventually_during":
            t_start = clause.time_window[0]
            t_end   = clause.time_window[1]
            return temporal_logic.eventually_during(rho_trace, trace.time, t_start, t_end)

        elif clause.operator == "until":
            # Until expects two predicates: (left, right)
            left_fn  = self.predicate_registry[clause.predicate[0]]
            right_fn = self.predicate_registry[clause.predicate[1]]
            rho_phi  = left_fn(trace, **clause.parameters["left_params"])
            rho_psi  = right_fn(trace, **clause.parameters["right_params"])
            return temporal_logic.until(rho_phi, rho_psi)

        else:
            raise NotImplementedError(
                f"Operator {clause.operator} not supported yet."
            )