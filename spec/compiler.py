# spec/compiler.py

from logic import predicates
from logic import temporal_logic
import numpy as np


class Compiler:

    def __init__(self, predicate_registry):
        self.predicate_registry = predicate_registry

    def compile(self, taskspec):

        tau_budget = float(taskspec.horizon_sec)   # normalises the time penalty

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

            # --- Stiffness rate penalty (penalises steep jumps in tr(K)) ---
            # PI2 sees a real cost for trajectories where tr(K) changes rapidly,
            # which physically means the impedance controller would demand sudden
            # torque spikes.  We penalise the RMS of the finite-difference derivative
            # of tr(K) along the trajectory.
            #
            #   cost += RATE_WEIGHT * sqrt( mean( (Δtr(K)/Δt)² ) )
            #
            # dt=0.01s so a jump of 680 N/m in one step → rate = 68000 N/m/s
            # RATE_WEIGHT=5e-4: at rms=668 N/m/s → penalty ≈ 0.33  (gentle signal)
            # Enough to distinguish spiky trajectories from smooth ones without
            # destabilising the PI2 landscape.
            RATE_WEIGHT = 5e-4
            if hasattr(trace, 'gains') and trace.gains is not None:
                K_arr = trace.gains["K"]                          # (T,3,3)
                trK   = np.array([np.trace(K_arr[i]) for i in range(len(K_arr))])
                dt    = float(trace.time[1] - trace.time[0]) if len(trace.time) > 1 else 0.01
                dtrK_dt = np.diff(trK) / dt                       # (T-1,) N/m/s
                total_cost += RATE_WEIGHT * float(np.sqrt(np.mean(dtrK_dt**2)))

            # --- Time penalty (only active when tau is learnable) ---
            # Penalises long trajectories: cost += TIME_WEIGHT * (tau / tau_budget)
            #   tau == tau_budget  → +TIME_WEIGHT   (worst, full horizon used)
            #   tau == tau_min     → +TIME_WEIGHT * tau_min/tau_budget  (best)
            # The hard VelocityLimit REQUIRE clause prevents the optimizer from
            # going arbitrarily fast — it finds the shortest tau that satisfies
            # all constraints.
            TIME_WEIGHT = 5.0
            if hasattr(trace, 'tau') and trace.tau is not None:
                total_cost += TIME_WEIGHT * float(trace.tau) / tau_budget

            return total_cost

        return objective

    def _evaluate_clause(self, trace, clause):

        predicate_fn = self.predicate_registry[clause.predicate]
        rho_trace = predicate_fn(trace, **clause.parameters)

        if clause.operator == "eventually":
            return temporal_logic.eventually(rho_trace)

        elif clause.operator == "always":
            return temporal_logic.always(rho_trace)

        else:
            raise NotImplementedError(
                f"Operator {clause.operator} not supported yet."
            )