# spec/compiler.py

from logic import predicates
from logic import temporal_logic


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
            # ---- Time pressure penalty ----
            # if hasattr(taskspec, "horizon_sec") and taskspec.horizon_sec is not None:
            #     tau_actual = trace.time[-1]
            #     time_violation = max(0.0, tau_actual - taskspec.horizon_sec)
            #     total_cost += 200.0 * (time_violation ** 2)

            # Hard clause check (optional logging for now)
            # --- Hard clauses with slack relaxation ---
            SLACK_WEIGHT = 500.0  # λ_s (tuneable)

            for clause in hard_clauses:
                rho = self._evaluate_clause(trace, clause)

                # Slack variable s >= 0
                s = max(0.0, -rho)

                # Quadratic slack penalty
                total_cost += SLACK_WEIGHT * (s ** 2)

            # Soft clauses contribute to cost
            for clause in soft_clauses:
                rho = self._evaluate_clause(trace, clause)
                J = max(0.0, -rho)
                total_cost += clause.weight * J

            return total_cost

        return objective

    def _evaluate_clause(self, trace, clause):

        predicate_fn = self.predicate_registry[clause.predicate]
        rho_trace = predicate_fn(trace, **clause.parameters)

        # Apply time window if specified
        if hasattr(clause, "time_window") and clause.time_window is not None:
            t_start, t_end = clause.time_window
            times = trace.time
            mask = (times >= t_start) & (times <= t_end)

            # IMPORTANT: apply mask
            rho_trace = rho_trace[mask]

            # Edge case: empty window
            if len(rho_trace) == 0:
                return -1e6  # strong violation

        if clause.operator == "eventually":
            return temporal_logic.eventually(rho_trace)

        elif clause.operator == "always":
            return temporal_logic.always(rho_trace)

        else:
            raise NotImplementedError(
                f"Operator {clause.operator} not supported yet."
            )