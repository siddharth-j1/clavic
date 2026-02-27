"""
Optimize and save the tau=0.5s trajectory checkpoint.

Run this once to generate short_tau_checkpoint.npz.
Then run compare_tau_initialization.py to plot both tau=2s and tau=0.5s.

Usage:
    python optimize_short_tau.py
"""

import numpy as np

from core.certified_policy import CertifiedPolicy
from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from logic.predicates import at_goal_pose, human_comfort_distance, velocity_limit, human_body_exclusion
from optimization.optimizer import PI2
from experiment_checkpoint_warmstart import save_checkpoint


TAU_SHORT        = 0.5
N_SAMPLES        = 12
N_UPDATES        = 80
CHECKPOINT_PATH  = "short_tau_checkpoint.npz"


def main():
    # Load taskspec but override tau
    taskspec = load_taskspec_from_json("spec/example_task.json")
    taskspec.horizon_sec = TAU_SHORT          # override tau only — all predicates stay the same

    predicate_registry = {
        "AtGoalPose":            at_goal_pose,
        "HumanComfortDistance":  human_comfort_distance,
        "HumanBodyExclusion":    human_body_exclusion,
        "VelocityLimit":         velocity_limit,
    }
    objective_fn = Compiler(predicate_registry).compile(taskspec)

    policy    = CertifiedPolicy(tau=TAU_SHORT)
    theta_dim = policy.parameter_dimension()

    theta_init = np.zeros(theta_dim)
    sigma_init = policy.structured_sigma()   # uniform σ=5.0, fresh start
    pi2 = PI2(theta=theta_init, sigma=sigma_init, lam=0.01, decay=0.98)

    best_cost  = float("inf")
    best_theta = theta_init.copy()

    print(f"Optimizing tau={TAU_SHORT}s  ({N_UPDATES} updates × {N_SAMPLES} samples) ...")
    for i in range(N_UPDATES):
        samples = pi2.sample(N_SAMPLES)
        costs   = np.array([objective_fn(policy.rollout(s)) for s in samples])
        pi2.update(samples, costs)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        print(f"  Update {i+1:02d} | Min: {costs.min():.4f} | Mean: {costs.mean():.4f} | Best: {best_cost:.4f}")

    trace = policy.rollout(best_theta)
    save_checkpoint(best_theta, TAU_SHORT, best_cost, trace, checkpoint_path=CHECKPOINT_PATH)
    print(f"Done — best cost: {best_cost:.4f}")


if __name__ == "__main__":
    main()
