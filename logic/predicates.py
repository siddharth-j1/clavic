# logic/predicates.py

import numpy as np


def at_goal_pose(trace, target, tolerance=0.02):
    pos = trace.position
    d = np.linalg.norm(pos - target, axis=1)
    return tolerance - d


def human_comfort_distance(trace, human_position, preferred_distance):
    pos = trace.position
    d = np.linalg.norm(pos - human_position, axis=1)
    return d - preferred_distance


def human_body_exclusion(trace, human_position, body_radius):
    """
    Hard exclusion zone — the physical body of the human.
    Robot must NEVER enter this radius. Use with modality=REQUIRE.
    rho > 0  means safe (outside body)
    rho < 0  means collision (inside body)
    """
    pos = trace.position
    d = np.linalg.norm(pos - human_position, axis=1)
    return d - body_radius


def velocity_limit(trace, vmax):
    if trace.velocity is None:
        raise ValueError("Velocity not available in trace.")
    v_norm = np.linalg.norm(trace.velocity, axis=1)
    return vmax - v_norm


def early_completion(trace, tau_budget):
    """
    Rewards finishing early within the allowed time budget.

    rho > 0  →  robot completed faster than budget  (good)
    rho < 0  →  robot took longer than budget        (bad)

    Use with modality=PREFER so the optimizer is incentivized to reduce tau
    while still satisfying all hard constraints (velocity, exclusion, goal).

    @param tau_budget   (float)
        Maximum allowed duration (e.g. 2.0 s).  The optimizer will try to
        beat this by choosing a smaller learned tau.
    """
    tau_actual = getattr(trace, 'tau', tau_budget)
    # Return a scalar wrapped in an array so temporal_logic operators work
    return np.array([tau_budget - tau_actual])