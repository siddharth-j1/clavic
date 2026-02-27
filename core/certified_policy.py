import numpy as np

from core.cgms.dmp_with_gain import DMPWithGainScheduling
from core.cgms.dynamical_systems import DynamicalSystems


class Trace:
    """
    Lightweight container for trajectory trace.
    """
    def __init__(self, time, position, velocity, gains):
        self.time = time
        self.position = position
        self.velocity = velocity
        self.gains = gains


class CertifiedPolicy:

    def __init__(self, tau):

        # ---- Hardcoded for now ----
        start_pos = np.array([0.55, 0.00, 0.11])
        goal_pos  = np.array([0.05, 0.72, 0.11])

        # ---- Same hyperparameters as CGMS ----
        self.TAU = tau
        self.DT = 0.01
        self.ALPHA = 0.05
        self.K0 = 200.0
        self.D0 = 30.0

        # ---- Instantiate CGMS DMP ----
        self.dmp = DMPWithGainScheduling(
            start=start_pos,
            end=goal_pos,
            tau=self.TAU,
            dt=self.DT,
            n_bfs_traj=51,
            n_bfs_slack=7,
            K0=self.K0,
            D0=self.D0,
            alpha=self.ALPHA,
            H=np.eye(3)
        )

        # ---- Extract parameter vector info ----
        theta_init, n_traj, n_damp, n_stiff = self.dmp.initial_weights()
        self.theta_dim = len(theta_init)
        self.sizes = (n_traj, n_damp, n_stiff)

    # def parameter_dimension(self):
    #     # +1 for time scaling parameter
    #     return self.dmp.param_dim
    
    def parameter_dimension(self):
    # +1 for time scaling parameter
        return self.theta_dim 

    # def rollout(self, theta):

    #     # ----------------------------
    #     # Split parameters
    #     # ----------------------------
    #     theta_dmp = theta[:-1]   # all except last
    #     theta_time = theta[-1]   # last element controls time

    #     # ----------------------------
    #     # Map time parameter to bounded duration
    #     # ----------------------------
    #     tau_min = 1.0
    #     tau_max = 6.0

    #     # sigmoid mapping to keep tau positive and bounded
    #     tau = tau_min + (tau_max - tau_min) * (1 / (1 + np.exp(-theta_time)))

    #     # Set DMP duration
    #     self.dmp.tau = tau
    #     self.dmp.tau = tau
    #     self.dmp.ts = np.arange(0.0, tau + 1e-12, self.dmp.dt)
    #     self.dmp.T  = self.dmp.ts.size
    #     self.dmp.ds = DynamicalSystems(tau)

    #     # ----------------------------
    #     # Set DMP parameters
    #     # ----------------------------
    #     self.dmp.set_theta(theta_dmp, self.sizes)

    #     # ----------------------------
    #     # Rollout
    #     # ----------------------------
    #     plan = self.dmp.rollout_traj()

    #     trace = Trace(
    #         time=plan["ts"],
    #         position=plan["y_des"],
    #         velocity=plan["yd_des"],
    #         gains={
    #             "K": plan["K"],
    #             "D": plan["D"]
    #         }
    #     )

    #     return trace


    def rollout(self, theta):

        # ----------------------------
        # Use full theta as DMP weights
        # ----------------------------
        theta_dmp = theta

        # ----------------------------
        # Deterministic time scaling
        # ----------------------------
        tau = self.TAU   # fixed duration (e.g., 2.0, 5.0, 10.0)

        # Update DMP internal timing
        self.dmp.tau = tau
        self.dmp.ts = np.arange(0.0, tau + 1e-12, self.dmp.dt)
        self.dmp.T  = self.dmp.ts.size
        self.dmp.ds = DynamicalSystems(tau)

        # ----------------------------
        # Set DMP parameters
        # ----------------------------
        self.dmp.set_theta(theta_dmp, self.sizes)

        # ----------------------------
        # Rollout
        # ----------------------------
        plan = self.dmp.rollout_traj()

        trace = Trace(
            time=plan["ts"],
            position=plan["y_des"],
            velocity=plan["yd_des"],
            gains={
                "K": plan["K"],
                "D": plan["D"]
            }
        )

        return trace