import numpy as np

from core.cgms.dmp_with_gain import DMPWithGainScheduling
from core.cgms.dynamical_systems import DynamicalSystems


class Trace:
    """
    Lightweight container for trajectory trace.
    """
    def __init__(self, time, position, velocity, gains,
                 raw_sk_weights=None, raw_sd_weights=None, tau=None):
        self.time = time
        self.position = position
        self.velocity = velocity
        self.gains = gains
        # Pre-clip raw weights — used by compiler for honest stiffness penalty
        self.raw_sk_weights = raw_sk_weights
        self.raw_sd_weights = raw_sd_weights
        # Actual duration used for this rollout — used by time penalty in compiler
        self.tau = tau


class CertifiedPolicy:

    def __init__(self, tau, tau_learnable=False, tau_min=0.5, tau_max=None):

        # ---- Hardcoded for now ----
        start_pos = np.array([0.55, 0.00, 0.11])
        goal_pos  = np.array([0.05, 0.72, 0.11])

        # ---- Hyperparameters ----
        self.TAU = tau
        self.DT = 0.01
        self.ALPHA = 0.05
        self.K0 = 200.0
        self.D0 = 30.0

        # ---- Learnable tau settings ----
        # When tau_learnable=True, one extra scalar is appended to theta.
        # It is sigmoid-mapped to [tau_min, tau_max] so tau stays bounded.
        self.tau_learnable = tau_learnable
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max) if tau_max is not None else float(tau)

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
        # +1 for the learnable tau scalar when tau_learnable=True
        return self.theta_dim + (1 if self.tau_learnable else 0)

    def structured_sigma(self, sigma_traj_xy=5.0, sigma_traj_z=5.0,
                         sigma_sd=5.0, sigma_sk=5.0, sigma_tau=1.0):
        """
        Build a per-parameter exploration noise vector.

        Parameter layout (total = parameter_dimension()):
          [traj_X (51)] [traj_Y (51)] [traj_Z (51)] [SD (42)] [SK (42)] [tau? (1)]

        When tau_learnable=True, the last element is the tau parameter.
        sigma_tau=1.0 explores ≈ ±1 in sigmoid-input space, which maps to
        roughly ±(tau_max-tau_min)/4 seconds around the midpoint.
        """
        n_traj, n_sd, n_sk = self.sizes
        n_per_axis = n_traj // 3

        sigma = np.empty(self.theta_dim)
        off = 0
        sigma[off:off + n_per_axis] = sigma_traj_xy;  off += n_per_axis  # X
        sigma[off:off + n_per_axis] = sigma_traj_xy;  off += n_per_axis  # Y
        sigma[off:off + n_per_axis] = sigma_traj_z;   off += n_per_axis  # Z
        sigma[off:off + n_sd]       = sigma_sd;        off += n_sd        # SD
        sigma[off:off + n_sk]       = sigma_sk;        off += n_sk        # SK

        if self.tau_learnable:
            sigma = np.append(sigma, sigma_tau)

        return sigma

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
        # Split out tau if learnable
        # ----------------------------
        if self.tau_learnable:
            theta_dmp = theta[:-1]          # DMP weights
            theta_tau = float(theta[-1])    # raw scalar → sigmoid → bounded tau
            sig = 1.0 / (1.0 + np.exp(-theta_tau))
            tau = self.tau_min + (self.tau_max - self.tau_min) * sig
        else:
            theta_dmp = theta
            tau = self.TAU

        # ----------------------------
        # Update DMP internal timing
        # ----------------------------
        self.dmp.tau = tau
        self.dmp.ts = np.arange(0.0, tau + 1e-12, self.dmp.dt)
        self.dmp.T  = self.dmp.ts.size
        self.dmp.ds = DynamicalSystems(tau)

        # ----------------------------
        # Set DMP parameters
        # ----------------------------
        # Extract raw SK/SD weights BEFORE set_theta clips them — needed for
        # the honest stiffness penalty in compiler.py.
        n_traj, n_sd, n_sk = self.sizes
        raw_sd = theta_dmp[n_traj:n_traj + n_sd].copy()
        raw_sk = theta_dmp[n_traj + n_sd:n_traj + n_sd + n_sk].copy()

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
            },
            raw_sk_weights=raw_sk,
            raw_sd_weights=raw_sd,
            tau=tau,
        )

        return trace