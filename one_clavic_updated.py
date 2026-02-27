# ==========================================
# core/certified_policy.py
# ==========================================
import numpy as np

from core.cgms.dmp_with_gain import DMPWithGainScheduling


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

    def __init__(self):

        # ---- Hardcoded for now ----
        start_pos = np.array([0.55, 0.00, 0.11])
        goal_pos  = np.array([0.05, 0.72, 0.11])

        # ---- Same hyperparameters as CGMS ----
        self.TAU = 5.0
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
    #     return self.dmp.param_dim + 1
    
    def parameter_dimension(self):
    # +1 for time scaling parameter
        return self.theta_dim + 1

    def rollout(self, theta):

        # ----------------------------
        # Split parameters
        # ----------------------------
        theta_dmp = theta[:-1]   # all except last
        theta_time = theta[-1]   # last element controls time

        # ----------------------------
        # Map time parameter to bounded duration
        # ----------------------------
        tau_min = 1.0
        tau_max = 6.0

        # sigmoid mapping to keep tau positive and bounded
        tau = tau_min + (tau_max - tau_min) * (1 / (1 + np.exp(-theta_time)))

        # Set DMP duration
        self.dmp.tau = tau

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

# ==========================================
# core/cgms/dmp_with_gain.py
# ==========================================
import numpy as np
from .utils import lt_pack, lt_unpack, sym
from .minimum_jerk import MinimumJerk
from .dynamical_systems import DynamicalSystems
from .function_approximator import FunctionApproximatorRBFN

class DMPWithGainScheduling:
    """
    DMP with gain scheduling via RBFs for trajectory and gain modulation.
    """
    def __init__(self, start, end, tau, dt, n_bfs_traj, n_bfs_slack, K0, D0, alpha, H, 
                normalize_rbfs_traj=True, normalize_rbfs_slack=True, 
                slack_mag=20.0, slack_rate_limit=200.0
        ):
        """
        @param start                    (np.ndarray)
            Start position vector. 
        @param end                      (np.ndarray)
            End position vector.
        @param tau                      (float)
            Duration of the trajectory.
        @param dt                       (float)   
            Time step for discretization.
        @param n_bfs_traj               (int)
            Number of RBFs for trajectory generation.
        @param n_bfs_slack              (int)    
            Number of RBFs for gain scheduling.
        @param K0                       (float)   
            Nominal stiffness gain.
        @param D0                       (float)
            Nominal damping gain.
        @param alpha                    (float)
            Gain scheduling scaling factor.
        @param H                        (np.ndarray)
            Desired stiffness matrix for the task.
        @param normalize_rbfs_traj      (bool)
            Whether to normalize trajectory RBF outputs.
        @param normalize_rbfs_slack     (bool)
            Whether to normalize gain scheduling RBF outputs.
        @param slack_mag                (float)    
            Maximum magnitude for slack variables.
        @param slack_rate_limit         (float) 
            Maximum rate of change for slack variables.
        """
        self.start      = np.asarray(start, float).reshape(3)
        self.end        = np.asarray(end, float).reshape(3)
        self.tau        = float(tau)
        self.dt         = float(dt)
        self.ts         = np.arange(0.0, self.tau+1e-12, self.dt)
        self.T          = self.ts.size
        self.alpha      = float(alpha)
        self.H          = np.asarray(H, float).reshape(3, 3)
        self.K0         = float(K0)
        self.D0         = float(D0)
        self.slack_mag  = float(slack_mag)
        self.slack_rate = float(slack_rate_limit)
        self.ds         = DynamicalSystems(self.tau)
        
        y, yd, ydd, ts  = MinimumJerk(self.start, self.end, self.tau, self.dt).generate()
        phase           = self.ds.time_system(ts)
        goal            = self.ds.polynomial_system(ts, self.start, self.end, 3)

        d, m = 20.0, 1.0
        k    = (d**2) / 4.0
        
        """
        Initialize trajectory RBFs by computing target forcing term
        """
        self.rbf_traj   = [FunctionApproximatorRBFN(n_bfs_traj, normalize=normalize_rbfs_traj, intersection_height = 0.95) for _ in range(3)]
        spring      = k * (y - goal)
        damper      = d * self.tau * yd
        f_target    = (self.tau**2) * ydd + (spring + damper) / m
        f_target    = f_target / (phase[:,None] + 1e-12)
        for i in range(3): 
            self.rbf_traj[i].train(phase, f_target[:,i])

        """
        Initialize slacks for constant gains pre-sampling 
        """
        self.rbf_SD = FunctionApproximatorRBFN(n_bfs_slack, normalize=normalize_rbfs_slack, intersection_height = 0.7)
        self.rbf_SK = FunctionApproximatorRBFN(n_bfs_slack, normalize=normalize_rbfs_slack, intersection_height = 0.7)
        I = np.eye(3)
        H = self.H
        # We want to find SK0 such that SK0^2 = 2*alpha*K0*I or SK0 = sqrt(2*alpha*K0)*I
        SK0 = np.sqrt(max(0.0, 2*alpha*K0)) * I
        # We want to find SD0 such that SD0^2 = D0*I - alpha*H or SD0 = sqrt(D0*I - alpha*H)
        # Perform eigen-decomposition to calculate square root of SD0
        w, V = np.linalg.eigh(sym(D0 * I - alpha*H))
        w = np.clip(w, 0, None)
        SD0 = (V * np.sqrt(w)) @ V.T
        SK = np.tile(lt_pack(SK0)[None,:], (ts.size,1))
        SD = np.tile(lt_pack(SD0)[None,:], (ts.size,1))
        self.rbf_SK.train(phase, SK)
        self.rbf_SD.train(phase, SD)
        
    def initial_weights(self):
        """
        @brief
            Concatenate the weight matrices into a single vector for optimization.
        """
        theta = np.concatenate([r.W.ravel() for r in self.rbf_traj] + [self.rbf_SD.W.ravel(), self.rbf_SK.W.ravel()])
        n_forcing_weights   = sum(r.W.size for r in self.rbf_traj)
        n_damping_weights   = self.rbf_SD.W.size
        n_stiffness_weights = self.rbf_SK.W.size
        return theta, n_forcing_weights, n_damping_weights, n_stiffness_weights
    
    def set_theta(self, theta, sizes):
        """
        @brief
            Slice the flat theta back into the weight matrices in the same order as initial_weights().

        @param theta (np.ndarray)
            Flat weight vector.
        @param sizes (Tuple[int, int, int])
            Sizes of the weight matrices: (n_forcing_weights, n_damping_weights, n_stiffness_weights).
        """
        _, n_damping_weights, n_stiffness_weights = sizes
        off = 0
        for r in self.rbf_traj:
            n   = r.W.size
            r.W = theta[off:off + n].reshape(r.W.shape)
            off += n
        self.rbf_SD.W   = theta[off:off + n_damping_weights].reshape(self.rbf_SD.W.shape)
        off             +=n_damping_weights
        self.rbf_SK.W   = theta[off:off + n_stiffness_weights].reshape(self.rbf_SK.W.shape)

    def rollout_traj(self, sample_unsafe: bool = False):
        ts   = self.ts
        T    = self.T
        y    = np.zeros((T,3))
        yd   = np.zeros((T,3))
        ydd  = np.zeros((T,3))
        y[0] = self.start

        def dmp(t, y, yd):
            phase = self.ds.time_system(np.array([t]))[0]
            gate  = phase
            goal  = self.ds.polynomial_system(np.array([t]), self.start, self.end, 3)[0]
            fhat  = np.array([self.rbf_traj[i].predict(phase)[0,0] for i in range(3)])
            d, m  = 20.0, 1.0
            k     = (d**2)/4.0
            spring = k * (y - goal)
            damper = d * self.tau * yd
            return ((fhat * gate) - (spring + damper) / m) / (self.tau**2)

        for k in range(T-1):
            t0 = ts[k]; h = ts[k+1] - ts[k]
            k1y = yd[k];               k1v = dmp(t0,            y[k],                    yd[k])
            k2y = yd[k] + 0.5*h*k1v;   k2v = dmp(t0 + 0.5*h,    y[k] + 0.5*h*k1y,        yd[k] + 0.5*h*k1v)
            k3y = yd[k] + 0.5*h*k2v;   k3v = dmp(t0 + 0.5*h,    y[k] + 0.5*h*k2y,        yd[k] + 0.5*h*k2v)
            k4y = yd[k] + 1.0*h*k3v;   k4v = dmp(t0 + 1.0*h,    y[k] + 1.0*h*k3y,        yd[k] + 1.0*h*k3v)
            y[k+1]  = y[k]  + (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
            yd[k+1] = yd[k] + (h/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
            ydd[k]  = k1v
        ydd[-1] = dmp(ts[-1], y[-1], yd[-1])

        x    = self.ds.time_system(ts)
        xdot = -np.ones_like(x) / self.tau

        SD_vecs, SDdot_vecs = self.rbf_SD.predict_with_time_derivative(x, xdot)     # (T,6), (T,6)
        SK_vecs              = self.rbf_SK.predict(x)                               # (T,6)

        SD   = np.array([lt_unpack(v) for v in SD_vecs])        # (T,3,3)
        SDot = np.array([lt_unpack(v) for v in SDdot_vecs])     # (T,3,3)
        SK   = np.array([lt_unpack(v) for v in SK_vecs])        # (T,3,3)

        H = self.H
        D    = np.array([sym(self.alpha*H + SD[k]@SD[k].T) for k in range(T)])                  # (T,3,3)
        Ddot = np.array([SDot[k]@SD[k].T + SD[k]@SDot[k].T for k in range(T)])                  # (T,3,3)

        def _B_at(t):
            x    = max(0.0, 1.0 - t/self.tau); xdot = -1.0/self.tau
            SDv, SDdv = self.rbf_SD.predict_with_time_derivative(np.array([x]), np.array([xdot]))
            SKv       = self.rbf_SK.predict(np.array([x]))
            SDt   = lt_unpack(SDv[0]); SDdt = lt_unpack(SDdv[0]); SKt = lt_unpack(SKv[0])
            Ddt   = SDdt@SDt.T + SDt@SDdt.T
            return sym(-self.alpha*Ddt - SKt@SKt.T)

        Q    = np.zeros((T,3,3))
        Q[0] = np.linalg.cholesky(sym(self.K0*np.eye(3)) + 1e-9*np.eye(3))

        def fQ(Qk, t):
            Bk = _B_at(t)
            X  = np.linalg.solve(Qk.T, Bk)         # Q^{-T} B
            return self.alpha*Qk + 0.5*X

        for k in range(T-1):
            t = ts[k]; h = ts[k+1]-ts[k]
            k1 = fQ(Q[k],            t)
            k2 = fQ(Q[k] + 0.5*h*k1, t + 0.5*h)
            k3 = fQ(Q[k] + 0.5*h*k2, t + 0.5*h)
            k4 = fQ(Q[k] + h*k3,     t + h)
            Q[k+1] = Q[k] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        K = np.array([sym(Q[k].T @ Q[k]) for k in range(T)])

        return {
            "ts": ts, "y_des": y, "yd_des": yd, "ydd_des": ydd,
            "SD": SD, "SK": SK,            
            "D": D, "Ddot": Ddot, "K": K
        }

# ==========================================
# core/cgms/dynamical_systems.py
# ==========================================
import numpy as np

class DynamicalSystems:
    """
    Generates the canonical phase and gating signal for DMPs.
    """
    def __init__(self, tau, decay = 0.1, D0 = 1e-7):
        """
        @param tau (float)
            Time constant for the phase variable.
        @param decay (float)
            Decay rate for the sigmoid gating function.
        @param D0 (float)
            Offset for the sigmoid gating function.
        """
        self.tau    = float(tau)
        self.decay  = float(decay) 
        self.K      = 1.0 + D0 
        self.D0     = D0
        num         = ((self.K / self.decay) - 1.0) / self.D0
        self.r      = -np.log(num) / self.tau

    def time_system(self, ts): 
        """
        @brief
            A dynamical system with linear decay.
            Dynamics:               x'   = -1 / tau
            Analytical solution:    x(t) = 1 - (t / tau)

        @param ts (np.ndarray)
            Time stamps.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts = np.asarray(ts, float)
        return np.clip(1.0 - ts/self.tau, 0.0, 1.0) 
    
    def sigmoid_system(self, ts):
        """
        @brief
            A dynamical system with sigmoid decay.
            Dynamics:               x'   = r * x * (1 - x / K)
            Analytical solution:    x(t) = K / (1 + D0 * exp(-r * t))

        @param ts (np.ndarray)
            Time stamps.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts = np.asarray(ts, float)
        return self.K / (1.0 + self.D0*np.exp(-self.r*ts))
    
    def exponential_system(self, ts, start, goal, alpha=15.0):
        """
        @brief
            A dynamical system with exponential decay.
            Dynamics:               x'   = -(alpha / tau) * (x - goal)
            Analytical solution:    x(t) = goal + (start - goal) * exp(-(alpha / tau) * t)

        @param ts (np.ndarray)
            Time stamps.
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param alpha (float)
            Decay rate.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts      = np.asarray(ts, float)
        start   = np.asarray(start, float).reshape(3) 
        goal    = np.asarray(goal, float).reshape(3)
        return goal[None,:] + (start - goal)[None,:] * np.exp(-(alpha / self.tau) * ts)[:,None]
    
    def polynomial_system(self, ts, start, goal, alpha=15.0):
        """
        @brief
            A dynamical system with polynomial decay.
            Dynamics:               x'   = -(alpha / tau) * (x - goal)^(1 - 1/alpha)
            Analytical solution:    x(t) = goal + (start - goal) * (1 - t / tau)^alpha

        @param ts (np.ndarray)
            Time stamps.
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param alpha (float)
            Decay rate.

        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts    = np.asarray(ts, float)
        s     = np.clip(ts / self.tau, 0.0, 1.0)          
        start = np.asarray(start, float).reshape(3)
        goal  = np.asarray(goal,  float).reshape(3)
        return goal[None,:] + (start - goal)[None,:] * (1.0 - s)[:,None]**alpha

# ==========================================
# core/cgms/function_approximator.py
# ==========================================
import numpy as np

class FunctionApproximatorRBFN:
    """
    Radial Basis Function (RBF) approximator with Gaussian kernels.
    """
    def __init__(self, n_bfs: int, intersection_height: float = 0.7, regularization: float = 1e-6, normalize: bool = True):
        """
        @param n_bfs (int)
            Number of RBFs.
        @param intersection_height (float)
            Height at which two neighbouring basis functions intersect.
        @param regularization (float)
            Regularization term for the least squares solution.
        @param normalize (bool)
            Whether to normalize the RBF outputs to sum to 1.
        """
        self.M          = int(n_bfs)
        self.h          = float(intersection_height)
        self.reg        = float(regularization)
        self.normalize  = bool(normalize)

        self.centers = None    # shape (M, 1)
        self.widths  = None    # shape (M, 1)
        self.W       = None    # shape (M, D)

    def _compute_centers_widths(self):
        """
        @brief
            Evenly space centers in [0, 1], calculate widths from intersection height h.
        """
        if self.M > 1:
            self.centers    = np.linspace(0, 1, self.M).reshape(-1,1)
            delta           = self.centers[1] - self.centers[0]
            # Standard deviation (sigma) for Gaussian basis functions such that two neighbouring functions intersect at height h
            sigma           = float(delta) / np.sqrt(-8.0 * np.log(self.h))
            self.widths     = np.full((self.M, 1), sigma)
        else:
            self.centers = np.array([[0.5]])
            self.widths  = np.array([[1.0]])

    def _activations(self, x):
        """
        @brief
            Compute the RBF activations for input x.

        @param x (np.ndarray)
            Input values, shape (T,).
        """
        X   = np.asarray(x, float).reshape(-1,1)        # shape (T, 1)
        C   = self.centers.T                            # shape (1, M)
        W   = self.widths.T                             # shape (1, M)
        phi = np.exp(-0.5 * ((X - C) / W)**2)           # shape (T, M)
        if self.normalize:
            s   = np.sum(phi, axis=1, keepdims=True) + 1e-12
            phi = phi/s
        return phi
    
    def train(self, x, fx):
        """
        @brief
            Train the RBF weights W such that psi(x) @ W = fx, using regularized least squares.

        @param x (np.ndarray)
            Input values, shape (T,).
        @param fx (np.ndarray)
            Target output values, shape (T, D).
        """
        x   = np.asarray(x, float).reshape(-1)
        FX  = np.asarray(fx, float)

        if FX.ndim == 1:
            FX = FX[:,None]

        # 1) set up Gaussians 
        self._compute_centers_widths()

        # 2) build activation matrix
        PSI = self._activations(x)

        # 3) solve for W using regularized least squares
        A   = PSI.T @ PSI + self.reg * np.eye(self.M)
        B   = PSI.T @ FX
        try: 
            self.W = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.W, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    def predict(self,x):
        """
        @brief
            Predict output for input x using the trained RBF weights.

        @param x (np.ndarray)
            Input values, shape (T,).
            
        @return (np.ndarray)
            Predicted output values, shape (T, D).
        """
        if self.W is None: 
            raise RuntimeError("RBF not trained")
        PSI = self._activations(np.asarray(x, float).reshape(-1))
        return PSI @ self.W
    
    def activations_and_time_derivative(self, x, xdot):
        X  = np.asarray(x, float).reshape(-1, 1)            # (T,1)
        xd = np.asarray(xdot, float).reshape(-1, 1)         # (T,1)
        C  = self.centers.T                                 # (1,M)
        W  = self.widths.T                                  # (1,M)

        G   = np.exp(-0.5 * ((X - C) / W)**2)               # (T,M)
        dGdx = G * (-(X - C) / (W**2))                      # (T,M)
        dGdt = dGdx * xd                                    # (T,M)  chain rule

        if self.normalize:
            s   = np.sum(G, axis=1, keepdims=True) + 1e-12  # (T,1)
            sd  = np.sum(dGdt, axis=1, keepdims=True)       # (T,1)
            Phi = G / s
            dPhi_dt = (dGdt * s - G * sd) / (s**2)          # quotient rule
        else:
            Phi = G
            dPhi_dt = dGdt

        return Phi, dPhi_dt

    def predict_with_time_derivative(self, x, xdot):
        if self.W is None:
            raise RuntimeError("RBF not trained")
        Phi, dPhi_dt = self.activations_and_time_derivative(x, xdot)
        return Phi @ self.W, dPhi_dt @ self.W

# ==========================================
# core/cgms/minimum_jerk.py
# ==========================================
import numpy as np

class MinimumJerk:
    """
    Generate minimum-jerk trajectory from start to goal in time tau with dt step.
    """
    def __init__(self, start, goal, tau, dt):
        """
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param tau (float)
            Duration of the trajectory.
        @param dt (float)
            Time step for discretization.
        """
        self.start  =   np.asarray(start, float).reshape(3)
        self.goal   =   np.asarray(goal, float).reshape(3)
        self.tau    =   float(tau)
        self.dt     =   float(dt)
        self.ts     =   np.arange(0.0, self.tau+1e-12, self.dt)

    def generate(self):
        """
        @return:
            y   (np.ndarray): Positions over time, shape (T, 3)
            yd  (np.ndarray): Velocities over time, shape (T, 3)
            ydd (np.ndarray): Accelerations over time, shape (T, 3)
            ts  (np.ndarray): Time stamps, shape (T,)
        """
        ts = self.ts
        tau_safe = max(self.tau, np.finfo(float).eps)
        s = ts / tau_safe                       
        A = (self.goal - self.start)[None, :]       

        phi   = 10*s**3 - 15*s**4 + 6*s**5
        dphi  = (30*s**2 - 60*s**3 + 30*s**4) / tau_safe
        ddphi = (60*s - 180*s**2 + 120*s**3) / (tau_safe**2)

        y   = self.start[None, :] + phi[:, None]  * A
        yd  = dphi[:, None] * A
        ydd = ddphi[:, None]* A
        return y, yd, ydd, ts

# ==========================================
# core/cgms/utils.py
# ==========================================
import numpy as np
"""
Helper functions
"""
def sym(M: np.ndarray) -> np.ndarray:
    """
    @brief
        Symmetrize a square matrix.

    @param M (np.ndarray)
        Input square matrix.

    @return (np.ndarray)
        Symmetrized matrix ( (M + M.T) / 2 ).
    """
    return 0.5 * (M + M.T)

def finite_diff(Y: np.ndarray, dt: float) -> np.ndarray:
    """
    @brief
        Compute finite differences along the first axis of an array.

    @param Y (np.ndarray)
        Input array. Can be 1D, 2D, or 3D.
    @param dt (float)
        Step size.

    @return (np.ndarray)
        Array of the same shape as Y, containing finite differences.
        Central differences are used for interior points.
        Forward and backward differences are used at the boundaries.
    """
    Y = np.asarray(Y, float)
    dY = np.zeros_like(Y)

    if Y.ndim == 1:
        dY[1:-1] = (Y[2:] - Y[:-2]) / (2 * dt)
        dY[0]    = (Y[1] - Y[0]) / dt
        dY[-1]   = (Y[-1] - Y[-2]) / dt

    elif Y.ndim == 2:
        dY[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / (2 * dt)
        dY[0, :]    = (Y[1, :] - Y[0, :]) / dt
        dY[-1, :]   = (Y[-1, :] - Y[-2, :]) / dt

    elif Y.ndim == 3:
        dY[1:-1, :, :] = (Y[2:, :, :] - Y[:-2, :, :]) / (2 * dt)
        dY[0, :, :]    = (Y[1, :, :] - Y[0, :, :]) / dt
        dY[-1, :, :]   = (Y[-1, :, :] - Y[-2, :, :]) / dt

    else:
        raise ValueError("finite_diff: unsupported ndim")

    return dY

def lt_pack(L: np.ndarray) -> np.ndarray:
    """
    @brief
        Pack the lower-triangular entries of a 3x3 matrix into a vector.

    @param L (np.ndarray)
        Input 3x3 matrix.

    @return (np.ndarray)
        Vector [L00, L10, L11, L20, L21, L22].
    """
    return np.array([L[0, 0],
                     L[1, 0], L[1, 1],
                     L[2, 0], L[2, 1], L[2, 2]], float)

def lt_unpack(v: np.ndarray) -> np.ndarray:
    """
    @brief
        Unpack a 6-element vector into the lower-triangular part of a 3x3 matrix.

    @param v (np.ndarray)
        Input vector of length 6, in the format [L00, L10, L11, L20, L21, L22].

    @return (np.ndarray)
        3x3 lower-triangular matrix.
    """
    v = np.asarray(v, float).reshape(-1)
    if v.shape[0] != 6:
        raise ValueError("lt_unpack: input vector must have length 6")
    L = np.zeros((3, 3), float)
    L[0, 0] = v[0]
    L[1, 0], L[1, 1] = v[1], v[2]
    L[2, 0], L[2, 1], L[2, 2] = v[3], v[4], v[5]
    return L

# ==========================================
# core/trace.py
# ==========================================
# core/trace.py

import numpy as np


class Trace:
    def __init__(self, time, position, orientation=None, velocity=None, gains=None):
        self.time = np.array(time)
        self.position = np.array(position)
        self.orientation = orientation
        self.velocity = velocity
        self.gains = gains

# ==========================================
# logic/predicates.py
# ==========================================
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


def velocity_limit(trace, vmax):
    if trace.velocity is None:
        raise ValueError("Velocity not available in trace.")
    v_norm = np.linalg.norm(trace.velocity, axis=1)
    return vmax - v_norm

# ==========================================
# logic/temporal_logic.py
# ==========================================
# logic/temporal_logic.py

import numpy as np


def smooth_min(values, k=40.0):
    values = np.array(values)
    return -1.0 / k * np.log(np.sum(np.exp(-k * values)))


def smooth_max(values, k=40.0):
    values = np.array(values)
    weights = np.exp(k * values)
    return np.sum(values * weights) / np.sum(weights)


def eventually(rho, k=20.0):
    return smooth_max(rho, k)


def always(rho, k=20.0):
    return smooth_min(rho, k)


def until(rho_phi, rho_psi, k1=20.0, k2=20.0):
    T = len(rho_phi)
    values = []

    for t_prime in range(T):
        if t_prime == 0:
            min_before = np.inf
        else:
            min_before = smooth_min(rho_phi[:t_prime], k=k1)

        inner = smooth_min([rho_psi[t_prime], min_before], k=k1)
        values.append(inner)

    return smooth_max(values, k=k2)

# ==========================================
# main.py
# ==========================================
import numpy as np

# ---- Spec ----
from spec.taskspec import TaskSpec, Clause
from spec.compiler import Compiler

# ---- Logic ----
from logic.predicates import at_goal_pose, human_comfort_distance

# ---- Core ----
from core.certified_policy import CertifiedPolicy

# ---- Optimizer ----
from optimization.optimizer import PI2

from spec.json_parser import load_taskspec_from_json



# ----------------------------
# Predicate Registry
# ----------------------------
def build_predicate_registry():
    return {
        "AtGoalPose": at_goal_pose,
        "HumanComfortDistance": human_comfort_distance
    }


# # ----------------------------
# # TaskSpec Builder
# # ----------------------------
# def build_taskspec(goal_pose, human_position):

#     clauses = [
#         Clause(
#             operator="eventually",
#             predicate="AtGoalPose",
#             weight=5.0,
#             modality="REQUIRE",
#             parameters={
#                 "target": goal_pose,
#                 "tolerance": 0.05
#             }
#         ),
#         Clause(
#             operator="always",
#             predicate="HumanComfortDistance",
#             weight=3.0,
#             modality="PREFER",
#             parameters={
#                 "human_position": human_position,
#                 "preferred_distance": 0.15  # realistic in this workspace
#             }
#         )
#     ]

#     return TaskSpec(
#         horizon_sec=5.0,
#         clauses=clauses
#     )


# ----------------------------
# Main Execution
# ----------------------------
def main():

    # ---- Scene Setup (MATCHES CGMS BACKBONE) ----
    goal_pose = np.array([0.05, 0.72, 0.11])   # same as CertifiedPolicy
    human_position = np.array([0.30, 0.40, 0.11])  # inside reachable workspace

    #taskspec = build_taskspec(goal_pose, human_position)
    taskspec = load_taskspec_from_json("spec/example_task.json")

    predicate_registry = build_predicate_registry()

    compiler = Compiler(predicate_registry)
    objective_fn = compiler.compile(taskspec)

    certified_policy = CertifiedPolicy()

    theta_dim = certified_policy.parameter_dimension()

    theta_init = np.zeros(theta_dim)
    sigma_init = np.ones(theta_dim) * 5.0

    pi2 = PI2(
        theta=theta_init,
        sigma=sigma_init,
        lam=0.01,
        decay=0.98
    )

    # ----------------------------
    # DEBUG: Check nominal trajectory robustness
    # ----------------------------
    trace = certified_policy.rollout(np.zeros(theta_dim))

    rho_goal = at_goal_pose(trace, goal_pose, 0.05)
    rho_human = human_comfort_distance(trace, human_position, 0.15)

    print("\n--- Nominal (theta = 0) robustness ---")
    print("Nominal max goal robustness:", np.max(rho_goal))
    print("Nominal min human robustness:", np.min(rho_human))
    print("----------------------------------------\n")

    print("Starting Optimization...")

    N_SAMPLES = 12
    N_UPDATES = 40

    best_cost = float("inf")

    current_mean = theta_init.copy()

    # ----------------------------
    # Optimization Loop
    # ----------------------------
    for update_idx in range(N_UPDATES):

        samples = pi2.sample(N_SAMPLES)
        costs = []

        for i in range(N_SAMPLES):
            theta = samples[i]
            trace = certified_policy.rollout(theta)
            cost = objective_fn(trace)
            costs.append(cost)

        costs = np.array(costs)

        # IMPORTANT: store updated mean
        current_mean, new_sigma, weights = pi2.update(samples, costs)

        best_cost = min(best_cost, costs.min())

        print(
            f"Update {update_idx+1:02d} | "
            f"Min: {costs.min():.4f} | "
            f"Mean: {costs.mean():.4f} | "
            f"BestSoFar: {best_cost:.4f}"
        )

    print("Optimization Complete.")
    trace_final = certified_policy.rollout(current_mean)
    rho_human_final = human_comfort_distance(trace_final, human_position, 0.15)
    print("Final min human robustness:", np.min(rho_human_final))

    print("Learned tau:", certified_policy.dmp.tau)

    goal_rho_trace = at_goal_pose(trace_final, goal_pose, 0.05)
    goal_times = trace_final.time
    goal_satisfied_indices = goal_rho_trace > 0

    if goal_satisfied_indices.any():
        first_hit_time = goal_times[goal_satisfied_indices][0]
        print("Goal first satisfied at time:", first_hit_time)
    else:
        print("Goal never satisfied")

    # ----------------------------
    # Visualize Nominal vs Learned
    # ----------------------------
    import matplotlib.pyplot as plt

    trace_nominal = certified_policy.rollout(np.zeros(theta_dim))
    trace_learned = certified_policy.rollout(current_mean)

    pos_nom = trace_nominal.position
    pos_learned = trace_learned.position

    plt.figure(figsize=(6,6))

    plt.plot(pos_nom[:,0], pos_nom[:,1], 'b--', label="Nominal")
    plt.plot(pos_learned[:,0], pos_learned[:,1], 'r', label="Learned")

    plt.scatter(pos_nom[0,0], pos_nom[0,1], c='green', s=100, label="Start")
    plt.scatter(goal_pose[0], goal_pose[1], c='black', s=100, label="Goal")
    plt.scatter(human_position[0], human_position[1], c='orange', s=100, label="Human")

    circle = plt.Circle(
        (human_position[0], human_position[1]),
        0.15,
        color='orange',
        fill=False,
        linestyle=':'
    )
    plt.gca().add_patch(circle)

    plt.legend()
    plt.axis('equal')
    plt.title("Nominal vs Learned Trajectory")
    plt.show()
    plt.savefig("traj.png", dpi=300)
    print("Saved traj.png")


if __name__ == "__main__":
    main()

# ==========================================
# optimization/objective_interface.py
# ==========================================
# optimization/objective_interface.py

class ObjectiveInterface:

    def __init__(self, certified_policy, objective_function):
        self.certified_policy = certified_policy
        self.objective_function = objective_function

    def evaluate(self, xi):
        trace = self.certified_policy.rollout(xi)
        return self.objective_function(trace)

# ==========================================
# optimization/optimizer.py
# ==========================================
import numpy as np

class PI2:
    """
    PI^2 style updater over a parameter vector theta.
    """
    def __init__(self, theta, sigma, lam=1.0, decay=0.98, seed=0):
        """
        @param theta    (np.ndarray)
            Initial guess of parameters
        @param sigma    (np.ndarray)
            Exploration noise per parameter 
        @param lam      (float)
            Temperature parameter 
        @param decay    (float)
            Moving average for smoothing of covariance
        """
        self.mean  = theta.copy()
        self.sigma = sigma.copy()
        self.lam   = float(lam)
        self.decay = float(decay)
        self.rng   = np.random.default_rng(int(seed))

    def sample(self, n):
        """
        @param n    (float)
            Number of sample in each iteration

        @return     (np.ndarray (n, p))
            Samples of p dimensional parameter vector drawn from ~ N(mean, diag(covariance))
        """
        # unscaled noise matrix of shape (n, p) where each element is ~ N(0, 1)
        z = self.rng.normal(0.0, 1.0, size=(n, self.mean.size))
        return self.mean[None, :] + z * self.sigma[None, :]

    def _weights_from_costs(self, costs):
        """
        @param      (np.ndarray (n, 1))
            Vector of cost for each sample of an iteration
        """
        # PI^2 uses a temperature λ without max-min range scaling
        cmin = float(np.min(costs))
        lam  = max(1e-12, self.lam)
        w = np.exp(-(costs - cmin) / lam)
        return w / (np.sum(w) + 1e-12)

    def update(self, samples, costs):
        """
        @brief 
            Update the mean and standard deviation of the distribution based on sample rollouts.

        @param samples      (np.ndarray (n, p))
            Sample drawn from ~ N(mean, diag(covariance))
        @param costs        (np.ndarray (n,))
            Cost associated with each sampled parameter vector
        
        @return new_mean    (np.ndarray, shape = (p,))
            Updated parameter mean.
        @return new_sigma   (np.ndarray, shape = (p,))
            Updated standard deviation per parameter dimension.
        @return w           (np.ndarray, shape = (n,))
            Normalized importance weights for each sample.
        """
        w = self._weights_from_costs(costs)                             # (N,)
        new_mean = np.sum(samples * w[:, None], axis=0)                 # (P,)
        # weighted diagonal covariance estimate with exponential weights
        diff2 = np.sum(w[:, None] * (samples - new_mean[None, :])**2, axis=0)
        new_sigma = np.sqrt(self.decay * self.sigma**2 + (1.0 - self.decay) * diff2 + 1e-12)
        self.mean, self.sigma = new_mean, new_sigma
        return new_mean, new_sigma, w


class PIBB:
    """
    PI-BB style updater over a parameter vector theta.
    """
    def __init__(self, theta, sigma, beta=8.0, decay=0.98, seed=0):
        """
        @param theta    (np.ndarray)
            Initial guess of parameters
        @param sigma    (np.ndarray)
            Exploration noise per parameter 
        @param beta     (float)
            Softmax sharpness in cost-to-weight 
        @param decay    (float)
            Moving average for smoothing of covariance
        """
        self.mean  = theta.copy()
        self.sigma = sigma.copy()
        self.beta  = float(beta)
        self.decay = float(decay)
        self.rng   = np.random.default_rng(int(seed))

    def sample(self, n):
        """
        @param n    (float)
            Number of sample in each iteration

        @return     (np.ndarray (n, p))
            Samples of p dimensional parameter vector drawn from ~ N(mean, diag(covariance))
        """
        # unscaled noise matrix of shape (n, p) where each element is ~ N(0, 1)
        z = self.rng.normal(0.0, 1.0, size=(n, self.mean.size))
        return self.mean[None, :] + z * self.sigma[None, :]

    def _weights_from_costs(self, costs):
        """
        @param      (np.ndarray (n, 1))
            Vector of cost for each sample of an iteration
        """
        # minimum and maximum cost values among all sampled rollouts within a single iteration 
        cmin, cmax = float(np.min(costs)), float(np.max(costs))
        scale = max(1e-12, cmax - cmin)
        w = np.exp(-self.beta * (costs - cmin) / scale)
        return w / (np.sum(w) + 1e-12)

    def update(self, samples, costs):
        """
        @brief 
            Update the mean and standard deviation of the distribution based on sample rollouts.

        @param samples      (np.ndarray (n, p))
            Sample drawn from ~ N(mean, diag(covariance))
        @param costs        (np.ndarray (n,))
            Cost associated with each sampled parameter vector
        
        @return new_mean    (np.ndarray, shape = (p,))
            Updated parameter mean.
        @return new_sigma   (np.ndarray, shape = (p,))
            Updated standard deviation per parameter dimension.
        @return w           (np.ndarray, shape = (n,))
            Normalized importance weights for each sample.
        """
        w = self._weights_from_costs(costs)                             # (N,)
        new_mean = np.sum(samples * w[:, None], axis=0)                 # (P,)
        diff2 = np.sum(w[:, None] * (samples - new_mean[None, :])**2, axis=0)
        new_sigma = np.sqrt(self.decay * self.sigma**2 + (1.0 - self.decay) * diff2 + 1e-12)
        self.mean, self.sigma = new_mean, new_sigma
        return new_mean, new_sigma, w

# ==========================================
# spec/compiler.py
# ==========================================
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

# ==========================================
# spec/json_parser.py
# ==========================================
import json
from spec.taskspec import TaskSpec, Clause


def load_taskspec_from_json(path):

    with open(path, "r") as f:
        data = json.load(f)

    horizon_sec = data["horizon_sec"]
    bindings = data.get("bindings", {})

    clauses = []

    for c in data["clauses"]:

        operator = c["type"]
        weight = c["weight"]
        modality = c["modality"]

        # Handle unary operators
        if operator in ["always", "eventually"]:

            predicate = c["predicate"]

            parameters = extract_parameters(predicate, bindings)

            clause = Clause(
                operator=operator,
                predicate=predicate,
                weight=weight,
                modality=modality,
                parameters=parameters
            )

        # Handle until operator
        elif operator == "until":

            left = c["left"]
            right = c["right"]

            parameters = {
                "left_params": extract_parameters(left, bindings),
                "right_params": extract_parameters(right, bindings)
            }

            clause = Clause(
                operator=operator,
                predicate=(left, right),
                weight=weight,
                modality=modality,
                parameters=parameters
            )

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        clauses.append(clause)

    return TaskSpec(
        horizon_sec=horizon_sec,
        clauses=clauses
    )


def extract_parameters(predicate_name, bindings):

    params = {}

    for key, value in bindings.items():
        if key.startswith(predicate_name + "."):
            param_name = key.split(".")[1]
            params[param_name] = value

    return params

# ==========================================
# spec/taskspec.py
# ==========================================
# spec/taskspec.py

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Clause:
    operator: str          # "eventually", "always", "until"
    predicate: str         # name of predicate
    weight: float          # logic weight
    modality: str          # "REQUIRE" or "PREFER"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    horizon_sec: float
    clauses: List[Clause]
    auxiliary_weights: Dict[str, float] = field(default_factory=dict)
