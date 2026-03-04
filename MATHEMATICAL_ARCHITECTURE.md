# Complete Mathematical Architecture: Certified Impedance Control for Manipulation

**Target Audience**: Beginner in robotics domain, UG mathematics level  
**Goal**: Understand EVERY equation, EVERY concept, WHAT goes into WHAT, and WHY

---

## TABLE OF CONTENTS

1. **Problem Setup** — What are we trying to do?
2. **Section A: Dynamic Movement System (DMP)** — How robot learns smooth paths
3. **Section B: Obstacle Avoidance Layer (Repulsive Forcing)** — How we steer around obstacles
4. **Section C: Certified Gains (CGMS)** — How we guarantee robot is safe & stable
5. **Section D: Temporal Logic Specification** — How we define what robot MUST do
6. **Section E: Three-Layer Safety Architecture** — How all pieces fit together
7. **Section F: Optimization Pipeline** — How we find the best trajectory

---

# SECTION 0: PROBLEM SETUP

## What is a Robot Manipulation Task?

You have a robot arm holding a coffee mug. The task is:
- **Start position**: Pick up mug from position A
- **Intermediate**: Move around an obstacle (table edge)
- **End position**: Hand the mug to a human at position B
- **Constraint**: Keep the mug upright (don't spill!)
- **Safety**: Don't crash into the obstacle, don't move too fast, keep soft interaction with human

### The Core Challenge

The robot must simultaneously satisfy:
1. **Geometric constraint**: Stay away from obstacles
2. **Kinematic constraint**: Move along a smooth path
3. **Stiffness constraint**: Reduce stiffness (K) when near human, stay stiff elsewhere
4. **Stability constraint**: Never oscillate (K must be positive definite)

Our solution has THREE layers:

| Layer | What | When | Guarantee |
|---|---|---|---|
| **Layer 1** | DMP repulsion (inside ODE) | during path generation | Soft — path naturally bends away |
| **Layer 2** | Soft optimization cost (PIBB) | during learning | Soft — optimizer nudged to avoid |
| **Layer 3** | Hard projector (post-rollout) | after trajectory generated | Hard — unconditional geometry |

---

# SECTION A: DYNAMIC MOVEMENT PRIMITIVES (DMP)

## What is DMP? (Purpose)

A DMP is a **learnable trajectory generator**. It learns to produce smooth, repeatable movements. Think of it as a robot that has learned "how to move smoothly from point A to point B."

### The Physics: Second-Order ODE with Forcing

The fundamental equation is:

$$\tau^2 \ddot{y} + \tau d \dot{y} + k(y - g) = f_{\text{learned}}(x) + f_{\text{repulse}}(\mathbf{y})$$

where:

| Symbol | Meaning | Units | Role |
|---|---|---|---|
| $y$ | Position (what we want) | meters | State variable |
| $\dot{y}$ | Velocity | m/s | First derivative of position |
| $\ddot{y}$ | Acceleration | m/s² | Second derivative (force) |
| $\tau$ | Time scaling factor | seconds | Slows down/speeds up whole trajectory |
| $d$ | Damping coefficient | (dimensionless) | Smoothness — larger = slower decay of oscillations |
| $k$ | Spring constant | N/m | Stiffness of attractor pull toward goal |
| $g$ | Goal position | meters | Where we want to reach |
| $f_{\text{learned}}$ | Learned forcing term | N/m | What the robot learned to do |
| $f_{\text{repulse}}$ | Obstacle repulsion | N/m | Push away from obstacles (NEW) |
| $x$ | Phase variable | 0 to 1 | Time-like variable that goes 0→1 over trajectory |

### Breaking Down the Equation

Rearrange to solve for acceleration:

$$\ddot{y} = \frac{1}{\tau^2} \left[ f_{\text{learned}}(x) + f_{\text{repulse}}(\mathbf{y}) - \tau d \dot{y} - k(y-g) \right]$$

This says: **acceleration** = (forcing term - damping - spring restoring force) / inertia

**Three forces acting on the position:**

1. **Spring force**: $-k(y - g)$
   - Always pulls $y$ toward goal $g$
   - Magnitude: $k \times \text{distance to goal}$
   - Direction: toward goal
   - **Purpose**: Ensures robot doesn't wander, always finishes trajectory

2. **Damping force**: $-\tau d \dot{y}$
   - Opposes motion (friction-like)
   - Magnitude: proportional to velocity
   - **Purpose**: Smooths trajectory, prevents oscillations, speeds up convergence

3. **Learning force**: $+f_{\text{learned}}(x)$
   - This is what the neural network learns
   - Guides the trajectory away from straight line to goal
   - Allows avoiding obstacles, navigating around clutter
   - **Purpose**: Shape the trajectory from generic spring-damper into a learned behavior

---

## How DMP Learns: RBF Neural Networks

The learned forcing term is represented as a **Radial Basis Function (RBF) network**:

$$f_{\text{learned}}(x) = \frac{\sum_{i=1}^{N} w_i \phi_i(x) \cdot x \cdot (g - y_0)}{\sum_{i=1}^{N} \phi_i(x)}$$

where:

| Symbol | Meaning | Purpose |
|---|---|---|
| $w_i$ | Weight of basis function $i$ | **These are what we optimize!** Each weight is a parameter that PIBB learns |
| $\phi_i(x)$ | Gaussian basis function centered at phase $x_i$ | Localized in time — active only during certain time intervals |
| $x$ | Phase variable (0 to 1) | Time in normalized units — lets us scale the whole trajectory |
| $(g - y_0)$ | Distance to goal | Scales learned forcing by how far we need to go |

### Example: 3 Basis Functions

Imagine we discretize the trajectory into 3 regions:
- **Early phase** ($x \in [0, 0.3]$): Start moving
- **Middle phase** ($x \in [0.3, 0.7]$): Avoid obstacle
- **Late phase** ($x \in [0.7, 1.0]$): Finish & settle at goal

Each basis function $\phi_i$ is a Gaussian centered at one of these regions. The network learns:
- $w_1$ = how much to move in early phase
- $w_2$ = how much to deviate during obstacle zone (learned by PIBB to dodge obstacle)
- $w_3$ = how much to correct trajectory at end to reach goal exactly

**Why RBF?** Gaussians are smooth and local — moving one weight doesn't wildly affect the whole trajectory.

---

## Summary of DMP Concept

**Input**: Phase variable $x(t)$ goes 0→1 over time  
**Process**: At each time step, compute:
1. Spring force toward goal
2. Damping to smooth motion
3. Learned force (weighted combination of basis functions)
4. **PLUS** new repulsive force (Layer 1)

**Output**: Smooth trajectory $y(t)$ that goes from start → goal while avoiding obstacles and following learned shape

---

# SECTION B: OBSTACLE AVOIDANCE — REPULSIVE FORCING (LAYER 1)

## The Problem This Solves

Without repulsion, the DMP follows the straight-line spring-damper dynamics to the goal. If the goal is on the other side of an obstacle, the trajectory goes straight through it!

**Solution**: Add an extra force that pushes away from obstacles **during** the trajectory generation.

---

## The Repulsive Force Equation

For each obstacle sphere, compute:

$$f_{\text{repulse}} = \begin{cases}
s \cdot k_{\text{dmp}} \cdot \alpha(d) \cdot \mathbf{n} & \text{if } d < r_{\text{infl}} \\
0 & \text{otherwise}
\end{cases}$$

where:

| Symbol | Meaning | Value in our case |
|---|---|---|
| $s$ | Repulsion strength (tunable) | 0.05 (soft parameter) |
| $k_{\text{dmp}}$ | DMP spring constant | 100 N/m (from $d^2/4$ formula) |
| $d$ | Distance from $\mathbf{y}$ to obstacle center $\mathbf{c}$ | $\|\|\mathbf{y} - \mathbf{c}\|\|$ (meters) |
| $r_{\text{infl}}$ | Influence radius (2× safe radius) | 0.24 meters (obstacle safe = 0.12 m) |
| $r$ | Safe radius of obstacle | 0.12 meters |
| $\alpha(d)$ | Taper function (0 to 1) | See below |
| $\mathbf{n}$ | Outward unit normal | $\frac{\mathbf{y} - \mathbf{c}}{d}$ |

### The Taper Function (Critical!)

$$\alpha(d) = \left( \frac{r_{\text{infl}} - d}{r_{\text{infl}} - r} \right)^3$$

This is a **cubic polynomial** that smoothly goes from:
- $\alpha = 1$ when $d = r$ (at obstacle surface)
- $\alpha = 0$ when $d = r_{\text{infl}}$ (at influence boundary)

**Why cubic and not linear?**
- Cubic gives smooth 2nd derivatives (smooth acceleration)
- Avoids sudden jerks in the motion
- Smoothness = natural, comfortable motion

### Magnitude Interpretation

$$\text{Force magnitude} = s \cdot k_{\text{dmp}} \cdot \alpha = 0.05 \times 100 \times \alpha = 5 \cdot \alpha \text{ (N/m)}$$

At different positions:
- **At surface** ($d = 0.12$ m): $\alpha = 1.0$ → force = 5.0 N/m (strongest)
- **At boundary** ($d = 0.24$ m): $\alpha = 0.0$ → force = 0.0 N/m (zero)
- **Outside boundary**: force = 0 (no repulsion)

**Why scale by $k_{\text{dmp}}$?**
- Spring force has magnitude $k(y - g)$ ≈ 100 × (distance to goal)
- Repulsion should be comparable magnitude to have effect
- If repulsion too weak: ignored by spring attractor
- If repulsion too strong: overshoots and oscillates

---

## Direction: Outward Normal

$$\mathbf{n} = \frac{\mathbf{y} - \mathbf{c}}{\|\mathbf{y} - \mathbf{c}\|}$$

This is the **unit vector pointing away from obstacle center**.

- If robot at $(0.4, 0.3, 0.3)$ (obstacle center) and $\mathbf{y} = (0.5, 0.3, 0.3)$ (1 meter away in +X):
- Then $\mathbf{n} = (1, 0, 0)$ — repulsion pushes in +X (away from obstacle)

---

## How Repulsion Modifies the DMP Equation

Original DMP:
$$\ddot{y} = \frac{1}{\tau^2} \left[ f_{\text{learned}} - \tau d \dot{y} - k(y-g) \right]$$

With repulsion (Layer 1):
$$\ddot{y} = \frac{1}{\tau^2} \left[ f_{\text{learned}} + f_{\text{repulse}}(\mathbf{y}) - \tau d \dot{y} - k(y-g) \right]$$

**What this means physically:**
- When trajectory would go through obstacle, repulsion force appears
- Spring attractor still pulls toward goal
- Damping still smooths motion
- Result: smooth arc around obstacle (not C-turn on surface)

---

## Hard vs Soft Obstacles

**Hard obstacle** (`hard=True`):
- DMP repulsion steers trajectory away (Layer 1)
- After trajectory is generated, hard radial projector clamps any remaining violations (Layer 3)
- **Guarantee**: $\|\mathbf{p}(t) - \mathbf{c}\| \geq r$ for ALL $t$ (unconditional)

**Soft obstacle** (`hard=False`):
- DMP repulsion steers trajectory away (Layer 1)
- Optimizer gets soft cost signal (ObstacleAvoidance weight × penalty)
- **No hard guarantee**: projector is skipped
- Ball may penetrate zone if optimizer finds it worthwhile

**Key insight:** DMP repulsion ALONE (even soft) is effective because it acts DURING trajectory generation when spring-damper dynamics are active.

---

# SECTION C: CERTIFIED GAINS (CGMS) — LAYER 2

## What is Impedance Control?

**Impedance** = how robot responds when a human touches it

Formally: $\mathbf{F}_{\text{command}} = K (\mathbf{y}_{\text{desired}} - \mathbf{y}_{\text{actual}}) + D (\dot{\mathbf{y}}_{\text{desired}} - \dot{\mathbf{y}}_{\text{actual}})$

where:
- **K** = stiffness (position error gain) — stiffer = more resistant to motion
- **D** = damping (velocity error gain) — more damping = smoother deceleration

**Why impedance?** When human gently pushes a soft robot, robot should yield. Impedance control lets us set "softness."

---

## The Challenge: Certified K > 0

Here's the problem: **K must be positive definite always.**

$$K = \begin{pmatrix} K_{xx} & K_{xy} & K_{xz} \\ K_{yx} & K_{yy} & K_{yz} \\ K_{zx} & K_{zy} & K_{zz} \end{pmatrix}$$

If $K$ is NOT positive definite (has a negative eigenvalue), the system is **unstable** — it will oscillate forever or even explode!

**But**: We also want K to be **learnable** — we want an optimizer (PIBB) to adjust K over time to fit different parts of the task.

### The Solution: Cholesky Decomposition

Use the mathematical fact that **any positive definite matrix can be written as:**

$$K = Q^T Q$$

where $Q$ is an **upper triangular matrix** (all entries below diagonal are zero).

**Genius insight:**
- If $Q$ is ANY triangular matrix, then $K = Q^T Q$ is **automatically** positive definite
- We learn $Q$ instead of $K$
- By construction, $K > 0$ always!

---

## The CGMS ODE: Learning Q(t)

We can't just set $K$ constant — it needs to **change over time**. Near the human we want low K (soft). Far away we want high K (stiff).

The differential equation for $Q$ is:

$$\dot{Q} = \alpha Q + 0.5 Q^{-T} B$$

where:

| Symbol | Meaning | Notes |
|---|---|---|
| $\dot{Q}$ | How $Q$ changes over time | We solve this ODE |
| $\alpha$ | Time constant | 0.05 (how fast K adapts) |
| $Q^{-T}$ | Inverse transpose | Inverse of $(Q^T)$ |
| $B$ | Forcing term | Encoded impedance goal |

### What is B?

$$B = -\alpha \dot{D} - SK \cdot SK^T$$

where:
- $\dot{D}$ = time derivative of desired damping
- $SK$ = learned slack matrix (from neural network)

**Interpretation:**
- If we want D to increase, $\dot{D} > 0$, so $B$ pushes Q to increase
- The $SK \cdot SK^T$ term encodes learned slack variables

### The Dynamics of K

Since $K = Q^T Q$, we can compute:

$$\dot{K} = \dot{Q}^T Q + Q^T \dot{Q} = 2 Q^T \dot{Q}$$

(This is the chain rule applied to $K = Q^T Q$)

Substituting the ODE for $\dot{Q}$:

$$\dot{K} = 2 Q^T \left( \alpha Q + 0.5 Q^{-T} B \right)$$

$$= 2 \alpha (Q^T Q) + 2 \times 0.5 (Q^T Q^{-T} B)$$

$$= 2 \alpha K + B$$

**This says**: K changes based on:
1. Term $2\alpha K$: Natural decay/growth proportional to current K
2. Term $B$: Control input from learning

---

## Boundary Condition: Initial K

At $t = 0$, we set:

$$K(0) = K_0 I = 300 \times I$$

where $I$ is the 3×3 identity matrix and $K_0 = 300$ N/m.

**Why identity matrix?** Equal stiffness in all directions (isotropic).

To find initial $Q$, we use Cholesky decomposition:

$$Q(0) = \text{cholesky}(K(0))$$

For identity, this gives $Q(0) = \sqrt{300} \times I$.

---

## Human Proximity Penalty (Implicit in Compiler)

The compiler adds a **soft cost** when the robot is close to the human:

$$\text{Cost}_{\text{K-proximity}} = \text{weight} \times \sum_{\text{near human}} \left( \frac{\max(0, K_{ii} - K_{\text{limit}})}{K_{\text{limit}}} \right)^2$$

**This means:**
- If $K_{ii} \leq K_{\text{limit}} = 100$ N/m near human: no cost
- If $K_{ii} > 100$ N/m near human: cost increases quadratically
- The optimizer (PIBB) learns to keep K low near human automatically

**Why not hard constraint?** Because K is determined by the Cholesky ODE. We can't arbitrarily set it. Instead, we guide PIBB through soft cost.

---

## Summary: CGMS Concept

**Input**: Desired damping trajectory $D(t)$ and learned slack variables $SK(t)$  
**Process**:
1. Compute damping from RBF: $D(t) = \alpha H + SK \cdot SK^T$
2. Build forcing term: $B = -\alpha \dot{D} - SK \cdot SK^T$
3. Solve ODE: $\dot{Q} = \alpha Q + 0.5 Q^{-T} B$
4. Compute K: $K(t) = Q(t)^T Q(t)$ (always positive definite by construction)

**Output**: Certified stiffness $K(t)$ that:
- Is always positive definite (guaranteed!)
- Adapts smoothly over time
- Respects human proximity constraints (through optimizer)

---

# SECTION D: TEMPORAL LOGIC SPECIFICATION

## What is Temporal Logic?

**Temporal logic** lets you write what the robot MUST do using Boolean logic + time constraints.

### Example Statements

1. **"Reach goal during carry phase"**
   - Formally: $\Diamond_{[0,7]} (\|\mathbf{y} - g\| < 0.05)$
   - Meaning: **Eventually** (at some time) in interval [0,7s], be within 5cm of goal
   - Symbol: $\Diamond$ = "eventually"

2. **"Never touch obstacle"**
   - Formally: $\Box (\|\mathbf{y} - c\| \geq 0.12)$
   - Meaning: **Always** (at all times), stay ≥ 12cm from obstacle
   - Symbol: $\Box$ = "always"

3. **"Hold position during pour phase"**
   - Formally: $\Box_{[7,10]} (\|\mathbf{y} - g\| < 0.05 \land \|\dot{\mathbf{y}}\| < 0.05)$
   - Meaning: **Always** during [7,10s], be near goal AND almost stationary

---

## Predicate = Atomic Test

Each statement boils down to a **predicate** — a function that outputs a number:

$$\rho(t) = \text{desired quantity} - \text{constraint}$$

Examples:

| Predicate | Equation | Meaning |
|---|---|---|
| At goal | $\rho = 0.05 - \|\mathbf{y} - g\|$ | $\rho > 0$ if within 5cm of goal |
| Avoid obstacle | $\rho = \|\mathbf{y} - c\| - 0.12$ | $\rho > 0$ if more than 12cm away |
| Speed limit | $\rho = 0.8 - \|\dot{\mathbf{y}}\|$ | $\rho > 0$ if moving slower than 0.8 m/s |

**Interpretation:**
- $\rho(t) > 0$ = constraint satisfied at time $t$
- $\rho(t) < 0$ = constraint violated at time $t$
- $\rho(t) = 0$ = right on the boundary

---

## Temporal Operators

### Eventually: $\Diamond_{[t_1, t_2]} \rho(t)$

"At least once during [t₁, t₂], the predicate is positive"

$$\Diamond_{[t_1, t_2]} \rho := \max_{t \in [t_1, t_2]} \rho(t)$$

**Cost calculation**: If this is a **REQUIRE** clause with weight $w$:

$$\text{Cost} = w \times \max(0, -\max_t \rho(t))^2$$

If maximum $\rho > 0$, cost is 0 (satisfied). If all $\rho < 0$, cost is high (not satisfied).

### Always: $\Box_{[t_1, t_2]} \rho(t)$

"At every time during [t₁, t₂], the predicate is positive"

$$\Box_{[t_1, t_2]} \rho := \min_{t \in [t_1, t_2]} \rho(t)$$

**Cost calculation**: If this is a **REQUIRE** clause with weight $w$:

$$\text{Cost} = w \times \max(0, -\min_t \rho(t))^2$$

If minimum $\rho > 0$ for entire interval, cost is 0 (always satisfied). If any $\rho < 0$, cost is high.

---

## Hard vs Soft: REQUIRE vs PREFER

### REQUIRE Clause (Hard)

$$\text{Cost}_{\text{hard}} = w_{\text{slack}} \times s^2$$

where $s = \max(0, -\rho_{\min})$ is the **slack** (how much constraint is violated)

- Very large weight: $w_{\text{slack}} = 500$ N/m
- Optimizer strongly penalizes violations
- For "always avoid obstacle" — this is appropriate

### PREFER Clause (Soft)

$$\text{Cost}_{\text{soft}} = w \times \max(0, -\rho)$$

where $w$ is modest weight (e.g., 6.0)

- Much smaller weight than REQUIRE
- Optimizer balances this against other costs
- For "try to avoid obstacle" on harmless ball — allows penetration if worthwhile

---

## Full Cost Function

$$J(\text{trajectory}) = \sum_{\text{REQUIRE clauses}} w_s \times s^2 + \sum_{\text{PREFER clauses}} w_p \times j + \text{other costs}$$

The optimizer (PIBB) searches for DMP weights that **minimize** this total cost.

---

# SECTION E: THREE-LAYER SAFETY ARCHITECTURE

## Complete Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    PIBB OPTIMIZER                            │
│        (Searches for best DMP weights θ)                     │
└────────┬────────────────────────────────────────────────────┘
         │ θ = [trajectory RBF weights, damping RBF, stiffness RBF]
         ↓
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 1: DMP GENERATION                     │
│  Spring-damper + learned forcing + REPULSIVE OBSTACLE FORCE  │
│  ODE: τ²ÿ + τdẏ + k(y-g) = f_learned + f_repulse          │
│  Output: y(t), ẏ(t) trajectory, smooth arc around obstacle  │
└────────┬────────────────────────────────────────────────────┘
         │ y(t) [may have points inside obstacle zone]
         ↓
┌─────────────────────────────────────────────────────────────┐
│           LAYER 2: CGMS GAIN SCHEDULING                      │
│  ODE: Q̇ = αQ + 0.5Q⁻ᵀB                                      │
│  K = QᵀQ > 0 by construction (Cholesky property)             │
│  Output: K(t), D(t) stiffness & damping, always stable      │
└────────┬────────────────────────────────────────────────────┘
         │ K(t), D(t) [certified positive definite]
         ↓
┌─────────────────────────────────────────────────────────────┐
│         LAYER 3: HARD RADIAL PROJECTOR                       │
│  ∀i: if ‖p[i] - c‖ < r  then  p[i] ← c + r·n[i]           │
│  Unconditional geometric guarantee                           │
│  Output: p_safe(t), v_safe(t)                                │
└────────┬────────────────────────────────────────────────────┘
         │ p_safe(t), v_safe(t), K(t), D(t) [TRACE]
         ↓
┌─────────────────────────────────────────────────────────────┐
│              COMPILER: COST COMPUTATION                      │
│  Evaluate temporal logic clauses                             │
│  Cost = sum of all constraint violations                     │
└────────┬────────────────────────────────────────────────────┘
         │ Cost = scalar
         ↓
    [Loop back to PIBB]
```

---

## What Each Layer Guarantees

| Layer | What it does | Type of guarantee | Why needed |
|---|---|---|---|
| **1: DMP Repulsion** | Steers trajectory away during generation | Soft (probabilistic) | Smooth path, reduces reliance on projector |
| **2: CGMS Gains** | Computes K(t) = Q^T Q | Hard (by construction) | K always positive definite (stability) |
| **3: Projector** | Hard-clamps positions outside sphere | Hard (geometric) | Unconditional safety: no point can be inside |

---

## Data Flow Through Layers

```
START: Optimizer chooses RBF weights θ
  ↓
LAYER 1 (DMP):
  Input: θ_traj (trajectory weights), phase variable x(t)
  Compute: spring, damping, learned forcing, repulsive forcing
  Output: y_raw(t), ẏ(t) [may violate obstacle constraint]
  
LAYER 2 (CGMS):
  Input: θ_SK, θ_SD (stiffness/damping RBF weights), y_raw(t)
  Compute: Q(t) via Cholesky ODE, K(t) = Q^T Q
  Output: K(t) > 0, D(t) [guaranteed certified]
  
LAYER 3 (Projector):
  Input: y_raw(t), v_raw(t), obstacle sphere (hard=True only)
  Compute: For each point, if inside obstacle, push to surface
  Output: y_safe(t), v_safe(t) [guaranteed outside obstacle]
  
COMPILER:
  Input: y_safe(t), v_safe(t), K(t), D(t), temporal logic clauses
  Compute: Cost = sum of all violations
  Output: scalar cost value
  
PIBB:
  Input: Cost values from many trajectory samples
  Update: Mean and covariance of θ distribution
  Output: New θ sample for next iteration
```

---

# SECTION F: OPTIMIZATION PIPELINE (PIBB)

## What is PIBB?

**PIBB** = "Policy Improvement with Path Integrals and Probabilistic Inference"

It's a **black-box optimizer** that doesn't need gradients. You just give it:
- Cost function (what we measure)
- Current distribution of θ

And it returns:
- Improved distribution (shifted toward lower-cost θ)

---

## Core Idea: Sampling & Weighting

### Step 1: Sample θ from Distribution

$$\theta_i \sim \mathcal{N}(\mu, \Sigma) \quad \text{for } i = 1, \ldots, N_{\text{samples}}$$

Sample $N = 30$ different parameter vectors from a Gaussian distribution.

### Step 2: Evaluate Each Sample

For each sample $\theta_i$:
1. Run all three layers (DMP → CGMS → Projector)
2. Compute compiler cost: $J_i = J(\theta_i)$

Now we have pairs: $(\theta_1, J_1), (\theta_2, J_2), \ldots, (\theta_{30}, J_{30})$

### Step 3: Reweight Based on Cost

Compute **importance weights** using exponential family:

$$w_i = \frac{\exp(-\beta J_i)}{\sum_{j=1}^{N} \exp(-\beta J_j)}$$

where $\beta = 8.0$ (inverse temperature).

**Interpretation:**
- Low-cost samples get **high** weight ($w_i$ close to 1)
- High-cost samples get **low** weight ($w_i$ close to 0)
- Sum of all weights = 1 (they're probabilities)

**Why exponential?** It's the form that maximizes entropy while respecting cost differences. (Information theory result.)

### Step 4: Update Distribution

$$\mu_{\text{new}} = \sum_{i=1}^{N} w_i \theta_i$$

$$\Sigma_{\text{new}} = \sum_{i=1}^{N} w_i (\theta_i - \mu_{\text{new}}) (\theta_i - \mu_{\text{new}})^T$$

**Meaning:**
- New mean is weighted average of all samples (weighted toward low-cost ones)
- New covariance is weighted spread around new mean
- Good samples pull the mean in their direction

### Step 5: Regularization (Decay)

$$\Sigma_{\text{new}} \leftarrow \text{decay} \times \Sigma_{\text{new}}$$

where decay = 0.99.

**Purpose:** Gradually narrow the distribution as we get closer to optimal θ. This prevents infinite exploration.

---

## Full Loop

```
Initialize: μ = 0, Σ = identity × σ_init

FOR each UPDATE (1 to 70):
  
  FOR each SAMPLE (1 to 30):
    θ_i ~ N(μ, Σ)
    trace_i ← rollout(θ_i)  [Layers 1, 2, 3]
    cost_i ← objective(trace_i)  [Compiler]
  
  w_i ← exp(-β cost_i) / Z  [Reweight]
  
  μ ← Σ w_i θ_i  [Update mean]
  Σ ← Σ w_i (θ_i - μ)ᵀ(θ_i - μ)  [Update covariance]
  Σ ← 0.99 × Σ  [Decay variance]
  
  Print: min(cost), mean(cost), best(cost) so far

RETURN: θ_best (the θ that gave lowest cost ever seen)
```

---

## Why PIBB Works

1. **No gradients needed**: PIBB only needs to evaluate cost at sample points
2. **Handles discontinuities**: Unlike gradient descent, can jump over discontinuities
3. **Explores naturally**: High variance at start (explore), low variance at end (exploit)
4. **Robust**: Doesn't get stuck in shallow local minima as easily

---

## Convergence Example: Scene 3

```
Update  Best Cost  What happened
──────  ──────────  ─────────────────────────────────
  1      11812.88   Random initial samples, terrible
  10      2500.00   Settling down, finding reasonable regions
  30       150.00   Good trajectory shape found
  50        10.00   Fine-tuning positions
  70         2.60   Converged (0 pinned points, 7.8 cm clearance)
```

**Key insight:** Cost drops smoothly because PIBB is climbing a smooth "cost landscape" toward the global best.

---

# SECTION G: COMPLETE EXAMPLE — SCENE 3B (BALL DELIVERY)

## Task Definition

**Temporal Logic Specification** (JSON):

```
REQUIRE: Eventually reach goal during [0, 7s]  (carry phase)
REQUIRE: Always hold position during [7, 10s]  (hold phase)
PREFER: Obstacle avoidance (soft — ball can penetrate)
REQUIRE: Keep speed < 0.8 m/s always
PREFER: Keep upright (soft — ball can tilt)
```

---

## Mathematical Flow

### Step 1: Initialize

- $\mu = 0$ (mean of θ distribution)
- $\Sigma = \text{diag}(3.0, 3.0, 0.5, 2.0, 2.0, 1.5)$ (exploration noise)
  - Larger noise in xy trajectory (allow big changes)
  - Smaller noise in z (keep height constant)
  - Moderate noise in learning weights

### Step 2: Sample & Rollout (Update 1, Sample 1)

Sample $\theta_1 \sim N(\mu, \Sigma)$

**Layer 1 — DMP**: For each time $t$:

$$\ddot{y} = \frac{1}{\tau^2} \left[ f_{\text{learned}}(\theta_1, x) + f_{\text{repulse}}(\mathbf{y}) - \tau d \dot{y} - k(y - g) \right]$$

Integrate over 7 seconds → get $y_1(t)$, $\dot{y}_1(t)$

**Layer 2 — CGMS**: Solve Cholesky ODE with $\theta_{SK}$, $\theta_{SD}$ from $\theta_1$:

$$\dot{Q} = \alpha Q + 0.5 Q^{-T} B$$

Integrate → get $K_1(t)$, $D_1(t)$ (guaranteed positive definite)

**Layer 3 — Projector**: For hard=False, skip projection. (No hard guarantee.)

### Step 3: Compute Cost

Evaluate 6 temporal logic clauses:

| Clause | Predicate | Evaluation |
|---|---|---|
| Reach goal | $\rho_1(t) = 0.05 - \|\mathbf{y}_1 - \mathbf{g}\|$ | $\text{max}_t \rho_1 = -0.03$ (not reached) |
| Hold position | $\rho_2(t) = 0.05 - \|\mathbf{y}_1 - \mathbf{g}\|$ at $t \in [7,10]$ | $\text{min}_t \rho_2 = 0.02$ (satisfied) |
| Obstacle (soft) | $\rho_3(t) = \|\mathbf{y}_1 - \mathbf{c}\| - 0.12$ | $\text{min}_t \rho_3 = -0.02$ (penetrates 2cm) |
| Speed limit | $\rho_4(t) = 0.8 - \|\dot{\mathbf{y}}_1\|$ | $\text{min}_t \rho_4 = 0.5$ (satisfied) |
| Upright | $\rho_5(t) = 0.3 - \|\text{tilt}\|$ | $\text{min}_t \rho_5 = 0.25$ (satisfied) |
| Angular speed | $\rho_6(t) = 1.5 - \|\boldsymbol{\omega}\|$ | $\text{min}_t \rho_6 = 1.4$ (satisfied) |

**Cost aggregation:**

$$J_1 = \underbrace{500 \times 0.03^2}_{reach\ goal\ hard} + \underbrace{0}_{hold\ satisfied} + \underbrace{6.0 \times 0.02}_{obstacle\ soft} + 0 + 6.0 \times 0.02 + 0 = 0.45 + 0.12 + 0.12 = 0.69$$

This is the cost for sample $\theta_1$.

### Step 4: Repeat for All Samples

Repeat steps 2-3 for samples $\theta_2, \ldots, \theta_{30}$ → get costs $J_1, \ldots, J_{30}$

Suppose: min cost = 0.45 (good sample), max cost = 15.0 (bad sample)

### Step 5: Reweight & Update

$$w_i = \frac{\exp(-8.0 \times J_i)}{\sum_{j=1}^{30} \exp(-8.0 \times J_j)}$$

Low-cost sample ($J = 0.45$): $w \propto \exp(-3.6) \approx 0.027$ (high)  
High-cost sample ($J = 15.0$): $w \propto \exp(-120) \approx 0$ (near zero)

**Update mean:**
$$\mu_{\text{new}} = \sum_{i=1}^{30} w_i \theta_i$$

Mean shifts toward $\theta_1$ (good sample).

**Update variance:**
$$\Sigma_{\text{new}} = 0.99 \times (\text{weighted spread})$$

Covariance shrinks slightly — next samples explore less, focus on promising region.

### Step 6: Next Update

Use new $\mu, \Sigma$ to sample 30 more parameter vectors. Repeat process. Over 70 updates, cost steadily decreases.

---

## Final Result (Update 70)

Best trajectory found:
- **Cost**: 6.0869
- **Goal reached**: YES (t = 7.01 s)
- **Obstacle clearance**: 7.1 cm (soft — acceptable)
- **Points inside obstacle**: 0 (DMP repulsion was effective!)
- **Stiffness**: K reduced in human zone, high elsewhere
- **Stability**: K eigenvalue min = 6.2 > 0 ✓ (certified)

**Why did DMP repulsion alone work?** The obstacle center and goal are positioned such that the natural spring-damper path dodges the obstacle if it gets nudged early. PIBB learned RBF weights that, combined with repulsion, produced a 7.1 cm arc.

---

# SECTION H: KEY INSIGHTS & TAKEAWAYS

## Conceptual Hierarchy

```
High-level task (temporal logic clauses)
    ↓
Cost function (penalties for violations)
    ↓
Optimizer (PIBB searches for best parameters)
    ↓
Parameter vector θ (DMP RBF weights, CGMS RBF weights)
    ↓
Three-layer execution:
    Layer 1: DMP (spring-damper + learning + repulsion)
    Layer 2: CGMS (Cholesky ODE for stable gains)
    Layer 3: Hard projector (geometric safety)
    ↓
Certified trajectory (safe, stable, satisfies task)
```

---

## Why This Architecture is Robust

1. **DMP Repulsion (Layer 1)**: Acts DURING trajectory generation when spring-damper is active. Elegant because it uses existing dynamics.

2. **CGMS (Layer 2)**: Gains are certified positive definite by mathematical construction (Cholesky). No "hoping" for stability — it's guaranteed.

3. **Hard Projector (Layer 3)**: Unconditional geometric guarantee. Even if Layers 1-2 fail, this catches violations. Safety net.

4. **Three-level cost (Compiler)**: Hard penalties (REQUIRE) for absolute constraints, soft penalties (PREFER) for preferences. Optimizer balances them.

5. **PIBB Optimization**: Black-box, no-gradient approach means we can handle constraints that aren't differentiable (like hard projector).

---

## Hard vs Soft Constraints — When to Use Each

| Constraint | Type | Reason | Example |
|---|---|---|---|
| Never touch human body | REQUIRE + Hard projector | Life/safety critical | Obstacle avoidance (mug) |
| Prefer smooth motion | PREFER | Nice to have, can trade off | Comfort goal |
| Stay within workspace | REQUIRE | Physics limits | Joint limits |
| Reduce stiffness near human | PREFER (through cost) | Safety, but K determined by ODE | Scene 3 impedance control |

**Rule of thumb:** If violating the constraint causes **harm**, make it REQUIRE + hard. If violating it is just **suboptimal**, make it PREFER.

---

## Mathematical Elegance of Cholesky

$$K = Q^T Q \quad \Rightarrow \quad K \text{ is always positive definite}$$

This is pure mathematics — no physics assumption. It works because:

$$\det(K) = \det(Q^T Q) = \det(Q^T) \det(Q) = [\det(Q)]^2 \geq 0$$

And all eigenvalues of $K$ are non-negative by construction (they're $\lambda_i = \sigma_i^2$ where $\sigma_i$ are singular values of Q).

**Implication**: We can optimize over Q freely (it's unconstrained), and K will always be safe. This is much easier than constraining K directly!

---

## The Role of DMP Repulsion

Many roboticists ask: "Why DMP repulsion if we have the hard projector?"

**Answers:**

1. **Fewer projected points**: Repulsion steers away early, reducing reliance on projector. Fewer = cleaner trajectory.

2. **Smooth acceleration**: Repulsion is continuous → smooth motion. Hard projection can cause discontinuities.

3. **Natural dynamics**: Repulsion uses the spring-damper system already present. It's elegant.

4. **Soft obstacles**: For objects that can penetrate (ball), only repulsion acts. Projector can be disabled.

---

## Why Temporal Logic?

Instead of encoding "avoid this region" as code, we write it as a mathematical formula. This enables:

1. **Clarity**: Natural language "eventually reach goal" → formal $\Diamond_{[0,7]} \rho$
2. **Composability**: Can combine clauses with AND ($\land$), OR ($\lor$), NOT ($\neg$)
3. **Verification**: Can prove mathematically that trajectory satisfies specification
4. **Automation**: Can synthesize trajectories from specifications

---

# SUMMARY TABLE: All Variables

| Variable | Dimension | Meaning | Computed How |
|---|---|---|---|
| $\mathbf{y}(t)$ | 3 | Position (what we generate) | ODE integration in Layer 1 |
| $\dot{\mathbf{y}}(t)$ | 3 | Velocity | Derivative of $\mathbf{y}$ |
| $\ddot{\mathbf{y}}(t)$ | 3 | Acceleration | From DMP equation |
| $x(t)$ | 1 | Phase (0 to 1) | $\dot{x} = -1/\tau$ |
| $f_{\text{learn}}(\mathbf{y}, x)$ | 3 | Learned forcing | RBF network: $\sum w_i \phi_i(x) \times (\mathbf{g} - \mathbf{y}_0)$ |
| $f_{\text{repulse}}(\mathbf{y})$ | 3 | Repulsive obstacle force | $s \cdot k_{\text{dmp}} \cdot \alpha(d) \cdot \mathbf{n}$ |
| $K(t)$ | $3 \times 3$ | Stiffness matrix | $Q(t)^T Q(t)$ |
| $D(t)$ | $3 \times 3$ | Damping matrix | $\alpha H + SK \cdot SK^T$ |
| $Q(t)$ | $3 \times 3$ upper tri | Cholesky factor | ODE integration in Layer 2 |
| $\theta$ | 396 | Parameter vector | Optimized by PIBB |
| $\rho(t)$ | 1 | Predicate (constraint test) | Problem-specific |
| $J(\theta)$ | 1 | Total cost | Sum of all constraint violations |

---

# FINAL CONCEPTUAL MAP

```
                    ┌──────────────────────────────┐
                    │  TASK SPECIFICATION (JSON)   │
                    │ (7 temporal logic clauses)   │
                    └──────────┬───────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │   COMPILER (spec/compiler.py)   │
                │ Evaluates clauses → Cost        │
                └──────────────┬───────────────────┘
                               │
            ┌──────────────────▼──────────────────────┐
            │         PIBB OPTIMIZER                  │
            │ (Searches for best parameter vector θ) │
            └──────────────────┬──────────────────────┘
                               │ θ = [traj weights, gain weights]
                ┌──────────────▼──────────────────┐
                │    LAYER 1: DMP GENERATION      │
                │ Spring-damper + learning +      │
                │ repulsive obstacle forcing      │
                └──────────────┬───────────────────┘
                               │ y(t), ẏ(t)
                ┌──────────────▼──────────────────┐
                │   LAYER 2: CGMS GAINS           │
                │ Q̇ = αQ + 0.5Q⁻ᵀB               │
                │ K = QᵀQ (certified K > 0)       │
                └──────────────┬───────────────────┘
                               │ K(t), D(t)
                ┌──────────────▼──────────────────┐
                │  LAYER 3: HARD PROJECTOR        │
                │ Clamp positions inside obstacle │
                │ (only for hard=True)            │
                └──────────────┬───────────────────┘
                               │ y_safe(t), v_safe(t)
                ┌──────────────▼──────────────────┐
                │        FINAL TRACE              │
                │ Positions, velocities, gains    │
                │ + temporal logic evaluations    │
                └──────────────┬───────────────────┘
                               │
                    [Back to PIBB iteration]
```

---

# RECOMMENDED READING ORDER

1. **Section 0**: Problem Setup (understand the challenge)
2. **Section A**: DMP (core trajectory generator)
3. **Section B**: Repulsive Forcing (intuitive addition to DMP)
4. **Section C**: CGMS (understanding certified gains)
5. **Section D**: Temporal Logic (how to specify tasks formally)
6. **Section E**: Three-Layer Architecture (how pieces fit)
7. **Section F**: PIBB Optimization (how we search for solutions)
8. **Section G**: Complete Example (see it all together)
9. **Section H**: Key Insights (reflect on design choices)

---

# GLOSSARY OF SYMBOLS

| Symbol | Definition | Units |
|---|---|---|
| $y, \mathbf{y}$ | Position | meters |
| $\dot{y}, \dot{\mathbf{y}}$ | Velocity | m/s |
| $\ddot{y}, \ddot{\mathbf{y}}$ | Acceleration | m/s² |
| $\tau$ | Time constant (DMP duration) | seconds |
| $d$ | Damping coefficient | dimensionless |
| $k$ | Spring constant | N/m |
| $g, \mathbf{g}$ | Goal position | meters |
| $K$ | Stiffness matrix | N/m |
| $D$ | Damping matrix | Ns/m |
| $x$ | Phase variable (0 to 1) | dimensionless |
| $\phi_i$ | Radial basis function $i$ | dimensionless |
| $w_i$ | Weight of basis function $i$ | varies |
| $f$ | Forcing term (acceleration-like) | N/m |
| $\alpha$ | Time constant (CGMS decay) | 1/seconds |
| $Q$ | Cholesky factor of K | varies |
| $\theta$ | Parameter vector (what we optimize) | varies |
| $\rho$ | Predicate value (constraint test) | varies |
| $J$ | Cost (objective function) | varies |
| $\beta$ | Inverse temperature (PIBB) | 1/cost units |
| $\mu$ | Mean of parameter distribution | varies |
| $\Sigma$ | Covariance of parameter distribution | varies |

---

This document provides the **complete mathematical and conceptual foundation** for the system. Each concept builds on previous ones, from basic spring-damper mechanics up to optimization of complex multi-objective tasks.

