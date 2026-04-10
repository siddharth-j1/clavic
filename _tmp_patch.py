#!/usr/bin/env python3
"""Replace \section{EXPERIMENTS} body in paper.txt with new content."""

NEW_EXPERIMENTS = r"""
\begin{figure*}
    \centering
    \includegraphics[width=\linewidth]{Shreyas/images/real/1a.png}
    \caption{Hardware deployment on the Franka Emika Panda arm for Experiment~1a ($T=10$\,s). The robot routes around the human's comfort zone, exhibiting compliant impedance behaviour as it approaches the operator.}
    \label{fig:real1a}
\end{figure*}

\begin{figure*}
    \centering
    \includegraphics[width=\linewidth]{Shreyas/images/real/1b.png}
    \caption{Hardware deployment for Experiment~1b ($T=3$\,s). Under temporal pressure the robot traverses the comfort zone to meet the deadline while the hard body exclusion guarantee ($r=0.08$\,m) is preserved throughout.}
    \label{fig:real1b}
\end{figure*}

We evaluate the proposed framework across four manipulation scenarios designed to probe distinct capabilities: (i)~separation of hard geometric safety from soft semantic preferences, (ii)~temporal adaptation under time pressure, (iii)~multi-phase sequencing with hold and obstacle avoidance, and (iv)~semantic reasoning about object affordances that changes constraint enforcement modality. In all experiments the robot end-effector starts at $[0.55,\,0,\,0.30]$\,m. Optimization used PIBB with $M_{\mathrm{roll}}=30$ rollouts per iteration and 70 update steps. All specifications were compiled from the predicate catalogue described in Appendix~\ref{appendix:llm}.

% ------------------------------------------------------------------ %
\subsection{Shared Experimental Setup}

Table~\ref{tab:exp_overview} summarises the four experimental scenarios. Each scenario was compiled from a natural language instruction into a \texttt{TaskSpec}, validated by the two-pass firewall, and executed through the certified PIBB optimizer. Hardware results are shown in Figs.~\ref{fig:real1a}--\ref{fig:real1b}.

\begin{table}[h]
\caption{Summary of experimental scenarios.}
\label{tab:exp_overview}
\centering
\small
\begin{tabular}{lllll}
\toprule
Exp & Task & $T$ (s) & Key constraint & Modality \\
\midrule
1a & Carry to goal, avoid human & 10 & Body exclusion ($r=0.08$\,m) & \texttt{HARD} \\
1b & Same, tight deadline       &  3 & Body exclusion ($r=0.08$\,m) & \texttt{HARD} \\
2  & Carry $\to$ hold $\to$ continue & 11 & Obstacle ($r=0.12$\,m) & \texttt{HARD} \\
3a & Carry mug $\to$ pour at human & 10 & Laptop ($r=0.12$\,m) & \texttt{HARD} \\
3b & Carry ball $\to$ deliver to human & 10 & Laptop ($r=0.12$\,m) & \texttt{PREFER} \\
\bottomrule
\end{tabular}
\end{table}

% ------------------------------------------------------------------ %
\subsection{Experiment 1: Human-Aware Manipulation Under Temporal Trade-off}

\subsubsection{Setup}
A human operator is situated at $p_h=[0.30,\,0.30,\,0.30]$\,m, directly between the start and goal positions ($[0.55,\,0,\,0.30]$\,m and $[0.30,\,0.55,\,0.30]$\,m respectively). Two concentric constraints model the human:
\begin{itemize}
  \item \textbf{Body exclusion} ($r=0.08$\,m, \texttt{HARD}): enforced by DMP repulsive forcing and radial projection; the end-effector is geometrically guaranteed never to penetrate this radius under any rollout.
  \item \textbf{Comfort zone} ($r_c=0.19$\,m, \texttt{PREFER}, $w=15$): a soft preference for remaining outside the human's personal space.
\end{itemize}
Additionally, the compiler's human-proximity stiffness cost $J_{K\text{-human}}$ smoothly reduces per-axis stiffness toward $K^h=100$\,N/m as the end-effector approaches the human, encoding compliant interaction behaviour.

\subsubsection{Experiment 1a --- Relaxed Time Budget ($T=10$\,s)}

With a 10\,s horizon the optimizer has sufficient freedom to route the end-effector around the comfort zone while reaching the goal. As shown in Fig.~\ref{fig:compare_workspace}, the trajectory arcs away from the human, respecting both the hard body exclusion and the soft comfort preference. All three per-axis stiffness components drop smoothly toward the $K^h=100$\,N/m compliance target as the robot enters the ramp zone $[r_c,\,3r_c]$ (Fig.~\ref{fig:compare_stiffness}, solid lines). This emergent compliance arises entirely from $J_{K\text{-human}}$ without any hand-coded impedance schedule.

\subsubsection{Experiment 1b --- Tight Time Budget ($T=3$\,s)}

The same geometric scene is re-run with a 3\,s horizon. The tighter deadline dramatically increases the cost of detouring around the comfort zone, causing the optimizer to trade the soft comfort preference in favour of task completion. The resulting trajectory passes through the comfort zone (Fig.~\ref{fig:compare_workspace}, dashed) while the hard body guarantee is preserved at every timestep.

The stiffness profiles (Fig.~\ref{fig:compare_stiffness}, dashed) confirm the behavioural consequence: because the robot does not dwell in the comfort zone, the proximity cost never fully activates and stiffness remains at task-level values throughout. This experiment demonstrates \emph{principled, semantically-aware trade-off}: the language interface encodes which constraints are inviolable (\texttt{HARD}) and which are negotiable (\texttt{PREFER}), allowing the compiler to make physically consistent decisions under competing objectives.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 1/compare_topdown.png}
    \caption{Top-down comparison of optimised trajectories for Experiments~1a ($T=10$\,s, \textit{solid}) and 1b ($T=3$\,s, \textit{dashed}). The hard body exclusion zone ($r=0.08$\,m, red) is never violated by either trajectory. Under a relaxed deadline the robot routes around the soft comfort zone ($r_c=0.19$\,m); under time pressure it trades the comfort preference to meet the deadline while preserving the hard guarantee.}
    \label{fig:compare_workspace}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 1/compare_stiffness.png}
    \caption{Per-axis Cartesian stiffness ($K_{xx}$, $K_{yy}$, $K_{zz}$) for Experiments~1a (\textit{solid}) and 1b (\textit{dashed}). At $T=10$\,s all three axes drop toward the compliance target ($K^h=100$\,N/m, dotted) as the end-effector traverses the comfort zone. At $T=3$\,s the robot bypasses the comfort zone and stiffness remains high throughout, illustrating how task timing propagates to physical interaction behaviour. Lyapunov stability (Kronander--Billard conditions) is guaranteed by construction in both cases.}
    \label{fig:compare_stiffness}
\end{figure}

% ------------------------------------------------------------------ %
\subsection{Experiment 2: Multi-Phase Sequencing with Hard Obstacle Avoidance}

\subsubsection{Setup}
This experiment evaluates multi-phase temporal sequencing combined with certified obstacle avoidance. The task comprises three sequential phases:
\begin{enumerate}
  \item \textbf{Carry} (0--5\,s): move from $[0.55,\,0,\,0.30]$\,m toward the waypoint $[0.20,\,0.35,\,0.30]$\,m while avoiding a rigid obstacle at $[0.40,\,0.30,\,0.30]$\,m ($r=0.12$\,m, \texttt{HARD}).
  \item \textbf{Hold} (5--7\,s): remain stationary at the waypoint (\texttt{ZeroVelocity}, \texttt{REQUIRE}).
  \item \textbf{Continue} (7--11\,s): proceed to the final goal $[0.30,\,0.55,\,0.30]$\,m.
\end{enumerate}
No human operator is present; Cartesian orientation is held constant at $q=[1,0,0,0]$ throughout.

\subsubsection{Results}

The workspace trajectory (Fig.~\ref{fig:exp2_workspace}) confirms that the end-effector navigates around the obstacle during both the carry and continue phases. The \texttt{HARD} guarantee ($\|p(t)-c\|\geq r$, $\forall t$) is preserved across all 70 optimizer iterations through the DMP repulsive forcing and radial projection backstop.

The velocity profiles (Fig.~\ref{fig:exp2_velocity}) show smooth DMP dynamics during carry and continue, with all three components converging to near-zero during the hold phase (shaded), consistent with the \texttt{ZeroVelocity} (\texttt{REQUIRE}) clause. The $x$-axis velocity reverses sign at the start of the continue phase as the robot steers away from the obstacle toward the final goal. Since no human is present, the proximity cost $J_{K\text{-human}}$ is inactive and stiffness is maintained at task-appropriate levels throughout (Fig.~\ref{fig:exp2_stiffness}).

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 2/scene4_workspace.png}
    \caption{End-effector trajectory for Experiment~2 (carry--hold--continue). \textit{Phase~1} (0--5\,s, blue): carry toward the waypoint while routing around the hard obstacle (grey sphere, $r=0.12$\,m). \textit{Phase~2} (5--7\,s, orange): stationary hold at the waypoint. \textit{Phase~3} (7--11\,s, green): continue to the final goal. The \texttt{HARD} obstacle constraint $\|p(t)-c\|\geq r$ is satisfied at all timesteps.}
    \label{fig:exp2_workspace}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 2/exp2_velocity.png}
    \caption{Cartesian end-effector velocities for Experiment~2. All three components converge to near-zero during the hold phase (shaded, 5--7\,s), enforcing the \texttt{ZeroVelocity} (\texttt{REQUIRE}) clause. The $x$-velocity reverses in the continue phase as the robot steers around the obstacle to reach the final goal. Phase boundaries are cosine-tapered to eliminate kinematic discontinuities between independent DMP segments.}
    \label{fig:exp2_velocity}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 2/scene4_stiffness.png}
    \caption{Per-axis stiffness for Experiment~2. In the absence of a human operator the proximity cost $J_{K\text{-human}}$ is inactive, and all three stiffness axes remain at task-level values throughout. Lyapunov stability is guaranteed by the certified Cholesky parameterisation.}
    \label{fig:exp2_stiffness}
\end{figure}

% ------------------------------------------------------------------ %
\subsection{Experiment 3: Language-Driven Semantic Modality Selection}

Experiments~3a and~3b share the same spatial configuration but differ in the \emph{semantic description} of the object being carried provided to the language interface. This pair probes whether the LLM correctly infers the appropriate constraint modality from object semantics, and whether that modality difference produces the expected behavioural divergence.

\subsubsection{Shared Setup}
Both experiments use a two-phase structure: carry from $[0.55,\,0,\,0.30]$\,m to a human-positioned goal $[0.30,\,0.55,\,0.30]$\,m (0--7\,s), followed by a second phase at the goal (7--10\,s). A laptop obstacle is placed at $[0.40,\,0.30,\,0.30]$\,m ($r=0.12$\,m). Since the goal is co-located with the human, the proximity cost $J_{K\text{-human}}$ activates in both experiments as the robot approaches the goal.

\subsubsection{Experiment 3a --- Coffee Mug: Carry and Pour}

The language instruction specifies carrying a \emph{coffee mug} to a human colleague and pouring its contents at the goal. The LLM identifies the laptop as a hard obstacle (spill risk) and assigns \texttt{HARD} modality to \texttt{ObstacleAvoidance}. It also generates a two-phase orientation specification: the mug must remain upright ($q=[1,0,0,0]$) during carry, then tilt to the pour pose ($q=[0.7071,0,0.7071,0]$, $90^\circ$ about the $y$-axis) at the goal.

The workspace trajectory (Fig.~\ref{fig:exp3a_workspace}) confirms avoidance of the laptop throughout both phases. The orientation profile (Fig.~\ref{fig:exp3a_orientation}) shows the end-effector maintaining a level pose during carry, then smoothly rotating to the pour orientation via the $SO(3)$ log-map DMP, reaching near-zero geodesic error $\|e_R\|\to 0$ by the end of Phase~2. The stiffness schedule (Fig.~\ref{fig:exp3a_stiffness}) shows high values during carry for precise positioning, followed by a smooth reduction toward $K^h=100$\,N/m as the robot enters the human's proximity zone, encoding safe physical handover behaviour.

\subsubsection{Experiment 3b --- Rubber Ball: Semantic Softening of the Obstacle Constraint}

The same scene is re-run, but the language instruction now specifies a \emph{rubber ball}. Since a ball cannot spill or damage a laptop on contact, the LLM infers that obstacle avoidance should be treated as a soft preference (\texttt{PREFER}, $w=6$) rather than a hard requirement. There is no pouring action; the task is simply to deliver the ball to the human.

Without the hard obstacle guarantee, the optimizer is free to route the trajectory through the laptop's exclusion zone when this reduces overall cost. Fig.~\ref{fig:exp3b_workspace} shows this behavioural divergence: the ball delivery path penetrates the obstacle zone, whereas the mug trajectory in Experiment~3a routes cleanly around it. Despite this geometric difference, stiffness reduction near the human is preserved in both experiments (Fig.~\ref{fig:exp3b_stiffness_ori}), confirming that $J_{K\text{-human}}$ compliance behaviour is independent of the obstacle modality.

This experiment pair demonstrates a core capability of the proposed framework: \emph{the same physical scene, described with different object semantics, produces different constraint enforcement strategies through language-level reasoning}, without any manual re-specification of safety rules.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 3/scene3_workspace.png}
    \caption{Experiment~3a (coffee mug carry--pour). The end-effector routes around the laptop obstacle (\texttt{HARD}, $r=0.12$\,m) during Phase~1 (0--7\,s) and tilts to the pour pose at the human-positioned goal in Phase~2 (7--10\,s). The three-layer safety architecture guarantees $\|p(t)-c\|\geq r$ at all timesteps.}
    \label{fig:exp3a_workspace}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 3/scene3_stiffness.png}
    \caption{Per-axis stiffness for Experiment~3a. High stiffness during carry ensures precise obstacle-aware positioning; stiffness reduces smoothly toward $K^h=100$\,N/m as the end-effector approaches the human goal, encoding compliant handover behaviour through the $J_{K\text{-human}}$ proximity cost.}
    \label{fig:exp3a_stiffness}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 3/scene3_orientation.png}
    \caption{Orientation trajectory for Experiment~3a. The end-effector maintains an upright pose ($q=[1,0,0,0]$) throughout Phase~1 (\texttt{OrientationLimit}, \texttt{REQUIRE}). At the phase boundary the orientation DMP interpolates smoothly in $SO(3)$ log space to the pour orientation ($90^\circ$ $y$-axis tilt), reaching near-zero geodesic error $\|e_R\|\to 0$ by the end of Phase~2.}
    \label{fig:exp3a_orientation}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 3/scene3b_workspace.png}
    \caption{Experiment~3b (rubber ball delivery). Identical scene to Experiment~3a, but the LLM assigns \texttt{PREFER} modality to the laptop obstacle based on object semantics (ball is harmless on contact). The optimizer routes through the exclusion zone when this reduces cost, demonstrating the behavioural consequence of soft vs.\ hard constraint enforcement.}
    \label{fig:exp3b_workspace}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Shreyas/images/exp 3/scene3b_stiffness.png}
    \caption{Per-axis stiffness for Experiment~3b. Despite the change in obstacle modality, stiffness reduces near the human goal in the same manner as Experiment~3a, confirming that compliance behaviour is governed by $J_{K\text{-human}}$ independently of the obstacle constraint. The flat orientation profile (inset) confirms no pouring action was generated for the ball delivery task.}
    \label{fig:exp3b_stiffness_ori}
\end{figure}

"""

with open("paper.txt", "r") as f:
    lines = f.readlines()

# Find line indices (0-based)
exp_start = None
conclusion_start = None
for i, l in enumerate(lines):
    if r"\section{EXPERIMENTS}" in l and exp_start is None:
        exp_start = i
    if r"\section{CONCLUSION}" in l and conclusion_start is None:
        conclusion_start = i

print(f"exp_start={exp_start}, conclusion_start={conclusion_start}")

if exp_start is None or conclusion_start is None:
    print("ERROR: Could not find markers")
    exit(1)

# Build new content:
# lines before experiments section (including \clearpage lines)
# new experiments section header + new body
# then from \section{CONCLUSION} onward

# Find the \clearpage before \section{EXPERIMENTS}
clearpage_start = exp_start
for i in range(exp_start, max(exp_start-5, 0), -1):
    if r"\clearpage" in lines[i]:
        clearpage_start = i
        break

print(f"clearpage_start={clearpage_start}")

before = lines[:clearpage_start]
after  = lines[conclusion_start:]

new_content = (
    "".join(before)
    + "\\clearpage\n\n\n\\section{EXPERIMENTS}\n"
    + NEW_EXPERIMENTS
    + "\n\n"
    + "".join(after)
)

with open("paper.txt", "w") as f:
    f.write(new_content)

print("Done. paper.txt updated.")
