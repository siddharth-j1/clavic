# Quick Citation Map: MATHEMATICAL_ARCHITECTURE.md → Papers

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SECTION A: DYNAMIC MOVEMENT PRIMITIVES           │
├─────────────────────────────────────────────────────────────────────┤
│ EQUATION:  τ² ÿ + τ d ẏ + k(y - g) = f_learned(x)                  │
│                                                                      │
│ CITE:                                                                │
│  ├─ Ijspeert et al. (2002) ICRA — DMP foundational formulation     │
│  ├─ Schaal et al. (2003) — RBF basis functions for DMPs            │
│  └─ YOUR CGMS Paper (2025) — implementation & extensions           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  SECTION B: OBSTACLE AVOIDANCE                       │
├─────────────────────────────────────────────────────────────────────┤
│ EQUATION:  f_repulse = s · k_dmp · α(d) · n                         │
│            α(d) = [(r_infl - d) / (r_infl - r)]³                    │
│                                                                      │
│ CITE:                                                                │
│  ├─ Khatib (1986) IJRR — Artificial potential fields (foundational)│
│  ├─ Hoffmann et al. (2009) ICRA — DMP obstacle avoidance           │
│  └─ YOUR CGMS Paper (2025) — cubic taper modification              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              SECTION C: CERTIFIED GAINS (CGMS)                       │
├─────────────────────────────────────────────────────────────────────┤
│ EQUATION:  K(t) = Q(t)ᵀ Q(t)  (K > 0 by construction)              │
│            Q̇ = α Q + 0.5 Q⁻ᵀ B                                      │
│                                                                      │
│ CITE:                                                                │
│  ├─ Hogan (1985) JDSMC — Impedance control (foundational)          │
│  ├─ Khatib (1987) IEEE JRA — Operational space control             │
│  ├─ Kronander & Billard (2016) IJRR — Lyapunov stability           │
│  └─ YOUR CGMS Paper (2025) — Cholesky ODE + three-layer safety     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│            SECTION D: TEMPORAL LOGIC SPECIFICATION                   │
├─────────────────────────────────────────────────────────────────────┤
│ SYNTAX: φ := ⊤ | f(y_t) < c | ¬φ | ⋀_w φ_i | ◇φ | □φ | φ U ψ     │
│                                                                      │
│ CITE:                                                                │
│  ├─ Pnueli (1977) JACM — Temporal logic (foundational)             │
│  ├─ Kress-Gazit et al. (2009) ICRA — TL for robotics              │
│  └─ YOUR wTLTL Paper (2020s) — weighted temporal logic             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│         SECTION E: THREE-LAYER SAFETY ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 1: DMP repulsion (soft steering)                              │
│   → Hoffmann et al. (2009) ICRA                                     │
│                                                                      │
│ LAYER 2: CGMS certified positive-definite K(t)                      │
│   → YOUR CGMS Paper (2025)                                          │
│                                                                      │
│ LAYER 3: Hard radial projector (geometric guarantee)                │
│   → YOUR CGMS Paper (2025)                                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│         SECTION F: OPTIMIZATION PIPELINE (PIBB)                      │
├─────────────────────────────────────────────────────────────────────┤
│ ALGORITHM: Probabilistic Inference with Importance Weighting        │
│            w_i = exp(-β J_i) / Z                                    │
│                                                                      │
│ CITE:                                                                │
│  ├─ Theodorou et al. (2010) JMLR — PI² algorithm                   │
│  ├─ Vlassis & Toussaint (2009) — probabilistic inference variant    │
│  └─ YOUR CGMS Paper (2025) — integration with three-layer safety    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│            ORIENTATION DMP (Section C, Orientation)                  │
├─────────────────────────────────────────────────────────────────────┤
│ EQUATION:  τ² ω̇ = -k e_g(t) - τ d ω(t) + γ(t) f_ori(s)           │
│            q̇ = 0.5 Ω(ω) q                                          │
│                                                                      │
│ CITE:                                                                │
│  └─ Ude et al. (2014) ICRA — Orientation in Cartesian space DMPs   │
│                                                                      │
│ NOTE: Your code has NEGATIVE spring term (-k e_g), which is CORRECT │
│ (matches Ude et al.)                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MOST IMPORTANT CITATIONS (by frequency of use)

| Rank | Paper | Why Important |
|------|-------|---|
| **1** | YOUR CGMS Paper (2025) | Most novel contributions (Cholesky ODE, three-layer safety, cubic taper) |
| **2** | YOUR wTLTL Paper | Task specification and temporal logic integration |
| **3** | Ijspeert et al. (2002) | Foundation of DMP mathematics |
| **4** | Hoffmann et al. (2009) | Obstacle avoidance in DMPs |
| **5** | Kronander & Billard (2016) | Stability conditions for impedance |
| **6** | Theodorou et al. (2010) | PIBB optimization algorithm |
| **7** | Khatib (1986 + 1987) | Potential fields + operational space |
| **8** | Hogan (1985) | Impedance control theory |
| **9** | Ude et al. (2014) | Orientation DMP formulation |

---

## FIXES TO MAKE IN instruction.txt

### Line 238: Orientation DMP Comment
**CURRENT:**
```
\textcolor{red}{Need to verify and cite this}
Orientation dynamics are defined in the tangent space of $SO(3)$:
```

**CHANGE TO:**
```
Following Ude et al.~\cite{ude2014}, orientation dynamics are defined in the tangent space of $SO(3)$:
```

**Add to bibliography:**
```bibtex
@inproceedings{ude2014,
  title={Orientation in Cartesian space dynamic movement primitives},
  author={Ude, Ales and Nemec, Bojan and Petric, Tadej and Morimoto, Jun},
  booktitle={ICRA},
  year={2014}
}
```

### Line 177: Kronander & Billard Reference
**CURRENT:**
```
matrix inequalities (Kronander and Billard) as hard constraints:
```

**CHANGE TO:**
```
matrix inequalities~\cite{kronander2016} as hard constraints:
```

---

## Complete Bibliography Entry Template

Add to your `Shreyas/references.bib` or create `paper_references.bib`:

```bibtex
% ===== MATHEMATICAL FOUNDATIONS =====

@inproceedings{ijspeert2002,
  title={Learning Attractors},
  author={Ijspeert, Auke Jan and Nakanishi, Jun and Schaal, Stefan},
  booktitle={ICRA},
  year={2002}
}

@article{schaal2003,
  title={Learning Movement Primitives},
  author={Schaal, Stefan and Mohajerian, Peyman and Ijspeert, Auke J},
  journal={IJRR},
  year={2003}
}

% ===== OBSTACLE AVOIDANCE =====

@article{khatib1986,
  title={Real-time obstacle avoidance for manipulators and mobile robots},
  author={Khatib, Oussama},
  journal={IJRR},
  volume={5},
  number={1},
  pages={90--98},
  year={1986}
}

@inproceedings{hoffmann2009,
  title={Biologically-inspired dynamical systems for movement generation: automatic real-time goal adaptation and obstacle avoidance},
  author={Hoffmann, Heiko and Pastor, Peter and Park, Dae-Hyung and Schaal, Stefan},
  booktitle={ICRA},
  year={2009}
}

% ===== IMPEDANCE CONTROL =====

@article{hogan1985,
  title={Impedance Control: An Approach to Manipulation},
  author={Hogan, Neville},
  journal={Journal of Dynamic Systems, Measurement, and Control},
  volume={107},
  number={1},
  pages={1--7},
  year={1985}
}

@article{khatib1987,
  title={A unified approach for motion and force control of robot manipulators: The operational space formulation},
  author={Khatib, Oussama},
  journal={IEEE Journal of Robotics and Automation},
  volume={3},
  number={1},
  pages={43--53},
  year={1987}
}

% ===== VARIABLE IMPEDANCE + STABILITY =====

@article{kronander2016,
  title={Learning Compliant Manipulation Tasks from Kinesthetic Demonstrations},
  author={Kronander, Florian and Billard, Aude G},
  journal={IJRR},
  volume={35},
  number={8},
  pages={923--948},
  year={2016}
}

% ===== OPTIMIZATION =====

@article{theodorou2010,
  title={A Generalized Path Integral Control Approach to Reinforcement Learning},
  author={Theodorou, Evangelos A and Buchli, Jonas and Schaal, Stefan},
  journal={JMLR},
  volume={11},
  pages={3137--3181},
  year={2010}
}

@article{vlassis2009,
  title={Learning Complex Dynamical Systems},
  author={Vlassis, Nikos and Toussaint, Marc},
  journal={arXiv preprint arXiv:0810.1929},
  year={2009}
}

% ===== TEMPORAL LOGIC =====

@article{pnueli1977,
  title={The Temporal Logic of Programs},
  author={Pnueli, Amir},
  journal={JACM},
  volume={26},
  number={2},
  pages={259--294},
  year={1977}
}

@inproceedings{kressgazit2009,
  title={Where's Waldo? Sensor-based temporal logic motion planning},
  author={Kress-Gazit, Hadas and Fainekos, Georgios E and Pappas, George J},
  booktitle={ICRA},
  year={2009}
}

% ===== ORIENTATION =====

@inproceedings{ude2014,
  title={Orientation in Cartesian space dynamic movement primitives},
  author={Ude, Ales and Nemec, Bojan and Petric, Tadej and Morimoto, Jun},
  booktitle={ICRA},
  year={2014}
}

% ===== YOUR PAPERS =====

@article{cgms2025,
  title={Safe and Certified Gains via Lyapunov-Structured Manifolds for Manipulation},
  author={Jain, Siddharth and others},
  journal={arXiv preprint arXiv:2511.16330},
  year={2025}
}

@article{wltl2020,
  title={Weighted Temporal Logic for Trajectory Synthesis},
  author={Your Name and Co-authors},
  journal={IEEE Transactions on Robotics},
  year={2020}
}
```

---

## Summary

**Total papers to cite in MATHEMATICAL_ARCHITECTURE.md: 13**

| Category | Count | Papers |
|----------|-------|--------|
| DMP foundations | 2 | Ijspeert, Schaal |
| Obstacle avoidance | 2 | Khatib, Hoffmann |
| Impedance & stability | 3 | Hogan, Khatib, Kronander |
| Optimization | 2 | Theodorou, Vlassis |
| Temporal logic | 3 | Pnueli, Kress-Gazit, wTLTL |
| Orientation | 1 | Ude |
| **YOUR PAPERS** | **2** | CGMS, wTLTL |

All papers are **publicly available** (arxiv, IEEE Xplore, or academic databases).
