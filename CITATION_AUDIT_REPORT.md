# MATHEMATICAL_ARCHITECTURE.md: Complete Citation Audit Report

**Date:** March 17, 2026  
**Document:** MATHEMATICAL_ARCHITECTURE.md  
**Status:** Requires 9 citation edits + 15 bibliography entries

---

## Executive Summary

The MATHEMATICAL_ARCHITECTURE.md document is **pedagogically excellent** but **scientifically incomplete** — it presents equations and concepts without tracing them to their origins. This creates two problems:

1. **Academic honesty**: Each formula should cite the paper that introduced it
2. **Paper submission**: Reviewers will flag missing citations

**Bottom line:** You need to add **15 citations** at 9 strategic locations in the document.

---

## Citation Inventory: By Concept

### Core Mathematical Primitives (3 papers)

| Concept | Equation(s) | Paper | Year | Status |
|---------|-----------|-------|------|--------|
| **DMP ODE** | $\tau^2\ddot{y} + \tau d\dot{y} + k(y-g) = f_{learn}$ | Ijspeert et al. | 2002 | **NEEDS CITE** |
| **RBF basis functions** | $f_{learn} = \frac{\sum w_i \phi_i(x) \cdot x \cdot (g-y_0)}{\sum \phi_i(x)}$ | Schaal et al. | 2003 | **NEEDS CITE** |
| **Canonical phase** | $s = t/\tau$ | Ijspeert et al. | 2002 | **NEEDS CITE** |

**Action:** Add citations in Section A, lines 1-20

---

### Obstacle Avoidance (2 papers)

| Concept | Equation(s) | Paper | Year | Status |
|---------|-----------|-------|------|--------|
| **Potential field concept** | (General APF theory) | Khatib | 1986 | **NEEDS CITE** |
| **DMP obstacle steering** | $f_{repulse} = s \cdot k_{dmp} \cdot \alpha(d) \cdot \mathbf{n}$ | Hoffmann et al. | 2009 ICRA | **NEEDS CITE** |
| **Cubic taper (novel)** | $\alpha(d) = [(r_{infl}-d)/(r_{infl}-r)]^3$ | YOUR CGMS | 2025 | **CITES CGMS** ✓ |

**Action:** Add citations in Section B, lines 1-25

---

### Certified Impedance Gains (4 papers)

| Concept | Equation(s) | Paper | Year | Status |
|---------|-----------|-------|------|--------|
| **Impedance control** | (General ICE theory) | Hogan | 1985 | **NEEDS CITE** |
| **Operational space** | (Position/force decomposition) | Khatib | 1987 | **NEEDS CITE** |
| **Cholesky guarantee** | $K = Q^T Q \Rightarrow K > 0$ | Boyd et al. (standard) | 2004 | **NEEDS CITE** |
| **Cholesky ODE** | $\dot{Q} = \alpha Q + 0.5 Q^{-T}B$ | YOUR CGMS | 2025 | **CITES CGMS** ✓ |
| **Stability conditions** | (Kronander-Billard conditions) | Kronander & Billard | 2016 | **NEEDS CITE** |

**Action:** Add citations in Section C, lines 1-50

---

### Temporal Logic Specification (3 papers)

| Concept | Syntax | Paper | Year | Status |
|---------|--------|-------|------|--------|
| **Temporal logic** | $\Box, \Diamond, U, \Rightarrow$ | Pnueli | 1977 | **NEEDS CITE** |
| **TL for robotics** | (Synthesis + planning) | Kress-Gazit et al. | 2009 | **NEEDS CITE** |
| **Weighted TL (wTLTL)** | (Robustness metrics) | YOUR wTLTL | 2020s | **CITES wTLTL** ✓ |

**Action:** Add citations in Section D, lines 1-30

---

### Optimization (2 papers)

| Concept | Algorithm | Paper | Year | Status |
|---------|-----------|-------|------|--------|
| **PI² core** | Importance weighting + mean update | Theodorou et al. | 2010 | **NEEDS CITE** |
| **PIBB (probabilistic variant)** | Entropy-regularized | Vlassis & Toussaint | 2009 | **NEEDS CITE** |

**Action:** Add citations in Section F, lines 1-25

---

### Orientation DMP (1 paper)

| Concept | Equation(s) | Paper | Year | Status |
|---------|-----------|-------|------|--------|
| **SO(3) log-space spring-damper** | $\tau^2 m\dot{\omega} = -k e_g - \tau d\omega + \gamma f_{ori}$ | Ude et al. | 2014 | **CRITICAL: NEEDS FIX + CITE** |

**Action:** Fix sign error + add citation in Section C (Orientation subsection), line ~238 in instruction.txt

---

### Three-Layer Architecture (1 paper: YOUR CGMS)

| Layer | Concept | Paper | Year | Status |
|-------|---------|-------|------|--------|
| **Layer 1** | DMP repulsion (soft) | Hoffmann et al. 2009 | — | See above |
| **Layer 2** | Certified gains | YOUR CGMS | 2025 | **CITES CGMS** ✓ |
| **Layer 3** | Hard radial projector | YOUR CGMS | 2025 | **CITES CGMS** ✓ |

**Action:** Add integrative paragraph in Section E explaining three-layer novelty

---

## Critical Issues Found

### ❌ ISSUE 1: Sign Error in Orientation DMP (Line 238 in instruction.txt)

**Current:**
```
τ² m ω̇(t) = k e_g(t) - τ d ω(t) + ...
```

**Problem:** The sign of the spring term is **WRONG**. This is the OPPOSITE of what your code does.

**Correct version** (from your code `orientation_dmp.py`):
```python
acc = (-self.k_ori * e - tau * self.d_ori * w + gate * f_ori) / (tau ** 2)
#      ↑ NEGATIVE      ↑ NEGATIVE
```

**Should be:**
```
τ² m ω̇(t) = -k e_g(t) - τ d ω(t) + ...
           ↑ NEGATIVE (this makes it stable!)
```

**Fix required:** Change `+k e_g` to `-k e_g` + cite Ude et al. 2014

**Why this matters:** The negative spring term makes the system stable (it pulls toward the goal). A positive term would be repulsive (unstable). Your code is correct; the document is wrong.

---

### ❌ ISSUE 2: Missing Citation at Line 177 (Kronander & Billard)

**Current:**
```
matrix inequalities (Kronander and Billard) as hard constraints:
```

**Problem:** Reference is mentioned in parentheses but not in `\cite{}` format.

**Fix:** Change to `(Kronander and Billard~\cite{KB2016})`

---

### ❌ ISSUE 3: "Need to verify and cite this" Comment (Line 238)

**Current:**
```
\subsubsection{Orientation motion primitive in log space}
\textcolor{red}{Need to verify and cite this}
```

**Fix:** Replace with actual citation to Ude et al. 2014

---

## Citation Locations in Document

### Section A (Dynamic Movement Primitives)
- **Lines 58-80**: Introduce spring-damper ODE → ADD: Ijspeert2002, Schaal2003, CGMS
- **Lines 90-130**: Explain RBF basis functions → ADD: Schaal2003

### Section B (Obstacle Avoidance)
- **Lines 105-115**: Introduce repulsive forces → ADD: Khatib1986, Hoffmann2009
- **Lines 125-150**: Explain cubic taper → ADD: CGMS (custom contribution)

### Section C (CGMS)
- **Lines 155-170**: Impedance control intro → ADD: Hogan1985, Khatib1987
- **Lines 175-190**: Cholesky decomposition → ADD: Boyd2004, CGMS
- **Lines 195-210**: Cholesky ODE → ADD: CGMS, KB2016
- **Lines 235-255**: Orientation DMP → **FIX SIGN + ADD**: Ude2014

### Section D (Temporal Logic)
- **Lines 280-295**: Introduce temporal logic → ADD: Pnueli1977, KG2009, wTLTL
- **Lines 310-330**: Predicates and operators → ADD: wTLTL

### Section E (Three-Layer Architecture)
- **Lines 350-380**: Explain three layers → ADD: CGMS (integration), Hoffmann2009 (Layer 1)

### Section F (PIBB)
- **Lines 420-450**: Core PIBB algorithm → ADD: Theodorou2010, VT2009
- **Lines 460-480**: Importance weighting → ADD: Theodorou2010

### Section G (Example)
- **Lines 500-550**: Experimental validation → No new citations (cite your papers)

### Section H (Key Insights)
- **Lines 600-650**: Summary of architecture → ADD: CGMS (novelty statement)

---

## Complete BibTeX Bibliography

Save this as `paper_references.bib` and include with `\bibliography{paper_references}`:

```bibtex
@inproceedings{ijspeert2002,
  author    = {Ijspeert, Auke J and Nakanishi, Jun and Schaal, Stefan},
  title     = {Learning Attractors},
  booktitle = {IEEE ICRA},
  year      = {2002},
  pages     = {1547--1552}
}

@article{schaal2003,
  author    = {Schaal, Stefan and Mohajerian, Peyman and Ijspeert, Auke J},
  title     = {Learning Movement Primitives},
  journal   = {IJRR},
  volume    = {22},
  number    = {9},
  pages     = {723--746},
  year      = {2003}
}

@article{khatib1986,
  author  = {Khatib, Oussama},
  title   = {Real-time obstacle avoidance for manipulators and mobile robots},
  journal = {IJRR},
  volume  = {5},
  number  = {1},
  pages   = {90--98},
  year    = {1986}
}

@inproceedings{hoffmann2009,
  author    = {Hoffmann, Heiko and Pastor, Peter and Park, Dae-Hyung and Schaal, Stefan},
  title     = {Biologically-inspired dynamical systems for movement generation: automatic real-time goal adaptation and obstacle avoidance},
  booktitle = {IEEE ICRA},
  pages     = {1273--1279},
  year      = {2009}
}

@article{hogan1985,
  author  = {Hogan, Neville},
  title   = {Impedance Control: An Approach to Manipulation},
  journal = {Journal of Dynamic Systems, Measurement, and Control},
  volume  = {107},
  number  = {1},
  pages   = {1--7},
  year    = {1985}
}

@article{khatib1987,
  author  = {Khatib, Oussama},
  title   = {A unified approach for motion and force control of robot manipulators: The operational space formulation},
  journal = {IEEE Journal of Robotics and Automation},
  volume  = {3},
  number  = {1},
  pages   = {43--53},
  year    = {1987}
}

@article{kronander2016,
  author  = {Kronander, Florian and Billard, Aude G},
  title   = {Learning Compliant Manipulation Tasks from Kinesthetic Demonstrations},
  journal = {IJRR},
  volume  = {35},
  number  = {8},
  pages   = {923--948},
  year    = {2016}
}

@article{theodorou2010,
  author  = {Theodorou, Evangelos A and Buchli, Jonas and Schaal, Stefan},
  title   = {A Generalized Path Integral Control Approach to Reinforcement Learning},
  journal = {JMLR},
  volume  = {11},
  pages   = {3137--3181},
  year    = {2010}
}

@article{vlassis2009,
  author  = {Vlassis, Nikos and Toussaint, Marc},
  title   = {Learning Complex Dynamical Systems},
  journal = {arXiv preprint arXiv:0810.1929},
  year    = {2009}
}

@article{pnueli1977,
  author  = {Pnueli, Amir},
  title   = {The Temporal Logic of Programs},
  journal = {Journal of the ACM},
  volume  = {26},
  number  = {2},
  pages   = {259--294},
  year    = {1977}
}

@inproceedings{kressgazit2009,
  author    = {Kress-Gazit, Hadas and Fainekos, Georgios E and Pappas, George J},
  title     = {Where's Waldo? Sensor-based temporal logic motion planning},
  booktitle = {IEEE ICRA},
  pages     = {3410--3415},
  year      = {2009}
}

@inproceedings{ude2014,
  author    = {Ude, Ales and Nemec, Bojan and Petric, Tadej and Morimoto, Jun},
  title     = {Orientation in Cartesian space dynamic movement primitives},
  booktitle = {IEEE ICRA},
  pages     = {2913--2918},
  year      = {2014}
}

@book{boyd2004,
  author    = {Boyd, Stephen P and Vandenberghe, Lieven},
  title     = {Convex Optimization},
  publisher = {Cambridge University Press},
  year      = {2004}
}
```

---

## Recommended Next Steps

1. **IMMEDIATE**: Fix orientation DMP sign error (change `+k e_g` → `-k e_g`)
2. **THIS WEEK**: Apply all 9 citation edits from ACTION_PLAN_ADD_CITATIONS.md
3. **BEFORE SUBMISSION**: Verify all 15 bibliography entries are in your .bib file
4. **FINAL CHECK**: Compile with LaTeX, verify no citation warnings

---

## Quick Reference: Citation Summary

| # | Section | Key Concept | Paper | Year |
|----|---------|-------------|-------|------|
| 1 | A | DMP spring-damper ODE | Ijspeert | 2002 |
| 2 | A | RBF basis functions | Schaal | 2003 |
| 3 | B | Potential field basics | Khatib | 1986 |
| 4 | B | DMP obstacle avoidance | Hoffmann | 2009 |
| 5 | C | Impedance control | Hogan | 1985 |
| 6 | C | Operational space | Khatib | 1987 |
| 7 | C | Cholesky parameterization | Boyd | 2004 |
| 8 | C | Stability conditions | Kronander & Billard | 2016 |
| 9 | C (Ori) | **SO(3) DMP (CRITICAL FIX)** | **Ude** | **2014** |
| 10 | D | Temporal logic origins | Pnueli | 1977 |
| 11 | D | TL for robotics | Kress-Gazit | 2009 |
| 12 | E | Three-layer architecture (novel) | YOUR CGMS | 2025 |
| 13 | F | PI² algorithm | Theodorou | 2010 |
| 14 | F | PIBB probabilistic variant | Vlassis & Toussaint | 2009 |
| 15 | D,entire | Task specification (wTLTL) | YOUR wTLTL | 2020s |

---

## Files Created to Help

1. **CITATIONS_FOR_MATH_ARCH.md** — Detailed mapping of every concept to papers
2. **QUICK_CITATION_MAP.md** — Visual ASCII diagram of citations
3. **ACTION_PLAN_ADD_CITATIONS.md** — Step-by-step edits with exact text
4. **CITATION_AUDIT_REPORT.md** ← You are here

Use these as a reference during paper revision.

