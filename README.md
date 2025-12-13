## DISCLAIMER

I’m not an expert in physics or numerical methods, and a lot of this was built via iterative “vibe coding” (rapid prototyping + testing-by-looking). Still: the goal is **not** a throwaway toy. This is meant to become a **serious, reviewable** visualization/learning tool for relativistic hydrogenic states and E1-driven dynamics.

For now:

* Expect **approximations** (some intentional) and possible **numerical edge cases**.
* **Please don’t cite this repo as a validated physics reference yet.** Independently verify anything that affects grades, grants, or reputations.

If you find a bug (or a minus sign achieving escape velocity), open an issue/PR—help making this solid is welcome.

# Relativistic Dirac Orbital Visualizer

A high-performance Python application for visualizing hydrogen-like 4-component spinors and dipole-driven time evolution.

## Overview

This application provides interactive 3D visualization of hydrogenic states (Dirac quantum numbers) and their time evolution under an electric dipole (E1) driving field. It features:

- **Relativistic structure**: 4-component spinor fields on a 3D grid
- **3D isosurface rendering**: VTK-based visualization with phase/amplitude/spin coloring
- **Time evolution**: stationary phase evolution and driven two-state dynamics in a truncated basis
- **Selection rules**: automatic E1 validation + allowed Δm display
- **Performance**: Numba JIT (optional) + threaded numeric backends

## Physics / Model Notes

### Conceptual Hamiltonian

The intended physical model is the single-particle Dirac–Coulomb problem:

```

H = α·p + βm + V(r),    V(r) = -Zα/r

````

in natural units (ℏ = c = mₑ = 1).

**Important implementation note:** this repo does **not** solve a PDE eigenvalue problem on the 3D grid. It **constructs** spinor fields from analytic building blocks for visualization and uses 1D radial quadrature for dipole elements (with a fallback 3D grid integral for non-bound/missing-QN cases).

### What’s “exact” vs “approx” in the current code

- **Energies:** hydrogenic Dirac energies use the standard closed-form expression (point nucleus).
- **Wavefunctions:** the current radial construction in `hydrogenic_dirac_radial(...)` is a **simplified hydrogenic model** used to generate visually/qualitatively reasonable 4-spinors on the grid (i.e., treat the fields as **approximate** until further validated).
- **E1 elements:** computed via **radial integration** using the current radial model + angular factors (Wigner 3j–style machinery in code).

### Limitations (current scope)

- No QED radiative corrections (Lamb shift, vacuum polarization, etc.)
- Classical driving field only: `E(t) = E₀ cos(ωt)`
- Truncated basis dynamics (you evolve coefficients in the chosen state list)
- Point-nucleus Coulomb potential (no finite nuclear size)

## Installation

### Prerequisites

- **Python 3.10+** (GUI code uses `X | Y` type syntax)
- A display capable of OpenGL rendering (VTK)

### Required Dependencies

```bash
pip install numpy scipy PySide6 pyqtgraph vtk
````

### Optional (Recommended for Performance)

```bash
pip install numba
```

Numba enables JIT compilation and can provide significant speedups for the density/superposition/color-volume bottlenecks.

### Full Installation

```bash
# Clone the repository
git clone https://github.com/M3C3I/Relativistic-Dirac-Orbital-Visualizer.git
cd Relativistic-Dirac-Orbital-Visualizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python gui_app.py
```

### Adding States

1. Open the **States** tab
2. Set the quantum numbers:

   * **n**: principal quantum number (1, 2, 3, ...)
   * **κ**: Dirac quantum number (±1, ±2, ...; κ=0 is forbidden)
   * **mⱼ**: magnetic quantum number (−j to +j in half-integer steps)
3. Click **Add Bound State**

### Understanding κ (Kappa)

| κ  | l | j   | Spectroscopic |
| -- | - | --- | ------------- |
| -1 | 0 | 1/2 | s₁/₂          |
| +1 | 1 | 1/2 | p₁/₂          |
| -2 | 1 | 3/2 | p₃/₂          |
| +2 | 2 | 3/2 | d₃/₂          |
| -3 | 2 | 5/2 | d₅/₂          |

### Setting Up Transitions

1. Add at least 2 states
2. Open the **Transitions** tab
3. Select initial and final states
4. Choose polarization (z, x, or y)
5. Set field amplitude E₀
6. Click **Apply Transition**
7. Switch evolution mode to **Driven Transition**
8. Press **Play** to see time evolution

### Selection Rules

The GUI checks E1-style constraints and shows allowed Δm values based on polarization:

* z-polarized: Δmⱼ = 0
* x/y-polarized: Δmⱼ = ±1 (depending on spherical components present)

(See `check_e1_selection_rules(...)` in `dirac_core.py`.)

## Visualization Controls

### 3D View

* **Threshold mode**:

  * **Percentile (robust)**: chooses an isosurface intended to enclose a target fraction of total probability mass (heuristic, histogram-based)
  * **Fraction of max**: uses a simple `iso = f * max(density)`
* **Color mode**: Phase, Amplitude, or Spin
* **Reset Camera**
* Mouse controls: drag to rotate, scroll to zoom

### 2D Slice

* Select plane (xy, xz, yz)
* Choose quantity (total density, large, small)

### Grid Resolution

* Available: 32³, 48³, 64³, 96³, 128³, 256³
* Higher resolution looks better but is heavier on CPU/RAM.

## Performance Optimization

* **Numba JIT**: accelerates density/superposition/color volume
* **Threaded numeric backends**: environment variables are set early in `dirac_core.py`

If you want to override threads, set these **before importing** the module:

```python
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"
```

## API Reference (Minimal)

```python
from dirac_core import DiracSolver, DiracGridConfig, FieldConfig

grid = DiracGridConfig(nx=64, ny=64, nz=64)
solver = DiracSolver(grid, FieldConfig(Z=1), include_rest_mass=False)

solver.add_bound_state(n=1, kappa=-1, mj=0.5)
solver.add_bound_state(n=2, kappa=+1, mj=0.5)

solver.step(dt=1.0)
density = solver.density_3d_current()
```

## File Format

Saved configurations are `.npz` files containing:

* grid parameters
* field parameters (Z, Coulomb enable)
* state metadata (quantum numbers / free-state params)
* coefficients and time
* transition settings (if driven)

**Note:** spinor arrays are not stored; states are rebuilt from metadata on load.

## Troubleshooting

### “VTK not found”

```bash
pip install vtk
```

Linux may need OpenGL libs:

```bash
sudo apt-get install libgl1-mesa-dev
```

### Slow performance

* Install Numba: `pip install numba`
* Reduce grid size (64³ is a good default)
* Avoid 256³ unless you have plenty of RAM (it can require multiple GB depending on state count)

### Black/empty 3D view

* Add at least one state
* Adjust threshold (try 5–15%)
* Reset camera

## Contributing

Contributions welcome:

1. Fork the repo
2. Create a feature branch
3. Add tests / validation notes where possible
4. Submit a PR

## Citation / Referencing (for now)

This project is **not yet validated** for academic citation as a physics reference.

If you still need to reference it (e.g., for a demo/tools section), please cite the GitHub repo **with a commit hash** and treat it as software-in-progress:

```bibtex
@software{relativistic_dirac_orbital_visualizer,
  title = {Relativistic Dirac Orbital Visualizer},
  year = {2025},
  note = {Software in progress; cite with commit hash},
  url = {https://github.com/M3C3I/Relativistic-Dirac-Orbital-Visualizer.git}
}
```

## Acknowledgments

* Dirac matrix conventions: Bjorken & Drell
* Spherical harmonics: SciPy
* 3D rendering: VTK
* GUI: PySide6 (Qt for Python)

## References

1. Bjorken, J.D. & Drell, S.D. *Relativistic Quantum Mechanics* (1964)
2. Berestetskii, V.B. et al. *Quantum Electrodynamics* (1982)
3. Grant, I.P. *Relativistic Quantum Theory of Atoms and Molecules* (2007)
