"""
rendering.py — Visualization sampling utilities for Dirac bound states.

IMPORTANT:
- Everything in this module is **render-only approximation**.
- We interpolate exact radial solutions (from physics.py) onto a finite 3D grid for visualization.
- Energies, dipoles, selection rules, etc. must always come from physics.py (exact model),
  never from 3D sums.

Conventions (must match physics.py):
Ψ(r,θ,φ) = (1/r) [ G(r) Ω_{κ,m}(θ,φ),
                   i F(r) Ω_{-κ,m}(θ,φ) ]^T

GPU Acceleration:
- Uses CuPy for GPU acceleration when available
- Automatically falls back to NumPy if CuPy is not installed or no GPU is available
- All calculations remain identical regardless of backend
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple
import time
import warnings
import numpy as np
import physics

# GPU acceleration setup - try to import CuPy, fallback to NumPy
_GPU_AVAILABLE = False
_GPU_DEVICE_NAME = "CPU (NumPy)"

try:
    import cupy as cp
    # Test if GPU is actually available
    cp.cuda.runtime.getDeviceCount()
    _GPU_AVAILABLE = True
    _GPU_DEVICE_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
    xp = cp  # Use CuPy as array module
except (ImportError, Exception):
    xp = np  # Fall back to NumPy


def get_array_module():
    """Return the active array module (cupy or numpy)."""
    return xp


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


def get_device_name() -> str:
    """Get the name of the compute device being used."""
    return _GPU_DEVICE_NAME


def to_numpy(arr) -> np.ndarray:
    """Convert array to numpy (handles both numpy and cupy arrays)."""
    if _GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr):
    """Convert array to GPU if available, otherwise return as-is."""
    if _GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr

@dataclass(frozen=True)
class RenderGrid:
    """3D visualization grid definition."""
    nx: int
    ny: int
    nz: int
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]


def make_grid_fields(grid: RenderGrid) -> Dict[str, Any]:
    """Construct Cartesian + spherical fields on the render grid (render-only).

    Adds cached/common fields:
      - R_SAFE, INV_R, ORIGIN_MASK, R_FLAT, GRID_SHAPE
      - _cache: {"omega": {}, "radial_interp": {}}

    Notes:
    - dv can be 0 for degenerate grids (nx==1 etc.); integration routines must guard on dv.
    - Uses GPU acceleration when available via CuPy.
    """
    if grid.nx < 1 or grid.ny < 1 or grid.nz < 1:
        raise ValueError("Grid dimensions must be positive.")

    # Use GPU-accelerated array module if available
    x = xp.linspace(grid.x_range[0], grid.x_range[1], grid.nx, dtype=xp.float64)
    y = xp.linspace(grid.y_range[0], grid.y_range[1], grid.ny, dtype=xp.float64)
    z = xp.linspace(grid.z_range[0], grid.z_range[1], grid.nz, dtype=xp.float64)

    X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")
    R = xp.sqrt(X * X + Y * Y + Z * Z)

    # Safe spherical angles
    R_safe = xp.where(R > 0.0, R, 1.0)
    cos_th = xp.clip(Z / R_safe, -1.0, 1.0)
    THETA = xp.arccos(cos_th)
    PHI = xp.arctan2(Y, X)

    # Force angles to 0 at the origin (render-only convention)
    origin_mask = (R == 0.0)
    if xp.any(origin_mask):
        THETA = THETA.copy()
        PHI = PHI.copy()
        THETA[origin_mask] = 0.0
        PHI[origin_mask] = 0.0

    dx = (grid.x_range[1] - grid.x_range[0]) / (grid.nx - 1) if grid.nx > 1 else 0.0
    dy = (grid.y_range[1] - grid.y_range[0]) / (grid.ny - 1) if grid.ny > 1 else 0.0
    dz = (grid.z_range[1] - grid.z_range[0]) / (grid.nz - 1) if grid.nz > 1 else 0.0
    dv = float(dx * dy * dz)

    inv_r = (1.0 / R_safe).astype(xp.float64)

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "R": R,
        "THETA": THETA,
        "PHI": PHI,
        "dv": dv,
        # Cached/common grid fields
        "R_SAFE": R_safe,
        "INV_R": inv_r,
        "ORIGIN_MASK": origin_mask,
        "R_FLAT": R.ravel(),
        "GRID_SHAPE": R.shape,
        # Per-grid caches (render-only)
        "_cache": {
            "omega": {},         # (kappa, mj2) -> Ω array (...,2)
            "radial_interp": {}, # (id(sol), "G"/"F") -> interpolated array (nx,ny,nz)
        },
        # Track if GPU is being used
        "_gpu_enabled": _GPU_AVAILABLE,
    }

def assert_valid_volume_element(fields: Dict[str, np.ndarray | float], *, where: str = "") -> None:
    """Render-only guardrail: ensure dv > 0 for any grid-based integration.

    We do NOT enforce this in make_grid_fields() because degenerate grids (nx==1 etc.)
    can still be useful for visualization slices. But integrating on such grids is invalid.
    """
    dv = float(fields.get("dv", 0.0))
    if not np.isfinite(dv) or dv <= 0.0:
        ctx = f" in {where}" if where else ""
        raise ValueError(
            f"Render-grid integration requires dv>0{ctx}, but dv={dv}. "
            f"This usually means a degenerate grid (nx, ny, nz must all be >= 2) "
            f"or a reversed/zero range on an axis."
        )

def _grid_cache(fields: Dict[str, Any]) -> Dict[str, Dict[Any, Any]]:
    """Return per-grid cache dict; create if missing (works with older fields dicts)."""
    c = fields.get("_cache")
    if not isinstance(c, dict):
        c = {"omega": {}, "radial_interp": {}}
        fields["_cache"] = c
    c.setdefault("omega", {})
    c.setdefault("radial_interp", {})
    return c  # type: ignore[return-value]


def _ensure_common_grid_fields(fields: Dict[str, Any]) -> None:
    """Ensure INV_R / ORIGIN_MASK / R_FLAT / GRID_SHAPE exist (for backward compatibility).
    
    Handles both NumPy and CuPy arrays transparently.
    """
    if "R" not in fields:
        raise ValueError("fields missing required key 'R'")

    R = fields["R"]
    # Determine which array module to use based on the R array type
    arr_mod = cp if (_GPU_AVAILABLE and isinstance(R, cp.ndarray)) else np
    
    if "GRID_SHAPE" not in fields:
        fields["GRID_SHAPE"] = R.shape
    if "R_FLAT" not in fields:
        fields["R_FLAT"] = R.ravel()
    if "ORIGIN_MASK" not in fields:
        fields["ORIGIN_MASK"] = (R == 0.0)
    if "INV_R" not in fields:
        R_safe = arr_mod.where(R > 0.0, R, 1.0)
        fields["R_SAFE"] = R_safe
        fields["INV_R"] = (1.0 / R_safe).astype(arr_mod.float64)


def omega_cached(
    kappa: int,
    mj: float,
    fields: Dict[str, Any],
    *,
    profile: Optional[Dict[str, float]] = None,
):
    """Get cached Ω_{kappa,mj}(THETA,PHI) for this grid (render-only cache).

    Cache key: (kappa, mj2) where mj2 = int(round(2*mj)).
    Uses GPU acceleration when available.
    """
    _ensure_common_grid_fields(fields)
    cache = _grid_cache(fields)["omega"]

    mj2 = int(round(2.0 * float(mj)))
    key = (int(kappa), int(mj2))
    hit = cache.get(key)
    if hit is not None:
        return hit

    t0 = time.perf_counter() if profile is not None else 0.0
    
    # physics.spinor_spherical_harmonic requires numpy arrays
    THETA = to_numpy(fields["THETA"]).astype(np.float64)
    PHI = to_numpy(fields["PHI"]).astype(np.float64)
    Om = physics.spinor_spherical_harmonic(int(kappa), float(mj), THETA, PHI).astype(np.complex128)
    
    # Convert to GPU if available
    if _GPU_AVAILABLE:
        Om = cp.asarray(Om)
    
    cache[key] = Om

    if profile is not None:
        profile["angular_s"] = profile.get("angular_s", 0.0) + (time.perf_counter() - t0)

    return Om


def interp_radial_cached(
    sol: physics.RadialSolution,
    component: Literal["G", "F"],
    fields: Dict[str, Any],
    *,
    profile: Optional[Dict[str, float]] = None,
):
    """Interpolate sol.G or sol.F onto this grid's radii (render-only cache).

    Cache key: (id(sol), component). Invalidation happens automatically when sol object changes
    (e.g., recomputed on Z change) or when grid changes (fields replaced).
    Uses GPU acceleration when available.
    """
    _ensure_common_grid_fields(fields)
    cache = _grid_cache(fields)["radial_interp"]

    key = (id(sol), str(component))
    hit = cache.get(key)
    if hit is not None:
        return hit

    t0 = time.perf_counter() if profile is not None else 0.0

    # np.interp requires numpy arrays
    Rf = to_numpy(fields["R_FLAT"]).astype(np.float64)
    shape = tuple(fields["GRID_SHAPE"])

    r_src = np.asarray(sol.r, dtype=np.float64)
    y_src = np.asarray(sol.G if component == "G" else sol.F, dtype=np.float64)

    y = np.interp(Rf, r_src, y_src, left=0.0, right=0.0).reshape(shape).astype(np.float64)
    
    # Convert to GPU if available
    if _GPU_AVAILABLE:
        y = cp.asarray(y)
    
    cache[key] = y

    if profile is not None:
        dt = time.perf_counter() - t0
        profile["interp_s"] = profile.get("interp_s", 0.0) + dt
        profile[f"interp_{component}_s"] = profile.get(f"interp_{component}_s", 0.0) + dt

    return y

def probability_render_estimate(
    psi,
    fields: Dict[str, Any],
    *,
    warn_if_below: float = 0.9,
) -> Dict[str, float | str]:
    """Render-only probability estimate: P ≈ Σ ρ dv.

    Returns:
      {"render_estimate_probability": prob, "render_estimate_warning": "..."} (warning optional)

    Notes:
    - This is an approximation. It depends on grid extent/resolution and on how the user's
      superposition coefficients are normalized.
    - Handles both NumPy and CuPy arrays.
    """
    assert_valid_volume_element(fields, where="probability_render_estimate")
    dv = float(fields["dv"])
    rho = density(psi)
    
    # Get sum - convert to numpy scalar if needed
    rho_sum = to_numpy(rho).sum() if _GPU_AVAILABLE and hasattr(rho, 'get') else np.sum(rho)
    prob = float(rho_sum * dv)

    out: Dict[str, float | str] = {"render_estimate_probability": prob}

    if (not np.isfinite(prob)) or prob <= 0.0:
        msg = "render probability estimate is non-finite or non-positive (grid/integration issue)"
        out["render_estimate_warning"] = msg
        warnings.warn(msg, RuntimeWarning)
        return out

    if prob < float(warn_if_below):
        msg = (
            f"grid too small for normalization: render_estimate_probability={prob:.3f} < {warn_if_below:.3f}. "
            f"Increase grid extent and/or resolution."
        )
        out["render_estimate_warning"] = msg
        warnings.warn(msg, RuntimeWarning)

    return out

def _interp_radial_to_R(r_src, y_src, R):
    """Render-only interpolation from 1D radial grid to 3D radii array.

    Uses numpy.interp on flattened radii; values outside [r_min, r_max] are set to 0.
    """
    r_src = np.asarray(r_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    Rf = to_numpy(R).astype(np.float64).ravel()
    y = np.interp(Rf, r_src, y_src, left=0.0, right=0.0)
    return y.reshape(R.shape).astype(np.float64)


def sample_bound_spinor(
    sol: physics.RadialSolution,
    fields: Dict[str, Any],
    *,
    profile: Optional[Dict[str, float]] = None,
):
    """Sample a bound-state 4-spinor onto the 3D render grid (render-only approximation).

    Caches used (per grid):
    - Ω_{κ,m} (angular) via omega_cached(...)
    - Interpolated G(R), F(R) via interp_radial_cached(...)

    Profiling (optional, pass dict):
    - "angular_s": time spent computing Ω (cache misses only)
    - "interp_s", "interp_G_s", "interp_F_s": time spent interpolating (cache misses only)
    - "assemble_s": time spent assembling psi
    
    Uses GPU acceleration when available.
    """
    _ensure_common_grid_fields(fields)

    inv_r = fields["INV_R"]
    origin_mask = fields["ORIGIN_MASK"]

    # Render-only radial interpolation (cached, may be on GPU)
    G_R = interp_radial_cached(sol, "G", fields, profile=profile)
    F_R = interp_radial_cached(sol, "F", fields, profile=profile)

    # Exact angular algebra from physics.py, cached per-grid (may be on GPU)
    Om_k = omega_cached(sol.qn.kappa, sol.qn.mj, fields, profile=profile)
    Om_mk = omega_cached(-sol.qn.kappa, sol.qn.mj, fields, profile=profile)

    t0 = time.perf_counter() if profile is not None else 0.0

    # Determine array module based on data location
    arr_mod = cp if (_GPU_AVAILABLE and isinstance(G_R, cp.ndarray)) else np

    big = G_R * inv_r
    sml = F_R * inv_r

    psi = arr_mod.empty((4,) + big.shape, dtype=arr_mod.complex128)
    psi[0, ...] = big * Om_k[..., 0]
    psi[1, ...] = big * Om_k[..., 1]
    psi[2, ...] = 1j * sml * Om_mk[..., 0]
    psi[3, ...] = 1j * sml * Om_mk[..., 1]

    if arr_mod.any(origin_mask):
        psi[:, origin_mask] = 0.0 + 0.0j

    if profile is not None:
        profile["assemble_s"] = profile.get("assemble_s", 0.0) + (time.perf_counter() - t0)

    return psi


def density(psi):
    """Probability density ρ = Σ_a |ψ_a|^2 on the grid.
    
    Handles both NumPy and CuPy arrays.
    """
    # Determine array module based on input type
    arr_mod = cp if (_GPU_AVAILABLE and isinstance(psi, cp.ndarray)) else np
    psi = arr_mod.asarray(psi, dtype=arr_mod.complex128)
    return arr_mod.sum(arr_mod.abs(psi) ** 2, axis=0).astype(arr_mod.float64)


def _hsv_to_rgb(h, s, v):
    """Vectorized HSV->RGB, outputs (...,3) in [0,1].
    
    Handles both NumPy and CuPy arrays.
    """
    # Determine array module based on input type
    arr_mod = cp if (_GPU_AVAILABLE and isinstance(h, cp.ndarray)) else np
    
    h = arr_mod.mod(h, 1.0)
    s = arr_mod.clip(s, 0.0, 1.0)
    v = arr_mod.clip(v, 0.0, 1.0)

    i = arr_mod.floor(h * 6.0).astype(arr_mod.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = arr_mod.mod(i, 6)

    r = arr_mod.empty_like(v)
    g = arr_mod.empty_like(v)
    b = arr_mod.empty_like(v)

    mask = (i == 0)
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = (i == 1)
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = (i == 2)
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = (i == 3)
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = (i == 4)
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = (i == 5)
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return arr_mod.stack([r, g, b], axis=-1)


def color_volume(psi, mode: Literal["phase", "amplitude", "spin"] = "phase"):
    """Create an RGB color volume for visualization.

    Returns array (...,3) float32 in [0,1].
    Always returns NumPy array for VTK compatibility.

    Modes:
    - "phase": hue encodes local phase of the *dominant* spinor component; value encodes amplitude.
    - "amplitude": grayscale proportional to amplitude.
    - "spin": encodes local ⟨S_z⟩ (render-only proxy mapping) with brightness from amplitude.
    
    Uses GPU acceleration when available.
    """
    # Determine array module based on input type
    arr_mod = cp if (_GPU_AVAILABLE and isinstance(psi, cp.ndarray)) else np
    
    psi = arr_mod.asarray(psi, dtype=arr_mod.complex128)
    rho = density(psi)
    amp = arr_mod.sqrt(rho)
    amax = float(to_numpy(arr_mod.max(amp))) if arr_mod.any(amp) else 1.0
    v = arr_mod.clip(amp / (amax + 1e-30), 0.0, 1.0)

    if mode == "amplitude":
        rgb = arr_mod.stack([v, v, v], axis=-1)
        return to_numpy(rgb).astype(np.float32)

    if mode == "spin":
        # S_z = 1/2 diag(σ_z, σ_z)
        a0 = arr_mod.abs(psi[0]) ** 2
        a1 = arr_mod.abs(psi[1]) ** 2
        a2 = arr_mod.abs(psi[2]) ** 2
        a3 = arr_mod.abs(psi[3]) ** 2
        sz_num = 0.5 * ((a0 - a1) + (a2 - a3))
        sz = sz_num / (rho + 1e-30)  # in [-1/2, +1/2]
        t = arr_mod.clip((sz / 0.5 + 1.0) * 0.5, 0.0, 1.0)  # map to [0,1]
        rgb = arr_mod.stack([t, arr_mod.zeros_like(t), 1.0 - t], axis=-1)
        rgb = rgb * v[..., None]
        return to_numpy(arr_mod.clip(rgb, 0.0, 1.0)).astype(np.float32)

    # mode == "phase"
    abs2 = arr_mod.abs(psi) ** 2  # (4,nx,ny,nz)
    idx = arr_mod.argmax(abs2, axis=0)  # (nx,ny,nz)
    # gather selected component
    psi_sel = arr_mod.take_along_axis(psi, idx[None, ...], axis=0)[0]
    ph = arr_mod.angle(psi_sel)  # [-pi, pi]
    h = (ph + arr_mod.pi) / (2.0 * arr_mod.pi)  # [0,1]
    s = arr_mod.ones_like(h)
    rgb = _hsv_to_rgb(h, s, v)
    return to_numpy(rgb).astype(np.float32)


def radial_distribution_from_density(
    R: np.ndarray,
    dens: np.ndarray,
    dv: float,
    n_bins: int = 128,
) -> Dict[str, np.ndarray | float]:
    """Compute a 1D radial distribution from a 3D density (render-only postprocessing).

    We compute shell probability via histogram:
      P_bin ≈ Σ_{voxels in bin} density * dv
    and return an estimated radial probability density p(r) = P_bin / Δr.

    Returns dict with:
      r_centers, p_r, shell_prob, total_prob
    """
    if float(dv) <= 0.0 or not np.isfinite(float(dv)):
        raise ValueError(
            f"radial_distribution_from_density requires dv>0 (got dv={dv}). "
            "This indicates a degenerate grid (nx,ny,nz must all be >=2)."
        )

    Rf = np.asarray(R, dtype=np.float64).ravel()
    wf = (np.asarray(dens, dtype=np.float64).ravel()) * float(dv)

    rmax = float(np.max(Rf)) if Rf.size else 0.0
    edges = np.linspace(0.0, rmax, int(n_bins) + 1, dtype=np.float64)
    shell_prob, _ = np.histogram(Rf, bins=edges, weights=wf)

    dr = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p_r = shell_prob / (dr + 1e-300)

    total_prob = float(np.sum(shell_prob))
    return {
        "r_centers": centers,
        "p_r": p_r,
        "shell_prob": shell_prob,
        "total_prob": total_prob,
    }

if __name__ == "__main__":
    print("rendering.py sanity run (render-only sampling)")
    print("NOTE: This is a visualization-oriented approximation, NOT exact physics.")
    print("      Render probability is NOT expected to be exactly 1 unless the grid is large enough.\n")

    # A reasonable-ish grid for a quick sanity check (still approximate)
    grid = RenderGrid(
        nx=64, ny=64, nz=64,
        x_range=(-400.0, 400.0),
        y_range=(-400.0, 400.0),
        z_range=(-400.0, 400.0),
    )
    fields = make_grid_fields(grid)

    # Exact state from physics.py
    qn = physics.BoundQN(n=1, kappa=-1, mj=+0.5, Z=1)  # hydrogen 1s1/2
    sol = physics.solve_radial_analytic(qn, include_rest_mass=True, n_points=4000, grid="log")

    psi = sample_bound_spinor(sol, fields)
    info = probability_render_estimate(psi, fields, warn_if_below=0.9)

    prob = info["render_estimate_probability"]
    print(f"Render probability estimate for H 1s1/2 on 64^3 grid over [-400,400]^3: {prob:.6f}")
    if "render_estimate_warning" in info:
        print(f"WARNING: {info['render_estimate_warning']}")
    print("\n(If this is far below 1, expand the grid extent and/or increase resolution.)")
