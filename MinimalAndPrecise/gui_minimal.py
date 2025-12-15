# gui_minimal.py
from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QListWidget,
    QSplitter,
    QCheckBox,
    QMessageBox,
)

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
from vtkmodules.util import numpy_support as vtk_np
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingOpenGL2 import *  # noqa: F401,F403

import physics
import rendering
import model

# Import GPU helper functions from rendering
from rendering import to_numpy, is_gpu_available, get_device_name


# -----------------------------
# VTK view: density isosurface
# -----------------------------

def _grid_dxdy_dz(g: rendering.RenderGrid) -> tuple[float, float, float]:
    dx = (g.x_range[1] - g.x_range[0]) / max(g.nx - 1, 1)
    dy = (g.y_range[1] - g.y_range[0]) / max(g.ny - 1, 1)
    dz = (g.z_range[1] - g.z_range[0]) / max(g.nz - 1, 1)
    return float(dx), float(dy), float(dz)

def scale_grid_keep_spacing(grid: rendering.RenderGrid, extent_factor: float) -> rendering.RenderGrid:
    """
    Scale physical extent by extent_factor while keeping voxel spacing (dx,dy,dz) the same.
    This increases nx,ny,nz accordingly.
    """
    extent_factor = float(extent_factor)
    if extent_factor <= 0:
        raise ValueError("extent_factor must be > 0")

    # current spacing
    dx, dy, dz = _grid_dxdy_dz(grid)

    def _scale_range(rng: tuple[float, float]) -> tuple[float, float]:
        a, b = float(rng[0]), float(rng[1])
        c = 0.5 * (a + b)
        h = 0.5 * (b - a) * extent_factor
        return (c - h, c + h)

    xr = _scale_range(grid.x_range)
    yr = _scale_range(grid.y_range)
    zr = _scale_range(grid.z_range)

    # choose nx so that spacing stays ~dx (same detail)
    nx = int(round((xr[1] - xr[0]) / dx)) + 1 if dx > 0 else grid.nx
    ny = int(round((yr[1] - yr[0]) / dy)) + 1 if dy > 0 else grid.ny
    nz = int(round((zr[1] - zr[0]) / dz)) + 1 if dz > 0 else grid.nz

    nx = max(2, nx)
    ny = max(2, ny)
    nz = max(2, nz)

    return rendering.RenderGrid(nx=nx, ny=ny, nz=nz, x_range=xr, y_range=yr, z_range=zr)

def _iso_touches_boundary(density, iso_value: float, *, border: int = 0) -> bool:
    """
    True if the iso-surface (density >= iso_value) hits any face of the grid.
    border lets you treat a few voxels near the edge as "edge".
    
    Handles both NumPy and CuPy arrays.
    """
    # Convert to numpy for boundary checking (small slices, not worth GPU overhead)
    density = to_numpy(density)
    
    if density.size == 0:
        return False
    b = int(max(0, border))
    nx, ny, nz = density.shape
    b = min(b, nx - 1, ny - 1, nz - 1)

    # Fast path: check if max on any boundary exceeds threshold
    if np.max(density[: b + 1, :, :]) >= iso_value: return True
    if np.max(density[nx - 1 - b :, :, :]) >= iso_value: return True
    if np.max(density[:, : b + 1, :]) >= iso_value: return True
    if np.max(density[:, ny - 1 - b :, :]) >= iso_value: return True
    if np.max(density[:, :, : b + 1]) >= iso_value: return True
    if np.max(density[:, :, nz - 1 - b :]) >= iso_value: return True
    return False


def expand_grid_keep_spacing(grid: rendering.RenderGrid, expand_factor: float) -> rendering.RenderGrid:
    """
    Expand physical ranges by expand_factor about the center, and increase nx/ny/nz so dx/dy/dz stays the same.
    (This is basically your scale_grid_keep_spacing, just named to emphasize auto-growth.)
    """
    return scale_grid_keep_spacing(grid, expand_factor)


def _sample_rgb_at_points(points_xyz: np.ndarray, rgb_vol: np.ndarray, g: rendering.RenderGrid) -> np.ndarray:
    """
    Nearest-voxel sampling of rgb_vol (nx,ny,nz,3) at arbitrary xyz points.
    Returns uint8 colors (N,3).
    """
    nx, ny, nz, _ = rgb_vol.shape
    dx, dy, dz = _grid_dxdy_dz(g)

    x0, y0, z0 = g.x_range[0], g.y_range[0], g.z_range[0]
    ix = np.clip(np.round((points_xyz[:, 0] - x0) / dx).astype(np.int32), 0, nx - 1)
    iy = np.clip(np.round((points_xyz[:, 1] - y0) / dy).astype(np.int32), 0, ny - 1)
    iz = np.clip(np.round((points_xyz[:, 2] - z0) / dz).astype(np.int32), 0, nz - 1)

    rgb = rgb_vol[ix, iy, iz, :]  # float32 in [0,1]
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return rgb_u8


class Dirac3DView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        style = vtkInteractorStyleTrackballCamera()
        style.SetMotionFactor(10.0)
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        self.renderer.SetBackground(0.12, 0.14, 0.16)

        self._interactor_initialized = False
        self._image_data: Optional[vtkImageData] = None
        self._contour: Optional[vtkContourFilter] = None
        self._mapper: Optional[vtkPolyDataMapper] = None
        self._actor: Optional[vtkActor] = None
        self._camera_initialized = False

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._interactor_initialized:
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            if interactor is not None:
                interactor.Initialize()
            self._interactor_initialized = True

    def clear(self) -> None:
        if self._actor is not None:
            self.renderer.RemoveActor(self._actor)
            self._actor = None
        self.vtk_widget.GetRenderWindow().Render()

    def update_isosurface(
        self,
        density: np.ndarray,
        grid: rendering.RenderGrid,
        *,
        iso_fraction_of_max: float = 0.10,
        rgb_volume: Optional[np.ndarray] = None,  # (nx,ny,nz,3) float in [0,1]
        pad_voxels: int = 3,  # <-- NEW: border padding
    ) -> None:
        if density is None or density.size == 0 or not np.any(density > 0):
            self.clear()
            return

        iso_fraction_of_max = float(np.clip(iso_fraction_of_max, 0.001, 0.999))

        # Convert to numpy if GPU arrays (VTK requires numpy)
        density = to_numpy(density)
        if rgb_volume is not None:
            rgb_volume = to_numpy(rgb_volume)

        nx, ny, nz = density.shape
        dx, dy, dz = _grid_dxdy_dz(grid)

        p = int(max(0, pad_voxels))
        if p > 0:
            density_vtk = np.pad(density, ((p, p), (p, p), (p, p)), mode="constant", constant_values=0.0)
            if rgb_volume is not None and rgb_volume.shape[:3] == density.shape:
                rgb_vtk = np.pad(rgb_volume, ((p, p), (p, p), (p, p), (0, 0)),
                                mode="constant", constant_values=0.0)
            else:
                rgb_vtk = None

            # Expanded grid origin so world coords stay consistent
            grid_vtk = rendering.RenderGrid(
                nx=nx + 2*p, ny=ny + 2*p, nz=nz + 2*p,
                x_range=(grid.x_range[0] - p*dx, grid.x_range[1] + p*dx),
                y_range=(grid.y_range[0] - p*dy, grid.y_range[1] + p*dy),
                z_range=(grid.z_range[0] - p*dz, grid.z_range[1] + p*dz),
            )
        else:
            density_vtk = density
            rgb_vtk = rgb_volume
            grid_vtk = grid

        iso_value = iso_fraction_of_max * float(np.max(density_vtk))

        nx2, ny2, nz2 = density_vtk.shape

        if self._image_data is None:
            self._image_data = vtkImageData()
        image = self._image_data
        image.SetDimensions(nx2, ny2, nz2)
        image.SetOrigin(grid_vtk.x_range[0], grid_vtk.y_range[0], grid_vtk.z_range[0])
        image.SetSpacing(dx, dy, dz)

        flat = np.ascontiguousarray(density_vtk.ravel(order="F"), dtype=np.float32)
        vtk_arr = vtk_np.numpy_to_vtk(flat, deep=True)
        vtk_arr.SetName("density")
        image.GetPointData().SetScalars(vtk_arr)

        if self._contour is None:
            self._contour = vtkContourFilter()
        contour = self._contour
        contour.SetInputData(image)
        contour.SetValue(0, iso_value)
        contour.Update()

        if self._mapper is None:
            self._mapper = vtkPolyDataMapper()
        mapper = self._mapper
        mapper.SetInputConnection(contour.GetOutputPort())

        poly = contour.GetOutput()
        if rgb_vtk is not None and rgb_vtk.shape[:3] == density_vtk.shape:
            pts = poly.GetPoints()
            if pts is not None and pts.GetNumberOfPoints() > 0:
                pts_np = vtk_np.vtk_to_numpy(pts.GetData())
                rgb_u8 = _sample_rgb_at_points(pts_np, rgb_vtk, grid_vtk)

                colors = vtkUnsignedCharArray()
                colors.SetNumberOfComponents(3)
                colors.SetName("rgb")
                colors.SetNumberOfTuples(rgb_u8.shape[0])
                colors.SetArray(rgb_u8, rgb_u8.size, True)

                poly.GetPointData().SetScalars(colors)
                mapper.SetScalarModeToUsePointData()
                mapper.ScalarVisibilityOn()
                mapper.SetColorModeToDirectScalars()
            else:
                mapper.ScalarVisibilityOff()
        else:
            mapper.ScalarVisibilityOff()

        if self._actor is None:
            self._actor = vtkActor()
            self.renderer.AddActor(self._actor)
        actor = self._actor
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.85)

        if not mapper.GetScalarVisibility():
            actor.GetProperty().SetColor(0.0, 0.9, 0.9)

        if not self._camera_initialized:
            self.renderer.ResetCamera()
            self._camera_initialized = True

        self.vtk_widget.GetRenderWindow().Render()



    def reset_camera(self) -> None:
        self.renderer.ResetCamera()
        self._camera_initialized = True
        self.vtk_widget.GetRenderWindow().Render()


# -----------------------------
# Simple GUI
# -----------------------------

@dataclass
class TransitionUIState:
    i: int = 0
    f: int = 1
    pol: str = "z"


def _eps_from_choice(choice: str) -> np.ndarray:
    # Keep it simple: linear x/y/z + sigma+/sigma-
    if choice == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    if choice == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.complex128)
    if choice == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.complex128)
    if choice == "σ+":
        # (x + i y)/sqrt(2)
        v = np.array([1.0, 1.0j, 0.0], dtype=np.complex128)
        return v / math.sqrt(float(np.vdot(v, v).real))
    if choice == "σ-":
        # (x - i y)/sqrt(2)
        v = np.array([1.0, -1.0j, 0.0], dtype=np.complex128)
        return v / math.sqrt(float(np.vdot(v, v).real))
    return np.array([0.0, 0.0, 1.0], dtype=np.complex128)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dirac Orbital Viewer (minimal)")
        self.resize(1200, 800)

        # A reasonable default grid for H
        bohr = 1.0 / physics.ALPHA_FS
        self.grid_base = rendering.RenderGrid(
            nx=64, ny=64, nz=64,
            x_range=(-12 * bohr, 12 * bohr),
            y_range=(-12 * bohr, 12 * bohr),
            z_range=(-12 * bohr, 12 * bohr),
        )
        self.grid = self.grid_base

        self.solver = model.DiracSolver(grid=self.grid, Z=1, include_rest_mass=False)
        self.solver.add_bound_state(n=1, kappa=-1, mj=+0.5, amplitude=1.0, phase=0.0)

        self.trans_ui = TransitionUIState()

        self.view3d = Dirac3DView(self)

        self._playing = False
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._refresh_state_list()
        self._refresh_transition_combos()
        self.refresh_visual()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view3d)

        # Right control column
        controls = QWidget()
        v = QVBoxLayout(controls)

        # Atom / grid
        g_atom = QGroupBox("Atom / Grid")
        f_atom = QFormLayout(g_atom)
        self.spin_Z = QSpinBox()
        self.spin_Z.setRange(1, 137)
        self.spin_Z.setValue(self.solver.Z)
        self.spin_Z.valueChanged.connect(self._on_Z_changed)
        f_atom.addRow("Z", self.spin_Z)

        self.combo_grid = QComboBox()
        self.combo_grid.addItems(["48³", "64³", "96³"])
        self.combo_grid.setCurrentText("64³")
        self.combo_grid.currentIndexChanged.connect(self._on_grid_changed)
        f_atom.addRow("Grid", self.combo_grid)

        self.lbl_fit_info = QLabel("fit: --")
        self.lbl_fit_info.setStyleSheet("color: gray; font-size: 10px;")
        f_atom.addRow("Auto-fit", self.lbl_fit_info)

        # GPU status display
        gpu_status = "GPU: " + get_device_name() if is_gpu_available() else "GPU: Not available (using CPU)"
        self.lbl_gpu_info = QLabel(gpu_status)
        self.lbl_gpu_info.setStyleSheet("color: green; font-size: 10px;" if is_gpu_available() else "color: orange; font-size: 10px;")
        f_atom.addRow("Compute", self.lbl_gpu_info)

        self.btn_reset_cam = QPushButton("Reset camera")
        self.btn_reset_cam.clicked.connect(self.view3d.reset_camera)
        f_atom.addRow(self.btn_reset_cam)

        v.addWidget(g_atom)

        # Add state
        g_add = QGroupBox("Add bound state")
        f_add = QFormLayout(g_add)
        self.spin_n = QSpinBox(); self.spin_n.setRange(1, 30); self.spin_n.setValue(1)
        self.spin_k = QSpinBox(); self.spin_k.setRange(-30, 30); self.spin_k.setValue(-1)
        self.spin_mj = QDoubleSpinBox(); self.spin_mj.setRange(-30.0, 30.0); self.spin_mj.setSingleStep(0.5); self.spin_mj.setValue(0.5)
        self.spin_amp = QDoubleSpinBox(); self.spin_amp.setRange(0.0, 10.0); self.spin_amp.setValue(1.0); self.spin_amp.setSingleStep(0.1)
        self.spin_ph = QDoubleSpinBox(); self.spin_ph.setRange(-360.0, 360.0); self.spin_ph.setValue(0.0); self.spin_ph.setSingleStep(15.0)

        f_add.addRow("n", self.spin_n)
        f_add.addRow("κ", self.spin_k)
        f_add.addRow("mⱼ", self.spin_mj)
        f_add.addRow("amp", self.spin_amp)
        f_add.addRow("phase (deg)", self.spin_ph)

        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._on_add_state)
        f_add.addRow(btn_add)

        v.addWidget(g_add)

        # States list
        g_states = QGroupBox("States")
        vv = QVBoxLayout(g_states)
        self.list_states = QListWidget()
        self.list_states.currentRowChanged.connect(lambda _: self.refresh_visual())
        vv.addWidget(self.list_states)

        hbtn = QHBoxLayout()
        btn_del = QPushButton("Remove")
        btn_del.clicked.connect(self._on_remove_state)
        btn_norm = QPushButton("Normalize coeffs")
        btn_norm.clicked.connect(self._on_normalize_coeffs)
        hbtn.addWidget(btn_del)
        hbtn.addWidget(btn_norm)
        vv.addLayout(hbtn)

        v.addWidget(g_states)

        # View settings
        g_view = QGroupBox("View")
        f_view = QFormLayout(g_view)

        self.combo_viewmode = QComboBox()
        self.combo_viewmode.addItems(["Superposition", "Selected state only"])
        self.combo_viewmode.currentIndexChanged.connect(lambda _: self.refresh_visual())
        f_view.addRow("Mode", self.combo_viewmode)

        self.combo_color = QComboBox()
        self.combo_color.addItems(["phase", "amplitude", "spin"])
        self.combo_color.currentIndexChanged.connect(lambda _: self.refresh_visual())
        f_view.addRow("Color", self.combo_color)

        self.spin_iso = QDoubleSpinBox()
        self.spin_iso.setRange(0.001, 0.9)
        self.spin_iso.setSingleStep(0.01)
        self.spin_iso.setValue(0.10)
        self.spin_iso.valueChanged.connect(lambda _: self.refresh_visual())
        f_view.addRow("Iso frac (×max)", self.spin_iso)

        self.check_play = QCheckBox("Play (phase evolution)")
        self.check_play.toggled.connect(self._on_play_toggled)
        f_view.addRow(self.check_play)

        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(0.001, 10.0)
        self.spin_dt.setValue(1.0)
        self.spin_dt.setSingleStep(0.1)
        f_view.addRow("dt", self.spin_dt)

        self.lbl_time = QLabel("t = 0.0")
        f_view.addRow(self.lbl_time)

        v.addWidget(g_view)

        # Exact E1 info (optional, but matches your current core)
        g_e1 = QGroupBox("Exact E1 (report only)")
        f_e1 = QFormLayout(g_e1)

        self.combo_i = QComboBox()
        self.combo_f = QComboBox()
        self.combo_pol = QComboBox()
        self.combo_pol.addItems(["z", "x", "y", "σ+", "σ-"])

        self.combo_i.currentIndexChanged.connect(self._update_e1_info)
        self.combo_f.currentIndexChanged.connect(self._update_e1_info)
        self.combo_pol.currentIndexChanged.connect(self._update_e1_info)

        f_e1.addRow("i", self.combo_i)
        f_e1.addRow("f", self.combo_f)
        f_e1.addRow("pol", self.combo_pol)

        self.lbl_e1 = QLabel("--")
        self.lbl_e1.setWordWrap(True)
        f_e1.addRow("info", self.lbl_e1)

        v.addWidget(g_e1)

        v.addStretch(1)

        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)
        self.setCentralWidget(central)

    # -----------------------------
    # Actions
    # -----------------------------

    def _on_Z_changed(self, Z: int) -> None:
        try:
            self.solver.set_nuclear_charge(int(Z))
            self._refresh_state_list()
            self._refresh_transition_combos()
            self.refresh_visual()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_grid_changed(self) -> None:
        # Base resolution change (keeps base extent)
        size = {"48³": 48, "64³": 64, "96³": 96}[self.combo_grid.currentText()]

        gb = self.grid_base
        self.grid_base = rendering.RenderGrid(
            nx=size, ny=size, nz=size,
            x_range=gb.x_range, y_range=gb.y_range, z_range=gb.z_range,
        )

        # Always auto-fit: start from base and let refresh_visual expand as needed
        self.grid = self.grid_base

        try:
            self.solver.update_grid(self.grid)
            self.refresh_visual()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    def _on_add_state(self) -> None:
        n = int(self.spin_n.value())
        k = int(self.spin_k.value())
        mj = float(self.spin_mj.value())
        amp = float(self.spin_amp.value())
        ph = math.radians(float(self.spin_ph.value()))

        if k == 0:
            QMessageBox.warning(self, "Invalid κ", "κ cannot be 0.")
            return

        try:
            self.solver.add_bound_state(n=n, kappa=k, mj=mj, amplitude=amp, phase=ph)
            self._refresh_state_list()
            self._refresh_transition_combos()
            self.refresh_visual()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_remove_state(self) -> None:
        idx = self.list_states.currentRow()
        if idx < 0:
            return
        try:
            self.solver.remove_state(idx)
            self._refresh_state_list()
            self._refresh_transition_combos()
            self.refresh_visual()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_normalize_coeffs(self) -> None:
        # GUI-side normalization: c_k /= sqrt(sum |c_k|^2)
        if not self.solver.states:
            return
        norm2 = sum(abs(st.coeff) ** 2 for st in self.solver.states)
        if norm2 <= 0:
            return
        norm = math.sqrt(norm2)
        for st in self.solver.states:
            st.coeff /= norm
        self._refresh_state_list()
        self.refresh_visual()

    def _on_play_toggled(self, playing: bool) -> None:
        self._playing = bool(playing)
        if self._playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_tick(self) -> None:
        dt = float(self.spin_dt.value())
        self.solver.step(dt)
        self.lbl_time.setText(f"t = {self.solver.t:.3f}")
        self.refresh_visual()

    # -----------------------------
    # Rendering / updates
    # -----------------------------

    def _refresh_state_list(self) -> None:
        self.list_states.blockSignals(True)
        self.list_states.clear()
        for row in self.solver.level_summary():
            # concise label
            self.list_states.addItem(
                f"[{row['index']}] n={row['n']} κ={row['kappa']} mⱼ={row['mj']:+.1f}  E={row['E']:+.6e}"
            )
        if self.list_states.count() > 0 and self.list_states.currentRow() < 0:
            self.list_states.setCurrentRow(0)
        self.list_states.blockSignals(False)

    def refresh_visual(self) -> None:
        if not self.solver.states:
            self.view3d.clear()
            self._update_e1_info()
            if hasattr(self, "lbl_fit_info"):
                self.lbl_fit_info.setText("fit: --")
            return

        view_selected_only = (self.combo_viewmode.currentIndex() == 1)
        idx = self.list_states.currentRow()
        if idx < 0:
            idx = 0

        def _compute_psi_and_density() -> tuple[np.ndarray, np.ndarray]:
            if view_selected_only:
                st = self.solver.states[idx]
                if st.psi_cache is None:
                    st.psi_cache = rendering.sample_bound_spinor(st.sol, self.solver.fields)
                psi_local = st.coeff * st.psi_cache
            else:
                psi_local = self.solver.total_spinor_current()
            dens_local = rendering.density(psi_local)
            return psi_local, dens_local

        psi, dens = _compute_psi_and_density()

        # --- ALWAYS AUTO-FIT (expand only; keep spacing fixed) ---
        iso_frac = float(self.spin_iso.value())

        max_iter = 12          # more attempts = more “make my computer cry”
        expand_factor = 1.50   # more aggressive growth per step
        safety_border = 2      # treat 2 voxels near edge as “edge”
        did_expand = False

        for _ in range(max_iter):
            # Handle GPU arrays - convert to numpy for max calculation
            dens_np = to_numpy(dens)
            iso_value = iso_frac * float(np.max(dens_np))
            if iso_value <= 0.0 or not np.isfinite(iso_value):
                break

            if not _iso_touches_boundary(dens, iso_value, border=safety_border):
                break

            did_expand = True
            self.grid = expand_grid_keep_spacing(self.grid, expand_factor)
            self.solver.update_grid(self.grid)      # invalidates caches
            psi, dens = _compute_psi_and_density()  # resample on the bigger grid

        # One extra small “margin” expansion after it fits, so it doesn’t look tight.
        if did_expand:
            self.grid = expand_grid_keep_spacing(self.grid, 1.08)
            self.solver.update_grid(self.grid)
            psi, dens = _compute_psi_and_density()
        # --- end auto-fit ---

        # small status line so you can see what the fit decided
        if hasattr(self, "lbl_fit_info"):
            dx, dy, dz = _grid_dxdy_dz(self.grid)
            self.lbl_fit_info.setText(
                f"fit: {self.grid.nx}×{self.grid.ny}×{self.grid.nz}  "
                f"dx≈{dx:.3g}  x=[{self.grid.x_range[0]:.1f},{self.grid.x_range[1]:.1f}]"
            )

        rgb = self.solver.compute_color_volume(psi, mode=self.combo_color.currentText())

        self.view3d.update_isosurface(
            dens,
            self.grid,
            iso_fraction_of_max=iso_frac,
            rgb_volume=rgb,
        )

        self._update_e1_info()


    def _refresh_transition_combos(self) -> None:
        self.combo_i.blockSignals(True)
        self.combo_f.blockSignals(True)

        self.combo_i.clear()
        self.combo_f.clear()
        for row in self.solver.level_summary():
            self.combo_i.addItem(f"[{row['index']}] n={row['n']} κ={row['kappa']} mⱼ={row['mj']:+.1f}")
            self.combo_f.addItem(f"[{row['index']}] n={row['n']} κ={row['kappa']} mⱼ={row['mj']:+.1f}")

        if self.combo_i.count() >= 2:
            self.combo_i.setCurrentIndex(0)
            self.combo_f.setCurrentIndex(1)
        elif self.combo_i.count() == 1:
            self.combo_i.setCurrentIndex(0)
            self.combo_f.setCurrentIndex(0)

        self.combo_i.blockSignals(False)
        self.combo_f.blockSignals(False)

    def _update_e1_info(self) -> None:
        n = len(self.solver.states)
        if n < 2:
            self.lbl_e1.setText("Add at least 2 states to see an i→f E1 report.")
            return

        i = self.combo_i.currentIndex()
        f = self.combo_f.currentIndex()
        if i < 0 or f < 0 or i >= n or f >= n:
            self.lbl_e1.setText("--")
            return
        if i == f:
            self.lbl_e1.setText("Pick two different states.")
            return

        st_i = self.solver.states[i]
        st_f = self.solver.states[f]
        eps = _eps_from_choice(self.combo_pol.currentText())

        sel = physics.check_e1_selection_rules(st_i.qn, st_f.qn, polarization_xyz=eps)
        d_xyz = physics.dipole_e1(st_i.sol, st_f.sol)
        omega0 = abs(float(st_f.sol.E) - float(st_i.sol.E))

        epsdotd = physics.project_dipole(d_xyz, eps)
        if not sel.get("allowed", False):
            epsdotd = 0.0 + 0.0j

        self.lbl_e1.setText(
            f"ω₀ = {omega0:.6e}\n"
            f"allowed = {sel.get('allowed', False)}  ({sel.get('reason','')})\n"
            f"|d| = {float(np.linalg.norm(d_xyz)):.6e}\n"
            f"|ε·d| = {abs(epsdotd):.6e}"
        )


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
