# gui_app.py - Relativistic Dirac Orbital Visualizer
"""
PySide6/VTK GUI for visualizing Dirac equation solutions
and dipole transitions in hydrogenic atoms.

PERFORMANCE OPTIMIZATIONS:
- Vectorized VTK color volume computation
- Reduced memory allocations in render loop
- Optimized radial distribution binning
- Uses solver's optimized compute_color_volume method
"""
from __future__ import annotations
import sys
from typing import Dict, Tuple, Optional
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QDockWidget, QTabWidget, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QListWidget, QFormLayout, QGroupBox,
    QCheckBox, QComboBox, QSlider, QMessageBox, QFileDialog,
)
import pyqtgraph as pg

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.util import numpy_support as vtk_np
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingOpenGL2 import *

from dirac_core import (
    DiracSolver, DiracGridConfig, FieldConfig, TransitionConfig,
    kappa_to_l_j, ALPHA_FS, check_e1_selection_rules, get_performance_info,
    NUMBA_AVAILABLE
)


def compute_isosurface_colors_vectorized(
    points_array: np.ndarray,
    color_volume: np.ndarray,
    grid: DiracGridConfig,
    skip_normalization: bool = False
) -> np.ndarray:
    """
    Vectorized computation of colors for isosurface points.
    
    FIX GUI-003: Added skip_normalization parameter for phase/spin modes
    where color volume is already normalized to [0,1].
    """
    if points_array is None or len(points_array) == 0:
        return np.array([], dtype=np.float32)
    
    nx, ny, nz = color_volume.shape
    
    # FIX GUI-NEW-003: Guard against zero range to avoid division by zero
    eps = 1e-12
    x_range = grid.x_max - grid.x_min
    y_range = grid.y_max - grid.y_min
    z_range = grid.z_max - grid.z_min
    
    dx = x_range / max(nx - 1, 1) if abs(x_range) > eps else 1.0
    dy = y_range / max(ny - 1, 1) if abs(y_range) > eps else 1.0
    dz = z_range / max(nz - 1, 1) if abs(z_range) > eps else 1.0
    
    ix = np.clip(np.round((points_array[:, 0] - grid.x_min) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.round((points_array[:, 1] - grid.y_min) / dy).astype(int), 0, ny - 1)
    iz = np.clip(np.round((points_array[:, 2] - grid.z_min) / dz).astype(int), 0, nz - 1)
    
    color_vals = color_volume[ix, iy, iz].astype(np.float32)
    
    # FIX GUI-003: Only normalize for amplitude mode, not phase/spin
    if not skip_normalization:
        cmin, cmax = color_vals.min(), color_vals.max()
        if cmax > cmin:
            color_vals = (color_vals - cmin) / (cmax - cmin)
        else:
            color_vals[:] = 0.5
    else:
        # Just clamp to [0, 1] for pre-normalized volumes
        color_vals = np.clip(color_vals, 0.0, 1.0)
    
    return color_vals


def radial_distribution_optimized(R: np.ndarray, density: np.ndarray, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized radial distribution using numpy histogram."""
    r_flat = R.ravel()
    d_flat = density.ravel()
    r_max = float(np.max(r_flat))
    if r_max <= 0:
        return np.zeros(n_bins), np.zeros(n_bins)
    hist, bin_edges = np.histogram(r_flat, bins=n_bins, range=(0, r_max), weights=d_flat)
    counts, _ = np.histogram(r_flat, bins=n_bins, range=(0, r_max))
    with np.errstate(invalid='ignore', divide='ignore'):
        radial_prof = np.where(counts > 0, hist / counts, 0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, radial_prof


class Dirac3DView(QWidget):
    """VTK-based 3D isosurface visualization."""
    
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
        self.colors = vtkNamedColors()
        self.renderer.SetBackground(self.colors.GetColor3d("DarkSlateGray"))
        self.lut = vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.Build()
        self.scalar_bar = vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut)
        self.scalar_bar.SetTitle("Color")
        self.scalar_bar.SetNumberOfLabels(4)
        self.scalar_bar.SetPosition(0.88, 0.1)
        self.scalar_bar.SetWidth(0.08)
        self.scalar_bar.SetHeight(0.8)
        self.renderer.AddViewProp(self.scalar_bar)
        self._actor = None
        self._camera_initialized = False
        self._last_diag = 0.0
        self._image_data = None
        self._contour_filter = None
        self._interactor_initialized = False  # FIX VTK-001

    def showEvent(self, event):
        """FIX VTK-001: Initialize VTK interactor on first show."""
        super().showEvent(event)
        if not self._interactor_initialized:
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            if interactor:
                interactor.Initialize()
            self._interactor_initialized = True

    def update_from_density(self, density, grid, iso_fraction=0.1, color_volume=None, color_mode="phase", iso_mode="percentile"):
        if density is None or not np.any(density > 0):
            if self._actor:
                self.renderer.RemoveActor(self._actor)
                self._actor = None
            self.vtk_widget.GetRenderWindow().Render()
            return
        iso_fraction = float(np.clip(iso_fraction, 0.01, 0.99))
        if iso_mode == "percentile":
            positive_vals = density[density > 1e-15]
            if len(positive_vals) > 0:
                percentile = 80 + (iso_fraction - 0.01) * (99.9 - 80) / 0.98
                iso_value = float(np.percentile(positive_vals, percentile))
            else:
                iso_value = 0.0
        else:
            iso_value = iso_fraction * float(np.max(density))
        nx, ny, nz = density.shape
        # FIX BUG-002: Guard against zero spacing (VTK requires positive spacing)
        eps = 1e-10
        x_range = grid.x_max - grid.x_min
        y_range = grid.y_max - grid.y_min
        z_range = grid.z_max - grid.z_min
        dx = x_range / max(nx - 1, 1) if abs(x_range) > eps else 1.0
        dy = y_range / max(ny - 1, 1) if abs(y_range) > eps else 1.0
        dz = z_range / max(nz - 1, 1) if abs(z_range) > eps else 1.0
        if self._image_data is None:
            self._image_data = vtkImageData()
        image = self._image_data
        image.SetDimensions(nx, ny, nz)
        image.SetOrigin(grid.x_min, grid.y_min, grid.z_min)
        image.SetSpacing(dx, dy, dz)
        flat_density = np.ascontiguousarray(density.ravel(order="F"), dtype=np.float32)
        vtk_density = vtk_np.numpy_to_vtk(flat_density, deep=True)
        vtk_density.SetName("density")
        image.GetPointData().SetScalars(vtk_density)
        if self._contour_filter is None:
            self._contour_filter = vtkContourFilter()
        contour = self._contour_filter
        contour.SetInputData(image)
        contour.SetValue(0, iso_value)
        contour.Update()
        self._configure_lut(color_mode)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        if color_volume is not None and color_volume.shape == density.shape:
            polydata = contour.GetOutput()
            points = polydata.GetPoints()
            if points and points.GetNumberOfPoints() > 0:
                points_array = vtk_np.vtk_to_numpy(points.GetData())
                # FIX GUI-003: Skip normalization for phase/spin modes (already [0,1])
                skip_norm = color_mode in ("phase", "spin")
                color_vals = compute_isosurface_colors_vectorized(
                    points_array, color_volume, grid, skip_normalization=skip_norm
                )
                vtk_colors = vtk_np.numpy_to_vtk(color_vals, deep=True)
                vtk_colors.SetName("colors")
                polydata.GetPointData().SetScalars(vtk_colors)
                mapper.SetScalarModeToUsePointData()
                mapper.ScalarVisibilityOn()
                mapper.SetLookupTable(self.lut)
                mapper.SetScalarRange(0.0, 1.0)
            else:
                mapper.ScalarVisibilityOff()
        else:
            mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.8)
        # FIX GUI-005: Set consistent fallback color when scalar visibility is off
        if color_volume is None or not mapper.GetScalarVisibility():
            actor.GetProperty().SetColor(self.colors.GetColor3d("Cyan"))
        if self._actor:
            self.renderer.RemoveActor(self._actor)
        self.renderer.AddActor(actor)
        self._actor = actor
        bounds = actor.GetBounds()
        diag = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
        if not self._camera_initialized or diag > 1.5 * self._last_diag:
            self.renderer.ResetCamera()
            self._camera_initialized = True
        self._last_diag = diag
        self._update_clipping(bounds, diag)
        titles = {"amplitude": "|ψ|²", "phase": "arg ψ₀", "spin": "P↑ − P↓"}
        self.scalar_bar.SetTitle(titles.get(color_mode, "Color"))
        self.vtk_widget.GetRenderWindow().Render()

    def _configure_lut(self, color_mode):
        self.lut.SetNumberOfTableValues(256)
        if color_mode == "amplitude":
            self.lut.SetHueRange(0.667, 0.667)
            self.lut.SetValueRange(0.2, 1.0)
        else:
            self.lut.SetHueRange(0.667, 0.0)
            self.lut.SetValueRange(1.0, 1.0)
        self.lut.SetSaturationRange(1.0, 1.0)
        self.lut.Build()

    def _update_clipping(self, bounds, diag):
        camera = self.renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition())
        obj_center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
        dist = np.linalg.norm(cam_pos - obj_center)
        near = max(1.0, dist - diag)
        far = dist + diag * 3
        camera.SetClippingRange(near, far)

    def reset_camera(self):
        self.renderer.ResetCamera()
        self._camera_initialized = True
        if self._actor:
            bounds = self._actor.GetBounds()
            diag = self._last_diag or 1.0
            self._update_clipping(bounds, diag)
        self.vtk_widget.GetRenderWindow().Render()


class DiracSliceView(pg.ImageView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()

    def update_from_slice(self, slice_data):
        # FIX GUI-NEW-001: Use safe empty image instead of potentially missing clear()
        if slice_data is None:
            self.setImage(np.zeros((2, 2), dtype=np.float32), autoLevels=False)
        else:
            self.setImage(slice_data.T, autoLevels=True)


class DiracLineView(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLabel("bottom", "Coordinate / Radius")
        self.setLabel("left", "Probability")
        self.addLegend()
        self._curve_line = None
        self._curve_radial = None

    def update_from_profiles(self, profiles):
        # FIX GUI-001: Handle empty profiles to clear view
        if not profiles:
            if self._curve_line is not None:
                self._curve_line.setData([], [])
            if self._curve_radial is not None:
                self._curve_radial.setData([], [])
            return
        if "line" in profiles:
            x, y = profiles["line"]
            if self._curve_line is None:
                self._curve_line = self.plot(x, y, pen="y", name="Line")
            else:
                self._curve_line.setData(x, y)
        if "radial" in profiles:
            r, P = profiles["radial"]
            if self._curve_radial is None:
                self._curve_radial = self.plot(r, P, pen="c", name="Radial")
            else:
                self._curve_radial.setData(r, P)


class StateControlPanel(QWidget):
    stateChanged = Signal()

    def __init__(self, solver, parent=None):
        super().__init__(parent)
        self.solver = solver
        self._updating = False
        self._build_ui()
        self.refresh_state_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        g_charge = QGroupBox("Nuclear Charge")
        f_charge = QFormLayout(g_charge)
        self.spin_Z = QSpinBox()
        self.spin_Z.setRange(1, 137)
        self.spin_Z.setValue(self.solver.field.Z)
        self.spin_Z.valueChanged.connect(self._on_Z_changed)
        f_charge.addRow("Z", self.spin_Z)
        layout.addWidget(g_charge)
        g_bound = QGroupBox("Add Bound State")
        f_bound = QFormLayout(g_bound)
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 20)
        self.spin_n.setValue(1)
        self.spin_kappa = QSpinBox()
        self.spin_kappa.setRange(-10, 10)
        self.spin_kappa.setValue(-1)
        self.spin_mj = QDoubleSpinBox()
        self.spin_mj.setRange(-10.0, 10.0)
        self.spin_mj.setSingleStep(0.5)
        self.spin_mj.setValue(0.5)
        f_bound.addRow("n", self.spin_n)
        f_bound.addRow("κ", self.spin_kappa)
        f_bound.addRow("mⱼ", self.spin_mj)
        btn_add = QPushButton("Add Bound State")
        btn_add.clicked.connect(self._on_add_bound)
        f_bound.addRow(btn_add)
        layout.addWidget(g_bound)
        g_list = QGroupBox("States")
        v_list = QVBoxLayout(g_list)
        self.list_states = QListWidget()
        self.list_states.currentRowChanged.connect(self._on_state_selected)
        v_list.addWidget(self.list_states)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._on_remove)
        v_list.addWidget(btn_remove)
        f_coeff = QFormLayout()
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.0, 10.0)
        self.spin_amp.setSingleStep(0.1)
        self.spin_amp.valueChanged.connect(self._on_coeff_changed)
        self.spin_phase = QDoubleSpinBox()
        self.spin_phase.setRange(-360, 360)
        self.spin_phase.setSingleStep(15)
        self.spin_phase.valueChanged.connect(self._on_coeff_changed)
        f_coeff.addRow("Amplitude", self.spin_amp)
        f_coeff.addRow("Phase (°)", self.spin_phase)
        v_list.addLayout(f_coeff)
        btn_norm = QPushButton("Normalize")
        btn_norm.clicked.connect(self._on_normalize)
        v_list.addWidget(btn_norm)
        layout.addWidget(g_list)
        layout.addStretch()

    def _on_Z_changed(self, value):
        try:
            self.solver.set_nuclear_charge(value)
            self.solver.ensure_grid_for_bound_states()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        self.refresh_state_list()
        self.stateChanged.emit()

    def _on_add_bound(self):
        n, kappa, mj = self.spin_n.value(), self.spin_kappa.value(), self.spin_mj.value()
        if kappa == 0:
            QMessageBox.warning(self, "Invalid κ", "κ cannot be 0")
            return
        try:
            self.solver.add_bound_state(n, kappa, mj, amplitude=1.0)
            self.solver.superposition.normalize()
            self.solver.ensure_grid_for_bound_states()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        self.refresh_state_list()
        self.stateChanged.emit()

    def _on_remove(self):
        idx = self.list_states.currentRow()
        if idx >= 0:
            self.solver.remove_state(idx)
            self.refresh_state_list()
            self.stateChanged.emit()

    def _on_state_selected(self, index):
        if index < 0 or index >= self.solver.superposition.n_states():
            return
        self._updating = True
        amp, phase = self.solver.superposition.get_coeff_polar(index)
        self.spin_amp.setValue(amp)
        self.spin_phase.setValue(np.degrees(phase))
        self._updating = False

    def _on_coeff_changed(self):
        if self._updating:
            return
        idx = self.list_states.currentRow()
        if idx >= 0:
            amp = self.spin_amp.value()
            phase = np.radians(self.spin_phase.value())
            self.solver.superposition.set_coeff_polar(idx, amp, phase)
            self.stateChanged.emit()

    def _on_normalize(self):
        self.solver.superposition.normalize()
        idx = self.list_states.currentRow()
        if idx >= 0:
            self._on_state_selected(idx)
        self.stateChanged.emit()

    def refresh_state_list(self):
        self.list_states.clear()
        for line in self.solver.level_summary():
            self.list_states.addItem(line)


class TransitionControlPanel(QWidget):
    transitionChanged = Signal()

    def __init__(self, solver, parent=None):
        super().__init__(parent)
        self.solver = solver
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        g_states = QGroupBox("Transition States")
        f_states = QFormLayout(g_states)
        self.combo_i = QComboBox()
        self.combo_f = QComboBox()
        self.combo_i.currentIndexChanged.connect(self._check_selection_rules)
        self.combo_f.currentIndexChanged.connect(self._check_selection_rules)
        f_states.addRow("Initial", self.combo_i)
        f_states.addRow("Final", self.combo_f)
        self.lbl_rules = QLabel("--")
        f_states.addRow("Selection rules:", self.lbl_rules)
        self.lbl_allowed_dm = QLabel("--")
        f_states.addRow("Allowed Δm:", self.lbl_allowed_dm)
        layout.addWidget(g_states)
        g_field = QGroupBox("Driving Field")
        f_field = QFormLayout(g_field)
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.0, 100.0)
        self.spin_amp.setDecimals(4)
        self.spin_amp.setValue(0.001)
        f_field.addRow("Amplitude E₀", self.spin_amp)
        self.combo_pol = QComboBox()
        self.combo_pol.addItems(["z-polarized", "x-polarized", "y-polarized"])
        self.combo_pol.currentIndexChanged.connect(self._check_selection_rules)
        f_field.addRow("Polarization", self.combo_pol)
        self.check_resonant = QCheckBox("Resonant")
        self.check_resonant.setChecked(True)
        self.check_resonant.toggled.connect(self._on_resonant_toggled)
        f_field.addRow(self.check_resonant)
        self.spin_omega = QDoubleSpinBox()
        self.spin_omega.setRange(0.0, 100.0)
        self.spin_omega.setDecimals(6)
        self.spin_omega.setValue(0.0)
        self.spin_omega.setEnabled(False)
        f_field.addRow("Custom ω:", self.spin_omega)
        layout.addWidget(g_field)
        g_info = QGroupBox("Computed Parameters")
        f_info = QFormLayout(g_info)
        self.lbl_omega_0 = QLabel("--")
        self.lbl_detuning = QLabel("--")
        self.lbl_dipole = QLabel("--")
        f_info.addRow("ω₀:", self.lbl_omega_0)
        f_info.addRow("Detuning Δ:", self.lbl_detuning)
        f_info.addRow("Dipole |d|:", self.lbl_dipole)
        layout.addWidget(g_info)
        btn_apply = QPushButton("Apply Transition")
        btn_apply.clicked.connect(self._apply_transition)
        layout.addWidget(btn_apply)
        layout.addStretch()

    def _on_resonant_toggled(self, checked):
        self.spin_omega.setEnabled(not checked)

    def refresh_state_combos(self):
        self.combo_i.blockSignals(True)
        self.combo_f.blockSignals(True)
        self.combo_i.clear()
        self.combo_f.clear()
        for j, st in enumerate(self.solver.superposition.states):
            label = f"[{j}] {st.label}"
            self.combo_i.addItem(label)
            self.combo_f.addItem(label)
        n = self.solver.superposition.n_states()
        if n >= 2:
            self.combo_i.setCurrentIndex(0)
            self.combo_f.setCurrentIndex(1)
        self.combo_i.blockSignals(False)
        self.combo_f.blockSignals(False)
        self._check_selection_rules()

    def _check_selection_rules(self):
        i, f = self.combo_i.currentIndex(), self.combo_f.currentIndex()
        if i < 0 or f < 0 or i == f:
            self.lbl_rules.setText("--")
            self.lbl_rules.setStyleSheet("")
            self.lbl_allowed_dm.setText("--")
            return
        n = self.solver.superposition.n_states()
        if i >= n or f >= n:
            return
        st_i = self.solver.superposition.states[i]
        st_f = self.solver.superposition.states[f]
        pol = self._get_polarization()
        allowed, reason, allowed_dm = check_e1_selection_rules(st_i, st_f, pol)
        if allowed:
            self.lbl_rules.setText("✓ Allowed")
            self.lbl_rules.setStyleSheet("color: green;")
        else:
            self.lbl_rules.setText(f"✗ {reason}")
            self.lbl_rules.setStyleSheet("color: red;")
        dm_str = ", ".join([f"{dm:+d}" for dm in allowed_dm])
        self.lbl_allowed_dm.setText(dm_str)

    def _get_polarization(self):
        idx = self.combo_pol.currentIndex()
        return [np.array([0,0,1.]), np.array([1,0,0.]), np.array([0,1,0.])][idx]

    def _apply_transition(self):
        i, f = self.combo_i.currentIndex(), self.combo_f.currentIndex()
        if i < 0 or f < 0 or i == f:
            QMessageBox.warning(self, "Error", "Select two different states")
            return
        omega = None if self.check_resonant.isChecked() else self.spin_omega.value()
        config = TransitionConfig(
            state_i=i, state_f=f,
            field_amplitude=self.spin_amp.value(),
            field_polarization=self._get_polarization(),
            field_frequency=omega,
        )
        try:
            self.solver.set_evolution_mode("driven", config)
            self._update_display()
            self.transitionChanged.emit()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_display(self):
        info = self.solver.get_transition_info()
        if info:
            st_i = self.solver.superposition.states[info["state_i"]]
            st_f = self.solver.superposition.states[info["state_f"]]
            omega_0 = abs(st_f.energy - st_i.energy)
            self.lbl_omega_0.setText(f"{omega_0:.6e}")
            if info["detuning"] is not None:
                self.lbl_detuning.setText(f"{info['detuning']:.6e}")
            else:
                self.lbl_detuning.setText("0 (resonant)")
            # FIX GUI-004: Use 'is not None' instead of truthiness (0.0 is valid)
            if info["dipole_magnitude"] is not None:
                self.lbl_dipole.setText(f"{info['dipole_magnitude']:.6e}")
            else:
                self.lbl_dipole.setText("--")
        else:
            self.lbl_omega_0.setText("--")
            self.lbl_detuning.setText("--")
            self.lbl_dipole.setText("--")


class ViewControlPanel(QWidget):
    viewSettingsChanged = Signal()
    playbackToggled = Signal(bool)
    resetCameraRequested = Signal()
    gridChanged = Signal()  # FIX BUG-005: Signal for grid changes requiring panel refresh

    def __init__(self, solver, parent=None):
        super().__init__(parent)
        self.solver = solver
        self._iso_fraction = 0.10
        self._iso_mode = "percentile"
        self._slice_quantity = "density"
        self._slice_plane = "xy"
        self._dt = 1.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        perf_info = get_performance_info()
        g_perf = QGroupBox("Performance")
        f_perf = QFormLayout(g_perf)
        numba_status = "✓ Enabled" if perf_info["numba_available"] else "✗ Not available"
        self.lbl_numba = QLabel(numba_status)
        self.lbl_numba.setStyleSheet("color: green;" if perf_info["numba_available"] else "color: orange;")
        f_perf.addRow("Numba JIT:", self.lbl_numba)
        f_perf.addRow("Threads:", QLabel(str(perf_info["num_threads"])))
        layout.addWidget(g_perf)
        g_3d = QGroupBox("3D Isosurface")
        v_3d = QVBoxLayout(g_3d)
        h_iso_mode = QHBoxLayout()
        h_iso_mode.addWidget(QLabel("Threshold:"))
        self.combo_iso_mode = QComboBox()
        self.combo_iso_mode.addItems(["Percentile (robust)", "Fraction of max"])
        self.combo_iso_mode.currentIndexChanged.connect(self._on_iso_mode_changed)
        h_iso_mode.addWidget(self.combo_iso_mode)
        v_3d.addLayout(h_iso_mode)
        self.lbl_iso = QLabel("Iso level: 10%")
        v_3d.addWidget(self.lbl_iso)
        self.slider_iso = QSlider(Qt.Horizontal)
        self.slider_iso.setRange(1, 50)
        self.slider_iso.setValue(10)
        self.slider_iso.valueChanged.connect(self._on_iso_changed)
        v_3d.addWidget(self.slider_iso)
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Phase", "Amplitude", "Spin"])
        self.combo_color.currentIndexChanged.connect(lambda: self.viewSettingsChanged.emit())
        v_3d.addWidget(self.combo_color)
        btn_reset_cam = QPushButton("Reset Camera")
        btn_reset_cam.clicked.connect(self.resetCameraRequested.emit)
        v_3d.addWidget(btn_reset_cam)
        layout.addWidget(g_3d)
        g_grid = QGroupBox("Grid Resolution")
        f_grid = QFormLayout(g_grid)
        self.combo_grid_size = QComboBox()
        self.combo_grid_size.addItems(["32³", "48³", "64³", "96³", "128³", "256³"])
        self.combo_grid_size.setCurrentIndex(2)
        self.combo_grid_size.currentIndexChanged.connect(self._on_grid_size_changed)
        f_grid.addRow("Grid size:", self.combo_grid_size)
        self.lbl_grid_info = QLabel("~260k points")
        self.lbl_grid_info.setStyleSheet("color: gray; font-size: 10px;")
        f_grid.addRow("", self.lbl_grid_info)
        layout.addWidget(g_grid)
        g_2d = QGroupBox("2D Slice")
        f_2d = QFormLayout(g_2d)
        self.combo_quantity = QComboBox()
        self.combo_quantity.addItems(["Total density", "Large", "Small"])
        self.combo_quantity.currentIndexChanged.connect(self._on_slice_changed)
        self.combo_plane = QComboBox()
        self.combo_plane.addItems(["xy", "xz", "yz"])
        self.combo_plane.currentIndexChanged.connect(self._on_slice_changed)
        f_2d.addRow("Quantity", self.combo_quantity)
        f_2d.addRow("Plane", self.combo_plane)
        layout.addWidget(g_2d)
        g_time = QGroupBox("Time Evolution")
        f_time = QFormLayout(g_time)
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        f_time.addRow(self.btn_play)
        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(0.01, 10.0)
        self.spin_dt.setValue(1.0)
        self.spin_dt.valueChanged.connect(lambda v: setattr(self, '_dt', v))
        f_time.addRow("Δt", self.spin_dt)
        self.lbl_time = QLabel("t = 0.000")
        f_time.addRow(self.lbl_time)
        btn_reset = QPushButton("Reset Time")
        btn_reset.clicked.connect(self._on_reset_time)
        f_time.addRow(btn_reset)
        layout.addWidget(g_time)
        layout.addStretch()

    def _on_iso_changed(self, value):
        self._iso_fraction = value / 100.0
        if self._iso_mode == "percentile":
            percentile = 80 + (self._iso_fraction - 0.01) * (99.9 - 80) / 0.98
            self.lbl_iso.setText(f"Iso level: {value}% → {percentile:.1f}th percentile")
        else:
            self.lbl_iso.setText(f"Iso level: {value}% of max")
        self.viewSettingsChanged.emit()

    def _on_iso_mode_changed(self, index):
        self._iso_mode = "percentile" if index == 0 else "fraction"
        self._on_iso_changed(self.slider_iso.value())
        self.viewSettingsChanged.emit()

    def _on_grid_size_changed(self, index):
        grid_sizes = [32, 48, 64, 96, 128, 256]
        new_size = grid_sizes[index]
        n_points = new_size ** 3
        if n_points < 1_000_000:
            self.lbl_grid_info.setText(f"~{n_points//1000}k points")
        else:
            self.lbl_grid_info.setText(f"~{n_points//1_000_000:.1f}M points")
        current_grid = self.solver.grid
        new_grid = DiracGridConfig(
            nx=new_size, ny=new_size, nz=new_size,
            x_range=(current_grid.x_min, current_grid.x_max),
            y_range=(current_grid.y_min, current_grid.y_max),
            z_range=(current_grid.z_min, current_grid.z_max),
        )
        try:
            self.solver.update_grid(new_grid)
            self.gridChanged.emit()  # FIX BUG-005: Notify panels to refresh
            self.viewSettingsChanged.emit()
        except Exception as e:
            QMessageBox.warning(self, "Grid Update Error", f"Failed to update grid: {e}")

    def _on_slice_changed(self):
        self._slice_quantity = ["density", "large", "small"][self.combo_quantity.currentIndex()]
        self._slice_plane = self.combo_plane.currentText()
        self.viewSettingsChanged.emit()

    def _on_play_toggled(self, checked):
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        self.playbackToggled.emit(checked)

    def _on_reset_time(self):
        self.solver.reset_time()
        self.lbl_time.setText("t = 0.000")
        self.viewSettingsChanged.emit()

    @property
    def iso_fraction(self):
        return self._iso_fraction

    @property
    def iso_mode(self):
        return self._iso_mode

    @property
    def slice_quantity(self):
        return self._slice_quantity

    @property
    def slice_plane(self):
        return self._slice_plane

    @property
    def dt(self):
        return self._dt

    @property
    def color_mode(self):
        return ["phase", "amplitude", "spin"][self.combo_color.currentIndex()]

    def update_time_label(self, t):
        self.lbl_time.setText(f"t = {t:.3f}")


class DiagnosticsPanel(QWidget):
    def __init__(self, solver, parent=None):
        super().__init__(parent)
        self.solver = solver
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.lbl_t = QLabel("0.000")
        self.lbl_prob = QLabel("0.000")
        self.lbl_r = QLabel("0.000")
        self.lbl_Sz = QLabel("0.000")
        form.addRow("Time:", self.lbl_t)
        form.addRow("Prob:", self.lbl_prob)
        form.addRow("⟨r⟩:", self.lbl_r)
        form.addRow("⟨Sz⟩:", self.lbl_Sz)
        layout.addLayout(form)
        layout.addStretch()

    def update_from_solver(self):
        try:
            vals = self.solver.expectation_values()
            self.lbl_t.setText(f"{vals['t']:.3f}")
            self.lbl_prob.setText(f"{vals['probability']:.4f}")
            self.lbl_r.setText(f"{vals['r_mean']:.2f}")
            self.lbl_Sz.setText(f"{vals['Sz']:.4f}")
        except Exception:
            pass

    def clear_display(self):
        """FIX GUI-001: Clear all diagnostic displays."""
        self.lbl_t.setText("--")
        self.lbl_prob.setText("--")
        self.lbl_r.setText("--")
        self.lbl_Sz.setText("--")


class VisualizationController:
    def __init__(self, solver, view3d, view2d, view1d, view_panel, diag_panel):
        self.solver = solver
        self.view3d = view3d
        self.view2d = view2d
        self.view1d = view1d
        self.view_panel = view_panel
        self.diag_panel = diag_panel

    def refresh(self):
        # FIX GUI-001: Clear all views when no states remain
        if self.solver.superposition.n_states() == 0:
            self.view3d.update_from_density(None, self.solver.grid)
            self.view2d.update_from_slice(None)
            self.view1d.update_from_profiles({})
            self.diag_panel.clear_display()
            return
        psi = self.solver.total_spinor_current()
        density = self.solver._compute_density_optimized(psi)
        color_vol = self.solver.compute_color_volume(psi, self.view_panel.color_mode)
        self.view3d.update_from_density(
            density, self.solver.grid,
            self.view_panel.iso_fraction,
            color_vol, self.view_panel.color_mode,
            self.view_panel.iso_mode
        )
        
        # FIX GUI-002: Use slice_quantity from view panel instead of always using total density
        mid = (self.solver.grid.nx//2, self.solver.grid.ny//2, self.solver.grid.nz//2)
        plane = self.view_panel.slice_plane
        quantity = self.view_panel.slice_quantity
        
        # Compute the appropriate slice based on quantity
        if quantity == "large":
            slice_source = np.real(np.sum(np.abs(psi[:2])**2, axis=0))
        elif quantity == "small":
            slice_source = np.real(np.sum(np.abs(psi[2:])**2, axis=0))
        else:  # "density" or default
            slice_source = density
        
        if plane == "xy":
            slice_data = slice_source[:, :, mid[2]]
        elif plane == "xz":
            slice_data = slice_source[:, mid[1], :]
        else:
            slice_data = slice_source[mid[0], :, :]
        self.view2d.update_from_slice(slice_data)
        line_coord = np.linspace(self.solver.grid.z_min, self.solver.grid.z_max, self.solver.grid.nz)
        line_prof = density[mid[0], mid[1], :]
        rad_coord, rad_prof = self.solver.radial_distribution()
        self.view1d.update_from_profiles({"line": (line_coord, line_prof), "radial": (rad_coord, rad_prof)})
        self.diag_panel.update_from_solver()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Relativistic Dirac Orbital Visualizer")
        self.resize(1400, 900)
        self._playing = False
        self.solver = self._create_solver()
        self.view3d = Dirac3DView(self)
        self.view2d = DiracSliceView(self)
        self.view1d = DiracLineView(self)
        self.state_panel = StateControlPanel(self.solver, self)
        self.trans_panel = TransitionControlPanel(self.solver, self)
        self.view_panel = ViewControlPanel(self.solver, self)
        self.diag_panel = DiagnosticsPanel(self.solver, self)
        self.controller = VisualizationController(
            self.solver, self.view3d, self.view2d, self.view1d,
            self.view_panel, self.diag_panel
        )
        self._build_ui()
        self._connect_signals()
        self._setup_timer()
        if self.solver.superposition.n_states() > 0:
            self.trans_panel.refresh_state_combos()
            self.controller.refresh()
        else:
            self.trans_panel.refresh_state_combos()

    def _create_solver(self):
        bohr = 1.0 / ALPHA_FS
        grid = DiracGridConfig(
            nx=64, ny=64, nz=64,
            x_range=(-10*bohr, 10*bohr),
            y_range=(-10*bohr, 10*bohr),
            z_range=(-10*bohr, 10*bohr),
        )
        solver = DiracSolver(grid, FieldConfig(Z=1), include_rest_mass=False)
        return solver

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.addWidget(self.view3d)
        splitter_h = QSplitter(Qt.Horizontal)
        splitter_h.addWidget(self.view2d)
        splitter_h.addWidget(self.view1d)
        splitter_h.setStretchFactor(0, 3)
        splitter_h.setStretchFactor(1, 2)
        splitter_v.addWidget(splitter_h)
        splitter_v.setStretchFactor(0, 3)
        splitter_v.setStretchFactor(1, 2)
        layout.addWidget(splitter_v)
        self.setCentralWidget(central)
        tabs = QTabWidget()
        tabs.addTab(self.state_panel, "States")
        tabs.addTab(self.trans_panel, "Transitions")
        tabs.addTab(self.view_panel, "View")
        tabs.addTab(self.diag_panel, "Diagnostics")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Stationary", "Driven Transition"])
        dock_widget = QWidget()
        dock_layout = QVBoxLayout(dock_widget)
        dock_layout.addWidget(QLabel("<b>Evolution Mode:</b>"))
        dock_layout.addWidget(self.mode_combo)
        dock_layout.addWidget(tabs)
        dock = QDockWidget("Controls", self)
        dock.setWidget(dock_widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Save...", self._on_save)
        file_menu.addAction("Load...", self._on_load)

    def _connect_signals(self):
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.state_panel.stateChanged.connect(self._on_state_changed)
        self.trans_panel.transitionChanged.connect(self._on_transition_changed)
        self.view_panel.viewSettingsChanged.connect(self.controller.refresh)
        self.view_panel.playbackToggled.connect(self._on_playback_toggled)
        self.view_panel.resetCameraRequested.connect(self.view3d.reset_camera)
        # FIX BUG-005: Refresh state panels after grid changes
        self.view_panel.gridChanged.connect(self._on_grid_changed)

    def _on_grid_changed(self):
        """FIX BUG-005: Refresh UI panels after grid change."""
        self.state_panel.refresh_state_list()
        self.trans_panel.refresh_state_combos()

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._on_timer_tick)

    def _on_mode_changed(self, index):
        if index == 0:
            self.solver.set_evolution_mode("stationary")
        else:
            if self.solver.superposition.n_states() < 2:
                QMessageBox.warning(self, "Error", "Need at least 2 states for transitions")
                self.mode_combo.setCurrentIndex(0)
                return
            self.trans_panel._apply_transition()
        self.controller.refresh()

    def _on_state_changed(self):
        self.state_panel.refresh_state_list()
        self.trans_panel.refresh_state_combos()
        self.controller.refresh()

    def _on_transition_changed(self):
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.blockSignals(False)
        self.controller.refresh()

    def _on_playback_toggled(self, playing):
        self._playing = playing
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_timer_tick(self):
        self.solver.step(self.view_panel.dt)
        self.view_panel.update_time_label(self.solver.time)
        self.controller.refresh()

    def _on_save(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save", "", "NPZ files (*.npz)")
        if filename:
            try:
                self.solver.save_configuration(filename)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _on_load(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load", "", "NPZ files (*.npz)")
        if filename:
            try:
                self.solver.load_configuration(filename)
                self.state_panel.refresh_state_list()
                self.trans_panel.refresh_state_combos()
                self.controller.refresh()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
