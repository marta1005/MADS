from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

from multiads.assembly import (
    Aircraft as AssemblyAircraft,
    Environment,
    MADSComponent,
    Section as AssemblySection,
    Span as AssemblySpan,
    Wing as AssemblyWing,
)
from multiads.scenario.aero_derivatives import AeroDerivativesVariable
from multiads.scenario.mass_properties import MassPropertiesVariable
from multiads.solvers import SolverOptions

if TYPE_CHECKING:
    from scipy.interpolate import RBFInterpolator


def _ds_options(comp: MADSComponent) -> dict[str, Any]:
    options = {}
    if hasattr(comp, "options") and isinstance(comp.options, dict):
        options = comp.options.get("synthesis", {})
    return deepcopy(options)


@dataclass
class Section:
    twist: float

    @classmethod
    def from_component(cls, comp: AssemblySection) -> Self:
        return Section(comp.twist)


@dataclass
class Span:
    length: float
    sweep: float
    dihed: float

    @classmethod
    def from_component(cls, comp: AssemblySpan) -> Self:
        return Span(comp.length, comp.sweep, comp.dihed)


@dataclass
class Wing:
    name: str
    sections: list[Section] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    wing_displ_z: list[float] = field(default_factory=lambda: [0.0])
    wing_rot_y: list[float] = field(default_factory=lambda: [0.0])
    span_positions: list[float] = field(default_factory=list)

    @classmethod
    def from_component(
        cls,
        comp: AssemblyWing,
        wing_displ_z: list[float] | None = None,
        wing_rot_y: list[float] | None = None,
        span_positions: list[float] | None = None,
    ) -> Self:
        sections = [Section.from_component(s) for s in comp.sections]
        spans = [Span.from_component(s) for s in comp.spans]
        return cls(
            name=comp.name,
            sections=sections,
            spans=spans,
            wing_displ_z=wing_displ_z or [0.0],
            wing_rot_y=wing_rot_y or [0.0],
            span_positions=span_positions or [],
        )


@dataclass
class Aircraft:
    name: str
    aerodynamicproperties: Any = None
    massproperties: Any = None

    @classmethod
    def from_variables(
        cls,
        name: str,
        aerodynamicproperties: Any = None,
        massproperties: Any = None,
    ) -> Self:
        return cls(
            name=name,
            aerodynamicproperties=aerodynamicproperties,
            massproperties=massproperties,
        )


@dataclass
class PropellerSynthesis:
    """Synthesis data for a propeller.

    Attributes:
        name: Propeller name.
        global_pos: Base position [x, y, z].
        attached_to_wing: Wing name (string) this propeller is attached to.
        attachment_y: Y position on wing for attachment point.
    """

    name: str
    global_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attached_to_wing: str = ""
    attachment_y: float = 0.0


@dataclass
class FuselageSynthesis:
    """Synthesis data for a fuselage.

    Attributes:
        name: Fuselage name.
        global_pos: Base position [x, y, z].
        has_fem_structure: If True, use FEM displacement.
        attachment_points: Points with {name, pos} for deformation interpolation.
    """

    name: str
    global_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    has_fem_structure: bool = False
    attachment_points: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SynthesisComponents:
    """Container for synthesis components.

    Attributes:
        wings: List of wing synthesis data.
        propellers: List of propeller synthesis data.
        fuselages: List of fuselage synthesis data.
        interpolation_mode: Interpolation mode ("rbf" or "section").
    """

    wings: list[Wing] = field(default_factory=list)
    propellers: list[PropellerSynthesis] = field(default_factory=list)
    fuselages: list[FuselageSynthesis] = field(default_factory=list)
    interpolation_mode: str = "rbf"

    def update_displacements(
        self,
        displacements: NDArray[np.float64],
        rotations: NDArray[np.float64] | None = None,
    ) -> None:
        """Update wing displacements from structural analysis.

        Args:
            displacements: Structural displacements array.
            rotations: Structural rotations array (optional).
        """
        for wing in self.wings:
            if len(displacements) > 0:
                if len(displacements.shape) > 1:
                    wing.wing_displ_z = displacements[:, 2].tolist()
                else:
                    wing.wing_displ_z = displacements.tolist()


class Options(SolverOptions):
    def __init__(
        self,
        *,
        name: str = "synthesis",
        n_threads: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.n_threads = n_threads


@dataclass
class DSdriver:
    aircraft: list[Aircraft] = field(default_factory=list)
    wings: list[Wing] = field(default_factory=list)
    propellers: list[PropellerSynthesis] = field(default_factory=list)
    fuselages: list[FuselageSynthesis] = field(default_factory=list)
    environment: Environment | None = None
    options: Options = field(default_factory=Options)
    indices: list[int] | None = None
    spans_new: NDArray[np.float64] | None = None
    dihedrals_new: NDArray[np.float64] | None = None
    thetas_new: NDArray[np.float64] | None = None
    interpolation_mode: str = "rbf"
    _fem_displacements: NDArray[np.float64] | None = field(default=None, repr=False)

    def compute_aircraft_properties(self) -> None:
        """Transform aerodynamic derivatives from GLOBAL to BODY reference frame.

        Performs moment transport: body_moments = global_moments + x_cg * global_forces
        """
        if not self.aircraft or self.aircraft[0].aerodynamicproperties is None:
            return

        aero_var = self.aircraft[0].aerodynamicproperties

        x_cg = 0.0
        if self.aircraft[0].massproperties is not None:
            mp = self.aircraft[0].massproperties
            if (
                hasattr(mp, "massVector")
                and mp.massVector is not None
                and len(mp.massVector) > 1
            ):
                x_cg = float(mp.massVector[1])
            elif hasattr(mp, "cg") and mp.cg is not None and mp.cg.size > 0:
                x_cg = float(mp.cg[0, 0])

        global_matrix = deepcopy(aero_var.matrix)

        rows_with_moment_transport = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for row in rows_with_moment_transport:
            if np.any(global_matrix[row, :] != 0):
                global_fz = float(global_matrix[row, 2])
                global_my = float(global_matrix[row, 4])
                body_my = global_my + x_cg * global_fz

                global_cy = float(global_matrix[row, 1])
                global_mz = float(global_matrix[row, 5])
                body_mz = global_mz - x_cg * global_cy

                matrix = aero_var.matrix.copy()
                matrix[row, 4] = body_my
                matrix[row, 5] = body_mz
                aero_var.matrix = matrix

    def body_AeroDerivatives_alpha(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[0, :]
        return np.zeros(6)

    def body_AeroDerivatives_beta(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[1, :]
        return np.zeros(6)

    def body_AeroDerivatives_p(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[2, :]
        return np.zeros(6)

    def body_AeroDerivatives_q(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[3, :]
        return np.zeros(6)

    def body_AeroDerivatives_r(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[4, :]
        return np.zeros(6)

    def body_AeroDerivatives_alpha_dot(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[5, :]
        return np.zeros(6)

    def body_AeroDerivatives_beta_dot(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[6, :]
        return np.zeros(6)

    def body_AeroDerivatives_pitching(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[7, :]
        return np.zeros(6)

    def body_AeroDerivatives_yawing(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[8, :]
        return np.zeros(6)

    def body_AeroDerivatives_delta_aileron(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[9, :]
        return np.zeros(6)

    def body_AeroDerivatives_delta_elevator(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[10, :]
        return np.zeros(6)

    def body_AeroDerivatives_delta_rudder(self) -> NDArray[np.float64]:
        if self.aircraft and self.aircraft[0].aerodynamicproperties is not None:
            return self.aircraft[0].aerodynamicproperties.matrix[11, :]
        return np.zeros(6)

    def find_nodes(self) -> None:
        for wing in self.wings:
            if wing.wing_displ_z and len(wing.wing_displ_z) > 0:
                y_airfoils = [0.0]
                self.indices = list(range(len(wing.spans) + 1))

    def find_deform(self) -> None:
        for wing in self.wings:
            if wing.wing_displ_z and len(wing.wing_displ_z) > 0:
                spans_old = [span.length for span in wing.spans]
                thetas_old = [section.twist for section in wing.sections]
                dihedrals_old = [span.dihed for span in wing.spans]

                self.spans_new = np.array(spans_old)
                self.dihedrals_new = np.array(dihedrals_old)
                self.thetas_new = np.array(thetas_old)

                if len(wing.wing_rot_y) >= len(thetas_old):
                    for i in range(len(thetas_old)):
                        self.thetas_new[i] = thetas_old[i] + np.degrees(
                            wing.wing_rot_y[i]
                        )

                if len(wing.wing_displ_z) >= len(spans_old) + 1:
                    for i in range(len(spans_old)):
                        delta_z = wing.wing_displ_z[i + 1] - wing.wing_displ_z[i]
                        self.dihedrals_new[i] = (
                            dihedrals_old[i]
                            + np.degrees(np.arcsin(delta_z / spans_old[i]))
                            if spans_old[i] != 0
                            else dihedrals_old[i]
                        )

    def thetas_deformed(self) -> NDArray[np.float64]:
        return self.thetas_new if self.thetas_new is not None else np.zeros(1)

    def spans_deformed(self) -> NDArray[np.float64]:
        return self.spans_new if self.spans_new is not None else np.zeros(1)

    def dihedrals_deformed(self) -> NDArray[np.float64]:
        return self.dihedrals_new if self.dihedrals_new is not None else np.zeros(1)

    def set_interpolation_mode(self, mode: str) -> None:
        """Set the interpolation mode for synthesis updates.

        Args:
            mode: Interpolation mode ("rbf" or "section").
        """
        self.interpolation_mode = mode

    def get_wing_by_name(self, name: str) -> Wing | None:
        """Get wing synthesis by name.

        Args:
            name: Wing name.

        Returns:
            Wing synthesis or None if not found.
        """
        for wing in self.wings:
            if wing.name == name:
                return wing
        return None

    def update_propeller_position(
        self,
        prop: PropellerSynthesis,
        wing: Wing,
    ) -> np.ndarray:
        """Update propeller position based on wing deformation.

        Interpolates wing displacement at attachment point and applies
        rotation to get propeller position update.

        Args:
            prop: Propeller synthesis data.
            wing: Wing synthesis data.

        Returns:
            Updated position [x, y, z].
        """
        delta = self._interpolate_wing_displacement(
            wing,
            prop.attachment_y,
            mode=self.interpolation_mode,
        )

        base_pos = np.array(prop.global_pos)
        return base_pos + delta

    def update_fuselage_position(
        self,
        fuselage: FuselageSynthesis,
        fem_displacements: NDArray[np.float64] | None = None,
    ) -> np.ndarray:
        """Update fuselage position based on structural deformation.

        Args:
            fuselage: Fuselage synthesis data.
            fem_displacements: FEM displacement results (optional).

        Returns:
            Updated position [x, y, z].
        """
        if fuselage.has_fem_structure and fem_displacements is not None:
            delta = self._interpolate_fem_displacement(fuselage, fem_displacements)
        else:
            delta = np.zeros(3)

        base_pos = np.array(fuselage.global_pos)
        return base_pos + delta

    def _interpolate_wing_displacement(
        self,
        wing: Wing,
        y_target: float,
        mode: str = "rbf",
    ) -> np.ndarray:
        """Interpolate wing displacement at target y-position.

        Args:
            wing: Wing synthesis data.
            y_target: Target y-position for interpolation.
            mode: Interpolation mode ("rbf" or "section").

        Returns:
            Displacement vector [dx, dy, dz].
        """
        if not wing.span_positions or not wing.wing_displ_z:
            return np.zeros(3)

        span_pos = np.array(wing.span_positions)
        displ_z = np.array(wing.wing_displ_z)
        rot_y = np.array(wing.wing_rot_y) if wing.wing_rot_y else np.zeros_like(displ_z)

        if len(span_pos) < 2:
            return np.array([0.0, 0.0, displ_z[0] if len(displ_z) > 0 else 0.0])

        if mode == "rbf":
            from multiads.utilities.coupling.mesh_deformation import RBFDeformation

            if len(span_pos) >= 4:
                rbf = RBFDeformation(kernel="thin_plate_spline")
                rbf.fit(
                    span_pos.reshape(-1, 1),
                    displ_z.reshape(-1, 1),
                )
                dz = float(rbf.transform([[y_target]])[0, 0])
            else:
                dz = float(np.interp(y_target, span_pos, displ_z))

            if len(rot_y) >= 4:
                rbf_rot = RBFDeformation(kernel="thin_plate_spline")
                rbf_rot.fit(
                    span_pos.reshape(-1, 1),
                    rot_y.reshape(-1, 1),
                )
                dtheta = float(rbf_rot.transform([[y_target]])[0, 0])
            else:
                dtheta = float(np.interp(y_target, span_pos, rot_y))
        else:
            dz = float(np.interp(y_target, span_pos, displ_z))
            dtheta = float(np.interp(y_target, span_pos, rot_y))

        x_disp = 0.0
        y_disp = 0.0

        return np.array([x_disp, y_disp, dz])

    def _interpolate_fem_displacement(
        self,
        fuselage: FuselageSynthesis,
        fem_displacements: NDArray[np.float64] | None = None,
    ) -> np.ndarray:
        """Interpolate FEM displacement at fuselage attachment points.

        Args:
            fuselage: Fuselage synthesis data.
            fem_displacements: FEM displacement results.

        Returns:
            Displacement vector [dx, dy, dz].
        """
        if (
            not fuselage.attachment_points
            or fem_displacements is None
            or len(fem_displacements) == 0
        ):
            return np.zeros(3)

        delta = np.zeros(3)
        n_points = len(fuselage.attachment_points)

        for point in fuselage.attachment_points:
            if "pos" in point and len(point["pos"]) >= 3:
                pos = np.array(point["pos"])
                if len(fem_displacements.shape) > 1:
                    idx = min(
                        int(np.linalg.norm(pos).item()) % len(fem_displacements),
                        len(fem_displacements) - 1,
                    )
                    point_disp = fem_displacements[idx]
                else:
                    point_disp = np.array(
                        [
                            0.0,
                            0.0,
                            float(fem_displacements[0])
                            if len(fem_displacements) > 0
                            else 0.0,
                        ]
                    )
                delta += point_disp

        if n_points > 0:
            delta /= n_points

        return delta


Driver = DSdriver
