# ruff: noqa: N803, N806
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from multiads.solvers import SolverOptions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from multiads.assembly import Fuselage, Propeller, Wing

    MassComponent = Fuselage | Propeller | Wing


class Options(SolverOptions):
    def __init__(
        self,
        *,
        non_linear_inertia_factor: float = 1.0,
        inertia_vector: NDArray[np.float64] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        # Factor to account for non-linear inertial effects
        self.non_linear_inertia_factor = non_linear_inertia_factor
        # [mass, xCG, yCG, zCG, Jxx, Jyy, Jzz, Jxy, Jxz, Jyz]  # noqa: ERA001
        self.inertia_vector = np.zeros(10) if inertia_vector is None else inertia_vector


class Driver:
    """Computes the integral/global mass properties of the aircraft.

    global_massVector[0] = mass of the aircraft
    global_massVector[1] = xG   |
    global_massVector[2] = yG   |--> aircraft center of gravity, global coordinates
    global_massVector[3] = zG   |
    global_massVector[4] = Ixx    |
    global_massVector[5] = Iyy    |
    global_massVector[6] = Izz    |----> aircraft inertia tensor around its CG
    global_massVector[7] = Pxy    |
    global_massVector[8] = Pxz    |
    global_massVector[9] = Pyz    |

    With this definition the Inertia properties will be:

    mass    = global_massVector[0]

              | global_massVector[1] |
    cg      = | global_massVector[2] |
              | global_massVector[3] |

              |  global_massVector[4]  -global_massVector[7]  -global_massVector[8] |
    inertia = | -global_massVector[7]   global_massVector[5]  -global_massVector[9] |
              | -global_massVector[8]  -global_massVector[9]   global_massVector[6] |
    """

    def __init__(
        self,
        propellers: Sequence[Propeller],
        wings: Sequence[Wing],
        fuselage: Fuselage | None,
        options: Options,
    ) -> None:
        # components
        self.propellers = propellers
        self.wings = wings
        self.fuselage = fuselage
        self.options = options

        # initialise mass vectors
        self.global_massVector = np.zeros(10)
        self.prop_mass_vector = np.zeros(10)
        self.wing_mass_vector = np.zeros(10)
        self.fuselage_mass_vector = np.zeros(10)

    def run(self) -> None:
        """Cycle over all aircraft components to take into account their inertia."""
        self.prop_mass_vector[:] = 0.0
        self.wing_mass_vector[:] = 0.0
        self.fuselage_mass_vector[:] = 0.0

        for prop in self.propellers:
            # Present Method
            self.global_massVector = self.add_properties(
                self.global_massVector,
                prop,
            )
            self.prop_mass_vector = self.add_properties(self.prop_mass_vector, prop)

        for wing in self.wings:
            # Present Method
            self.global_massVector = self.add_properties(
                self.global_massVector,
                wing,
            )
            self.wing_mass_vector = self.add_properties(self.wing_mass_vector, wing)

        if fs := self.fuselage:
            # Present Method
            self.global_massVector = self.add_properties(self.global_massVector, fs)
            self.fuselage_mass_vector = self.add_properties(
                self.fuselage_mass_vector,
                fs,
            )

    def add_properties(
        self,
        current_massVector: NDArray[np.float64],
        other: MassComponent,
    ) -> NDArray[np.float64]:
        """Update the current mass properties with another component."""
        # Initialize global_massVector
        global_massVector = np.zeros(10)

        # Mass
        global_massVector[0] = current_massVector[0] + other.mass

        # Center of gravity (with this formulation, the CG of `other` should be in the
        # global reference. If not, the `pos` vector should be considered
        # (trasform from local to global)
        global_massVector[1] = (
            current_massVector[0] * current_massVector[1]
            + other.mass * other.global_pos[0]
        ) / global_massVector[0]
        global_massVector[2] = (
            current_massVector[0] * current_massVector[2]
            + other.mass * other.global_pos[1]
        ) / global_massVector[0]
        global_massVector[3] = (
            current_massVector[0] * current_massVector[3]
            + other.mass * other.global_pos[2]
        ) / global_massVector[0]

        # Inertia Tensor
        current_inertia_tensor_elements = self.get_inertia_tensor_about_point(
            global_massVector[1:4],
            current_massVector,
        )
        # include additional inertias if specified
        other_inertia_tensor_elements = self.get_inertia_tensor_about_point(
            global_massVector[1:4],
            self.options.inertia_vector,
        )
        # compose global mass vector
        global_massVector[4:] = [
            I__ + J__
            for I__, J__ in zip(
                current_inertia_tensor_elements,
                other_inertia_tensor_elements,
                strict=True,
            )
        ]

        return global_massVector

    def get_inertia_tensor_about_point(
        self,
        point: NDArray[np.float64],
        massVector: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the inertia tensor about an arbitrary point.

        Args:
            point: x,y,z position of the new point.
            massVector: mass properties of the component under investigation.

        Returns:
            New inertia tensor.

        """
        R = point - massVector[1:4]
        RdotR = np.dot(R, R)

        Jxx = massVector[4] + massVector[0] * (RdotR - R[0] ** 2)
        Jyy = massVector[5] + massVector[0] * (RdotR - R[1] ** 2)
        Jzz = massVector[6] + massVector[0] * (RdotR - R[2] ** 2)
        Jxy = massVector[7] - massVector[0] * R[0] * R[1]
        Jxz = massVector[8] - massVector[0] * R[2] * R[0]
        Jyz = massVector[9] - massVector[0] * R[1] * R[2]

        return np.array([Jxx, Jyy, Jzz, Jxy, Jxz, Jyz])
