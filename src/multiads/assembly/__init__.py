from __future__ import annotations

import copy
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, TypeVar

import numpy as np
import scipy.interpolate as sint
from numpy.typing import NDArray

from multiads.ambiance import Atmosphere
from multiads.scenario import N, V, ValueType, Variable
from multiads.utilities.cst import (
    evaluate_cst_surface,
    normalized_chordwise_points,
)

T = TypeVar("T")

Optimizable: TypeAlias = Variable[V, N] | T

OptimizableInt: TypeAlias = Optimizable[
    np.int32,
    NDArray[np.int32],
    int | np.int32,
]
OptimizableFloat: TypeAlias = Optimizable[
    np.float64,
    NDArray[np.float64],
    float | np.float64,
]
OptimizableIntNP: TypeAlias = Optimizable[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]
OptimizableFloatNP: TypeAlias = Optimizable[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


class ComponentOptions:
    pass


class MADSComponent:
    def __init__(
        self,
        name: str,
        options: Sequence[ComponentOptions] | None = None,
    ) -> None:
        self.name = name
        self.variables: dict[str, Variable] = {}
        self.options: Sequence[ComponentOptions] = options or []

    def __setattr__(self, name: str, val: Optimizable) -> None:
        if isinstance(val, Variable):
            self.variables[name] = val
            val = val.value
        super().__setattr__(name, val)

    def remove_variable(self, attr: str, value: ValueType) -> None:
        try:
            self.variables.pop(attr)
        except KeyError:
            msg = f"Attribute '{attr}' of component '{self.name}' is not a variable."
            raise RuntimeError(msg) from None
        setattr(self, attr, value)


def flatten_components(
    components: Iterable[MADSComponent],
) -> Sequence[MADSComponent]:
    components_flat: list[MADSComponent] = []
    _flatten_components(components, components_flat)
    return components_flat


def _flatten_components(
    components: Iterable[MADSComponent],
    flat_list: list[MADSComponent],
) -> None:
    for comp in components:
        # Avoid repetitions
        if comp in flat_list:
            continue
        flat_list.append(comp)

        # Loop over attributes
        for att in vars(comp):
            next_comps = _get_comps_in_attribute(comp, att)
            _flatten_components(next_comps, flat_list)


def copy_components(
    components: Iterable[MADSComponent],
) -> Sequence[MADSComponent]:
    # Mapping from original components to their copies
    original_to_copy: dict[MADSComponent, MADSComponent] = {}

    # Copied components
    copied_components: list[MADSComponent] = []

    # Helper to add a component and its references to the mapping
    def add_component(component: MADSComponent, is_first_level: bool) -> None:
        # Avoid duplicates
        if component in original_to_copy:
            return

        # Make a copy...
        copied_component = copy.deepcopy(component)
        original_to_copy[component] = copied_component

        # ...but keep the variables
        copied_component.variables = component.variables

        # Only return first-level components
        if is_first_level:
            copied_components.append(copied_component)

        # Recursively add referenced components
        for att in vars(component):
            next_comps = _get_comps_in_attribute(component, att)
            for comp in next_comps:
                add_component(comp, is_first_level=False)

    def update_attribute(  # noqa: ANN202
        original: MADSComponent,
        copied: MADSComponent,
        att: str,
    ):
        value_orig = getattr(original, att)
        value_copy = getattr(copied, att)

        if isinstance(value_orig, Sequence):
            for i, x in enumerate(value_orig):
                if isinstance(x, MADSComponent):
                    value_copy[i] = original_to_copy[x]
        elif isinstance(value_orig, MADSComponent):
            value_copy = original_to_copy[value_orig]

        setattr(copied, att, value_copy)

    # First pass: create 'shallow' copies of components and references
    for component in components:
        add_component(component, is_first_level=True)

    # Second pass: update references
    for original, copied in original_to_copy.items():
        for att in vars(original):
            update_attribute(original, copied, att)

    return copied_components


def update_components(
    components: Sequence[MADSComponent],
    variables: Mapping[str, NDArray],
) -> None:
    _components = flatten_components(components)
    for comp in _components:
        for att, var in comp.variables.items():
            if (v := variables.get(var.name)) is not None:
                var.value = v
            setattr(comp, att, var.value)


def _get_comps_in_attribute(
    parent: MADSComponent,
    att: str,
) -> Sequence[MADSComponent]:
    value = getattr(parent, att)
    next_obj = []
    if isinstance(value, Sequence):
        next_obj.extend([c for c in value if isinstance(c, MADSComponent)])
    elif isinstance(value, MADSComponent):
        next_obj.append(value)
    return next_obj


## Note: this value seems not consistent with the other values.
# Compare with ambiance package for checking. Also, it is
# not clear whether the dynamic or kinematic viscosity is meant.
# In general, we should define the units that are to be used
# for each parameter.
class Environment(MADSComponent):
    def __init__(
        self,
        name: str,
        height: OptimizableFloat,
        speed: OptimizableFloat,
        alpha: OptimizableFloat = 0.0,
        beta: OptimizableFloat = 0.0,
        gamma: OptimizableFloat = 0.0,
    ) -> None:
        super().__init__(name)
        self.height: float = height
        self.speed: float = speed
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma

    @property
    def density(self) -> float:
        return Atmosphere(self.height).density[0]

    @property
    def pressure(self) -> float:
        return Atmosphere(self.height).pressure[0]

    @property
    def temperature(self) -> float:
        return Atmosphere(self.height).temperature[0]

    @property
    def sound_speed(self) -> float:
        return Atmosphere(self.height).speed_of_sound[0]

    @property
    def dyn_viscosity(self) -> float:
        return Atmosphere(self.height).dynamic_viscosity[0]

    @property
    def kin_viscosity(self) -> float:
        return Atmosphere(self.height).kinematic_viscosity[0]

    @property
    def mach(self) -> float:
        return self.speed / self.sound_speed

    @property
    def matrix_body_wind(self) -> NDArray[np.float64]:
        alpha = np.radians(self.alpha)
        beta = np.radians(self.beta)
        gamma = np.radians(self.gamma)
        return np.array(
            [
                [
                    np.cos(beta) * np.cos(alpha),
                    -np.cos(gamma) * np.sin(beta) * np.cos(alpha)
                    + np.sin(gamma) * np.sin(alpha),
                    np.sin(gamma) * np.sin(beta) * np.cos(alpha)
                    + np.cos(gamma) * np.sin(alpha),
                ],
                [
                    np.sin(beta),
                    np.cos(gamma) * np.cos(beta),
                    -np.sin(gamma) * np.cos(beta),
                ],
                [
                    -np.cos(beta) * np.sin(alpha),
                    np.cos(gamma) * np.sin(beta) * np.sin(alpha)
                    + np.sin(gamma) * np.cos(alpha),
                    -np.sin(gamma) * np.sin(beta) * np.sin(alpha)
                    + np.cos(gamma) * np.cos(alpha),
                ],
            ],
        )

    @property
    def matrix_wind_body(self) -> NDArray[np.float64]:
        alpha = np.radians(self.alpha)
        beta = np.radians(self.beta)
        gamma = np.radians(self.gamma)
        return np.array(
            [
                [
                    np.cos(beta) * np.cos(alpha),
                    np.sin(beta),
                    -np.cos(beta) * np.sin(alpha),
                ],
                [
                    -np.cos(gamma) * np.sin(beta) * np.cos(alpha)
                    + np.sin(gamma) * np.sin(alpha),
                    np.cos(gamma) * np.cos(beta),
                    np.cos(gamma) * np.sin(beta) * np.sin(alpha)
                    + np.sin(gamma) * np.cos(alpha),
                ],
                [
                    np.sin(gamma) * np.sin(beta) * np.cos(alpha)
                    + np.cos(gamma) * np.sin(alpha),
                    -np.sin(gamma) * np.cos(beta),
                    -np.sin(gamma) * np.sin(beta) * np.sin(alpha)
                    + np.cos(gamma) * np.cos(alpha),
                ],
            ],
        )

    @property
    def velocity(self) -> NDArray[np.float64]:
        matrix = self.matrix_wind_body
        return np.dot(matrix, np.array([self.speed, 0.0, 0.0]))

    @velocity.setter
    def velocity(self, vel: NDArray[np.float64]) -> None:
        beta = np.arctan2(-vel[1], vel[0])
        alpha = np.arctan2(vel[2], vel[0] / np.cos(beta))
        self.speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        self.alpha = np.degrees(alpha)
        self.beta = np.degrees(beta)


class Airfoil(MADSComponent):
    def __init__(
        self,
        name: str,
        thickness_factor: OptimizableFloat = 1.0,
        camber_factor: OptimizableFloat = 1.0,
        polar_file: Path | str | None = None,
    ) -> None:
        super().__init__(name)
        self.thickness_factor: float = thickness_factor
        self.camber_factor: float = camber_factor
        self.polar_file: Path | None = None
        if polar_file is not None:
            self.polar_file = Path(polar_file).resolve()

    def coordinates(
        self,
        npoints: int = 100,
        *,
        distribution: str = "eq",
        chord: float = 1.0,
        twist: float = 0.0,
        xc_ref: float = 0.25,
    ) -> NDArray[np.float64]:
        # Get x/c coordinates of the nodes
        x = self._get_x_points(npoints, distribution)

        # Delegate to implementations
        xy = self._coordinates(x)

        # Rescale with chord
        if chord != 1.0:
            xy *= chord
            xc_ref *= chord

        # Rotate points
        if twist != 0.0:
            twist_rad = np.radians(twist)
            s = np.sin(twist_rad)
            c = np.cos(twist_rad)

            rot = np.array([[c, s], [-s, c]])

            xy[0, :] -= xc_ref
            xy = rot @ xy
            xy[0, :] += xc_ref

        return xy

    def thickness_distribution(
        self,
        npoints: int = 100,
        *,
        distribution: str = "eq",
        chord: float = 1.0,
    ) -> NDArray[np.float64]:
        x = self._get_x_points(npoints, distribution)
        t = self._thickness_distribution(x)
        return np.vstack((x, t)).T * chord

    def camber_distribution(
        self,
        npoints: int = 100,
        *,
        distribution: str = "eq",
        chord: float = 1.0,
    ) -> NDArray[np.float64]:
        x = self._get_x_points(npoints, distribution)
        c = self._camber_distribution(x)
        return np.vstack((x, c)).T * chord

    def _get_x_points(self, npoints: int, distribution: str) -> NDArray[np.float64]:
        npoints = npoints // 2 + np.mod(npoints, 2)
        dist = distribution.lower()
        if dist == "eq":
            x = np.linspace(0.0, 1.0, npoints)
        elif dist == "le":
            x = 1.0 - np.cos(np.linspace(0.0, np.pi / 2.0, npoints))
        elif dist == "te":
            x = np.sin(np.linspace(0.0, np.pi / 2.0, npoints))
        elif dist in ("le_te", "te_le"):
            x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, npoints)))
        else:
            msg = "Unkown distribution of points for the airfoil"
            raise ValueError(msg)
        return x

    @property
    @abstractmethod
    def thickness(self) -> float:
        pass

    @property
    @abstractmethod
    def camber(self) -> float:
        pass

    @abstractmethod
    def _thickness_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def _camber_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def _coordinates(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def airfoil_name(self) -> str:
        pass


class AirfoilNACA4(Airfoil):
    def __init__(
        self,
        name: str,
        m: OptimizableInt,
        p: OptimizableInt,
        t: OptimizableInt,
        thickness_factor: OptimizableFloat = 1.0,
        camber_factor: OptimizableFloat = 1.0,
        polar_file: str | Path | None = None,
    ) -> None:
        self.m: int = m
        self.p: int = p
        self.t: int = t

        if self.m < 0 or self.m > 9:  # noqa: PLR2004
            msg = f"Wrong camber: {self.m}"
            raise ValueError(msg)
        if self.p < 0 or self.p > 9:  # noqa: PLR2004
            msg = f"Wrong camber position: {self.p}"
            raise ValueError(msg)
        if self.t < 0 or self.t > 99:  # noqa: PLR2004
            msg = f"Wrong thickness: {self.t}"
            raise ValueError(msg)

        super().__init__(name, thickness_factor, camber_factor, polar_file)

    @property
    def airfoil_name(self) -> str:
        return f"NACA{self.m:01}{self.p:01}{self.t:02}"

    @property
    def naca_camber(self) -> float:
        return self.m * 0.01

    @property
    def camber(self) -> float:
        return self.naca_camber * self.camber_factor

    @property
    def camber_pos(self) -> float:
        return self.p * 0.1

    @property
    def naca_thickness(self) -> float:
        return self.t * 0.01

    @property
    def thickness(self) -> float:
        return self.naca_thickness * self.thickness_factor

    def _thickness_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            self.thickness
            / 0.2
            * (
                0.2969 * np.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )
        )

    def _camber_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        camber_distribution = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < self.p:
                camber_distribution[i] = (
                    self.m / (self.p**2) * (2 * self.p * xi - xi**2)
                )
            else:
                camber_distribution[i] = (
                    self.m
                    / ((1 - self.p) ** 2)
                    * (1 - 2 * self.p + 2 * self.p * xi - xi**2)
                )

        return camber_distribution

    def _coordinates(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Aliases
        m, p = self.camber, self.camber_pos

        # Expressions for NACA 4-digit airfoils
        if m == 0.0 or p == 0.0:
            yc = np.zeros_like(x)
            dyc = np.zeros_like(x)
        else:
            yc = np.where(
                x < self.camber_pos,
                m / p**2 * (2 * p * x - x**2),
                m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2),
            )
            dyc = np.where(
                x < self.camber_pos,
                m / p**2 * 2 * (p - x),
                m / (1 - p) ** 2 * 2 * (p - x),
            )

        yt = self._thickness_distribution(x)

        xyu = np.empty((len(x), 2))
        xyl = np.empty_like(xyu)

        # Returned 'x' is slightly different in non-symmetric airfoils
        for i, theta in enumerate(np.arctan(dyc)):
            xyu[i, 0] = x[i] - yt[i] * np.sin(theta)
            xyu[i, 1] = yc[i] + yt[i] * np.cos(theta)
            xyl[i, 0] = x[i] + yt[i] * np.sin(theta)
            xyl[i, 1] = yc[i] - yt[i] * np.cos(theta)

        return np.vstack((np.flipud(xyl[1:, :]), xyu))


class AirfoilFile(Airfoil):
    def __init__(
        self,
        name: str,
        filename: Path | str | None = None,
        thickness_factor: OptimizableFloat = 1.0,
        camber_factor: OptimizableFloat = 1.0,
        polar_file: str | Path | None = None,
    ) -> None:
        # The airfoil may not have a shape, only a polar file
        if filename is None and polar_file is None:
            msg = "An AirfoilFile needs, at least, a filename or a polar file"
            raise ValueError(msg)

        if filename is not None:
            self.filename: Path | None = Path(filename).resolve()
            self.file_thickness = np.max(self.thickness_distribution()[:, 1])
            self.file_camber = np.max(self.camber_distribution()[:, 1])
        else:
            self.filename = None
            self.file_thickness = 0.0
            self.file_camber = 0.0

        super().__init__(name, thickness_factor, camber_factor, polar_file)

    @property
    def airfoil_name(self) -> str:
        if self.filename:
            return str(self.filename)
        return ""

    @property
    def thickness(self) -> float:
        return self.file_thickness * self.thickness_factor

    @property
    def camber(self) -> float:
        return self.file_camber * self.camber_factor

    def _thickness_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Get coordinates
        xy = self._coordinates(x)

        # Split in lower and upper side (point @ LE included in both sides)
        npoints = int(np.ceil(xy.shape[0] / 2))
        xyl = np.flipud(xy[:npoints, :])
        xyu = xy[npoints - 1 :, :]

        # Approximate thickness
        return xyu[:, 1] - xyl[:, 1]

    def _camber_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Get coordinates
        xy = self._coordinates(x)

        # Split in lower and upper side (point @ LE included in both sides)
        npoints = int(np.ceil(xy.shape[0] / 2))
        xyl = np.flipud(xy[:npoints, :])
        xyu = xy[npoints - 1 :, :]

        # Approximate camber
        return (xyu[:, 1] + xyl[:, 1]) / 2

    def _coordinates(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.filename is None:
            msg = f"Airfoil '{self.name}' does not have an associated geometry file"
            raise RuntimeError(msg)

        # Read file coordinates
        nodes = np.loadtxt(self.filename, skiprows=1)

        # Normalize
        le_index = np.argmin(nodes[:, 0])
        chord = np.max(nodes[:, 0]) - nodes[le_index, 0]
        nodes[:, 0] -= nodes[le_index, 0]
        nodes /= chord

        # Separate upper and lower sides
        xy1 = nodes[:le_index, :]
        xy2 = nodes[le_index:, :]

        if xy1[0, 0] > 0.5:  # noqa: PLR2004
            xy1 = np.flipud(xy1)
        if xy2[0, 0] > 0.5:  # noqa: PLR2004
            xy2 = np.flipud(xy2)

        diff1 = np.max(xy1[:, 1]) - np.min(xy2[:, 1])
        diff2 = np.max(xy2[:, 1]) - np.min(xy1[:, 1])
        if diff1 > diff2:
            xyu = xy1
            xyl = xy2
        else:
            xyu = xy2
            xyl = xy1

        # Interpolate upper and lower sides separately
        try:
            thickness_scale = self.thickness / self.file_thickness
        except AttributeError:
            thickness_scale = 1.0  # Calling from __init__

        s = sint.CubicSpline(xyu[:, 0], xyu[:, 1] * thickness_scale)
        xyu = np.vstack((x, s(x))).T

        s = sint.CubicSpline(xyl[:, 0], xyl[:, 1] * thickness_scale)
        xyl = np.vstack((x, s(x))).T

        # Return a single array
        return np.vstack((np.flipud(xyl[1:, :]), xyu))


class AirfoilCST(Airfoil):
    def __init__(
        self,
        name: str,
        upper_coefficients: Sequence[OptimizableFloat],
        lower_coefficients: Sequence[OptimizableFloat],
        n1: OptimizableFloat = 0.5,
        n2: OptimizableFloat = 1.0,
        trailing_edge_thickness: OptimizableFloat = 0.0,
        thickness_factor: OptimizableFloat = 1.0,
        camber_factor: OptimizableFloat = 1.0,
        polar_file: str | Path | None = None,
    ) -> None:
        super().__init__(name, thickness_factor, camber_factor, polar_file)
        self._upper_coefficient_attrs = tuple(
            f"_upper_coefficient_{idx}" for idx, _ in enumerate(upper_coefficients)
        )
        self._lower_coefficient_attrs = tuple(
            f"_lower_coefficient_{idx}" for idx, _ in enumerate(lower_coefficients)
        )
        for attr_name, coefficient in zip(
            self._upper_coefficient_attrs,
            upper_coefficients,
            strict=True,
        ):
            setattr(self, attr_name, coefficient)
        for attr_name, coefficient in zip(
            self._lower_coefficient_attrs,
            lower_coefficients,
            strict=True,
        ):
            setattr(self, attr_name, coefficient)
        self.n1: float = n1
        self.n2: float = n2
        self.trailing_edge_thickness: float = trailing_edge_thickness

    @property
    def airfoil_name(self) -> str:
        return f"CST:{self.name}"

    @property
    def upper_coefficients(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, attr)) for attr in self._upper_coefficient_attrs)

    @property
    def lower_coefficients(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, attr)) for attr in self._lower_coefficient_attrs)

    @property
    def thickness(self) -> float:
        x = normalized_chordwise_points(101, "cosine")
        return float(np.max(self._thickness_distribution(x)))

    @property
    def camber(self) -> float:
        x = normalized_chordwise_points(101, "cosine")
        return float(np.max(self._camber_distribution(x)))

    def _raw_surfaces(
        self,
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        upper = evaluate_cst_surface(
            x,
            self.upper_coefficients,
            n1=self.n1,
            n2=self.n2,
            trailing_edge_thickness_over_c=self.trailing_edge_thickness,
            sign=1.0,
        )
        lower = evaluate_cst_surface(
            x,
            self.lower_coefficients,
            n1=self.n1,
            n2=self.n2,
            trailing_edge_thickness_over_c=self.trailing_edge_thickness,
            sign=-1.0,
        )
        return upper, lower

    def _scaled_surfaces(
        self,
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        upper_raw, lower_raw = self._raw_surfaces(x)
        camber = 0.5 * (upper_raw + lower_raw) * self.camber_factor
        thickness = (upper_raw - lower_raw) * self.thickness_factor
        upper = camber + 0.5 * thickness
        lower = camber - 0.5 * thickness
        return upper, lower

    def _thickness_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        upper, lower = self._scaled_surfaces(x)
        return upper - lower

    def _camber_distribution(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        upper, lower = self._scaled_surfaces(x)
        return 0.5 * (upper + lower)

    def _coordinates(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        upper, lower = self._scaled_surfaces(x)
        xyu = np.column_stack((x, upper))
        xyl = np.column_stack((x, lower))
        return np.vstack((np.flipud(xyl[1:, :]), xyu))


class Section(MADSComponent):
    def __init__(
        self,
        name: str,
        airfoil: Airfoil,
        chord: OptimizableFloat,
        twist: OptimizableFloat,
        options: Sequence[ComponentOptions] | None = None,
        spanwise_y_m: OptimizableFloat | None = None,
        leading_edge_x_m: OptimizableFloat = 0.0,
        leading_edge_z_m: OptimizableFloat = 0.0,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(name, options)
        self.airfoil = airfoil
        self.chord: float = chord
        self.twist: float = twist
        self.spanwise_y_m: float | None = spanwise_y_m
        self.leading_edge_x_m: float = leading_edge_x_m
        self.leading_edge_z_m: float = leading_edge_z_m
        self.metadata: dict[str, object] = {} if metadata is None else dict(metadata)

    @property
    def thickness_to_chord_ratio(self) -> float:
        return self.airfoil.thickness

    @property
    def camber_to_chord_ratio(self) -> float:
        return self.airfoil.camber

    @property
    def thickness(self) -> float:
        return self.airfoil.thickness * self.chord

    @property
    def camber_distribution(self) -> NDArray[np.float64]:
        return self.airfoil.camber_distribution()

    @property
    def rel_chord_position_max_thickness(self) -> float:
        xt = self.airfoil.thickness_distribution()
        max_thickness_index = np.argmax(xt[:, 1])
        return xt[max_thickness_index, 0]

    @property
    def rel_chord_position_max_camber(self) -> float:
        xt = self.airfoil.camber_distribution()
        max_camber_index = np.argmax(xt[:, 1])
        return xt[max_camber_index, 0]


class Span(MADSComponent):
    def __init__(
        self,
        name: str,
        length: OptimizableFloat,
        sweep: OptimizableFloat,
        dihed: OptimizableFloat,
        options: Sequence[ComponentOptions] | None = None,
        start_y_m: OptimizableFloat | None = None,
        end_y_m: OptimizableFloat | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(name, options)
        if start_y_m is not None and end_y_m is not None:
            length = _as_float_value(end_y_m) - _as_float_value(start_y_m)
        self.length: float = length
        self.sweep: float = sweep
        self.dihed: float = dihed
        self.start_y_m: float | None = start_y_m
        self.end_y_m: float | None = end_y_m
        self.metadata: dict[str, object] = {} if metadata is None else dict(metadata)

    def __setattr__(self, name: str, val: object) -> None:
        super().__setattr__(name, val)
        if (
            name in {"start_y_m", "end_y_m"}
            and getattr(self, "start_y_m", None) is not None
            and getattr(self, "end_y_m", None) is not None
        ):
            super().__setattr__(
                "length",
                _as_float_value(self.end_y_m) - _as_float_value(self.start_y_m),
            )


def _as_float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        inner_value = getattr(value, "value", None)
        if inner_value is None:
            raise
        return float(inner_value)


class MovableSurface(MADSComponent):
    def __init__(
        self,
        name: str,
        length: OptimizableFloat,
        pos_start: OptimizableFloatNP,
        pos_end: OptimizableFloatNP,
        ampl: OptimizableFloat,
        options: Sequence[ComponentOptions] | None = None,
    ) -> None:
        super().__init__(name, options)
        self.length: float = length
        self.ampl: float = ampl
        self.pos_start: NDArray[np.float64] = pos_start
        self.pos_end: NDArray[np.float64] = pos_end


class FuelTank(MADSComponent):
    def __init__(
        self,
        name: str,
        structure_mass: OptimizableFloat,
        width: OptimizableFloat = 0.0,
        height: OptimizableFloat = 0.0,
        length: OptimizableFloat = 0.0,
        volume: OptimizableFloat = 0.0,
        fuel_mass: OptimizableFloat = 0.0,
    ) -> None:
        super().__init__(name)
        self.width: float = width  # m
        self.height: float = height  # m
        self.length: float = length  # m
        self.volume: float = volume  # m**3
        self.structure_mass: float = structure_mass  # kg
        self.fuel_mass: float = fuel_mass  # kg


class Wing(MADSComponent):
    def __init__(
        self,
        name: str,
        sections: Sequence[Section],
        spans: Sequence[Span],
        movable_surfaces: Sequence[MovableSurface] | None = None,
        offset: OptimizableFloatNP | None = None,
        scaling: OptimizableFloat = 1.0,
        global_pos: OptimizableFloatNP | None = None,
        mass: OptimizableFloat = 0.0,
        alpha: OptimizableFloat = 0.0,
        beta: OptimizableFloat = 0.0,
        xc_ref: OptimizableFloat = 0.25,
        roll: OptimizableFloat = 0.0,
        structure_mass: OptimizableFloat = 0.0,
        fuel_mass_in_wing: OptimizableFloat = 0.0,  # kg
        portion_laminar_flow: OptimizableFloat = 1.0,  # [0, 1]
        a_cap_root: OptimizableFloat = 10.0e-3,
        a_stringer_root: OptimizableFloat = 1.0e-3,
        y_wing_fuselage_interface: OptimizableFloat = 0.0,
        case_name: str | None = None,
        symmetry: bool = False,
        mirror: bool = False,
        metadata: Mapping[str, object] | None = None,
        children: Sequence[MADSComponent] | None = None,
        options: Sequence[ComponentOptions] | None = None,
    ) -> None:
        super().__init__(name, options)

        movable_surfaces = [] if movable_surfaces is None else movable_surfaces
        offset = np.zeros(3) if offset is None else offset
        global_pos = np.zeros(3) if global_pos is None else global_pos
        children = [] if children is None else children

        self.sections = sections
        self.spans = spans
        self.movable_surfaces: Sequence[MovableSurface] = movable_surfaces
        self.mass: float = mass
        self.offset: NDArray[np.float64] = offset
        self.scaling: float = scaling
        self.mass: float = mass
        self.global_pos: NDArray[np.float64] = global_pos
        self.alpha: float = alpha
        self.beta: float = beta
        self.xc_ref: float = xc_ref
        self.roll: float = roll
        self.structure_mass: float = structure_mass
        self.fuel_mass_in_wing: float = fuel_mass_in_wing
        self.portion_laminar_flow: float = portion_laminar_flow
        self.a_cap_root: float = a_cap_root
        self.a_stringer_root: float = a_stringer_root
        self.y_wing_fuselage_interface: float = y_wing_fuselage_interface
        self.case_name = case_name
        self.symmetry = symmetry
        self.mirror = mirror
        self.metadata: dict[str, object] = {} if metadata is None else dict(metadata)
        self.children: Sequence[str] = [c.name for c in children]
        self.geometry_state = None
        self.export_state = None
        self.packaging_state = None

    def _overall_wing_data(self) -> tuple[float, float, float, float, float]:
        # initialise
        sum_all_mean_chords_weighted = 0.0
        sum_all_areas = 0.0
        overall_semi_span = 0.0
        sum_all_thick_to_chord_ratios_weighted = 0.0
        sum_all_rel_chord_position_max_thickness = 0.0
        mean_chord_span = 0.0

        for i, span in enumerate(self.spans):
            # obtain thickness to chord ratio of each span
            thick_to_chord_ratio_inner = self.sections[i].thickness_to_chord_ratio
            thick_to_chord_ratio_outer = self.sections[i + 1].thickness_to_chord_ratio

            # for each span: obtain  relative position of maximum thickness
            rel_chord_position_max_thickness_inner = self.sections[
                i
            ].rel_chord_position_max_thickness
            rel_chord_position_max_thickness_outer = self.sections[
                i + 1
            ].rel_chord_position_max_thickness

            # calculate area, taper ratio, and reference chord (MAC) of each span
            chord_section_inner = self.sections[i].chord
            chord_section_outer = self.sections[i + 1].chord
            area_span = 0.5 * span.length * (chord_section_inner + chord_section_outer)
            taper_ratio_span = chord_section_outer / chord_section_inner

            # get length of each semi span
            overall_semi_span += span.length

            # calculate MAC for each partition
            mean_chord_span = (
                (2.0 / 3.0)
                * chord_section_inner
                * (1.0 + taper_ratio_span + taper_ratio_span**2)
                / (1.0 + taper_ratio_span)
            )

            # calculate mean thickness to chord ratio of the span
            thick_to_chord_ratio_span = (
                thick_to_chord_ratio_outer + thick_to_chord_ratio_inner
            ) / 2

            # compute area weighted average of the thickness to chord ratio
            sum_all_thick_to_chord_ratios_weighted += (
                thick_to_chord_ratio_span * area_span
            )

            rel_chord_position_max_thickness_span = (
                rel_chord_position_max_thickness_outer
                + rel_chord_position_max_thickness_inner
            ) / 2

            # area-weighted average of the relative positions of maximum thickness
            sum_all_rel_chord_position_max_thickness += (
                rel_chord_position_max_thickness_span * area_span
            )

            # compute area weighted average of the mean aerodynamic chord (MAC)
            sum_all_mean_chords_weighted += mean_chord_span * area_span
            sum_all_areas += area_span

        return (
            sum_all_mean_chords_weighted,
            sum_all_areas,
            2 * overall_semi_span,
            sum_all_thick_to_chord_ratios_weighted,
            sum_all_rel_chord_position_max_thickness,
        )

    @property
    def mac(self) -> float:
        chords_weighted, areas, _, _, _ = self._overall_wing_data()
        return chords_weighted / areas

    @property
    def overall_thick_to_chord_ratio(self) -> float:
        _, areas, _, thick_to_chord_ratios_weighted, _ = self._overall_wing_data()
        return thick_to_chord_ratios_weighted / areas

    @property
    def overall_rel_chord_position_max_thick(self) -> float:
        (
            _,
            areas,
            _,
            _,
            sum_all_rel_chord_position_max_thickness,
        ) = self._overall_wing_data()
        return sum_all_rel_chord_position_max_thickness / areas

    @property
    def aspect_ratio(self) -> float:
        _, areas, span, _, _ = self._overall_wing_data()
        return span**2 / (2.0 * areas)

    @property
    def span(self) -> float:
        return 2 * sum(s.length for s in self.spans)

    # def mean_sweep_angle(self, x_nondim: float = 0.25) -> float:
    #     """
    #     Returns the mean sweep angle (in degrees) of the wing, relative to the x-axis.
    #     Positive sweep is backwards, negative sweep is forward.

    #     This is purely measured from root to tip, with no consideration for the sweep of the individual
    #     cross-sections in between.

    #     Args:

    #         x_nondim: The nondimensional x-coordinate of the cross-section to use for sweep angle computation.

    #             * If you provide 0, it will use the leading edge of the cross-section.

    #             * If you provide 0.25, it will use the quarter-chord point of the cross-section.

    #             * If you provide 1, it will use the trailing edge of the cross-section.

    #     Returns:

    #         The mean sweep angle, in degrees.
    #     """
    #     root_quarter_chord = self._compute_xyz_of_WingXSec(
    #         0, x_nondim=x_nondim, z_nondim=0
    #     )
    #     tip_quarter_chord = self._compute_xyz_of_WingXSec(
    #         -1, x_nondim=x_nondim, z_nondim=0
    #     )

    #     vec = tip_quarter_chord - root_quarter_chord
    #     vec_norm = vec / np.linalg.norm(vec)

    #     sin_sweep = vec_norm[0]  # from dot product with x_hat

    #     # sweep_deg = arcsind(sin_sweep)

    #     return sweep_deg

    @property
    def overall_sweep(self) -> float:
        """Return the mean sweep angle (in degrees) of the wing.

        Relative to the x-axis, positive sweep is backwards, negative sweep is forward.
        """
        # initialize
        sum_all_sweeps_weighted = 0.0

        # loop through all spans of the wing
        for span in self.spans:
            # compute sum of span weighted sweep angles
            sum_all_sweeps_weighted += span.sweep * span.length

        # divide sum of span weighted sweeps by total span to obtain overall span weighted sweep angle
        return sum_all_sweeps_weighted / self.span

    def sweep_at_chord_station(self, x_nondim: float = 0.25) -> float:
        """Return the sweep angle (in degrees) of the wing at the desired station.

        Positive sweep is backwards, negative sweep is forward.
        """
        # compute overall sweep
        overall_sweep = self.overall_sweep

        # convert sweep angle to provided x_nondim
        # "Designing Unmanned Aircraft Systems: A comprehensive Approach", p.89
        return overall_sweep + np.arctan(
            (4 / self.aspect_ratio)
            * (self.xc_ref - x_nondim)
            * (1 - self.taper_ratio)
            / (1 + self.taper_ratio),
        )

    @property
    def mean_sweep_at_max_thick_lines(self) -> float:
        """Return a mean sweep angle (in degrees) of the wing,.

        Relative to the x-axis; this sweep angle is a span-weighted
        sum of the sweeps given by the maximum thickness line of each span.
        Positive sweep is backwards, negative sweep is forward.

        Assumptions:
        - the "span" is the projected span; NOT the span along a swept line of the wing.
        - no twist is considered
        - no dihedral is considered
        - to include/clarify the above parameters, we first need to define how they are
          defined in the parameterization
        """
        # initialize
        sum_all_sweeps_weighted = 0.0

        for i, span in enumerate(self.spans):
            # relative position of maximum thickness and chord length of each span
            rel_pos_max_thick_inner = self.sections[i].rel_chord_position_max_thickness
            rel_pos_max_thick_outer = self.sections[
                i + 1
            ].rel_chord_position_max_thickness
            chord_section_inner = self.sections[i].chord
            chord_section_outer = self.sections[i + 1].chord

            # calculate area, aspect ratio and taper ratio of each span
            area_span = 0.5 * span.length * (chord_section_inner + chord_section_outer)
            taper_ratio_this_span = chord_section_outer / chord_section_inner
            aspect_ratio_this_span = span.length**2 / area_span

            # convert sweep angle of this span from specified to xc_ref position LE
            # "Designing Unmanned Aircraft Systems: A comprehensive Approach", p.89
            sweep_at_le_this_span = span.sweep + np.arctan(
                (4 / aspect_ratio_this_span)
                * (self.xc_ref - 0)  # 0 -> LE
                * (1 - taper_ratio_this_span)
                / (1 + taper_ratio_this_span),
            )

            # sweep of maximum thickness line of this span
            sweep_max_thick_this_span = np.arctan(
                -rel_pos_max_thick_inner
                + span.length * np.arctan(sweep_at_le_this_span)
                + rel_pos_max_thick_outer,
            )

            # compute sum of span weighted sweep angles
            sum_all_sweeps_weighted += sweep_max_thick_this_span * span.length

        return sum_all_sweeps_weighted / self.span

    @property
    def area(self) -> float:
        # recover component area from indipendent design variables
        aspect_ratio = self.aspect_ratio
        overall_span = self.span
        return overall_span**2 / aspect_ratio

    @property
    def wetted_area(self) -> float:
        return 2 * self.area

    @property
    def taper_ratio(self) -> float:
        chord_tip = self.sections[-1].chord
        chord_root = self.sections[0].chord
        return chord_tip / chord_root


class Blade(Wing):
    pass


class Propeller(MADSComponent):
    def __init__(
        self,
        name: str,
        blade: Blade,
        r_tip: OptimizableFloat,
        n_blades: OptimizableInt,
        pitch: OptimizableFloat = 0.0,
        rpm: OptimizableFloat = 0.0,
        reverse: bool = False,
        global_pos: OptimizableFloatNP | None = None,
        mass: OptimizableFloat = 0.0,
        alpha: OptimizableFloat = 0.0,
        beta: OptimizableFloat = 0.0,
        thrust: OptimizableFloat = 0.0,  # TODO @Simone: should be removed
        options: Sequence[ComponentOptions] | None = None,
    ) -> None:
        super().__init__(name, options)

        global_pos = np.zeros(3) if global_pos is None else global_pos

        self.blade = blade
        self.r_tip: float = r_tip
        self.n_blades: int = n_blades
        self.pitch: float = pitch
        self.rpm: float = rpm
        self.reverse = reverse
        self.global_pos: NDArray[np.float64] = global_pos
        self.mass: float = mass
        self.alpha: float = alpha
        self.beta: float = beta
        self.thrust: float = thrust

    @property
    def length(self) -> float:
        return sum(s.length for s in self.blade.spans) * self.blade.scaling

    @property
    def hub_offset(self) -> float:
        return self.r_tip - self.length

    def local_velocity(self, i: int, speed: float = 0.0) -> float:
        r_hub = self.r_tip - self.blade.span / 2.0
        pos = r_hub + sum(s.length for s in self.blade.spans[: i - 1])
        speed_t = pos * self.rpm * np.pi / 30.0
        return np.sqrt(speed_t**2 + speed**2)


class PropulsionSystem(MADSComponent):
    def __init__(
        self,
        name: str,
        type: str,  # allowed types: ['hydrogen','battery','fuelcell','kerosene-hydrogen','kerosene-battery','kerosene-fuelcell','hydrogen-battery','hydrogen-fuelcell','fuelcell-battery']
        power: OptimizableFloat = 0.0,
        shaft_power: OptimizableFloat = 0.0,
        electric_engine_power: OptimizableFloat = 0.0,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self.type = type
        self.power: float = power
        self.shaft_power: float = shaft_power
        self.electric_engine_power: float = electric_engine_power
        self.options = options or {}

class PowerManagSyst(MADSComponent):
    def __init__(
        self,
        name: str,
        type: str,  # allowed types: ['hydrogen','battery','fuelcell','kerosene-hydrogen','kerosene-battery','kerosene-fuelcell','hydrogen-battery','hydrogen-fuelcell','fuelcell-battery']
        hybrid_factor_engine: OptimizableFloat = 1.0,  # combustion to electric hybridization - set to 0 if pure electric
        hybrid_factor_electric_power: OptimizableFloat = 1.0,  # battery to fuel cell hybridazazion - set to 1 to have ful battery
        hybrid_factor_power: OptimizableFloat = 1.0,  # global hybridzytion - set to 0 if only combustion engine
        total_electrical_power: OptimizableFloat = 0.0,
        total_combustion_power: OptimizableFloat = 0.0,
        electrical_power_from_fc: OptimizableFloat = 0.0,
        electrical_power_from_battery: OptimizableFloat = 0.0,
    ) -> None:
        super().__init__(name)
        self.type = type
        self.hybrid_factor_engine: float = hybrid_factor_engine
        self.hybrid_factor_electric_power: float = hybrid_factor_electric_power
        self.hybrid_factor_power: float = hybrid_factor_power
        self.total_electrical_power: float = total_electrical_power
        self.total_combustion_power: float = total_combustion_power
        self.electrical_power_from_fc: float = electrical_power_from_fc
        self.electrical_power_from_battery: float = electrical_power_from_battery


class FlightControlSystem(MADSComponent):
    def __init__(self, name: str, mass: OptimizableFloat) -> None:
        super().__init__(name)
        self.mass: float = mass


class Fuselage(MADSComponent):
    def __init__(
        self,
        name: str,
        wetted_area: OptimizableFloat,  # m**2
        maximum_width: OptimizableFloat,  # m
        maximum_height: OptimizableFloat,  # m
        length: OptimizableFloat,  # m
        global_pos: OptimizableFloatNP | None = None,
        volume_pressurized_cabin: OptimizableFloat = 1.0,  # m**3
        mass: OptimizableFloat = 0.0,  # kg
        maximum_fuselage_pressure_differential: OptimizableFloat = 0.0,  # Pa
        portion_laminar_flow: OptimizableFloat = 1.0,
    ) -> None:
        super().__init__(name)

        global_pos = np.zeros(3) if global_pos is None else global_pos

        self.wetted_area: float = wetted_area
        self.maximum_width: float = maximum_width
        self.maximum_height: float = maximum_height
        self.length: float = length
        self.global_pos: NDArray[np.float64] = global_pos
        self.mass: float = mass
        self.volume_pressurized_cabin: float = volume_pressurized_cabin
        self.maximum_fuselage_pressure_differential: float = (
            maximum_fuselage_pressure_differential
        )
        self.portion_laminar_flow: float = portion_laminar_flow


class Nacelle(MADSComponent):
    def __init__(
        self,
        name: str,
        wetted_area: OptimizableFloat,  # nacelle wetted area in [m**2]
        maximum_width: OptimizableFloat,  # maximum width of the nacelle in [m]
        maximum_height: OptimizableFloat,  # maximum height of the nacelle in [m]
        length: OptimizableFloat,  # length of the nacelle in [m]
        mass: OptimizableFloat,  # nacelle mass in [kg]
        portion_laminar_flow: OptimizableFloat,  # fraction of laminar flow to turbulent flow on the nacelle [0, 1]
    ) -> None:
        super().__init__(name)
        self.wetted_area: float = wetted_area
        self.maximum_width: float = maximum_width
        self.maximum_height: float = maximum_height
        self.length: float = length
        self.mass: float = mass
        self.portion_laminar_flow: float = portion_laminar_flow


class ThermalSystem(MADSComponent):
    def __init__(
        self,
        name: str,
        fuel_tank_volume: OptimizableFloat,
        q_total: OptimizableFloat,  # W
    ) -> None:
        super().__init__(name)
        self.fuel_tank_volume: float = fuel_tank_volume
        self.q_total: float = q_total


class ElectricEngine(MADSComponent):
    def __init__(
        self,
        name: str,
        mass: OptimizableFloat,  # engine mass in [kg]
    ) -> None:
        super().__init__(name)
        self.mass: float = mass


class CombustionEngine(MADSComponent):
    def __init__(
        self,
        name: str,
        mass: OptimizableFloat,  # engine mass in [kg]
    ) -> None:
        super().__init__(name)
        self.mass: float = mass


class ElectronicSpeedController(MADSComponent):
    def __init__(
        self,
        name: str,
        mass: OptimizableFloat,  # engine mass in [kg]
    ) -> None:
        super().__init__(name)
        self.mass: float = mass


class Configuration(MADSComponent):
    def __init__(
        self,
        name: str,
        m_tom: OptimizableFloat,  # maximum takeoff mass in [kg]
        ultimate_load_factor: OptimizableFloat,  # ultimate load factor [-]
    ) -> None:
        super().__init__(name)
        self.m_tom: float = m_tom
        self.ultimate_load_factor: float = ultimate_load_factor


# [TODO][All] Place holder for any deviation into the assembly :(
class AllVariables(MADSComponent):
    def __init__(
        self,
        name: str,
        rotor_mass: OptimizableFloat,
    ) -> None:
        super().__init__(name)
        self.rotor_mass: float = rotor_mass


class Aircraft(MADSComponent):
    def __init__(
        self,
        name: str,
        global_pos: OptimizableFloatNP | None = None,
        mass: OptimizableFloat = 0.0,
        l_ref: float = 1.0,  # reference length [m]
        s_ref: float = 1.0,  # reference surface area [m^2]
    ) -> None:
        super().__init__(name)
        self.mass = mass
        self.global_pos = np.zeros(3) if global_pos is None else global_pos
        self.l_ref = l_ref
        self.s_ref = s_ref


class Battery(MADSComponent):
    def __init__(
        self,
        name: str,
        power: OptimizableFloat,  # W
        nominal_voltage: OptimizableFloat,
        flight_time: OptimizableFloat,
        mass: OptimizableFloat = 0.0,
        global_pos: OptimizableFloatNP | None = None,
    ) -> None:
        super().__init__(name)
        self.power: float = power
        self.nominal_voltage: float = nominal_voltage
        self.flight_time: float = flight_time
        self.mass: float = mass
        self.global_pos = np.zeros(3) if global_pos is None else global_pos


class FuelCell(MADSComponent):
    def __init__(
        self,
        name: str,
        power: OptimizableFloat,  # fc power [W]
        waste_heat: OptimizableFloat,  # Waste heat produced
        h2_fuel_flow: OptimizableFloat,  # h2 fuel flow
        mass: OptimizableFloat = 0.0,
        global_pos: OptimizableFloatNP | None = None,
    ) -> None:
        super().__init__(name)
        self.power: float = power
        self.waste_heat: float = waste_heat
        self.h2_fuel_flow: float = h2_fuel_flow
        self.mass: float = mass
        self.global_pos = np.zeros(3) if global_pos is None else global_pos


class PointMass(MADSComponent):
    def __init__(
        self,
        name: str,
        mass: OptimizableFloat = 0.0,
        global_pos: OptimizableFloatNP | None = None,
    ) -> None:
        super().__init__(name)

        global_pos = np.zeros(3) if global_pos is None else global_pos

        self.mass: float = mass
        self.global_pos: NDArray[np.float64] = global_pos
