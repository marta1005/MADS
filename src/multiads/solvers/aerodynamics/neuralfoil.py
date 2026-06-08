from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aerosandbox as asb
import numpy as np

from multiads.assembly import (
    Airfoil,
    AirfoilNACA4,
    Environment,
    Propeller,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario.polars import POLAR_DEFAULT_AOA, PolarVariable
from multiads.solvers import BaseSolver, SolverOptions
from multiads.solvers.aerodynamics import SectionOptions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.assembly import MADSComponent
    from multiads.scenario import BaseVariable


class Options(SolverOptions):
    def __init__(
        self,
        *,
        aoa: NDArray[np.float64] | None = None,
        model: str = "xxxlarge",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.aoa = aoa
        self.model = model


class Neuralfoil(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.environment: Environment | None = None
        self.wings: Sequence[Wing] | None = None
        self.propellers: Sequence[Propeller] | None = None
        self.polars: dict[str, PolarVariable] | None = None

        aoa = self.options.aoa
        self.alphas = aoa if aoa is not None else POLAR_DEFAULT_AOA

    @staticmethod
    def station_coordinates(station: Any) -> NDArray[np.float64]:
        """Return normalized airfoil coordinates from a resolved geometry station."""

        order = np.argsort(np.asarray(station.x_over_c, dtype=float))
        x = np.asarray(station.x_over_c, dtype=float)[order]
        upper = np.asarray(station.upper_z_over_c, dtype=float)[order]
        lower = np.asarray(station.lower_z_over_c, dtype=float)[order]
        upper_coords = np.column_stack((x[::-1], upper[::-1]))
        lower_coords = np.column_stack((x[1:], lower[1:]))
        return np.vstack((upper_coords, lower_coords))

    @staticmethod
    def local_alpha_deg(
        station: Any,
        *,
        global_alpha_deg: float,
        mode: str = "global-plus-twist",
    ) -> float:
        """Return the 2D airfoil angle used for a resolved station."""

        twist = float(getattr(station, "twist_deg", 0.0))
        if mode == "global":
            return float(global_alpha_deg)
        if mode == "global-plus-twist":
            return float(global_alpha_deg + twist)
        if mode == "global-minus-twist":
            return float(global_alpha_deg - twist)
        msg = f"Unknown profile alpha mode: {mode}"
        raise ValueError(msg)

    @classmethod
    def estimate_profile_drag_from_resolved_stations(
        cls,
        stations: Sequence[Any],
        environment: Environment,
        *,
        s_ref_m2: float,
        y_min_m: float = 0.0,
        alpha_mode: str = "global-plus-twist",
        model: str = "large",
        n_crit: float = 9.0,
        station_stride: int = 1,
        metric_prefix: str = "neuralfoil",
    ) -> dict[str, float]:
        """Estimate an integrated profile-drag contribution from resolved stations.

        This is solver-level postprocessing, not a CTA-specific model. The caller
        decides which spanwise region is meaningful by setting ``y_min_m`` and
        the metric prefix used in the returned dictionary.
        """

        selected = [
            station
            for station in stations
            if float(station.spanwise_y_m) >= max(0.0, float(y_min_m)) - 1.0e-9
        ]
        selected = sorted(selected, key=lambda station: float(station.spanwise_y_m))
        if not selected:
            return cls.zero_profile_drag_metrics(metric_prefix, y_min_m)

        stride = max(1, int(station_stride))
        if stride > 1 and len(selected) > 2:
            sampled = selected[::stride]
            if sampled[-1] is not selected[-1]:
                sampled.append(selected[-1])
            selected = sampled

        y = np.asarray([float(station.spanwise_y_m) for station in selected], dtype=float)
        chord = np.asarray([float(station.chord_m) for station in selected], dtype=float)
        speed = float(environment.speed)
        mach = float(environment.mach)
        reynolds = speed * chord / float(environment.kin_viscosity)

        cd = np.zeros_like(y)
        cl = np.zeros_like(y)
        cm = np.zeros_like(y)
        alpha_local = np.zeros_like(y)
        for idx, station in enumerate(selected):
            alpha_local[idx] = cls.local_alpha_deg(
                station,
                global_alpha_deg=float(environment.alpha),
                mode=alpha_mode,
            )
            airfoil = asb.Airfoil(
                name=f"{station.name}",
                coordinates=cls.station_coordinates(station),
            )
            aero = cls.compute_aero_from_airfoil(
                airfoil,
                alphas=float(alpha_local[idx]),
                mach=mach,
                reynolds=float(reynolds[idx]),
                model=model,
                n_crit=float(n_crit),
                include_360_deg_effects=True,
            )
            cd[idx] = float(np.ravel(aero["CD"])[0])
            cl[idx] = float(np.ravel(aero["CL"])[0])
            cm[idx] = float(np.ravel(aero["CM"])[0])

        cd_profile = 2.0 * float(np.trapezoid(cd * chord, y)) / float(s_ref_m2)
        q_pa = 0.5 * float(environment.density) * speed**2
        drag_n = q_pa * float(s_ref_m2) * cd_profile
        return {
            f"{metric_prefix}_profile_y_min_m": float(y_min_m),
            f"{metric_prefix}_profile_cd": cd_profile,
            f"{metric_prefix}_profile_drag_n": drag_n,
            f"{metric_prefix}_station_count": float(len(selected)),
            f"{metric_prefix}_section_cd_min": float(np.min(cd)),
            f"{metric_prefix}_section_cd_mean": float(np.mean(cd)),
            f"{metric_prefix}_section_cd_max": float(np.max(cd)),
            f"{metric_prefix}_section_cl_min": float(np.min(cl)),
            f"{metric_prefix}_section_cl_mean": float(np.mean(cl)),
            f"{metric_prefix}_section_cl_max": float(np.max(cl)),
            f"{metric_prefix}_section_cm_mean": float(np.mean(cm)),
            f"{metric_prefix}_re_min": float(np.min(reynolds)),
            f"{metric_prefix}_re_mean": float(np.mean(reynolds)),
            f"{metric_prefix}_re_max": float(np.max(reynolds)),
            f"{metric_prefix}_alpha_local_min_deg": float(np.min(alpha_local)),
            f"{metric_prefix}_alpha_local_mean_deg": float(np.mean(alpha_local)),
            f"{metric_prefix}_alpha_local_max_deg": float(np.max(alpha_local)),
        }

    @staticmethod
    def zero_profile_drag_metrics(
        metric_prefix: str = "neuralfoil",
        y_min_m: float = 0.0,
    ) -> dict[str, float]:
        """Return zero-valued profile-drag metrics with the standard keys."""

        return {
            f"{metric_prefix}_profile_y_min_m": float(y_min_m),
            f"{metric_prefix}_profile_cd": 0.0,
            f"{metric_prefix}_profile_drag_n": 0.0,
            f"{metric_prefix}_station_count": 0.0,
            f"{metric_prefix}_section_cd_min": 0.0,
            f"{metric_prefix}_section_cd_mean": 0.0,
            f"{metric_prefix}_section_cd_max": 0.0,
            f"{metric_prefix}_section_cl_min": 0.0,
            f"{metric_prefix}_section_cl_mean": 0.0,
            f"{metric_prefix}_section_cl_max": 0.0,
            f"{metric_prefix}_section_cm_mean": 0.0,
            f"{metric_prefix}_re_min": 0.0,
            f"{metric_prefix}_re_mean": 0.0,
            f"{metric_prefix}_re_max": 0.0,
            f"{metric_prefix}_alpha_local_min_deg": 0.0,
            f"{metric_prefix}_alpha_local_mean_deg": 0.0,
            f"{metric_prefix}_alpha_local_max_deg": 0.0,
        }

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        # Filter components
        _components = copy_components(components)
        components_flat = flatten_components(_components)
        self.wings = [c for c in components_flat if type(c) is Wing]
        self.propellers = [c for c in components_flat if type(c) is Propeller]

        try:
            self.environment = next(
                c for c in components_flat if type(c) is Environment
            )
        except StopIteration:
            msg = f"An environment must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        # Inputs and outputs
        self.inputs: list[BaseVariable] = []
        self.polars = {}

        for wing in self.wings:
            self._update_wing_io(wing, self.inputs, self.polars)

        for prop in self.propellers:
            self._update_wing_io(prop.blade, self.inputs, self.polars)

        self.outputs = list(self.polars.values())

        return [*self.wings, *self.propellers]

    def _update_wing_io(  # noqa: C901
        self,
        wing: Wing,
        inputs: list[BaseVariable],
        outputs: dict[str, PolarVariable],
    ) -> None:
        for sec in wing.sections:
            # Design variables
            if v := sec.variables.get("chord"):
                inputs.append(v)

            foil = sec.airfoil

            if v := foil.variables.get("thickness_factor"):
                inputs.append(v)
            if v := foil.variables.get("camber_factor"):
                inputs.append(v)

            if type(foil) is AirfoilNACA4:
                if v := foil.variables.get("m"):
                    inputs.append(v)
                if v := foil.variables.get("p"):
                    inputs.append(v)
                if v := foil.variables.get("t"):
                    inputs.append(v)

            # Allocate polars
            try:
                opts = next(o for o in sec.options if type(o) is SectionOptions)
                if opts.polar:
                    if opts.polar_length != len(self.alphas):
                        msg = (
                            f"'SectionOptions' of section '{sec.name}' have a "
                            "different polar length than specified in 'Neuralfoil'"
                        )
                        raise ValueError(msg)

                    outputs[sec.name] = PolarVariable.from_num_points(
                        f"{sec.name}.polar",
                        len(self.alphas),
                    )

            except StopIteration:
                msg = (
                    f"Cannot compute polars of section '{sec.name}' "
                    "as it has no 'SectionOptions'"
                )
                raise RuntimeError(msg) from None

    def _run(self) -> None:
        if (
            self.environment is None
            or self.polars is None
            or self.wings is None
            or self.propellers is None
        ):
            msg = f"The solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        speed = self.environment.speed
        sound_speed = self.environment.sound_speed
        kin_viscosity = self.environment.kin_viscosity

        for wing in self.wings:
            for sec in wing.sections:
                airfoil = self.make_airfoil(sec.airfoil)
                mach = speed / sound_speed
                reynolds = speed * sec.chord / kin_viscosity
                self._compute_airfoil(
                    airfoil,
                    self.alphas,
                    mach,
                    reynolds,
                    self.options.model,
                    self.polars[sec.name],
                )

        for prop in self.propellers:
            for i, sec in enumerate(prop.blade.sections):
                airfoil = self.make_airfoil(sec.airfoil)
                local_speed = prop.local_velocity(i, speed)
                mach = local_speed / sound_speed
                reynolds = local_speed * sec.chord / kin_viscosity
                self._compute_airfoil(
                    airfoil,
                    self.alphas,
                    mach,
                    reynolds,
                    self.options.model,
                    self.polars[sec.name],
                )

    def compute_output(self) -> None:
        # `self.outputs` was already updated in `_run`
        if self.outputs is None:
            msg = f"The outputs of solver '{type(self).__name__} are not initialized"
            raise RuntimeError(msg)

    @classmethod
    def make_airfoil(cls, airfoil: Airfoil) -> asb.Airfoil:
        coords = airfoil.coordinates()
        coords = np.flipud(coords)
        return asb.Airfoil(name=airfoil.airfoil_name, coordinates=coords)

    @classmethod
    def compute_aero_from_coordinates(
        cls,
        *,
        name: str,
        coordinates: NDArray[np.float64],
        alphas: NDArray[np.float64] | float,
        mach: float,
        reynolds: float,
        model: str = "large",
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        include_360_deg_effects: bool = True,
    ) -> dict[str, NDArray[np.float64] | float]:
        """Evaluate NeuralFoil for explicit airfoil coordinates."""

        airfoil = asb.Airfoil(name=name, coordinates=np.asarray(coordinates, dtype=float))
        return cls.compute_aero_from_airfoil(
            airfoil,
            alphas=alphas,
            mach=mach,
            reynolds=reynolds,
            model=model,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            include_360_deg_effects=include_360_deg_effects,
        )

    @classmethod
    def compute_aero_from_airfoil(
        cls,
        airfoil: asb.Airfoil,
        *,
        alphas: NDArray[np.float64] | float,
        mach: float,
        reynolds: float,
        model: str = "large",
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        include_360_deg_effects: bool = True,
    ) -> dict[str, NDArray[np.float64] | float]:
        """Evaluate NeuralFoil for an AeroSandbox airfoil."""

        return airfoil.get_aero_from_neuralfoil(
            alpha=alphas,
            Re=reynolds,
            mach=mach,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            model_size=model,
            include_360_deg_effects=include_360_deg_effects,
        )

    @classmethod
    def _compute_airfoil(
        cls,
        airfoil: asb.Airfoil,
        alphas: NDArray[np.float64],
        mach: float,
        reynolds: float,
        model: str,
        polar: PolarVariable,
    ) -> None:
        out = cls.compute_aero_from_airfoil(
            airfoil,
            alphas=alphas,
            reynolds=reynolds,
            mach=mach,
            model=model,
            include_360_deg_effects=True,
        )
        polar.mach = np.array([mach])
        polar.reynolds = np.array([reynolds])
        polar.aoa = alphas
        polar.cl = np.asarray(out["CL"])
        polar.cd = np.asarray(out["CD"])
        polar.cm = np.asarray(out["CM"])

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
