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
