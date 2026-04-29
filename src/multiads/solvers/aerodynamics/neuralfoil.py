from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aerosandbox as asb
import numpy as np

from multiads.assembly import (
    Airfoil,
    AirfoilFile,
    AirfoilNACA4,
    Environment,
    Propeller,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario.polars import POLAR_DEFAULT_AOA, PolarVariable
from multiads.solvers import BaseSolver, SolverOptions

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
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.aoa = aoa


class Neuralfoil(BaseSolver):
    AIRFOIL = asb.Airfoil

    def __init__(self, options: Options) -> None:
        super().__init__()
        self.options: Options = options
        self.environment: Environment | None = None
        self.wings: Sequence[Wing] | None = None
        self.propellers: Sequence[Propeller] | None = None
        self.polars: dict[str, PolarVariable] | None = None

        self.alphas = options.aoa if options.aoa is not None else POLAR_DEFAULT_AOA

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

    def _update_wing_io(
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
            outputs[sec.name] = PolarVariable.from_num_points(f"{sec.name}.polar")

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
                    self.polars[sec.name],
                )

        for prop in self.propellers:
            for i, sec in enumerate(prop.blade.sections):
                airfoil = self.make_airfoil(sec.airfoil)
                speed = prop.local_velocity(i, speed)
                mach = speed / sound_speed
                reynolds = speed * sec.chord / kin_viscosity
                self._compute_airfoil(
                    airfoil,
                    self.alphas,
                    mach,
                    reynolds,
                    self.polars[sec.name],
                )

    def compute_output(self) -> None:
        # `self.outputs` was already updated in `_run`
        if self.outputs is None:
            msg = f"The outputs of solver '{type(self).__name__} are not initialized"
            raise RuntimeError(msg)

    @classmethod
    def make_airfoil(cls, airfoil: Airfoil) -> AIRFOIL:
        if type(airfoil) is AirfoilNACA4:
            return cls.AIRFOIL(name=airfoil.airfoil_name.lower())
        if type(airfoil) is AirfoilFile:
            return cls.AIRFOIL(name=airfoil.airfoil_name, coordinats=airfoil.filename)
        msg = (
            f"Airfoils of type '{type(airfoil).__name__}' are not supported "
            f"by '{cls.__name__}'."
        )
        raise RuntimeError(msg)

    @classmethod
    def _compute_airfoil(
        cls,
        airfoil: AIRFOIL,
        alphas: NDArray[np.float64],
        mach: float,
        reynolds: float,
        polar: PolarVariable,
    ) -> None:
        out = airfoil.get_aero_from_neuralfoil(
            alpha=alphas,
            Re=reynolds,
            mach=mach,
            model_size="xxxlarge",  # The largest model is still fast
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
