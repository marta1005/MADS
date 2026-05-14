from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from multiads.assembly import (
    Environment,
    Fuselage,
    MADSComponent,
    Nacelle,
    Wing,
    Aircraft,
    copy_components,
    flatten_components,
)
from multiads.scenario import InnerVariableFloat
from multiads.scenario.span_loads import (
    SPANLOAD_DEFAULT_NUM_STATIONS,
    SpanLoadsVariable,
)
from multiads.solvers import BaseSolver
from multiads.solvers.aerodynamics.aero_low_fidelity_lib import (
    Driver,
    Options,
)
from multiads.solvers.aerodynamics.loads_aggregator import SpanloadsOptions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.scenario import BaseVariable


class AeroLowFidelity(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.environment: Environment | None = None
        self.wings: Sequence[Wing] | None = None
        self.fuselages: Sequence[Fuselage] | None = None
        self.nacelles: Sequence[Nacelle] | None = None
        self.aircraft: Aircraft | None = None
        self.outputs_map: dict[str, InnerVariableFloat | SpanLoadsVariable] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        _components = copy_components(components)
        comp_flat = flatten_components(_components)

        try:
            self.environment = next(
                c for c in comp_flat if type(c) is Environment
            )
        except StopIteration:
            msg = f"An environment must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        self.wings = [c for c in comp_flat if type(c) is Wing]
        self.fuselages = [c for c in comp_flat if type(c) is Fuselage]
        self.nacelles = [c for c in comp_flat if type(c) is Nacelle]

        inputs: list[BaseVariable] = []

        if v := self.environment.variables.get("speed"):
            inputs.append(v)
        if v := self.environment.variables.get("density"):
            inputs.append(v)
        if v := self.environment.variables.get("alpha"):
            inputs.append(v)

        wing_like: list[MADSComponent] = []
        outputs: list[InnerVariableFloat | SpanLoadsVariable] = []

        for wing in self.wings:
            if v := wing.variables.get("alpha"):
                inputs.append(v)
            if v := wing.variables.get("cd0"):
                inputs.append(v)

            for sec in wing.sections:
                if v := sec.variables.get("chord"):
                    inputs.append(v)
                if v := sec.variables.get("twist"):
                    inputs.append(v)

            for span in wing.spans:
                if v := span.variables.get("length"):
                    inputs.append(v)
                if v := span.variables.get("sweep"):
                    inputs.append(v)
                if v := span.variables.get("dihed"):
                    inputs.append(v)

            wing_like.append(wing)

            opts = wing.options
            spanloads_opts = next(
                (o for o in opts if type(o) is SpanloadsOptions), None
            )

            if spanloads_opts:
                for load_name, load_size in spanloads_opts.loads.items():
                    outputs.append(
                        SpanLoadsVariable.from_num_stations(
                            f"{wing.name}.{load_name}",
                            load_size,
                        )
                    )

        for fuselage in self.fuselages:
            wing_like.append(fuselage)

        for nacelle in self.nacelles:
            wing_like.append(nacelle)

        if self.wings:
            for wing in self.wings:
                outputs.extend(
                   [
                            InnerVariableFloat(f"{wing.name}.lift", 0.0),
                            InnerVariableFloat(f"{wing.name}.drag", 0.0),
                            InnerVariableFloat(f"{wing.name}.efficiency", 0.0),
                            InnerVariableFloat(f"{wing.name}.my", 0.0),
                    ]
                )

        self.inputs = inputs
        self.outputs = outputs
        self.outputs_map = {v.name: v for v in outputs}

        return [*self.wings, *self.fuselages, *self.nacelles]

    def _run(self) -> None:
        if (
            self.environment is None
            or self.wings is None
        ):
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        self.driver = Driver(
            environment=self.environment,
            wings=self.wings,
            fuselages=self.fuselages,
            nacelles=self.nacelles,
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The outputs of solver '{type(self).__name__}' are not initialized"
            raise RuntimeError(msg)

        total_lift = 0.0
        total_drag = 0.0
        total_my = 0.0

        for wing in self.wings:
            wing_name = wing.name

            lift = self.driver.lift([wing_name])
            drag = self.driver.drag([wing_name])
            efficiency = self.driver.efficiency([wing_name])
            my = self.driver.moment_y([wing_name])

            self.outputs_map[f"{wing.name}.lift"].value = lift
            self.outputs_map[f"{wing.name}.drag"].value = drag
            self.outputs_map[f"{wing.name}.efficiency"].value = efficiency
            self.outputs_map[f"{wing.name}.my"].value = my

            total_lift += lift
            total_drag += drag
            total_my += my

            spanloads_opts = wing.options
            spanloads_opts = next(
                (o for o in spanloads_opts if type(o) is SpanloadsOptions), None
            )

            if spanloads_opts:
                for load_name in spanloads_opts.loads.keys():
                    span_var = self.outputs_map.get(f"{wing_name}.{load_name}")
                    if span_var and type(span_var) is SpanLoadsVariable:
                        span_data = self.driver.spanloads([wing_name])
                        if span_data and len(span_data) >= 5:
                            span_var.centers = np.array(span_data[0])
                            span_var.spans = np.array(span_data[1])
                            span_var.fx = np.array(span_data[2])
                            span_var.fy = np.array(span_data[3])
                            span_var.fz = np.array(span_data[4])
                            if len(span_data) > 5:
                                span_var.m = np.array(span_data[5])

        if self.aircraft:
            total_efficiency = total_lift / total_drag if total_drag != 0 else 0.0

            self.outputs_map["lift"].value = total_lift
            self.outputs_map["drag"].value = total_drag
            self.outputs_map["efficiency"].value = total_efficiency
            self.outputs_map["my"].value = total_my

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
