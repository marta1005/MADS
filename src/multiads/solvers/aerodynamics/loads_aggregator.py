from __future__ import annotations

from typing import TYPE_CHECKING, Any

from multiads.assembly import (
    ComponentOptions,
    MADSComponent,
    Propeller,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario.span_loads import (
    SPANLOAD_DEFAULT_NUM_STATIONS,
    SpanLoadsGroupVariable,
    SpanLoadsVariable,
)
from multiads.solvers import BaseSolver

# NOTE @Andres: This should be generic

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.scenario import BaseVariable


class SpanloadsOptions(ComponentOptions):
    def __init__(self, loads: Mapping[str, int] | None = None) -> None:
        if loads is None:
            loads = {"span_load": SPANLOAD_DEFAULT_NUM_STATIONS}
        self.loads = {
            k: v if v < 1 else SPANLOAD_DEFAULT_NUM_STATIONS for k, v in loads.items()
        }

    @property
    def num_loads(self) -> int:
        return len(self.loads)


class LoadsAggregator(BaseSolver):
    def __init__(self) -> None:
        super().__init__()
        self.wings: Sequence[Wing] | None = None
        self.propellers: Sequence[Propeller] | None = None
        self.inputs_map: dict[str, SpanLoadsVariable] | None = None
        self.outputs_map: dict[str, SpanLoadsGroupVariable] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        # Parse variables and create local components
        _components = copy_components(components)
        comp_flat = flatten_components(_components)
        self.wings = [c for c in comp_flat if type(c) is Wing]
        self.propellers = [c for c in comp_flat if type(c) is Propeller]
        wing_like = [*self.wings, *self.propellers]

        # Inputs and outputs
        inputs: list[SpanLoadsVariable] = []
        outputs: list[SpanLoadsGroupVariable] = []

        # TODO @Andres: automate this
        loads: list[SpanLoadsVariable] = []
        for wing in wing_like:
            opts = wing.options
            opts = next((o for o in opts if type(o) is SpanloadsOptions), None)

            if opts := opts:
                for load_name, load_size in opts.loads.items():
                    loads.append(
                        SpanLoadsVariable.from_num_stations(
                            f"{wing.name}.{load_name}",
                            load_size,
                        ),
                    )
                    inputs.append(loads[-1])

            outputs.append(
                SpanLoadsGroupVariable.from_loads(
                    f"{wing.name}.span_loads_group",
                    loads,
                ),
            )

        self.inputs = inputs
        self.outputs = outputs
        self.inputs_map = {i.name: i for i in inputs}
        self.outputs_map = {o.name: o for o in outputs}

        return wing_like

    def _run(self) -> None:
        # We are just routing variables
        pass

    def compute_output(self) -> None:
        if (
            self.wings is None
            or self.propellers is None
            or self.inputs_map is None
            or self.outputs_map is None
        ):
            msg = f"The solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        for wing in [*self.wings, *self.propellers]:
            loads = self.inputs_map[f"{wing.name}.span_load"]
            group = self.outputs_map[f"{wing.name}.span_loads_group"]
            group.span_loads[0].value = loads.value

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
