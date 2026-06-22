from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import multiads.solvers.aerodynamics.dust_lib as dl
from multiads.assembly import (
    AirfoilNACA4,
    Environment,
    Fuselage,
    MADSComponent,
    Propeller,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario import (
    BaseVariable,
    InnerVariable,
    InnerVariableFloat,
    InnerVariableFloatNP,
)
from multiads.scenario.polars import PolarVariable
from multiads.scenario.span_loads import SpanLoadsVariable
from multiads.solvers import BaseSolver

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class DUST(BaseSolver):
    def __init__(self, options: dl.Options | None = None) -> None:
        super().__init__()
        self.options: dl.Options = options or dl.Options()
        self.driver: dl.Driver | None = None
        self.environment: Environment | None = None
        self.wings: Sequence[dl.Wing] | None = None
        self.propellers: Sequence[dl.Propeller] | None = None
        self.fuselages: Sequence[dl.Fuselage] | None = None
        self.polars: dict[str, PolarVariable] | None = None
        self.outputs_map: dict[str, InnerVariable] | None = None
        self.run_directory: Path | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        # Initialize components
        self.environment, wings, propellers, fuselages = self._initialize_components(
            components,
        )
        self.wings, self.propellers, self.fuselages = self._remap_components(
            wings,
            propellers,
            fuselages,
        )

        # Inputs
        polars = self._create_input_polars(self.wings, self.propellers)
        inputs = self._create_design_variables(
            self.environment,
            wings,
            propellers,
            fuselages,
        )
        self.polars = {p.name: p for p in polars}
        self.inputs = [*polars, *inputs]

        # Outputs
        self.outputs = self._create_output_variables(self.wings, self.propellers)
        self.outputs_map = {v.name: v for v in self.outputs}

        return [self.environment, *wings, *propellers, *fuselages]

    def _initialize_components(
        self,
        components: Sequence[MADSComponent],
    ) -> tuple[Environment, Sequence[Wing], Sequence[Propeller], Sequence[Fuselage]]:
        _components = copy_components(components)
        comp_flat = flatten_components(_components)

        wings = [c for c in comp_flat if type(c) is Wing]
        propellers = [c for c in comp_flat if type(c) is Propeller]
        fuselages = [c for c in comp_flat if type(c) is Fuselage]

        try:
            environment = next(c for c in comp_flat if type(c) is Environment)
        except StopIteration:
            msg = f"An environment must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        return environment, wings, propellers, fuselages

    def _remap_components(
        self,
        wings: Sequence[Wing],
        propellers: Sequence[Propeller],
        fuselages: Sequence[Fuselage],
    ) -> tuple[Sequence[dl.Wing], Sequence[dl.Propeller], Sequence[dl.Fuselage]]:
        wings_mapped = [dl.Wing.from_component(w) for w in wings]
        propellers_mapped = [dl.Propeller.from_component(p) for p in propellers]
        fuselages_mapped = [dl.Fuselage.from_component(f) for f in fuselages]
        return wings_mapped, propellers_mapped, fuselages_mapped

    def _create_input_polars(
        self,
        wings: Sequence[dl.Wing],
        propellers: Sequence[dl.Propeller],
    ) -> Sequence[PolarVariable]:
        polars: list[PolarVariable] = []
        for wing in wings:
            polars.extend(
                PolarVariable.from_num_points(f"{sec.name}.polar", sec.polar_length)
                for sec in wing.sections
                if sec.polar
            )
        for prop in propellers:
            polars.extend(
                PolarVariable.from_num_points(f"{sec.name}.polar", sec.polar_length)
                for sec in prop.blade.sections
                if sec.polar
            )
        return polars

    def _create_design_variables(
        self,
        environment: Environment,
        wings: Sequence[Wing],
        propellers: Sequence[Propeller],
        fuselages: Sequence[Fuselage],
    ) -> Sequence[BaseVariable]:
        def add_vars(
            component: MADSComponent,
            keys: Sequence[str],
            var_list: list[BaseVariable],
        ) -> None:
            var_list.extend(v for k, v in component.variables.items() if k in keys)

        inputs: list[BaseVariable] = []

        # Environment
        env_list = ["height", "speed", "alpha", "beta", "gamma"]
        add_vars(environment, env_list, inputs)

        # Wings
        wing_list = [
            "offset",
            "scaling",
            "global_pos",
            "alpha",
            "beta",
            "xc_ref",
            "roll",
        ]
        span_list = ["length", "sweep", "dihed"]
        sec_list = ["chord", "twist"]
        airfoil_list = ["thickness_factor", "camber_factor"]
        naca_list = ["m", "p", "t"]

        wing_like = [*wings, *[p.blade for p in propellers]]
        for l_wing in wing_like:
            add_vars(l_wing, wing_list, inputs)
            for l_span in l_wing.spans:
                add_vars(l_span, span_list, inputs)
            for l_sec in l_wing.sections:
                add_vars(l_sec, sec_list, inputs)
                add_vars(l_sec.airfoil, airfoil_list, inputs)
                if type(l_sec.airfoil) is AirfoilNACA4:
                    add_vars(l_sec.airfoil, naca_list, inputs)

        # Propellers
        prop_list = ["r_tip", "n_blades", "pitch", "rpm", "global_pos", "alpha", "beta"]
        for l_prop in propellers:
            add_vars(l_prop, prop_list, inputs)

        # Fuselages
        fus_list = ["maximum_width", "maximum_height", "length", "global_pos"]
        for l_fus in fuselages:
            add_vars(l_fus, fus_list, inputs)

        return inputs

    def _create_output_variables(
        self,
        wings: Sequence[dl.Wing],
        propellers: Sequence[dl.Propeller],
    ) -> Sequence[InnerVariable]:
        outputs: list[InnerVariable] = []
        wing_like = [*wings, *[p.blade for p in propellers]]
        for wing in wing_like:
            if wing.options.compute_loads:
                outputs.extend(
                    [
                        InnerVariableFloatNP(f"{wing.name}.force", np.zeros(3)),
                        InnerVariableFloat(f"{wing.name}.lift", 0.0),
                        InnerVariableFloat(f"{wing.name}.drag", 0.0),
                        InnerVariableFloat(f"{wing.name}.efficiency", 0.0),
                        InnerVariableFloatNP(f"{wing.name}.moment", np.zeros(3)),
                        InnerVariableFloat(f"{wing.name}.my", 0.0),
                    ],
                )

            if wing.options.compute_spanwise:
                outputs.append(
                    SpanLoadsVariable.from_num_stations(
                        f"{wing.name}.span_load",
                        wing.options.spanwise_size,
                    ),
                )

        for prop in propellers:
            if prop.options.compute_loads:
                outputs.extend(
                    [
                        InnerVariableFloatNP(f"{prop.name}.force", np.zeros(3)),
                        InnerVariableFloat(f"{prop.name}.thrust", 0.0),
                        InnerVariableFloatNP(f"{prop.name}.moment", np.zeros(3)),
                        InnerVariableFloat(f"{prop.name}.torque", 0.0),
                        InnerVariableFloat(f"{prop.name}.power", 0.0),
                    ],
                )

        if self.options.output_options.compute_loads:
            outputs.extend(
                [
                    InnerVariableFloatNP("force", np.zeros(3)),
                    InnerVariableFloat("lift", 0.0),
                    InnerVariableFloat("drag", 0.0),
                    InnerVariableFloat("efficiency", 0.0),
                    InnerVariableFloatNP("moment", np.zeros(3)),
                    InnerVariableFloat("my", 0.0),
                ],
            )

        return outputs

    def set_state(self, components: Sequence[MADSComponent]) -> None:
        if self.wings is None or self.propellers is None or self.fuselages is None:
            msg = f"Components of solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        local_components = [*self.wings, *self.propellers, *self.fuselages]
        for comp in components:
            for l_comp in local_components:
                if comp.name == l_comp.name:
                    l_comp.update(comp)  # type: ignore[invalid-argument-type]

        if self.options.run_directory is not None:
            self.run_directory = self.options.run_directory
            self.run_directory.mkdir(parents=True, exist_ok=True)
        else:
            self.options.work_dir.mkdir(parents=True, exist_ok=True)
            self.run_directory = Path(
                tempfile.mkdtemp(
                    dir=self.options.work_dir,
                    prefix=self.options.name,
                ),
            )

    def _run(self) -> None:
        if (
            self.environment is None
            or self.wings is None
            or self.propellers is None
            or self.fuselages is None
            or self.run_directory is None
        ):
            msg = f"The solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        # Change to local directory and run
        main_dir = Path.cwd()
        os.chdir(self.run_directory)

        self.driver = dl.Driver(
            environment=self.environment,
            propellers=self.propellers,
            wings=self.wings,
            fuselages=self.fuselages,
            options=self.options,
        )

        self._write_polars()

        try:
            self.driver.preprocess()
            if self.options.post_preprocess_hook is not None:
                self.options.post_preprocess_hook(self.run_directory)
            self.driver.run()
        finally:
            # Back to main directory
            os.chdir(main_dir)

    def _write_polars(self) -> None:
        if self.wings is None or self.propellers is None or self.polars is None:
            return

        wing_like = [*self.wings, *(p.blade for p in self.propellers)]

        for wing in wing_like:
            for sec in wing.sections:
                if sec.polar:
                    polar = self.polars[f"{sec.name}.polar"]
                    dl.write_polar(polar, Path(f"{sec.name}.c81"))

    def compute_output(self) -> None:
        if (
            self.environment is None
            or self.wings is None
            or self.propellers is None
            or self.fuselages is None
        ):
            msg = f"Components of solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The interface of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.driver is None or self.run_directory is None:
            msg = f"Solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        #  Run DUST analyses
        loads_analyses = self._create_loads_analyses(self.wings, self.propellers)
        span_analyses = self._create_spanwise_analyses(self.wings, self.propellers)
        viz_analysis = self._create_visualization_analysis()

        if viz_analysis:
            analyses = [
                *loads_analyses.values(),
                viz_analysis,
                *span_analyses.values(),
            ]
        else:
            analyses = [*loads_analyses.values(), *span_analyses.values()]

        # Change to local directory and run
        main_dir = Path.cwd()
        try:
            os.chdir(self.run_directory)

            self.driver.postprocess(analyses)

            # Map analyises to output variables
            self._process_loads_results(
                loads_analyses,
                self.environment,
                self.wings,
                self.propellers,
                self.outputs_map,
            )
            self._process_spanwise_results(
                span_analyses,
                self.wings,
                self.propellers,
                self.outputs_map,
            )
            self._add_thrust_to_spanwise_results(
                self.propellers,
                self.outputs_map,
            )
            self._process_global_results(
                self.environment,
                loads_analyses,
                self.outputs_map,
            )
        finally:
            # Back to main directory
            os.chdir(main_dir)
        if not self.options.keep_run_directory:
            shutil.rmtree(self.run_directory)

    def _create_loads_analyses(
        self,
        wings: Sequence[dl.Wing],
        propellers: Sequence[dl.Propeller],
    ) -> dict[str, dl.PostLoads]:
        analyses: dict[str, dl.PostLoads] = {}
        wing_like = [*wings, *[p.blade for p in propellers]]

        for wing in wing_like:
            if wing.options.compute_loads:
                analyses[wing.name] = dl.PostLoads(
                    name=wing.name + "_loads",
                    average=wing.options.loads_avg,
                    start_res=wing.options.loads_start,
                    end_res=wing.options.loads_end,
                    step_res=wing.options.loads_step,
                    reference=wing.options.loads_reference,
                    components=[wing.name],
                )

        for prop in propellers:
            if prop.options.compute_loads:
                analyses[prop.name] = dl.PostLoads(
                    name=prop.name + "_loads",
                    average=prop.options.loads_avg,
                    start_res=prop.options.loads_start,
                    end_res=prop.options.loads_end,
                    step_res=prop.options.loads_step,
                    reference=prop.options.loads_reference,
                    components=[prop.name],
                )

        if self.options.output_options.compute_loads:
            analyses["all"] = dl.PostLoads(
                name="all_loads",
                average=self.options.output_options.loads_avg,
                start_res=self.options.output_options.loads_start,
                end_res=self.options.output_options.loads_end,
                step_res=self.options.output_options.loads_step,
                reference=self.options.output_options.loads_reference,
                components=["all"],
            )

        return analyses

    def _create_spanwise_analyses(
        self,
        wings: Sequence[dl.Wing],
        propellers: Sequence[dl.Propeller],
    ) -> dict[str, dl.PostSpanwiseLoads]:
        analyses: dict[str, dl.PostSpanwiseLoads] = {}
        wing_like = [*wings, *[p.blade for p in propellers]]
        for wing in wing_like:
            if wing.options.compute_spanwise:
                analyses[wing.name] = dl.PostSpanwiseLoads(
                    name=wing.name + "_spanwise",
                    average=wing.options.spanwise_avg,
                    resolution=wing.options.spanwise_size,
                    axis_nod=wing.options.spanwise_axis_node,
                    axis_dir=wing.options.spanwise_axis_dir,
                    start_res=wing.options.spanwise_start,
                    end_res=wing.options.spanwise_end,
                    step_res=wing.options.spanwise_step,
                    components=[wing.name],
                    symmetric_geo=wing.symmetry,
                    lifting_line_data=wing.options.spanwise_lifting_line_data,
                    vortex_lattice_data=wing.options.spanwise_vortex_lattice_data,
                )
        return analyses

    def _create_visualization_analysis(self) -> dl.PostViz | None:
        opts = self.options.output_options
        if not opts.visualization:
            return None

        return dl.PostViz(
            name="visualization",
            average=opts.viz_avg,
            start_res=opts.viz_start,
            end_res=opts.viz_end,
            step_res=opts.viz_step,
            fmt=opts.viz_fmt,
            wake=opts.viz_wake,
            separate_wake=opts.viz_separate_wake,
            variables=opts.viz_variables,
            components=["all"],
        )

    def _process_loads_results(
        self,
        analyses: Mapping[str, dl.PostLoads],
        environment: Environment,  # noqa: ARG002
        wings: Sequence[dl.Wing],
        propellers: Sequence[dl.Propeller],
        outputs: Mapping[str, InnerVariable],
    ) -> None:
        wing_like = [*wings, *[p.blade for p in propellers]]

        for wing in wing_like:
            if loads := analyses.get(wing.name, None):
                force = np.asarray(loads.f[-1], dtype=float)
                lift = float(force[2])
                drag = float(-force[0])
                outputs[f"{wing.name}.force"].value = force
                outputs[f"{wing.name}.lift"].value = lift
                outputs[f"{wing.name}.drag"].value = drag
                outputs[f"{wing.name}.efficiency"].value = (
                    lift / drag if drag != 0.0 else 0.0
                )

                moment = np.asarray(loads.m[-1], dtype=float)
                outputs[f"{wing.name}.moment"].value = moment
                outputs[f"{wing.name}.my"].value = moment[1]

        for prop in propellers:
            if loads := analyses.get(prop.name, None):
                force = np.dot(environment.matrix_body_wind, loads.f[-1])
                outputs[f"{prop.name}.force"].value = force
                outputs[f"{prop.name}.thrust"].value = -force[0]

                m_sign = 1.0 if prop.reverse else -1.0

                moment = np.dot(environment.matrix_body_wind, loads.m[-1])
                outputs[f"{prop.name}.moment"].value = moment
                outputs[f"{prop.name}.torque"].value = m_sign * moment[0]
                outputs[f"{prop.name}.power"].value = (
                    m_sign * moment[0] * prop.rpm * np.pi / 30.0
                )

    def _process_spanwise_results(
        self,
        analyses: Mapping[str, dl.PostSpanwiseLoads],
        wings: Sequence[dl.Wing],
        propellers: Sequence[dl.Propeller],
        outputs: Mapping[str, InnerVariable],
    ) -> None:
        wing_like = [*wings, *[p.blade for p in propellers]]
        for wing in wing_like:
            if (loads := analyses.get(wing.name, None)) and (
                type(var := outputs.get(f"{wing.name}.span_load", None))
                is SpanLoadsVariable
            ):
                if not hasattr(loads, "y_cen"):
                    continue
                var.centers = loads.y_cen
                var.spans = loads.y_span
                var.fx = loads.fx[-1]
                var.fy = loads.fy[-1]
                var.fz = loads.fz[-1]
                var.m = loads.mo[-1]

    def _add_thrust_to_spanwise_results(
        self,
        propellers: Sequence[dl.Propeller],
        outputs: Mapping[str, InnerVariable],
    ) -> None:
        for prop in propellers:
            thrust = outputs.get(f"{prop.name}.thrust", None)
            wing = prop.parent
            if thrust and wing:
                var = outputs.get(f"{wing}.span_load", None)
                if type(var) is SpanLoadsVariable:
                    var.thrust = thrust.value

    def _process_global_results(
        self,
        environment: Environment,  # noqa: ARG002
        analyses: Mapping[str, dl.PostLoads],
        outputs: Mapping[str, InnerVariable],
    ) -> None:
        if loads := analyses.get("all", None):
            force = np.asarray(loads.f[-1], dtype=float)
            lift = float(force[2])
            drag = float(-force[0])
            outputs["force"].value = force
            outputs["lift"].value = lift
            outputs["drag"].value = drag
            outputs["efficiency"].value = lift / drag if drag != 0.0 else 0.0

            moment = np.asarray(loads.m[-1], dtype=float)
            outputs["moment"].value = moment
            outputs["my"].value = moment[1]

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
