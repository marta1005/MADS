from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gemseo.core.discipline import Discipline

from multiads.assembly import MADSComponent, update_components
from multiads.assembly.envelope import MADSPhase, Segment, update_segments

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.scenario import BaseVariable
    from multiads.solvers import BaseSolver


# TODO @Andres: Inherit from MADSDiscipline
class UserDefined(Discipline):
    def __init__(
        self,
        name: str,
        inputs: Sequence[BaseVariable],
        outputs: Sequence[BaseVariable],
        expression: str | Callable,
    ) -> None:
        if callable(expression):
            self.expression = expression
        else:
            body = f"lambda {', '.join(i.name for i in inputs)}: {expression}"
            self.expression = eval(body)  # NOTE @Andres: Maybe use `ast.literal_eval`

        super().__init__(name)
        self.input_grammar.update_from_data({i.name: i.value_np for i in inputs})
        self.output_grammar.update_from_data({o.name: o.value_np for o in outputs})

        # Set default Jacobian settings
        self.set_jacobian_approximation(
            jac_approx_type="finite_differences",
            jax_approx_step=1e-04,
            jac_approx_n_processes=1,
            jac_approx_use_threading=False,
            jac_approx_wait_time=0,
        )

    def _run(self, input_data: Mapping[str, NDArray]) -> dict[str, NDArray]:
        outputs = self.expression(*input_data.values())
        output_names = list(self.output_grammar.names)
        if len(output_names) == 1:
            outputs = (outputs,)
        outputs = [np.atleast_1d(o) for o in outputs]
        return dict(zip(output_names, outputs, strict=True))


class MADSDiscipline(Discipline):
    def __init__(
        self,
        name: str,
        components: Sequence[MADSComponent],
        segments: Sequence[MADSPhase] | None = None,
        solver: BaseSolver | None = None,
    ) -> None:
        super().__init__(name)
        self.solver = solver

        # Assign components and variables
        self.components = self.solver.parse_variables(components, segments)

        # TODO @Andres: This is a quick fix!
        try:
            self.segments = self.solver.segments
        except AttributeError:
            self.segments = []

        if self.solver.inputs is None:
            msg = f"The solver of discipline '{name}' did not update its inputs."
            raise RuntimeError(msg)

        if self.solver.outputs is None:
            msg = f"The solver of discipline '{name}' did not update its outputs."
            raise RuntimeError(msg)

        self.input_grammar_update({v.name: v.value_np for v in self.solver.inputs})
        self.output_grammar_update({v.name: v.value_np for v in self.solver.outputs})

        # Settings for the Jacobian
        self.set_jacobian_approximation(
            jac_approx_type=self.solver.options.jac_approx_type,
            jax_approx_step=self.solver.options.jac_approx_step,
            jac_approx_n_processes=self.solver.options.jac_approx_n_processes,
            jac_approx_use_threading=self.solver.options.jac_approx_use_threading,
            jac_approx_wait_time=self.solver.options.jac_approx_wait_time,
        )

    def input_grammar_update(self, inputs: Mapping[str, NDArray]) -> None:
        self.input_grammar.update_from_data(inputs)
        self.default_input_data.update(inputs)

    def output_grammar_update(self, outputs: Mapping[str, NDArray]) -> None:
        self.output_grammar.update_from_data(outputs)
        self.default_output_data.update(outputs)

    def update_variables(self, input_data: Mapping[str, NDArray]) -> None:
        if self.solver.inputs is None:
            msg = (
                f"The inputs of solver '{type(self.solver).__name__}' "
                "are not initialized"
            )
            raise RuntimeError(msg)

        for var in self.solver.inputs:
            if (value := input_data.get(var.name, None)) is not None:
                var.value_np[:] = value

    def _run(self, input_data: Mapping[str, NDArray]) -> Mapping[str, NDArray]:
        # update data
        self.update_variables(input_data)
        # update components with input data
        update_components(self.components, input_data)
        update_segments(self.segments, input_data)
        # call the solver
        self.solver.run(self.components)
        # compute the output
        self.solver.compute_output()

        if self.solver.outputs is None:
            msg = f"The solver of discipline '{self.name}' did not update its outputs."
            raise RuntimeError(msg)

        return {var.name: var.value_np for var in self.solver.outputs}

    # TODO @Simone: function to compute derivatives
    def _derive_state(self) -> Mapping[str, Any]:
        input_names = self.get_input_data_names()
        output_names = self.get_output_data_names()
        return self.solver.compute_sensitivities(
            input_names,
            self.inputs,
            output_names,
            self.outputs,
        )

    # TODO @Simone: Original definitions of functions in GEMSEO
    def _compute_jacobian(self, inputs=None, outputs=None) -> None:
        """Compute the jacobian.

        :param inputs: linearization should be performed with respect
          to inputs list. If None, linearization should
          be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)

                # ------------------------------------------------
                # example: complete computation of the jacobian
                # self.jac['max_resp']['wing_sweep'] = atleast_2d()
                # self.jac['max_resp']['k_actuator'] =
                # ------------------------------------------------
        """
        self._init_jacobian(inputs, outputs, with_zeros=True)
        ## compute Jacobian sourcing the correspongin solver solver
        # get the inputs
        inputs = self.get_inputs_by_name(*self.inputs)
        # [SM]
        outputs = self.get_outputs_by_name(*self.outputs)
        # launch the compuation of the sensitivities
        # return a structure of the jacobian consistent for the definition
        # in GEMSEO
        jac = self._derive_state(inputs, outputs)
        # define Jacobian in GEMSEO
        self.jac = jac
        # for var_in in self.inputs:
        #    self.jac[var_in] = {}
        #    for var_out in self.outputs:
        #        self.jac[var_in][var_out] = atleast_2d(
        #            array(jac[])
        #        )
