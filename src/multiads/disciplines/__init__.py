from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gemseo.core.discipline import Discipline

from multiads.assembly import MADSComponent, update_components

try:
    from multiads.assembly.envelope import MADSPhase, Segment, update_segments
except ImportError:  # pragma: no cover - optional mission/envelope layer
    MADSPhase = Any
    Segment = Any

    def update_segments(segments: Any, input_data: Any) -> None:  # noqa: ARG001
        return None

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.scenario import BaseVariable
    from multiads.solvers import BaseSolver

# Logger
from gemseo import LOGGER

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
        if self.solver.options.jac_approx_type=="user":
            LOGGER.info(f'SOLVER :: {type(self.solver).__name__} using user defined JACOBIAN')
        else: 
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
        input_names = list(self.input_grammar.names)
        output_names = list(self.output_grammar.names)
        
        solver_inputs = [inp for inp in self.solver.inputs if inp.name in input_names]
        solver_outputs = [out for out in self.solver.outputs if out.name in output_names]
        
        return self.solver.compute_sensitivities(
            input_names,
            solver_inputs,
            output_names,
            solver_outputs,
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
        """
        if inputs is None:
            input_names = list(self.input_grammar.names)
        else:
            input_names = list(inputs)
            
        if outputs is None:
            output_names = list(self.output_grammar.names)
        else:
            output_names = list(outputs)
        
        solver_inputs = [inp for inp in self.solver.inputs if inp.name in input_names]
        solver_outputs = [out for out in self.solver.outputs if out.name in output_names]
        
        jac = self.solver.compute_sensitivities(
            input_names,
            solver_inputs,
            output_names,
            solver_outputs,
        )
        
        self.jac = {}
        for out_name in output_names:
            self.jac[out_name] = {}
            if out_name in jac:
                for inp_var in solver_inputs:
                    inp_name = inp_var.name
                    if inp_name in jac[out_name]:
                        value = jac[out_name][inp_name]
                        if value is not None and hasattr(value, 'shape'):
                            if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
                                self.jac[out_name][inp_name] = np.atleast_2d(value.flatten())
                            else:
                                self.jac[out_name][inp_name] = np.atleast_2d(value)
                    else:
                        self.jac[out_name][inp_name] = np.zeros((1, len(inp_var.value_np)))
            else:
                for inp_var in solver_inputs:
                    self.jac[out_name][inp_var.name] = np.zeros((1, len(inp_var.value_np)))
