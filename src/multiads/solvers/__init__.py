from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.assembly import MADSComponent
    from multiads.scenario import BaseVariable, InnerVariable


class SolverOptions:
    def __init__(
        self,
        *,
        jac_approx_type: str = "finite_differences",
        jac_approx_step: float = 1e-04,
        jac_approx_n_processes: int = 1,
        jac_approx_use_threading: bool = False,
        jac_approx_wait_time: int = 0,
    ) -> None:
        self.jac_approx_type = jac_approx_type
        self.jac_approx_step = jac_approx_step
        self.jac_approx_n_processes = jac_approx_n_processes
        self.jac_approx_use_threading = jac_approx_use_threading
        self.jac_approx_wait_time = jac_approx_wait_time


class BaseSolver(ABC):
    def __init__(self) -> None:
        self.options = SolverOptions()
        self.inputs: Sequence[BaseVariable] | None = None
        self.outputs: Sequence[InnerVariable] | None = None

    def get_input_variables(self, names: Sequence[str]) -> Sequence[BaseVariable]:
        if self.inputs is None:
            return []
        var_map = {v.name: v for v in self.inputs}
        return [var_map[name] for name in names if name in var_map]

    def get_output_variables(self, names: Sequence[str]) -> Sequence[InnerVariable]:
        if self.outputs is None:
            return []
        var_map = {v.name: v for v in self.outputs}
        return [var_map[name] for name in names if name in var_map]

    @abstractmethod
    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401
    ) -> Sequence[MADSComponent]:
        """Update the list of components and return the I/O variables."""

    def run(self, components: Sequence[MADSComponent]) -> None:
        self.set_state(components)
        self._run()

    def set_state(self, components: Sequence[MADSComponent]) -> None:  # noqa: B027
        pass

    @abstractmethod
    def _run(self) -> None:
        pass

    @abstractmethod
    def compute_output(self) -> None:
        pass

    @abstractmethod
    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[BaseVariable],
        output_names: Sequence[str],
        outputs: Sequence[BaseVariable],
    ) -> Mapping[str, NDArray]:
        """Compute the Jacobian matrix."""
