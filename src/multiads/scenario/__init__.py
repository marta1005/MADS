from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
from gemseo import create_parameter_space, create_scenario, generate_n2_plot
from gemseo.core.coupling_structure import CouplingStructure
from numpy.typing import NDArray

from multiads.design_space.cpacs_structure_data import CPACSStructureData

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.base_post import BasePost
    from gemseo.scenarios.base_scenario import BaseScenario
    from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult


N = TypeVar("N", NDArray[np.int32], NDArray[np.float64])
V = TypeVar("V", np.float64, np.int32, NDArray[np.int32], NDArray[np.float64])
ValueType = np.int32 | NDArray[np.int32] | float | np.float64 | NDArray[np.float64]


class BaseVariable(Generic[V, N]):
    def __init__(
        self,
        name: str,
        value: V,
        value_np: N,
        is_array: bool,
        cpacs: Mapping[str, str],
    ) -> None:
        self.name = name
        self.value_np: N = value_np
        self.is_array = is_array
        self.cpacs = cpacs

        # 'Ghost' attribute, needed to parametrize an instance -> V
        self._value_type: V = value

    @property
    def value(self) -> V:
        return self.cast_value(self.value_np)

    @value.setter
    def value(self, val: ValueType) -> None:
        self.value_np[:] = np.ravel(val)

    def cast_value(self, val: N) -> V:
        if self.is_array:
            return np.asarray(val)  # type: ignore[return-value]
        return val[0]


class Variable(BaseVariable[V, N]):
    def __init__(
        self,
        name: str,
        value: ValueType,
        lb: ValueType | None = None,
        ub: ValueType | None = None,
        cpacs: Mapping[str, str] | None = None,
    ) -> None:
        # Cast value to numpy type
        _value: V
        if is_array := isinstance(value, np.ndarray):
            _value = np.ravel(value)  # type: ignore[assignment]
        elif type(value) is float:
            _value = np.float64(value)  # type: ignore[assignment]
        elif type(value) is int:
            _value = np.int32(value)  # type: ignore[assignment]
        else:
            _value = value  # type: ignore[assignment]

        value_np: N = np.ravel(_value)

        super().__init__(
            name=name,
            value=_value,
            value_np=value_np,
            is_array=is_array,
            cpacs=cpacs or {},
        )

        self.lb_np: N = -np.inf * np.ones_like(value_np) if lb is None else np.ravel(lb)
        self.ub_np: N = np.inf * np.ones_like(value_np) if ub is None else np.ravel(ub)

    @property
    def lb(self) -> V:
        return self.cast_value(self.lb_np)

    @lb.setter
    def lb(self, val: ValueType) -> None:
        self.lb_np = np.ravel(val)

    @property
    def ub(self) -> V:
        return self.cast_value(self.ub_np)

    @ub.setter
    def ub(self, val: ValueType) -> None:
        self.ub_np = np.ravel(val)


VariableInt = Variable[np.int32, NDArray[np.int32]]
VariableFloat = Variable[np.float64, NDArray[np.float64]]
VariableIntNP = Variable[NDArray[np.int32], NDArray[np.int32]]
VariableFloatNP = Variable[NDArray[np.float64], NDArray[np.float64]]


class RandomVariable(BaseVariable[V, N]):
    def __init__(
        self,
        name: str,
        value: ValueType,
        sigma: ValueType,
        distribution: str,
        cpacs: Mapping[str, str] | None = None,
    ) -> None:
        _value: V
        if is_array := isinstance(value, np.ndarray):
            _value = np.ravel(value)  # type: ignore[assignment]
        elif type(value) is float:
            _value = np.float64(value)  # type: ignore[assignment]
        elif type(value) is int:
            _value = np.int32(value)  # type: ignore[assignment]
        else:
            _value = value  # type: ignore[assignment]

        value_np = np.ravel(value)

        super().__init__(
            name=name,
            value=_value,
            value_np=value_np,
            is_array=is_array,
            cpacs=cpacs or {},
        )

        self.sigma_np: N = np.ravel(sigma)
        self.distribution = distribution

    @property
    def sigma(self) -> V:
        return self.cast_value(self.sigma_np)

    @sigma.setter
    def sigma(self, val: ValueType) -> None:
        self.sigma_np = np.ravel(val)


RandomVariableInt = RandomVariable[np.int32, NDArray[np.int32]]
RandomVariableFloat = RandomVariable[np.float64, NDArray[np.float64]]
RandomVariableIntNP = RandomVariable[NDArray[np.int32], NDArray[np.int32]]
RandomVariableFloatNP = RandomVariable[NDArray[np.float64], NDArray[np.float64]]


class InnerVariable(BaseVariable[V, N]):
    def __init__(
        self,
        name: str,
        value: ValueType,
    ) -> None:
        _value: V
        if is_array := isinstance(value, np.ndarray):
            _value = np.ravel(value)  # type: ignore[assignment]
        elif type(value) is float:
            _value = np.float64(value)  # type: ignore[assignment]
        elif type(value) is int:
            _value = np.int32(value)  # type: ignore[assignment]
        else:
            _value = value  # type: ignore[assignment]

        value_np: N = np.ravel(_value)

        super().__init__(
            name=name,
            value=_value,
            value_np=value_np,
            is_array=is_array,
            cpacs={},
        )


InnerVariableInt = InnerVariable[np.int32, NDArray[np.int32]]
InnerVariableFloat = InnerVariable[np.float64, NDArray[np.float64]]
InnerVariableIntNP = InnerVariable[NDArray[np.int32], NDArray[np.int32]]
InnerVariableFloatNP = InnerVariable[NDArray[np.float64], NDArray[np.float64]]


class ConstraintType(Enum):
    EQ = "eq"
    INEQ = "ineq"


class Constraint:
    def __init__(
        self,
        output_name: str,
        constraint_type: ConstraintType,
        value: float = 0.0,
        positive: bool = False,
    ) -> None:
        self.output_name = output_name
        self.constraint_type = constraint_type
        self.value = value
        self.positive = positive


class MADSScenario:
    def __init__(self) -> None:
        self.parameter_space = create_parameter_space()
        self.scenario: BaseScenario | None = None
        self._disciplines: Sequence[Discipline] | None = None

    def fill_parameter_space(self, variables: Sequence[BaseVariable]) -> None:
        """Fill the design space with `variables`.

        Args:
            variables: the variables defined according the dictionary structure
                       compatible with the assembly

        """
        for ov in variables:
            # Variables to be created from CPACS files
            if cpacs := ov.cpacs:
                try:
                    input_file = cpacs["input_file"]
                    xpath = cpacs["xpath"]
                    marker = cpacs["cpacs_marker"]
                except KeyError:
                    msg = f"Incomplete CPACS options for variable '{ov.name}'"
                    raise ValueError(msg) from None

                cpacs_var = CPACSStructureData(input_file)
                cpacs_var.select_variable_from_xpath(xpath, marker)
                ov.value = cpacs_var.get_value(marker)

            if isinstance(ov, Variable):
                self.parameter_space.add_variable(
                    ov.name,
                    ov.value.size,
                    lower_bound=ov.lb,  # type: ignore[invalid-argument-type]
                    upper_bound=ov.ub,  # type: ignore[invalid-argument-type]
                    value=ov.value,
                )
            elif isinstance(ov, RandomVariable):
                self.parameter_space.add_random_variable(
                    ov.name,
                    ov.distribution,  # type: ignore[invalid-argument-type]
                    ov.value.size,
                    mu=ov.value,
                    sigma=ov.sigma,
                )
            else:
                msg = (
                    f"Variable '{ov.name}' must be either a "
                    "'Variable' or a 'RandomVariable'"
                )
                raise TypeError(msg)

    def create_scenario(
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        name: str = "",
        formulation: str = "DisciplinaryOpt",
        scenario_type: str = "MDO",
        maximize_objective: bool = False,
        **formulation_options: Any,  # noqa: ANN401
    ) -> None:
        # Include inner variables first
        structure = CouplingStructure(disciplines)
        for variable_name in structure.all_couplings:
            for discipline in structure.strongly_coupled_disciplines:
                if (
                    variable_name in discipline.default_input_data
                    and variable_name not in self.parameter_space
                ):
                    variable = discipline.default_input_data[variable_name]
                    self.parameter_space.add_variable(
                        name=variable_name,
                        size=variable.size,
                        value=variable,
                    )
                    break

        # TODO @All: Check inputs/outputs

        self.scenario = create_scenario(
            disciplines=disciplines,
            objective_name=objective_name,
            design_space=self.parameter_space,
            name=name,
            scenario_type=scenario_type,
            maximize_objective=maximize_objective,
            formulation_name=formulation,
            **formulation_options,
        )
        self._disciplines = disciplines

    def add_constraints(self, constraints: Sequence[Constraint]) -> None:
        if self.scenario is not None:
            for constraint in constraints:
                self.scenario.add_constraint(
                    output_name=constraint.output_name,
                    constraint_type=constraint.constraint_type.value,
                    value=constraint.value,
                    positive=constraint.positive,
                )

    def add_observables(self, observables: Sequence[str]) -> None:
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        for o in observables:
            self.scenario.add_observable(o)

    def execute(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        self.scenario.execute(*args, **kwargs)

    def get_optim_variable_names(self) -> Sequence[str]:
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        return self.scenario.get_optim_variable_names()

    def get_result(self, *args, **kwargs) -> ScenarioResult | None:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        return self.scenario.get_result(*args, **kwargs)

    def post_process(self, *args, **kwargs) -> BasePost:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        return self.scenario.post_process(*args, **kwargs)

    def print_execution_metrics(self) -> None:
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        self.scenario.print_execution_metrics()

    def save_optimization_history(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        self.scenario.save_optimization_history(*args, **kwargs)

    def set_algorithm(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        self.scenario.set_algorithm(*args, **kwargs)

    def set_differentiation_method(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        self.scenario.set_differentiation_method(*args, **kwargs)

    def set_optimization_history_backup(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        self.scenario.set_optimization_history_backup(*args, **kwargs)

    def to_dataset(self, *args, **kwargs) -> Dataset:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        return self.scenario.to_dataset(*args, **kwargs)

    def xdsmize(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.scenario is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        self.scenario.xdsmize(*args, **kwargs)

    def n2_plot(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.scenario is None or self._disciplines is None:
            msg = f"'{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        generate_n2_plot(self._disciplines, *args, **kwargs)
