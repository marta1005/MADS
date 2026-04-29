from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Mapping
import numpy as np
from numpy.typing import NDArray
import copy

from multiads.assembly import (
    PropulsionSystem,
    Propeller,
    Environment,
    MADSComponent,
    copy_components,
    flatten_components,
)
from multiads.scenario import BaseVariable, InnerVariableFloat, VariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.propulsion.thermal_engine_low_fidelity_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Sequence, Mapping

class ThermalEngineLowFidelity(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.engines: Sequence[PropulsionSystem] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv,
    ) -> Sequence[MADSComponent]:
        # Filter components
        c_flat = flatten_components(components)
        # assign components
        self.engines = copy_components(c for c in c_flat if isinstance(c, PropulsionSystem))
        ## environment
        if env := next((c for c in c_flat if isinstance(c, Environment)),None):
            self.environment = env
        else:
            print(f"warning...model 'EngineLowFidelity' using DEFAULT Environment()")
            self.environment = Environment(name="Default :: Environment", height=0.0,speed=150.0)
        # check if propeller needs to map propeller properties
        if props := next((c for c in c_flat if isinstance(c, Propeller)),None):
            # self.props =  [c for c in c_flat if isinstance(c, Propeller)]
            self.propellers = copy_components(c for c in c_flat if isinstance(c, Propeller))



        # Inputs and outputs
        self.inputs: list[BaseVariable] = []
        self.outputs: list[BaseVariable] = []

        def update_io(
            engine: PropulsionSystem,
            inputs: list[BaseVariable],
            outputs: list[BaseVariable],
        ) -> None:
            # Design variables
            if v := engine.variables.get("power"):
                self.inputs.append(v)
            if v := engine.variables.get("shaft_power"):
                self.inputs.append(v)
            # generate output grammar
            # self.outputs.extend([InnerVariableFloat(f"{engine.name}.temperature", 0.0)])
            self.outputs.extend([InnerVariableFloat(f"{engine.name}.fuel_consumption", 0.0)])
            self.outputs.extend([InnerVariableFloat(f"{engine.name}.co2_emissions", 0.0)])
            self.outputs.extend([InnerVariableFloat(f"{engine.name}.mass", 0.0)])

        # props
        if self.propellers:
            for prop in self.propellers:
                self.inputs.append(InnerVariableFloat(f"{prop.name}.shaft_power",0.0))

        # variable form the engine
        for engine in self.engines:
            update_io(engine, self.inputs, self.outputs)

        self.outputs_map = {v.name: v for v in self.outputs}

        # always return the component
        return [*self.engines,*self.propellers]

    def _run(self) -> None:
        if self.engines is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        
        # recover and map inner variable if propeller is specified :: shaft_power
        if self.propellers:
            # loop over engine - ensure that mapping is consistent
            for i, engine in enumerate(self.engines):
                prop = self.propellers[i]
                shaft_power = copy.deepcopy(
            next(i for i in self.inputs if i.name == f"{prop.name}.shaft_power"),
        )
                self.engines[i].power = shaft_power.value

        self.driver = Driver(
            engines=self.engines,
            environment=self.environment,
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None:
            msg = f"The outputs of solver '{type(self).__name__}' are not initialized"
            raise RuntimeError(msg)

        masses = self.driver.mass
        fuel_consumptions = self.driver.fuel_consumption
        co2_emissions= self.driver.co2_emission


        for i, engine in enumerate(self.engines):
            print("fuel consumption",fuel_consumptions[i])
            self.outputs_map[f"{engine.name}.mass"].value = masses[i]
            self.outputs_map[f"{engine.name}.fuel_consumption"].value = fuel_consumptions[i]
            self.outputs_map[f"{engine.name}.co2_emissions"].value = co2_emissions[i]


    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[BaseVariable],
        output_names: Sequence[str],
        outputs: Sequence[BaseVariable],
    ) -> Mapping[str, NDArray]:
        """compute sensitivities of the inputs with respect to the outputs

        Args:
            input_names ([type]): [description]
            inputs ([type]): [description]
            requested_outputs ([type]): [description]
            mapped_outputs ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        sensitivity = self.driver._get_sensitivity()
        jac = {}
        for out in self.outputs:
            for inp in inputs:
                # create dictionary key based on output
                jac.setdefault(out, {})
                try:
                    # create key for the input from the sensitivity matrix
                    jac.setdefault(out, {}).update(inp, sensitivity[out][inp])
                except KeyError:
                    raise ValueError(
                        f"'The Jacobian for the ThermalEngineLowFidelity is not computed for input '{inp}' and output '{out}' "
                    )
        return jac