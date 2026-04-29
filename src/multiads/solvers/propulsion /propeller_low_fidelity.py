from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from multiads.assembly import (
    Propeller,
    Environment,
    MADSComponent,
    copy_components,
    flatten_components,
)

from multiads.scenario import BaseVariable, InnerVariableFloat, VariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.propulsion.propeller_low_fidelity_lib import Driver, Options


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

import numpy as np
from numpy.typing import NDArray

class PropellerLowFidelity(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.propellers: Sequence[Propeller] | None = None
    
    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv,
    ) -> Sequence[MADSComponent]:
        # Filter components
        c_flat = flatten_components(components)
        # assign components
        self.propellers = copy_components(c for c in c_flat if isinstance(c, Propeller))
        ## environment
        if env := next((c for c in c_flat if isinstance(c, Environment)),None):
            self.environment = env
        else:
            print(f"warning...model 'PropellerLowFidelity' using DEFAULT Environment()")
            self.environment = Environment(name="Default :: Environment", height=0.0,speed=150.0)


        # Inputs and outputs
        self.inputs: list[BaseVariable] = []
        self.outputs: list[BaseVariable] = []


        def update_io(
            prop: Propeller,
            inputs: list[BaseVariable],
            outputs: list[BaseVariable],
        ) -> None:
            for sec in prop.blade.sections:
                # Design variables
                if v := sec.variables.get("chord"):
                    self.inputs.append(v)
                if v := sec.variables.get("twist"):
                    self.inputs.append(v)
                    
            # propeller specific inputs
            if p := prop.variables.get("r_tip"):
                self.inputs.append(p)
            if p := prop.variables.get("pitch"):
                self.inputs.append(p)
            if p := prop.variables.get("rpm"):
                self.inputs.append(p)
                
            # generate output grammar
            self.outputs.extend([InnerVariableFloat(f"{prop.name}.shaft_power", 0.0)])
            self.outputs.extend([InnerVariableFloat(f"{prop.name}.prop_efficiency", 0.0)])
            self.outputs.extend([InnerVariableFloat(f"{prop.name}.mass", 0.0)])
        
        for prop in self.propellers:  
            update_io(prop,self.inputs,self.outputs)
            
            # if thrust is given assign to inputs
            if p := prop.thrust is None:
                print(f"warning :: thrust from propeller defined as InnerVariable")
                self.outputs.append(InnerVariableFloat(f"{prop.name}.thrust", 0.0))
            else:
                if p := prop.variables.get("thrust"):
                    self.inputs.append(p)
                
        self.outputs_map = {v.name: v for v in self.outputs}

        # print(self.inputs,self.outputs_map)
        #alway return the component
        return [*self.propellers]
 

    def _run(self) -> None:
        if self.propellers is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        self.driver = Driver(
            propellers=self.propellers,
            environment=self.environment,
            options=self.options,
        )
        self.driver.run()
    
        
    def compute_output(self) -> None:
        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None:
            msg = f"The outputs of solver '{type(self).__name__} are not initialized"
            raise RuntimeError(msg)

        thrusts = self.driver.thrust
        prop_efficiencies = self.driver.eta
        shaft_powers = self.driver.shaft_power
        masses = self.driver.mass

        for i, prop in enumerate(self.propellers):
            if self.outputs_map.get(f"{prop.name}.thrust") is not None:
                self.outputs_map[f"{prop.name}.thrust"].value = thrusts[i]
            self.outputs_map[f"{prop.name}.prop_efficiency"].value = prop_efficiencies[i]
            self.outputs_map[f"{prop.name}.shaft_power"].value = shaft_powers[i]
            self.outputs_map[f"{prop.name}.mass"].value = masses[i]


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
                        f"'The Jacobian for the FuelCellLowFidelity is not computed for input '{inp}' and output '{out}' "
                    )
        return jac