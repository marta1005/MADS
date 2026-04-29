from __future__ import annotations

import copy

from typing import TYPE_CHECKING
import numpy as np

from multiads.assembly import (
    Wing,
    Propeller,
    Fuselage,
    Aircraft,
    Environment,
    MADSComponent,
    copy_components,
    flatten_components,
)

from multiads.assembly.envelope import(
    MADSPhase,
    flatten_segments,
)

from multiads.assembly.envelope import MADSPhase, Segment

import numpy as np
from numpy.typing import NDArray

from multiads.scenario import BaseVariable, InnerVariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.mission.mission_low_fi_lib import Driver, Options

from multiads.scenario.mass_properties import MassPropertiesVariable
from multiads.scenario.aero_derivatives import AeroDerivativesVariable
from multiads.scenario.stability_control_properties import StabilityControlVariable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class MissionSizing(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.aircraft: Sequence[Aircraft] | None = None
        self.segments:  Sequence[MADSPhase] | None = None
        self.outputs_map: dict[str, InnerVariableFloat] | None = None
    
    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        segments: Sequence[MADSPhase],
    ) -> Sequence[MADSComponent]:
    
        # Filter components
        _components = copy_components(components)
        components_flat = flatten_components(_components)
        ## select aircraft
        if ac := next((c for c in components_flat if isinstance(c, Aircraft)), None):
            self.aircraft = ac
        else:
            print(f"warning...model 'SnC' using DEFAULT Aircraft()")
            self.aircraft = Aircraft(name="Default :: Aircraft")
        ## environment
        if env := next((c for c in components_flat if isinstance(c, Environment)),None):
            self.environment = env
        else:
            print(f"warning...model 'SnC' using DEFAULT Environment()")
            self.environment = Environment(name="Default :: Environment", height=0.0, speed=150.0)
            
        # create in/out variables list
        self.inputs: list[BaseVariable] = []
        self.outputs: list[InnerVariableFloat] = []
        # create segment list - local
        self.segments: list = []

        # filter segments
        _segments = copy.deepcopy(segments)
        segments_flat = flatten_segments(_segments)
        if sg := next((s for s in segments_flat if isinstance(s, Segment)), None):
            # segments as list
            self.segments.append(sg)
            # loop over segment
            for nn, seg in enumerate(self.segments):

                # print(f"{self.aircraft.name}.{seg.name}.load_factor")
                load_factor = InnerVariableFloat(f"{self.aircraft.name}.{seg.name}.load_factor", 0.0)
                self.outputs.append(load_factor)
                
                # climb angle of the mission
                if ca := seg.variables.get("climb_angle"):
                    self.inputs.append(ca)
        else:
            # inner varaiable :: load factor independent of segment
            load_factor = InnerVariableFloat(f"{self.aircraft.name}.load_factor", 0.0)

        # Extract variables
        ## mass from the aircraft
        if fs := self.aircraft.variables.get("mass"):
            self.inputs.append(fs)

        # include ass properties if needed # ! to be extended to segments
        self.inputs.append(
            MassPropertiesVariable.from_num_elements(
                f"{self.aircraft.name}.mass_properties", 1,
            )
        )

        # consolidate inputs
        # stability_conrolproperties neeed to extract the aerodynamic forces # ! to be extended to segments
        self.stabilityproperties = {}
        self.inputs.append(
            StabilityControlVariable.zeros(
                f"{self.aircraft.name}.stability_control_properties",
            )
        )

        
        # print(dir(self.aircraft))
        # print(self.aircraft.variables)

        # update in/out dictionary
        def update_io(
            comp: None,
            inputs: dict[str,AeroDerivativesVariable],
            outputs: dict[str, InnerVariableFloat],
        ) -> None:
            # update component with mass properties
            # inner variables allocation
            ## all aero derivatives if needed
            ## contains all the derivative see ./scenario/aerodynamic_derivatives
            #if fs:=comp.variables.get("aeroderivatives"):
            #    self.inputs.append(fs)
            #else:
            #    inputs[comp.name] = AeroDerivativesVariable(
            #            f"{comp.name}.aeroderivatives", np.zeros((12,6),dtype=float)
            #    )
            pass

        # create a map of the output
        self.outputs_map = {v.name: v for v in self.outputs}

        return [self.aircraft]
    
    def set_state(self, components):
        pass

    def _run(self)-> None:
        if self.aircraft is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)
        
        # get variables needed by the solver
        massproperties = copy.deepcopy(
            next(i for i in self.inputs if i.name == f"{self.aircraft.name}.mass_properties"),
        )
        
        stability_controlproperties = copy.deepcopy(
            next(i for i in self.inputs if i.name == f"{self.aircraft.name}.stability_control_properties"),
        )

        # Run MissionSizing driver
        self.driver = Driver(
            aircraft = self.aircraft,
            environment = self.environment,
            massproperties = massproperties,
            stability_controlproperties = stability_controlproperties,
            segments = self.segments,
            options = self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None:
            msg = f"The outputs of solver '{type(self).__name__} are not initialized"
            raise RuntimeError(msg)

        # loop over the segments to return the load factor
        for seg in self.segments:
            self.outputs_map[f"{self.aircraft.name}.{seg.name}.load_factor"].value = self.driver.load_factor

        # ....


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
