import copy
from dataclasses import dataclass, field
from typing_extensions import Self
import numpy as np
from scipy.optimize import fsolve

import importlib


from collections.abc import Sequence
from pathlib import Path
from typing import Any

from multiads.solvers import SolverOptions
from multiads.assembly import Aircraft, Environment

from multiads.scenario import BaseVariable, InnerVariableFloat

from multiads.solvers.mission.segments import *

from multiads.assembly.envelope import Segment

class Options(SolverOptions):
    def __init__(
        self,
        *,
        aero_property_type: str = "global", # define the property type for the mass properties ["global", "body","wind"]
        gravitation: float = 9.81,           # [m/s^2]
        area_ref: float = 1.0,               # area reference for non-dimensionalization [m^2]
        length_ref: float = 1.0,             # length reference for non-dimensionalization [m]
        phase_name: str = "climb",              # name of the phase to consider
        segment_name: str = "ConstMaConstAoA",  # semgent used for mission sizing
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.aero_property_type = aero_property_type
        self.gravitation = gravitation
        self.area_ref = area_ref
        self.length_ref = length_ref
        self.phase_name = phase_name
        self.segment_name = segment_name

class Driver:
    """_summary_

    Raises:
        RuntimeError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(
        self,
        aircraft: Aircraft,
        environment: Environment,
        massproperties: InnerVariableFloat,
        stability_controlproperties : InnerVariableFloat,
        segments: Sequence[Segment],
        options: Options,
    ) -> None:
        
        # collect options
        self.options = options
        # recover environment variables
        self.environment = environment
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.velocity
        
        # components
        self.aircraft = aircraft
        
        # segments
        self.segment = segments[0]
        
        # inner variables mapping
        self.massproperties = massproperties
        self.stability_controlproperties = stability_controlproperties
        
        # properties
        self.load_factor: Sequence[float] = []
    
    def run(self):

        # dynamic import of libraries
        # "MyModule"
        module_name = "multiads.solvers.mission.segments."+self.options.phase_name+"."+self.options.segment_name
        # "MyClass"
        class_name = self.options.segment_name                                                             

        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except ValueError as ve:
            print(f"{module} or {cls} not found in mission solver")
        
        # init the class with assigned input types
        # __init__(...,segment,environment,massproperties,stability_control_properties)
        # print(dir(self.segment))
        instance = cls(self.segment,self.environment,self.massproperties,self.stability_controlproperties)
        
        # run method        
        instance.run()  # Output:
        
        # collect output of the run
        self.load_factor = instance.load_factor
        self.delta_energy = instance.delta_energy
        self.mission_residual = instance.res_forces
 