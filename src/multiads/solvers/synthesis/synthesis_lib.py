from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess as sp
from typing import Any, Optional, Union
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

import assembly
from assembly import MADSComponent
from assembly import MassProperties
from assembly import AerodynamicProperties


def _ds_options(comp: MADSComponent) -> dict[str, Any]:
    options = {}
    if hasattr(comp, "options"):
        options = comp.options.get("synthesis", {})
    return deepcopy(options)


@dataclass
class Section:
    twist: float

    @classmethod
    def from_component(cls, comp: assembly.Section) -> Self:
        return Section(comp.twist)


@dataclass
class Span:
    length: float
    sweep: float
    dihed: float

    @classmethod
    def from_component(cls, comp: assembly.Span) -> Self:                
        return Span(comp.length, comp.sweep, comp.dihed)


@dataclass
class Wing:
    name: str
    sections: list[Section] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    beam_nodes: list[list[float]] = None

    @classmethod
    def from_component(cls, comp: assembly.Wing) -> Self:
        sections = [Section.from_component(s) for s in comp.sections]
        spans = [Span.from_component(s) for s in comp.spans]
        return Wing(
            name=comp.name,
            sections=sections,
            beam_nodes=comp.beam_nodes,
            spans=spans,
        )


@dataclass
class Aircraft:
    name: str
    aerodynamicproperties: list[AerodynamicProperties] = field(default_factory=list)
    massproperties: MassProperties = None
    wing_displ_z: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wing_rot_y: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @classmethod
    def from_component(cls, comp: assembly.Aircraft) -> Self:
        opts = _ds_options(comp)

        aircraft = Aircraft(
            name=comp.name,
            aerodynamicproperties=comp.aerodynamicproperties,
            massproperties=comp.massproperties,
            wing_displ_z=comp.wing_displ_z,
            wing_rot_y=comp.wing_rot_y,
            **opts,
        )

        return aircraft


@dataclass
class DSoptions:
    name: str = "synthesis"
    n_threads: int = 1


@dataclass
class DSdriver:
    aircraft: list[Aircraft] = field(default_factory=list)
    wings: list[Wing] = field(default_factory=list)
    environment: assembly.Environment = field(default_factory=assembly.Environment)
    options: DSoptions = field(default_factory=DSoptions)
    aero_prop_body: AerodynamicProperties = None
    aero_prop_global: AerodynamicProperties = None
    indices:  list[float] = None
    spans_new:  list[float] = None
    dihedrals_new:  list[float] = None
    thetas_new:  list[float] = None

    """  The design synthesis driver is used to compute the global properties of the aircraft, starting from the 
         Aircraft(MADSComponent) initial properties. the driver produces global aircraft output which permit to update and fill
         the aircraft assembly component.
         The Design Synthesis discipline, togheter with the Aircraft(MADSComponent) assembly component, provides the description of 
         the aircraft global properties. (global -> not associated to the single components (e.g. wing, fuselage, propellers...))
    """

    def compute_aircraft_properties(self) -> None:

        if self.aircraft[0].aerodynamicproperties is not None:
        
            # Extract x_cg
            x_cg = self.aircraft[0].massproperties.massVector[1]

            # extract the aerodynamic properties of the aircraft, distinguishing between global and cg reference provided by the user
            for aero_prop in self.aircraft[0].aerodynamicproperties:
                if aero_prop.reference_type == "global":
                    self.aero_prop_global = aero_prop
                elif aero_prop.reference_type == "body":
                    self.aero_prop_body = aero_prop

            self.aero_prop_body = deepcopy(self.aero_prop_global)

            # Moment transport equation in the center of gravity, applied to stability derivatives in global coordinates
            if self.aero_prop_body.AeroDerivatives_alpha is not None:
                self.aero_prop_body.AeroDerivatives_alpha[4] = self.aero_prop_global.AeroDerivatives_alpha[4] + x_cg * self.aero_prop_global.AeroDerivatives_alpha[2]
                self.aero_prop_body.AeroDerivatives_alpha[5] = self.aero_prop_global.AeroDerivatives_alpha[5] - x_cg * self.aero_prop_global.AeroDerivatives_alpha[1]

            if self.aero_prop_body.AeroDerivatives_beta is not None:
                self.aero_prop_body.AeroDerivatives_beta[4] = self.aero_prop_global.AeroDerivatives_beta[4] + x_cg * self.aero_prop_global.AeroDerivatives_beta[2]
                self.aero_prop_body.AeroDerivatives_beta[5] = self.aero_prop_global.AeroDerivatives_beta[5] - x_cg * self.aero_prop_global.AeroDerivatives_beta[1]

            if self.aero_prop_body.AeroDerivatives_p is not None:
                self.aero_prop_body.AeroDerivatives_p[4] = self.aero_prop_global.AeroDerivatives_p[4] + x_cg * self.aero_prop_global.AeroDerivatives_p[2]
                self.aero_prop_body.AeroDerivatives_p[5] = self.aero_prop_global.AeroDerivatives_p[5] - x_cg * self.aero_prop_global.AeroDerivatives_p[1]

            if self.aero_prop_body.AeroDerivatives_q is not None:
                self.aero_prop_body.AeroDerivatives_q[4] = self.aero_prop_global.AeroDerivatives_q[4] + x_cg * self.aero_prop_global.AeroDerivatives_q[2]
                self.aero_prop_body.AeroDerivatives_q[5] = self.aero_prop_global.AeroDerivatives_q[5] - x_cg * self.aero_prop_global.AeroDerivatives_q[1]

            if self.aero_prop_body.AeroDerivatives_r is not None:
                self.aero_prop_body.AeroDerivatives_r[4] = self.aero_prop_global.AeroDerivatives_r[4] + x_cg * self.aero_prop_global.AeroDerivatives_r[2]
                self.aero_prop_body.AeroDerivatives_r[5] = self.aero_prop_global.AeroDerivatives_r[5] - x_cg * self.aero_prop_global.AeroDerivatives_r[1]

            if self.aero_prop_body.AeroDerivatives_alpha_dot is not None:
                self.aero_prop_body.AeroDerivatives_alpha_dot[4] = self.aero_prop_global.AeroDerivatives_alpha_dot[4] + x_cg * self.aero_prop_global.AeroDerivatives_alpha_dot[2]
                self.aero_prop_body.AeroDerivatives_alpha_dot[5] = self.aero_prop_global.AeroDerivatives_alpha_dot[5] - x_cg * self.aero_prop_global.AeroDerivatives_alpha_dot[1]

            if self.aero_prop_body.AeroDerivatives_beta_dot is not None:
                self.aero_prop_body.AeroDerivatives_beta_dot[4] = self.aero_prop_global.AeroDerivatives_beta_dot[4] + x_cg * self.aero_prop_global.AeroDerivatives_beta_dot[2]
                self.aero_prop_body.AeroDerivatives_beta_dot[5] = self.aero_prop_global.AeroDerivatives_beta_dot[5] - x_cg * self.aero_prop_global.AeroDerivatives_beta_dot[1]

            if self.aero_prop_body.AeroDerivatives_pitching is not None:
                self.aero_prop_body.AeroDerivatives_pitching[4] = self.aero_prop_global.AeroDerivatives_pitching[4] + x_cg * self.aero_prop_global.AeroDerivatives_pitching[2]
                self.aero_prop_body.AeroDerivatives_pitching[5] = self.aero_prop_global.AeroDerivatives_pitching[5] - x_cg * self.aero_prop_global.AeroDerivatives_pitching[1]

            if self.aero_prop_body.AeroDerivatives_yawing is not None:
                self.aero_prop_body.AeroDerivatives_yawing[4] = self.aero_prop_global.AeroDerivatives_yawing[4] + x_cg * self.aero_prop_global.AeroDerivatives_yawing[2]
                self.aero_prop_body.AeroDerivatives_yawing[5] = self.aero_prop_global.AeroDerivatives_yawing[5] - x_cg * self.aero_prop_global.AeroDerivatives_yawing[1]

            if self.aero_prop_body.AeroDerivatives_delta_aileron is not None:
                self.aero_prop_body.AeroDerivatives_delta_aileron[4] = self.aero_prop_global.AeroDerivatives_delta_aileron[4] + x_cg * self.aero_prop_global.AeroDerivatives_delta_aileron[2]
                self.aero_prop_body.AeroDerivatives_delta_aileron[5] = self.aero_prop_global.AeroDerivatives_delta_aileron[5] - x_cg * self.aero_prop_global.AeroDerivatives_delta_aileron[1]

            if self.aero_prop_body.AeroDerivatives_delta_elevator is not None:
                self.aero_prop_body.AeroDerivatives_delta_elevator[4] = self.aero_prop_global.AeroDerivatives_delta_elevator[4] + x_cg * self.aero_prop_global.AeroDerivatives_delta_elevator[2]
                self.aero_prop_body.AeroDerivatives_delta_elevator[5] = self.aero_prop_global.AeroDerivatives_delta_elevator[5] - x_cg * self.aero_prop_global.AeroDerivatives_delta_elevator[1]

            if self.aero_prop_body.AeroDerivatives_delta_rudder is not None:
                self.aero_prop_body.AeroDerivatives_delta_rudder[4] = self.aero_prop_global.AeroDerivatives_delta_rudder[4] + x_cg * self.aero_prop_global.AeroDerivatives_delta_rudder[2]
                self.aero_prop_body.AeroDerivatives_delta_rudder[5] = self.aero_prop_global.AeroDerivatives_delta_rudder[5] - x_cg * self.aero_prop_global.AeroDerivatives_delta_rudder[1]

    # return derivatives in cg coordinates
    def body_AeroDerivatives_alpha(self):
        return self.aero_prop_body.AeroDerivatives_alpha

    def body_AeroDerivatives_beta(self):
        return self.aero_prop_body.AeroDerivatives_beta

    def body_AeroDerivatives_p(self):
        return self.aero_prop_body.AeroDerivatives_p

    def body_AeroDerivatives_q(self):
        return self.aero_prop_body.AeroDerivatives_q

    def body_AeroDerivatives_r(self):
        return self.aero_prop_body.AeroDerivatives_r
    
    def body_AeroDerivatives_alpha_dot(self):
        return self.aero_prop_body.AeroDerivatives_alpha_dot

    def body_AeroDerivatives_beta_dot(self):
        return self.aero_prop_body.AeroDerivatives_beta_dot

    def body_AeroDerivatives_pitching(self):
        return self.aero_prop_body.AeroDerivatives_pitching
    
    def body_AeroDerivatives_yawing(self):
        return self.aero_prop_body.AeroDerivatives_yawing

    def body_AeroDerivatives_delta_aileron(self):
        return self.aero_prop_body.AeroDerivatives_delta_aileron

    def body_AeroDerivatives_delta_elevator(self):
        return self.aero_prop_body.AeroDerivatives_delta_elevator

    def body_AeroDerivatives_delta_rudder(self):
        return self.aero_prop_body.AeroDerivatives_delta_rudder


    
    # Compute deformed aircraft main wing (new twists, dihedrals, sweeps, spans)
    
    def find_nodes(self):
        """
        *** Routine from LAG2DUST *** 
        It finds the nodes of the MBDYN nodes which are located at the DUST airfoil sections defined in the geom.in file. It then proceeds to locate the ones located at the leading and trailing edge as well as their indices wrt the BDF file
        """
        
        for wing in self.wings:
            if wing.beam_nodes is not None and wing.beam_nodes.size > 0:
                
                beam_nodes = wing.beam_nodes
                y_nodes = beam_nodes[:,1]
                y_airfoils = [0.]
                self.indices = []

                for i, span in enumerate(wing.spans):
                    y_airfoils.append(y_airfoils[-1] + span.length)
                    
                #print(y_airfoils)
                
                for y_dust in y_airfoils:
                    for i, y_mbdyn in enumerate(y_nodes):
                        if (round(y_dust) == round(y_mbdyn)):
                            index = i
                            
                    self.indices.append(index)
                    
                #print(self.indices)
                    
        

    def find_deform(self):
        """
        *** Routine from LAG2DUST *** 
        Using the indices calculated in the previous method, this step finds the deformation of the leading and trailing edge
        nodes in the z direction. Based on that, it calculates the new twists,dihedrals and spans
        """
        
        for wing in self.wings:
            if wing.beam_nodes is not None and wing.beam_nodes.size > 0:
                
                ################################# Information required ######################################

                dz = np.array([0.])
                dz = np.append(dz, self.aircraft[0].wing_displ_z)
                dtheta = np.array([0.])
                dtheta = np.append(dtheta, self.aircraft[0].wing_rot_y)
                
                #print(dz)
                #print(dtheta)
                
                # Old geometric data
                spans_old = []
                dihedrals_old = []
                thetas_old = []
                
                sections = wing.sections
                spans = wing.spans
                
                for i, h in enumerate(sections):
                    thetas_old.append(sections[i].twist)
                    
                for i, h in enumerate(spans):
                    spans_old.append(spans[i].length)
                    dihedrals_old.append(spans[i].dihed)
                    
                #print(thetas_old)
                #print(spans_old)
                #print(dihedrals_old)
                
                # New geometric data
                self.spans_new =  np.zeros(len(spans_old))
                self.dihedrals_new =  np.zeros(len(dihedrals_old))
                self.thetas_new =  np.zeros(len(thetas_old))
                
        
                ######################### spans & dihedrals ##########################

                for i in range(len(spans_old)):
                    
                    delta_z = (dz[self.indices[i+1]]) - (dz[self.indices[i]])
                    delta_y = spans_old[i]
                    
                    #print(delta_z)
                    #print(delta_y)
                    
                    self.dihedrals_new[i] = dihedrals_old[i] + np.degrees(np.arcsin(delta_z/delta_y))
                    self.spans_new[i] = delta_y * np.cos(np.radians(self.dihedrals_new[i]))
                    
                #print(self.dihedrals_new)
                #print(self.spans_new)

                ######################################## twists ###########################################
                
                for i in range(len(thetas_old)):
                    #print(i)
                    #print(self.indices[i])
                    self.thetas_new[i] = thetas_old[i] + np.degrees(dtheta[self.indices[i]])
                    
                #print(self.thetas_new)
                        
    
    # return deformed wing geometry
    def thetas_deformed(self):
        return self.thetas_new

    def spans_deformed(self):
        return self.spans_new

    def dihedrals_deformed(self):
        return self.dihedrals_new

