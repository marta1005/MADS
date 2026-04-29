from typing import Any

from ambiance import Atmosphere
from assembly import MissionAssembly, MissionSegment, PropulsionSystem
from scenario import Variable
from solvers import BaseSolver, SolverFactory
from solvers.propulsion.rubber_engine_uav_functions import *


# TODO @Tim: split discipline into smaller ones for each component
# TODO @Tim: include number of fuel tanks as an input.
@SolverFactory.register("propulsion", "rubber_engine_uav_sizing")
class rubber_engine_uav_sizing(BaseSolver):
    required_variables = {
        PropulsionSystem: ["type"],
        MissionAssembly: [
            "mission_segments",
            "fuel_used_total",
            "battery_mass_used_total",
        ],
        MissionSegment: [
            "type",
            "lift_drag_ratio",
            "airspeed_start",
            "hybridization",
            "altitude_start",
            "mass_start",
        ],
    }

    def __init__(self) -> None:
        #        # parameter dictionary
        #        self.parameters = None

        # components
        self.PropulsionSystems = None
        self.missions = None

        # state of the variables to be observed
        # self.observable_space = {}

        # inputs
        self.altitude_start = None
        self.mission_segment_types = None
        self.m_tom = None
        self.battery_mass_used_total = None
        self.fuel_used_total = None
        self.thrustSafetyFactor = None
        self.rotorRadius = None
        self.propellerEfficiency = None
        self.methodRotor = None
        self.number_of_blades = None
        self.estimatedGlideRatio = None
        self.airspeed = None
        self.hybridization = None
        self.density_air = None
        self.engine_combustion_efficiency = None
        self.engine_electric_efficiency = None
        self.power_specific_engine_mass_combustion = None
        self.power_specific_engine_volume_combustion = None
        self.fuelTankEmptyMassFraction = None
        self.fuelDensity = None
        self.number_of_engines_combustion = None
        self.escEfficiency = None
        self.method_components = None
        self.number_of_engines_electric = None
        self.escMassDensity = None
        self.batteryMassDensity = None
        self.powerSpecificESCMass = None
        self.power_specific_engine_mass_electric = None
        self.motor_mass_density_electric = None

        # outputs
        self.output = None

    def set_options(self, parameters):
        # parameters set from overall MDA(O)
        #        self.parameters = parameters
        self.fuelDensity = parameters["fuelDensity"]
        self.number_of_engines_electric = parameters["number_of_engines_electric"]
        self.number_of_engines_combustion = parameters["number_of_engines_combustion"]
        self.thrustSafetyFactor = parameters["thrustSafetyFactor"]
        self.rotorRadius = parameters["rotorRadius"]
        self.propellerEfficiency = parameters["propellerEfficiency"]
        self.methodRotor = parameters["methodRotor"]
        self.number_of_blades = parameters["number_of_blades"]
        self.engine_combustion_efficiency = parameters["engine_combustion_efficiency"]
        self.engine_electric_efficiency = parameters["engine_electric_efficiency"]
        self.power_specific_engine_mass_combustion = parameters[
            "power_specific_engine_mass_combustion"
        ]
        self.power_specific_engine_volume_combustion = parameters[
            "power_specific_engine_volume_combustion"
        ]
        self.fuelTankEmptyMassFraction = parameters["fuelTankEmptyMassFraction"]
        self.escEfficiency = parameters["escEfficiency"]
        self.method_components = parameters["method_components"]
        self.escMassDensity = parameters["escMassDensity"]
        self.batteryMassDensity = parameters["batteryMassDensity"]
        self.powerSpecificESCMass = parameters["powerSpecificESCMass"]
        self.power_specific_engine_mass_electric = parameters[
            "power_specific_engine_mass_electric"
        ]
        self.motor_mass_density_electric = parameters["motor_mass_density_electric"]
        self.propulsion_system_architecture = parameters[
            "propulsion_system_architecture"
        ]

    def get_state():
        return [self.PropulsionSystems, self.missions]

    def set_state(self, components):
        self.PropulsionSystems = list(
            filter(
                lambda x: isinstance(x, PropulsionSystem),
                components,
            ),
        )
        self.propellant_type = self.PropulsionSystems[0].type

        self.missions = list(
            filter(
                lambda x: isinstance(x, MissionAssembly),
                components,
            ),
        )
        self.mission_segment_types = [
            seg.type for seg in self.missions[0].mission_segments
        ]
        self.estimatedGlideRatio = [
            seg.lift_drag_ratio for seg in self.missions[0].mission_segments
        ]
        self.airspeed = [
            seg.airspeed_start for seg in self.missions[0].mission_segments
        ]
        self.hybridization = [
            seg.hybridization for seg in self.missions[0].mission_segments
        ]
        self.altitude_start = [
            seg.altitude_start for seg in self.missions[0].mission_segments
        ]
        self.fuel_used_total = self.missions[0].fuel_used_total
        self.battery_mass_used_total = self.missions[0].battery_mass_used_total
        self.m_tom = self.missions[0].mission_segments[0].mass_start

    def run(self):
        # initialize outputs dictionary
        outputs = {}

        # compute air density
        atmosphere = Atmosphere(self.altitude_start)
        self.density_air = atmosphere.density

        # Compute power demand for engines and rotors during cruise and hover
        (
            required_engine_power_total_combustion,
            required_engine_power_total_electric,
            real_power_per_rotor,
        ) = power_demand_mission_segments(
            mission_segment_types=self.mission_segment_types,
            m_tom=self.m_tom,
            estimatedGlideRatio=self.estimatedGlideRatio,
            thrustSafetyFactor=self.thrustSafetyFactor,
            rotorRadius=self.rotorRadius,
            airspeed=self.airspeed,
            propellerEfficiency=self.propellerEfficiency,
            hybridization=self.hybridization,
            engine_combustion_efficiency=self.engine_combustion_efficiency,
            engine_electric_efficiency=self.engine_electric_efficiency,
            density_air=self.density_air,
        )

        # in the serial case, the electric motors must be able to pass the entire power required by the rotors (taking into account their own efficiency!)
        if self.propulsion_system_architecture == "serial":
            required_engine_power_total_electric = (
                required_engine_power_total_electric
                + required_engine_power_total_combustion
                * self.engine_combustion_efficiency
                / self.engine_electric_efficiency
            )

        if "kerosene" in self.propellant_type:
            outputs_combustion = rubber_engine_sizing_combustion_uav(
                power_specific_engine_mass_combustion=self.power_specific_engine_mass_combustion,
                power_specific_engine_volume_combustion=self.power_specific_engine_volume_combustion,
                fuelTankEmptyMassFraction=self.fuelTankEmptyMassFraction,
                fuelDensity=self.fuelDensity,
                methodRotor=self.methodRotor,
                fuel_mass_total=self.fuel_used_total,
                number_of_blades=self.number_of_blades,
                rotorRadius=self.rotorRadius,
                number_of_engines_combustion=self.number_of_engines_combustion,
                required_engine_power_total_combustion=required_engine_power_total_combustion,
                real_power_per_rotor=real_power_per_rotor,
            )

        # update outputs dictionary
        outputs.update(outputs_combustion)

        if "battery" in self.propellant_type:
            outputs_electric = rubber_engine_sizing_electric_uav(
                escEfficiency=self.escEfficiency,
                methodRotor=self.methodRotor,
                method_components=self.method_components,
                battery_mass_total=self.battery_mass_used_total,
                number_of_blades=self.number_of_blades,
                rotorRadius=self.rotorRadius,
                number_of_engines_electric=self.number_of_engines_electric,
                escMassDensity=self.escMassDensity,
                batteryMassDensity=self.batteryMassDensity,
                powerSpecificESCMass=self.powerSpecificESCMass,
                power_specific_engine_mass_electric=self.power_specific_engine_mass_electric,
                motor_mass_density_electric=self.motor_mass_density_electric,
                required_engine_power_total_electric=required_engine_power_total_electric,
                real_power_per_rotor=real_power_per_rotor,
            )
        # update outputs dictionary
        outputs.update(outputs_electric)

        # save outputs
        self.output = outputs

    def compute_output(
        self,
        requested_outputs: list[str],
        mapped_outputs: list[Variable],
    ) -> dict[str, Any]:
        """_summary_: post-process data if needed and collect results"""
        outputs = {}
        for out in requested_outputs:
            if output_var := next((o for o in mapped_outputs if o.name == out), False):
                out_type = output_var.output_type
                try:
                    outputs[out] = self.output[out_type]
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}",
                    )

        return outputs
