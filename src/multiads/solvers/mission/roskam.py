from typing import Any

import numpy as np
import solvers.mission.roskam_lib as rl
from assembly import (
    AerodynamicPolarData,
    Configuration,
    MissionAssembly,
    MissionSegment,
    PropulsionSystem,
    Section,
    Span,
    Wing,
)
from scenario import Variable
from solvers import BaseSolver, SolverFactory


def fuel_used_total(driver):
    return np.asarray([driver.fuel_used_total])


def battery_mass_used_total(driver):
    return np.asarray([driver.battery_mass_used_total])


def lift_to_drag_ratio(driver, segment_number: list[str]):
    lift_to_drag_ratio = driver.retrieve_lift_to_drag_ratio(segment_number)
    return np.atleast_1d(lift_to_drag_ratio)


def angle_of_attack(driver, segment_number: list[str]):
    angle_of_attack = driver.retrieve_angle_of_attack(segment_number)
    return np.atleast_1d(angle_of_attack)


def lift_coefficient(driver, segment_number: list[str]):
    lift_coefficient = driver.retrieve_lift_coefficient(segment_number)
    return np.atleast_1d(lift_coefficient)


@SolverFactory.register("mission", "roskam")
class Roskam(BaseSolver):
    required_variables = {
        PropulsionSystem: ["propellant_type"],
        MissionAssembly: ["mission_segments"],
        MissionSegment: [
            "type",
            "range",
            "mass_start",
            "mass_end",
            "endurance",
            "airspeed_start",
            "airspeed_end",
            "altitude_start",
            "altitude_end",
            #                         "lift_drag_ratio", # this one is removed because this is actually an output in the current setup.
            #                         "angle_of_attack", # this one is removed because this is actually an output in the current setup.
            "fuel_used",
            "battery_mass_used",
            "h2o_used",
            "hybridization",
            "lift_coefficient",
            # "environment", # TODO @Tim: add to assembly !
        ],
        Configuration: ["aerodynamic_polar"],
        AerodynamicPolarData: AerodynamicPolarData.attributes(),
        Wing: ["sections", "spans"],
        Section: Section.attributes(),
        Span: Span.attributes(),
    }

    implemented_outputs = {
        "fuel_used_total": fuel_used_total,
        "battery_mass_used_total": battery_mass_used_total,
        "lift_to_drag_ratio": lift_to_drag_ratio,
        "angle_of_attack": angle_of_attack,
        "lift_coefficient": lift_coefficient,
    }

    def __init__(self) -> None:
        # options of the solver in separated structure
        self.options = rl.Options()
        # simulation driver
        self.driver = None
        # components
        self.configuration = None
        self.propulsion_system = None
        self.mission_assembly = None
        self.wings = None

    def get_state(self):
        return [
            self.configuration,
            self.propulsion_system,
            self.mission_assembly,
            self.wings,
        ]

    def set_state(self, components):
        # extract items
        self.configuration = next(
            filter(lambda x: isinstance(x, Configuration), components)
        )
        self.configuration = rl.Configuration.from_component(self.configuration)
        self.propulsion_system = next(
            filter(lambda x: isinstance(x, PropulsionSystem), components)
        )
        self.propulsion_system = rl.PropulsionSystem.from_component(
            self.propulsion_system
        )
        self.mission_assembly = next(
            filter(lambda x: isinstance(x, MissionAssembly), components)
        )
        self.mission_assembly = rl.MissionAssembly.from_component(self.mission_assembly)
        self.wings = list(filter(lambda x: isinstance(x, Wing), components))

    def run(self):
        """Purpose: run the roskam mission model"""
        # initialise driver object
        self.driver = rl.Driver(
            configuration=self.configuration,
            propulsion_system=self.propulsion_system,
            mission_assembly=self.mission_assembly,
            wings=self.wings,
            options=self.options,
        )

        # run model
        self.driver.run()

    def compute_output(
        self,
        requested_outputs: list[str],
        mapped_outputs: list[Variable],
    ) -> dict[str, Any]:
        """_summary_: post-process data if needed and collect results"""
        outputs = {}
        for out in requested_outputs:
            if output_var := next((o for o in mapped_outputs if o.name == out), False):
                # Callback options
                segment_name = output_var.options.get("roskam", {}).get(
                    "segment_name", []
                )
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = Roskam.implemented_outputs[out_type]
                    outputs[out] = out_function(self.driver, *segment_name)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}",
                    )

        return outputs
