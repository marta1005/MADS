from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemseo_fmu.disciplines.fmu_discipline import FMUDiscipline
from gemseo_fmu.disciplines.static_fmu_discipline import StaticFMUDiscipline

from multiads.assembly import Battery
from multiads.solvers import SolverOptions

if TYPE_CHECKING:
    import numpy as np


class Options(SolverOptions):
    def __init__(
        self,
        *,
        static: bool = True,
        validate: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.static = static
        self.validate = validate


class Driver:
    def __init__(self, batteries: Sequence[Battery], options: Options) -> None:
        self.options = options
        self.batteries = batteries
        self.weights: list[np.float64] = []
        self.volumes: list[np.float64] = []
        self.voltages: list[np.float64] = []
        self.capacities: list[np.float64] = []

        fmu_filepath = Path(__file__).parent / "Model_BAT_V4.fmu"
        if options.static:
            self.fmu = StaticFMUDiscipline(fmu_filepath, validate=options.validate)
        else:
            self.fmu = FMUDiscipline(fmu_filepath, validate=options.validate)

    def run(self) -> None:
        for batt in self.batteries:
            output = self.fmu.execute(
                PowerKW=batt.power / 1000.0,  # kW
                NomVoltageV=batt.nominal_voltage,
                FlightTimeMin=batt.flight_time,
            )
            self.weights.append(output["WeightKg"][0])
            self.volumes.append(output["VolumeL"][0])
            self.voltages.append(output["voltage_output"][0])
            self.capacities.append(output["capacity_output"][0])
