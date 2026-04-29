from __future__ import annotations

from typing import TYPE_CHECKING

from multiads.assembly import (
    FuelCell,
    PowerManagSyst,
    Environment,
    MADSComponent,
    copy_components,
    flatten_components,
)

from multiads.scenario import BaseVariable, InnerVariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.power_supply.fuel_cell_low_fidelity_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class FuelCellLowFidelity(BaseSolver):
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.fuelcells: Sequence[FuelCell] | None = None
        self.powermanagment: Sequence[PowerManagSyst] | None = None
        self.outputs_map: dict[str, InnerVariable] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv,
    ) -> Sequence[MADSComponent]:
        # Filter fuel cells
        _components = copy_components(components)
        c_flat = flatten_components(_components)
        self.fuelcells = [c for c in c_flat if isinstance(c, FuelCell)]
        # iterate over additional dependencies
        ## power management
        if pwr := next((c for c in c_flat if isinstance(c, PowerManagSyst)), None):
            self.powermanagement = pwr
        else:
            print(f"warning...model 'FuelCellLowFidelity' using DEFAULT PowerManagSyst()")
            self.powermanagement = PowerManagSyst(name="Default :: Power Management",type="Hybrid")
        ## environment
        if env := next((c for c in c_flat if isinstance(c, Environment)),None):
            self.environment = env
        else:
            print(f"warning...model 'FuelCellLowFidelity' using DEFAULT Environment()")
            self.environment = Environment(name="Default :: Environment", height=0.0,speed=150.0)

        # Extract variables
        self.inputs: list[BaseVariable] = []
        for fuelcell in self.fuelcells:
            if v := fuelcell.variables.get("power"):
                self.inputs.append(v)

        self.outputs: list[InnerVariableFloat] = []
        for fuelcell in self.fuelcells:
            self.outputs.extend(
                [
                    InnerVariableFloat(f"{fuelcell.name}.mass", 0.0), 
                    InnerVariableFloat(f"{fuelcell.name}.h2_fuel_flow", 0.0),
                    InnerVariableFloat(f"{fuelcell.name}.waste_heat", 0.0),
                ],
            )

        self.outputs_map = {v.name: v for v in self.outputs}

        return self.fuelcells
    
    def _run(self) -> None:
        if self.fuelcells is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        self.driver = Driver(
            fuelcells=self.fuelcells,
            powermanag_sys=self.powermanagement,
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

        masses = self.driver.mass
        fuelflows = self.driver.h2_fuel_flow
        wasteheatfc = self.driver.waste_heat

        for i, fuelcell in enumerate(self.fuelcells):
            self.outputs_map[f"{fuelcell.name}.mass"].value = masses[i]
            self.outputs_map[f"{fuelcell.name}.h2_fuel_flow"].value = fuelflows[i]
            self.outputs_map[f"{fuelcell.name}.waste_heat"].value = wasteheatfc[i]


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