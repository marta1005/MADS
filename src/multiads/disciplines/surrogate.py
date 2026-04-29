import logging
from typing import Any, Callable, Optional

import numpy as np

# gemseo 6
from gemseo import compute_doe, create_design_space
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.surrogate import SurrogateDiscipline as GemseoSurrogate

# MDF settings
from gemseo.formulations.mdf import MDF_Settings

# gemseo 6
from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo as MLAlgo

# Disciplinary OPt
# DOE Settings - https://gemseo.readthedocs.io/en/6.0.0/algorithms/doe_algos.html
from gemseo.settings.doe import LHS_Settings
from numpy.typing import NDArray

from multiads.disciplines import MADSDiscipline
from multiads.scenario import BaseVariable, MADSScenario


class SurrogateDiscipline(GemseoSurrogate):
    def __init__(
        self,
        discipline: list[MADSDiscipline],
        regressor_name: str,
        num_samples: int = 0,
        inputs: list[BaseVariable] = None,
        outputs: list[BaseVariable] = None,
        train: Optional[bool] = True,
        new_doe: Optional[bool] = False,
        **regressor_args: Any,
    ):
        """_summary_

        Args:
            discipline (list[MADSDiscipline]): _description_
            regressor_name (str): _description_
            num_samples (int, optional): _description_. Defaults to 0.
            inputs (list[BaseVariable], optional): _description_. Defaults to None.
            outputs (list[BaseVariable], optional): _description_. Defaults to None.
            train (Optional[bool], optional): _description_. Defaults to True.
            new_doe (Optional[bool], optional): defines if a new doe must be executed. Defaults to False.

        """
        # Create discipline
        if len(discipline) == 1:
            self.discipline = discipline[0]

        if train == True:
            if new_doe == True:
                # check if a scenario must be created for the execution
                if len(discipline) == 1:
                    # data collector
                    data = IODataset()

                    print(
                        f"\n----- Processing Design Of Experiment for {self.discipline.name} discipline... -----\n",
                    )
                    inputs_doe = self.propose_doe(inputs, num_samples)

                    # Evaluate DOE
                    outputs_doe = self.evaluate_doe(
                        inputs=inputs_doe,
                        execute=self.discipline.execute,
                        num_samples=num_samples,
                    )
                    print("----- Design Of Experiment completed -----")

                    # Debug
                    # print(inputs_doe)
                    # print(outputs_doe)
                    # print(inputs)
                    # print(outputs)

                    for n, v in inputs_doe.items():
                        data.add_input_variable(n, v)

                    for n, v in outputs_doe.items():
                        data.add_output_variable(n, v)

                    # Train and make surrogate (reorganize data to make them compatible)
                    print("\n----- Saving Training Dataset -----\n")
                    # print(data)

                    with open(f"training_dataset_{self.discipline.name}.csv", "w") as f:
                        data.to_csv(f, index=False)

                    # Assign
                    super().__init__(
                        regressor_name,
                        data=data,
                        disc_name=self.discipline.name + " [Surrogate]",
                    )

                else:
                    print(
                        f"\n----- Processing Design Of Experiment for {discipline[0].name} discipline... -----\n",
                    )

                    # Create scenario for DOE training
                    mads_scenario = MADSScenario()
                    mads_scenario.fill_parameter_space(inputs)
                    mads_scenario.create_scenario(
                        disciplines=discipline,
                        formulation=MDF_Settings(
                            main_mda_name="MDAGaussSeidel",
                        ),  # old #'DisciplinaryOpt',#'BiLevel', #'MDF',
                        objective_name=[out.name for out in outputs],
                        scenario_type="DOE",
                    )

                    # Execute scenario
                    doe_settings = LHS_Settings(n_samples=num_samples)
                    mads_scenario.scenario.execute(doe_settings)
                    # gemseo #5
                    # mads_scenario.scenario.execute({"n_samples":num_samples, "algo":"LHS"})
                    data = mads_scenario.scenario.to_dataset(opt_naming=False)
                    print("----- Design Of Experiment completed -----")

                    # Train and make surrogate (reorganize data to make them compatible)
                    print("\n----- Training Dataset -----\n")

                    # to reorganize the data because output of data is a single vector
                    dataset = IODataset()
                    for n, v in data.input_dataset.items():
                        dataset.add_input_variable(n[1], v)
                    dataset.add_output_group(
                        data.output_dataset,
                        [out.name for out in outputs],
                        {out.name: len(out.value) for out in outputs},
                    )

                    print(dataset)

                    with open(f"training_dataset_{discipline[0].name}.csv", "w") as f:
                        dataset.to_csv(f, index=False)

                    # Assign
                    super().__init__(
                        regressor_name,
                        data=dataset,
                        disc_name=discipline[0].name + " [Surrogate]",
                    )

            elif new_doe == False:
                logging.info(f"reading training_dataset_{self.discipline.name}.csv")

                from pandas import read_csv

                # dataset should be a csv file with "," delimiter and headers
                # typical output from gemseo discipline
                #  e.g.
                #          inputs,inputs,inputs,outputs
                #          electrical_power_from_battery,batt_nominal_voltage,batt_flight_time,BATT_WEIGHT
                #          0,0,0,0
                #          345625.17458812945,279.216410313634,19.54279201936402,683.0418661723377
                try:
                    dataframe = read_csv(
                        f"training_dataset_{self.discipline.name}.csv",
                        delimiter=",",
                        header=[0, 1, 2],
                        index_col=None,
                    )
                except ValueError:
                    print(
                        f"the file training_dataset_{self.discipline.name}.csv might not be avaible or having the wrong *.csv format ",
                    )

                dataframe.columns = dataframe.columns.set_levels(
                    dataframe.columns.levels[2],
                    level=2,
                )
                data = IODataset(dataframe)

                # Assign
                super().__init__(
                    regressor_name,
                    data=data,
                    disc_name=self.discipline.name + " [Surrogate]",
                )

            else:
                raise ValueError(
                    "in surrogate.py set new DoE to True or provide dataset",
                )

        else:
            pass

    def propose_doe(
        self,
        inputs: list[BaseVariable],
        num_samples: int,
    ) -> dict[str, NDArray]:
        # Initialize design space
        space = create_design_space()

        # Create discipline
        for v in inputs:
            if not isinstance(v, np.ndarray):
                v_array = np.atleast_1d(v.value)
            else:
                v_array = v.value

            size = v_array.shape[0]

            lb = v.lb
            ub = v.ub
            val = v.value

            # Create bounds if doesnt have them (mandatory)
            lb = np.where(np.isneginf(lb), -3 * val, lb)
            ub = np.where(np.isposinf(ub), 3 * val, ub)

            # Manage the case val=0
            equal_bounds = lb == ub
            lb = np.where(equal_bounds, -1.0, lb)
            ub = np.where(equal_bounds, 1.0, ub)

            # Debug
            # print(v.name, val, lb, ub)

            space.add_variable(
                name=v.name,
                size=size,
                type_="float",
                value=val,
                lower_bound=float(lb),
                upper_bound=float(ub),
            )

        # Evaluate inputs set for doe
        doe = compute_doe(space, algo_name="LHS", n_samples=num_samples)
        # pydoe = PyDOE()
        # doe = pydoe.execute(space, algo_name="PYDOE_LHS", n_samples=100)
        inputs_doe = {}
        k = 0

        # Re-organize Inputs for doe (vectors are broken up)
        for v in inputs:
            v_array = np.atleast_1d(v.value)
            size = v_array.shape[0]

            for i in range(size):
                var_name = f"{v.name}[{i}]" if size > 1 else v.name
                inputs_doe[var_name] = doe[:, k]
                k += 1

        return inputs_doe

    def evaluate_doe(
        self,
        execute: Callable,
        inputs: dict[str, NDArray],
        num_samples: int,
    ) -> dict[str, NDArray]:
        outputs: dict[str, NDArray] = dict()

        for i in range(num_samples):
            # Assign inputs set for the DOE
            out = execute(
                {
                    name: np.array([vals[i]]) for name, vals in inputs.items()
                },  # this gives also the inputs in the dictionary
            )

            input_names = set(inputs.keys())

            for name, value in out.items():
                size = value.shape[0]

                # assign output only if different from input, and manage the output-vector case
                var_name = name
                if var_name not in input_names:
                    # Debug
                    # print(value)
                    # print("-------")

                    if size == 1:
                        outputs[var_name] = np.append(
                            outputs.get(var_name, np.array([])),
                            [value[0]],
                            axis=0,
                        )
                    else:
                        value = np.atleast_2d(value)
                        outputs[var_name] = np.vstack(
                            [
                                outputs.get(var_name, np.empty((0, value.shape[1]))),
                                value,
                            ],
                        )

                    # Debug
                    # print("-----------------")
                    # print(var_name)
                    # print(outputs[var_name])
                    # print("-----------------")

        return outputs

    def train_surrogate(
        self,
        ml_algorithm: Type[MLAlgo],
        inputs: dict[str, NDArray],
        outputs: dict[str, NDArray],
        **regressor_args: Any,
    ) -> None:
        data = IODataset()

        for n, v in inputs.items():
            data.add_input_variable(n, v)
        for n, v in outputs.items():
            data.add_output_variable(n, v)

        self.surrogate = ml_algorithm(data, **regressor_args)
        self.surrogate.learn()

    def predict(self, inputs: dict[str, NDArray]) -> dict[str, NDArray]:
        if self.surrogate is None:
            raise ValueError("Surrogate model is not trained yet.")
        return self.surrogate.predict(inputs)

    def verify_surrogate(self, error_type="R2Measure") -> float:
        """_summary_

        Returns:
            _type_: _description_

        """
        # introduce measure metric
        r2 = self.get_error_measure(error_type)
        # compute metric
        learning_r2 = r2.compute_learning_measure()

        return learning_r2
