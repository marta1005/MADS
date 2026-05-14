from typing import Any, Callable, Optional, Type
from gemseo.datasets.io_dataset import IODataset
from multiads.scenario import BaseVariable
import logging
from multiads.disciplines import MADSDiscipline
import numpy as np
from gemseo import create_design_space, create_surrogate, compute_doe
from numpy.typing import NDArray
import pandas as pd
# DOE Setting- https://gemseo.readthedocs.io/en/6.0.0/algorithms/doe_algos.html
from gemseo.settings.doe import LHS_Settings

from multiads.disciplines import MADSDiscipline
from multiads.scenario import BaseVariable
from multiads.scenario import MADSScenario


class DoEModelFactory:
    """
    A factory class for creating datasets for training and validation using Design of Experiments (DoE) methods.

    This class provides methods to generate or load datasets for both static and dynamic problems.
    It supports various DoE methods and handles the creation of design spaces, evaluation of disciplines,
    and storage of datasets for later use.
    """

    def __init__(self):
        """
        Initialize the DoEModelFactory.

        This method initializes the factory with no specific settings.
        """
        pass

    def dataset_for_training(self, new_doe: bool, doe_method: str, save_doe:bool,num_samples: int, discipline: Any, inputs: Optional[list[BaseVariable]] = None,outputs: Optional[list[BaseVariable]] = None) -> IODataset:
        """
        Create or load a dataset for training.

        This method generates a new dataset using the specified DoE method and discipline,
        or loads an existing dataset from a CSV file.

        Args:
            new_doe: Boolean flag indicating whether to create a new DoE or load an existing one.
            doe_method: String specifying the DoE method to use (e.g., "LHS", "DOE_LHS").
            num_samples: Integer specifying the number of samples for the DoE.
            discipline: The discipline to evaluate, which must have an `execute` method.
            inputs: Optional list of input variables. If None, the discipline's inputs are used.

        Returns:
            IODataset containing the input and output variables for training.

        Note:
            If `new_doe` is True, the method generates a new DoE, evaluates it on the discipline,
            and saves the dataset to a CSV file. If False, it loads the dataset from the CSV file.
            The CSV file is named based on the discipline's name.
        """
        if len(discipline) == 1:
            discipline = discipline[0]
            data = IODataset()
            self.save_doe=save_doe
            # Dataset generation or dataset loading
            if new_doe == True:
                print(f"\n----- Processing Design Of Experiment for {discipline.name} discipline... -----\n")
                # Proposing DoE in the design space
                inputs_doe = self.propose_doe(inputs, num_samples, doe_method)
                

                # Evaluating DoE obtained from inputs_doe
                outputs_doe = self.evaluate_doe(inputs=inputs_doe, execute=discipline.execute, num_samples=num_samples)
                print("----- Design Of Experiment completed -----")

                # Creating an IODataset
                for n, v in inputs_doe.items():
                    data.add_input_variable(n, v)

                for n, v in outputs_doe.items():
                    data.add_output_variable(n, v)

                print("\n----- Training Dataset -----\n")
                
                if save_doe==True:
                    with open(f"training_dataset_{discipline.name}.csv", "w") as f:
                        data.to_csv(f, index=False)

            if new_doe == False:
                # Loading dataset
                logging.info(f"reading training_dataset_{discipline.name}.csv")
                from pandas import read_csv

                dataframe = read_csv(
                    f"training_dataset_{discipline.name}.csv", delimiter=",", header=[0, 1, 2], index_col=None
                )
                dataframe.columns = dataframe.columns.set_levels(dataframe.columns.levels[2], level=2)
                data = IODataset(dataframe)

                print("\n----- Training Dataset -----\n")
            return data
        else:
                # Process Design of Experiment for multiple disciplines
                print(f"\n----- Processing Design Of Experiment for {discipline[0].name} discipline... -----\n")

                # Create scenario for DOE training
                mads_scenario = MADSScenario()
                mads_scenario.fill_parameter_space(inputs)
                mads_scenario.create_scenario(
                    disciplines=discipline,
                    #formulation=MDF_Settings(main_mda_name="MDAGaussSeidel"),  # old #'DisciplinaryOpt',#'BiLevel', #'MDF',
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

                # Reorganize the data because output of data is a single vector
                dataset = IODataset()
                for n, v in data.input_dataset.items():
                    dataset.add_input_variable(n[1], v)
                dataset.add_output_group(data.output_dataset, [out.name for out in outputs], {out.name: np.size(out.value) for out in outputs})
                
    
                # Filter out zeros by replacing them with the average
                #data_filter = dataset.get_columns(variable_names=[out.name for out in outputs])
                #average = np.mean(data_filter)
                #data_filter = np.where(data_filter == 0, average, data_filter)
                #dataset.update_data(data_filter, variable_names=outputs.name)

                # Save the dataset to a CSV file
                if save_doe==True:
                    with open(f"training_dataset_{discipline[0].name}.csv", "w") as f:
                        dataset.to_csv(f, index=False)
                return dataset
            
    def dataset_for_training_dynamic(self, new_doe: bool, doe_method: str, num_samples: int,save_doe:bool, discipline: Any, inputs: Optional[list[BaseVariable]] = None) -> tuple[NDArray, dict[str, NDArray]]:
        """
        Create or load a dataset for dynamic training.

        This method generates a new dataset for dynamic problems using the specified DoE method and discipline,
        or loads an existing dataset from CSV files.

        Args:
            new_doe: Boolean flag indicating whether to create a new DoE or load an existing one.
            doe_method: String specifying the DoE method to use (e.g., "LHS", "DOE_LHS").
            num_samples: Integer specifying the number of samples for the DoE.
            discipline: The discipline to evaluate, which must have an `execute` method.
            inputs: Optional list of input variables. If None, the discipline's inputs are used.

        Returns:
            A tuple containing:
            - Input data as a numpy array
            - Output data as a dictionary of numpy arrays

        Note:
            If `new_doe` is True, the method generates a new DoE, evaluates it on the discipline,
            and saves the input and output datasets to separate CSV files. If False, it loads the datasets from the CSV files.
            The CSV files are named based on the discipline's name.
        """
        data = IODataset()

        if new_doe == True:
            print(f"\n----- Processing Design Of Experiment for {discipline.name} discipline... -----\n")
            inputs_doe = self.propose_doe(inputs, num_samples, doe_method=doe_method)

            columns = list((inputs_doe.keys()))
            X_features = []
            for i in columns:
                X_input = (inputs_doe[i])
                X_features.append(X_input)
            X_features_matrix = np.column_stack(X_features)

            outputs_doe = self.evaluate_doe(inputs=inputs_doe, execute=discipline.execute, num_samples=num_samples)
            print("----- Design Of Experiment completed -----")

            # Train and make surrogate (reorganize data to make them compatible)
            print("\n----- Training Dataset -----\n")
            # data.columns = data.columns.set_levels(data.columns.levels[2],level=2)
            data = outputs_doe.values()
            data = np.vstack(list(data))

            data = pd.DataFrame(list(data))

            # data=pd.DataFrame(list(outputs_doe.values))
            if save_doe==True:
                with open(f"training_dataset_outputs_dynamic{discipline.name}.csv", "w") as f:
                    data.to_csv(f, index=False)

            data = inputs_doe.values()
            data = np.vstack(list(data))

            data = pd.DataFrame(list(data))
            if save_doe==True:
                with open(f"training_dataset_inputs_dynamic{discipline.name}.csv", "w") as f:
                    data.to_csv(f, index=False)

        if new_doe == False:
            logging.info(f"reading training_dataset_dynamic{discipline.name}.csv")
            from pandas import read_csv

            dataframe = read_csv(
                f"training_dataset_{discipline.name}.csv", delimiter=",", header=[0, 1, 2], index_col=None
            )
            dataframe.columns = dataframe.columns.set_levels(dataframe.columns.levels[2], level=2)
            data = IODataset(dataframe)

            print("\n----- Training Dataset -----\n")
        return X_features_matrix, outputs_doe

    def dataset_for_validation(self, new_doe: bool, num_samples: int, discipline: Any, inputs: Optional[list[BaseVariable]] = None, outputs: Optional[list[BaseVariable]] = None,doe_method: Optional[str] = "LHS") -> IODataset:
        """
        Create or load a dataset for validation.

        This method generates a new dataset for validation using the specified DoE method and discipline,
        or loads an existing dataset from a CSV file.

        Args:
            new_doe: Boolean flag indicating whether to create a new DoE or load an existing one.
            num_samples: Integer specifying the number of samples for the DoE.
            discipline: The discipline to evaluate, which must have an `execute` method.
            inputs: Optional list of input variables. If None, the discipline's inputs are used.
            doe_method: Optional string specifying the DoE method to use. Defaults to "LHS".

        Returns:
            IODataset containing the input and output variables for validation.

        Note:
            If `new_doe` is True, the method generates a new DoE, evaluates it on the discipline,
            and saves the dataset to a CSV file. If False, it loads the dataset from the CSV file.
            The CSV file is named based on the discipline's name.
        """
        if len(discipline) == 1:
            discipline = discipline[0]
            data = IODataset()
            # Dataset generation or dataset loading
            if new_doe == True:
                print(f"\n----- Processing Design Of Experiment for validation {discipline.name} discipline... -----\n")
                inputs_doe = self.propose_doe(inputs, num_samples, doe_method)

                # Evaluating DoE obtained from inputs_doe
                outputs_doe = self.evaluate_doe(inputs=inputs_doe, execute=discipline.execute, num_samples=num_samples)
                print("----- Design Of Experiment completed -----")

                # Creating an IODataset
                for n, v in inputs_doe.items():
                    data.add_input_variable(n, v)

                for n, v in outputs_doe.items():
                    data.add_output_variable(n, v)

                print("\n----- Training Dataset -----\n")
                if self.save_doe==True:
                    with open(f"training_dataset_validate_{discipline.name}.csv", "w") as f:
                        data.to_csv(f, index=False)

            if new_doe == False:
                # Loading dataset
                logging.info(f"reading training_dataset_validate_{discipline.name}.csv")
                from pandas import read_csv

                dataframe = read_csv(
                    f"training_dataset_validate_{discipline.name}.csv", delimiter=",", header=[0, 1, 2], index_col=None
                )
                dataframe.columns = dataframe.columns.set_levels(dataframe.columns.levels[2], level=2)
                data = IODataset(dataframe)

                print("\n----- Training Dataset -----\n")
            return data
        else:
            # Process Design of Experiment for multiple disciplines
                print(f"\n----- Processing Design Of Experiment for {discipline[0].name} discipline... -----\n")

                # Create scenario for DOE training
                mads_scenario = MADSScenario()
                mads_scenario.fill_parameter_space(inputs)
                mads_scenario.create_scenario(
                    disciplines=discipline,
                    #formulation=MDF_Settings(main_mda_name="MDAGaussSeidel"),  # old #'DisciplinaryOpt',#'BiLevel', #'MDF',
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

                # Reorganize the data because output of data is a single vector
                dataset = IODataset()
                for n, v in data.input_dataset.items():
                    dataset.add_input_variable(n[1], v)
                dataset.add_output_group(data.output_dataset, [out.name for out in outputs], {out.name: np.size(out.value) for out in outputs})

                # Filter out zeros by replacing them with the average
                #data_filter = dataset.get_columns(variable_names=[out.name for out in outputs])
                #average = np.mean(data_filter)
                #data_filter = np.where(data_filter == 0, average, data_filter)
                #dataset.update_data(data_filter, variable_names=outputs.name)

                # Save the dataset to a CSV file
                if self.save_doe==True:
                    with open(f"training_dataset_validate_{discipline[0].name}.csv", "w") as f:
                        dataset.to_csv(f, index=False)
                    
                return dataset

    def propose_doe(self, inputs: Optional[list[BaseVariable]], num_samples: int, doe_method: str) -> dict[str, NDArray]:
        """
        Propose a DoE based on the specified inputs and method.

        This method creates a design space based on the input variables and generates a DoE using the specified method.

        Args:
            inputs: Optional list of input variables. If None, the discipline's inputs are used.
            num_samples: Integer specifying the number of samples for the DoE.
            doe_method: String specifying the DoE method to use (e.g., "LHS", "DOE_LHS").

        Returns:
            Dictionary of input variables and their values for the DoE.

        Note:
            The method handles cases where input bounds are infinite by setting them to a reasonable range.
            It also manages cases where input values are zero by setting bounds to [-1, 1].
            The DoE is generated using the GEMSEO library.
        """
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

            # Create bounds if doesn't have them (mandatory)
            lb = np.where(np.isneginf(lb), val, lb)
            ub = np.where(np.isposinf(ub), val, ub)

            # Manage the case val=0
            equal_bounds = (lb == ub)
            lb = np.where(equal_bounds, lb, lb)
            ub = np.where(equal_bounds, ub, ub)

            # Debug
            # print(v.name, val, lb, ub)

            if isinstance(v.value,np.floating):
                space.add_variable(name=v.name, size=size, type_="float", value=val, lower_bound=float(lb), upper_bound=float(ub))
            elif isinstance(v.value,np.ndarray):
                space.add_variable(name=v.name, size=size, type_="float", value=np.atleast_1d(val).astype(float), lower_bound=np.atleast_1d(lb).astype(float), upper_bound=np.atleast_1d(ub).astype(float))
                
            #    space.add_variable(name=v.name, size=size, type_="Array", value=val, lower_bound=(lb), upper_bound=(ub))
                
        # Evaluate inputs set for DoE
        doe = compute_doe(space, algo_name=doe_method, n_samples=num_samples)
        # pydoe = PyDOE()
        # doe = pydoe.execute(space, algo_name="PYDOE_LHS", n_samples=100)
        inputs_doe = {}
        k = 0

        # Re-organize Inputs for DoE (vectors are broken up)
        for v in inputs:
            v_array = np.atleast_1d(v.value)
            size = v_array.shape[0]

            for i in range(size):
                var_name = f"{v.name}[{i}]" if size > 1 else v.name
                inputs_doe[var_name] = doe[:, k]
                k += 1
        return inputs_doe

    def evaluate_doe(self, execute: Callable, inputs: dict[str, NDArray], num_samples: int) -> dict[str, NDArray]:
        """
        Evaluate the DoE on the specified discipline.

        This method evaluates the input data on the discipline's execute method and collects the outputs.

        Args:
            execute: Callable function that represents the discipline's execute method.
            inputs: Dictionary of input variables and their values for the DoE.
            num_samples: Integer specifying the number of samples for the DoE.

        Returns:
            Dictionary of output variables and their values.

        Note:
            The method handles both scalar and vector outputs. It skips outputs that are also inputs.
            The outputs are stored in a dictionary with the output names as keys and numpy arrays as values.
        """
        outputs: dict[str, NDArray] = dict()

        for i in range(num_samples):
            # print({name: np.array([vals[i]]) for name, vals in inputs.items()})
            # Assign inputs set for the DoEls  

            out = execute(  # Execute the DoE on the discipline
                {name: np.array([vals[i]]) for name, vals in inputs.items()}  # This gives also the inputs in the dictionary
            )
            input_names = set(inputs.keys())
            
            for name, value in out.items():
                size = value.shape[0]

                # Assign output only if different from input, and manage the output-vector case
                var_name = name
                if var_name not in input_names:

                    if size == 1:
                        outputs[var_name] = np.append(
                            outputs.get(var_name, np.array([])), [value[0]], axis=0
                        )
                    else:
                        value = np.atleast_2d(value)
                        outputs[var_name] = np.vstack([
                            outputs.get(var_name, np.empty((0, value.shape[1]))), value]
                        )
        return outputs