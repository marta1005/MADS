from typing import Any, Callable, Optional, Type
from gemseo.datasets.io_dataset import IODataset
from multiads.scenario import BaseVariable
import logging
from multiads.disciplines import MADSDiscipline
import numpy as np
from gemseo import create_design_space, create_surrogate, compute_doe
from numpy.typing import NDArray
import pyLOM, pyLOM.NN
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
import optuna
from ..models.regressor_training_models import RegressorTrainingModelFactory
from ..utils.data_utils import GemesoRegressorWrapper
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings

from ..utils.data_utils import DataUtils
from ..models.regressor_training_models import RegressorTrainingModelFactory

class HPOModelFactory:
    """
    A factory class for hyperparameter optimization (HPO) of various types of regressors from different libraries (GEMSEO, scikit-learn, pyLOM).

    This class provides methods to perform hyperparameter optimization using different approaches:
    - RandomizedSearchCV and GridSearchCV for scikit-learn and GEMSEO regressors
    - Bayesian optimization using Optuna for scikit-learn and GEMSEO regressors
    - In-built optimizer for pyLOM regressors
    """

    def __init__(self):
        """
        Initialize the HPOModelFactory.

        This method initializes the factory with no specific settings.
        """
        pass

    def HyperParameterOptimizationFactory(self, regressor_name: str, X_train: NDArray, Y_train: NDArray, HPO_space: dict, **extra_args) -> Any:
        """
        Perform hyperparameter optimization for the specified regressor.

        This method selects the appropriate optimization method based on the regressor type and the specified optimizer.

        Args:
            regressor_name: Name of the regressor to optimize.
            X_train: Training input data as a numpy array.
            Y_train: Training output data as a numpy array.
            HPO_space: Dictionary specifying the hyperparameter space for optimization.
            **extra_args: Additional arguments for the regressor, including the optimizer type.

        Returns:
            The regressor fitted with the best hyperparameters.

        Note:
            For pyLOM regressors, the method uses the in-built optimizer.
            For scikit-learn and GEMSEO regressors, it uses either RandomizedSearchCV, GridSearchCV, or Bayesian optimization.
        """
        self.regressor_dict = RegressorTrainingModelFactory()
        self.dr_utils = DataUtils()

        if regressor_name in self.regressor_dict.SM_pylom.keys():  # pylom has different approach to HPO
            SM = self.regressor_dict.SM_pylom[regressor_name]
            # In-built optimizer for pyLOM
            self.surrogate_train = self.HyperparameterOpt_pylom(
                inputs=X_train,
                outputs=Y_train,
                regressor_name=regressor_name,
                regressor_settings=SM,
                **extra_args
            )
            self.inputs_scaler_pylom = self.inputs_scaler_py
            self.outputs_scaler_pylom = self.outputs_scaler_py
        else:
            self.common_dict_SM = self.regressor_dict.common_dict_SM
            # Find the optimizer in extra args
            optimizer = extra_args["Optimizer"]
            print(self.common_dict_SM)
            # Similar approach for both RandomizedSearchCV and GridSearchCV for scikit and gemseo
            if optimizer == "RandomizedSearchCV" or optimizer == "GridSearchCV":
                self.surrogate_train = self.HyperparameterOpt(
                    regressor_name=regressor_name,
                    input_data=X_train,
                    output_data=Y_train,
                    HPO_parameter_space=HPO_space,
                    **extra_args
                )
            # Optuna Bayesian Optimization for scikit and gemseo
            elif optimizer == "BayesianOptimization":
                self.surrogate_train = self.HyperparameterOpt_bayesian(
                    regressor_name=regressor_name,
                    input_data=X_train,
                    output_data=Y_train,
                    HPO_parameter_space=HPO_space,
                    **extra_args
                )

        return self.surrogate_train

    def HyperparameterOpt_bayesian(self, regressor_name: str, input_data: np.array, output_data: np.array, HPO_parameter_space: Any, **extra_parameters: Any) -> Any:
        """
        Perform Bayesian hyperparameter optimization using Optuna.

        This method uses Optuna to find the best hyperparameters for the specified regressor.
        It supports both scikit-learn and GEMSEO regressors, including special cases like MultiOutputRegressor and RegressorChain.

        Args:
            regressor_name: Name of the regressor to optimize.
            input_data: Input data for training as a numpy array.
            output_data: Output data for training as a numpy array.
            HPO_parameter_space: Dictionary specifying the hyperparameter space for optimization.
            **extra_parameters: Additional parameters for the optimization, including settings for Optuna.

        Returns:
            The regressor fitted with the best hyperparameters.

        Note:
            The method uses cross-validation to evaluate the performance of each hyperparameter set.
            For GEMSEO regressors, it uses the GemesoRegressorWrapper to make them compatible with scikit-learn's cross_val_score.
        """
        settings = extra_parameters["HPO_settings"]  # Settings to be used
        n_iter = settings["n_iter"]  # Iterations
        cv = settings["cv"]  # Cross-validation
        scoring = settings["scoring"]  # Scoring of the iterations
        print(HPO_parameter_space.items())

        def objective(trial):  # Trial of the hyperparameters
            params = {}
            for name, config in HPO_parameter_space.items():  # Defining the structure of params as expected by Optuna
                print(HPO_parameter_space.items())
                if config["type"] == "float":
                    params[name] = trial.suggest_float(name, config["lb"], config["ub"])
                elif config["type"] == "int":
                    params[name] = trial.suggest_int(name, config["lb"], config["ub"])
                elif config["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, config["choices"])
                    print(params)

                if regressor_name == "GradientBoosterRegressor_scikit" or regressor_name == "SVMRegressor_scikit":  # Special type using MultiOutputRegressor
                    SM = self.regressor_dict.SM_scikit[regressor_name]
                    model = MultiOutputRegressor(SM(**params))
                    scores = cross_val_score(model, input_data, output_data, cv=cv, scoring=scoring)
                if regressor_name == "RegressorChain_scikit":
                    RG = self.regressor_dict.SM_scikit[self.common_dict_SM["base_estimator"]]
                    model = RegressorChain(RG(**params))
                    scores = cross_val_score(model, input_data, output_data, cv=cv, scoring=scoring)

                elif regressor_name in self.regressor_dict.SM_scikit:  # All other regressors from scikit
                    SM = self.regressor_dict.SM_scikit[regressor_name]
                    model = SM(**params)
                    scores = cross_val_score(model, input_data, output_data, cv=cv, scoring=scoring)
                elif regressor_name in self.regressor_dict.SM_gemseo:  # Default gemseo using GemseoRegressorWrapper
                    SM = self.regressor_dict.SM_gemseo[regressor_name]
                    regressor_settings = self.regressor_dict.SM_gemseo_settings[regressor_name]
                    settings = regressor_settings(**self.common_dict_SM)
                    model = GemesoRegressorWrapper(regressor=SM, settings_model=settings, regressor_name=regressor_name, **extra_parameters)
                    scores = cross_val_score(model, input_data, output_data, cv=cv, scoring=scoring)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_iter)
        best_HP = study.best_params  # Best parameters
        print(f"Best Parameters:{best_HP}")


        # Fitting the best parameters to the regressors used
        if regressor_name in self.regressor_dict.SM_scikit:
            SM = self.regressor_dict.SM_scikit[regressor_name]
        elif regressor_name in self.regressor_dict.SM_gemseo:
            SM = self.regressor_dict.SM_gemseo[regressor_name]
            regressor_settings = self.regressor_dict.SM_gemseo_settings[regressor_name]

        if regressor_name == "RegressorChain_scikit":
            RG = self.regressor_dict.SM_scikit[self.common_dict_SM["base_estimator"]]
            Best_SM = RegressorChain(RG(**best_HP))
            Best_SM.fit(input_data, output_data)
        elif regressor_name == "GradientBoosterRegressor_scikit" or regressor_name == "SVMRegressor_scikit":
            Best_SM = SM(**best_HP)
            Best_SM = MultiOutputRegressor(SM(**best_HP))
            Best_SM.fit(input_data, output_data)
        elif regressor_name in self.regressor_dict.SM_scikit:
            Best_SM = SM(**best_HP)
            Best_SM.fit(input_data, output_data)
        elif regressor_name in self.regressor_dict.SM_gemseo:
            data = IODataset()
            for n, v in input_data.items():
                data.add_input_variable(n, v)
            for n, v in output_data.items():
                data.add_output_variable(n, v)

            # Special case because PCE requires distribution as inputs for the choice of basis
            if regressor_name == "PCERegressor_gemseo":
                parameter_space = ParameterSpace()
                PDFdist = (extra_parameters["input_distribution"])
                if len(PDFdist) == input_data.shape[1]:
                    for i in range(input_data.shape[1]):
                        inp = (input_data.columns[i])
                        parameter_space.add_random_variable(str(inp), PDFdist[i - 1])
                settings1 = PCERegressor_Settings(probability_space=parameter_space, **best_HP)
                Best_SM = PCERegressor(data=data, settings_model=settings1)
                Best_SM.learn()
            elif regressor_name in self.regressor_dict.SM_gemseo.keys():
                # Default gemseo implementation for SM
                SM = self.regressor_dict.SM_gemseo[regressor_name]
                settings1 = regressor_settings(**best_HP)
                Best_SM = SM(data=data, settings_model=settings1)
                Best_SM.learn()
        return Best_SM  # Returning the regressor fitted with best HP

    def HyperparameterOpt_pylom(self, inputs: dict[str, NDArray], outputs: dict[str, NDArray], regressor_name: str, regressor_settings: Optional[Any], **regressor_args: Any) -> Any:
        """
        Perform hyperparameter optimization for pyLOM regressors.

        This method uses pyLOM's in-built optimizer to find the best hyperparameters for the specified regressor.
        It handles the data preparation, model training, and evaluation.

        Args:
            inputs: Dictionary of input variables and their values as numpy arrays.
            outputs: Dictionary of output variables and their values as numpy arrays.
            regressor_name: Name of the regressor to optimize.
            regressor_settings: Settings for the regressor.
            **regressor_args: Additional arguments for the regressor, including optimization parameters.

        Returns:
            The regressor fitted with the best hyperparameters.

        Note:
            The method uses Optuna for the optimization process and evaluates the model using a regression evaluator.
            It also handles the scaling of inputs and outputs.
        """
        model = regressor_settings
        optimization_params = regressor_args["optimization_params"]  # The parameters to be optimized
        pylom_args = self.dr_utils.dataset_for_pylom(inputs.values, outputs.values)
        self.inputs_scaler_py = pylom_args["inputs_scaler"]
        self.outputs_scaler_py = pylom_args["outputs_scaler"]

        dataset = pyLOM.NN.Dataset(**pylom_args)  # Tensor dataset for pyLOM
        td_train, td_test = dataset.get_splits_by_parameters([0.8, 0.2])

        if "n_trials" in regressor_args:
            n_trials = regressor_args["n_trials"]
        else:
            n_trials = 30  # Default value
        if "pruner" in regressor_args:
            pruner = regressor_args["pruner"]
        else:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5,  # Default pruner
                                                n_warmup_steps=5,
                                                interval_steps=1)

        optimizer = pyLOM.NN.OptunaOptimizer(optimization_params=optimization_params,
                                             n_trials=n_trials,
                                             direction="minimize",
                                             pruner=pruner)

        pipeline = pyLOM.NN.Pipeline(train_dataset=td_train, test_dataset=td_test, valid_dataset=None,
                                     model_class=model, optimizer=optimizer)
        training_logs = pipeline.run()
        model = pipeline.model

        preds = model.predict(td_test)  # Similar logic as prediction from pyLOM
        scaled_preds = self.outputs_scaler_py.inverse_transform([preds])
        scaled_y = self.outputs_scaler_py.inverse_transform([td_test[:][1]])
        evaluator = pyLOM.NN.RegressionEvaluator(tolerance=1e-10)
        evaluator(scaled_preds, scaled_y)
        evaluator.print_metrics()
        return model

    def HyperparameterOpt(self, regressor_name: str, input_data: np.array, output_data: np.array, HPO_parameter_space: Any, **extra_parameters: Any) -> Any:
        """
        Perform hyperparameter optimization using RandomizedSearchCV or GridSearchCV.

        This method uses either RandomizedSearchCV or GridSearchCV to find the best hyperparameters for the specified regressor.
        It supports both scikit-learn and GEMSEO regressors, including special cases like MultiOutputRegressor and RegressorChain.

        Args:
            regressor_name: Name of the regressor to optimize.
            input_data: Input data for training as a numpy array.
            output_data: Output data for training as a numpy array.
            HPO_parameter_space: Dictionary specifying the hyperparameter space for optimization.
            **extra_parameters: Additional parameters for the optimization, including settings for the search.

        Returns:
            The regressor fitted with the best hyperparameters.

        Note:
            For GEMSEO regressors, the method uses the GemesoRegressorWrapper to make them compatible with scikit-learn's search methods.
            The method handles the special cases of MultiOutputRegressor and RegressorChain separately.
        """
        # Define default inputs to the regressor, optional
        if "HPO_dict" in extra_parameters.keys():
            HPO_dict = extra_parameters["HPO_dict"]
        # Define settings of HPO
        sett = extra_parameters["HPO_settings"]
        # Update the parameter space
        if "RandomizedSearchCV" in extra_parameters["Optimizer"]:
            opt = RandomizedSearchCV
            sett.update({"param_distributions": HPO_parameter_space})  # RandomizedSearchCV needs parameters under param_distributions
        elif "GridSearchCV" in extra_parameters["Optimizer"]:
            opt = GridSearchCV
            sett.update({"param_grid": HPO_parameter_space})  # GridSearchCV needs parameters under param_grid

        # Scikit-learn regressors with support of MultiOutputRegressor
        if regressor_name == "GradientBoosterRegressor_scikit" or regressor_name == "SVMRegressor_scikit":
            SM = self.regressor_dict.SM_scikit[regressor_name]
            sett = extra_parameters["HPO_settings"]
            # MultiOutputRegressor for each feature
            multi_GBR = MultiOutputRegressor(SM())
            randomsearch = opt(estimator=multi_GBR,
                               **sett)  # Function doing the HPO
            BP = randomsearch.fit(input_data, output_data)  # Finding the best HPO

            print("Best Parameters:")
            print(BP.best_params_)
            return BP.best_estimator_  # Returning the regressor fitted with the best HP
        # Different implementation because of base regressor
        elif regressor_name == "RegressorChain_scikit":
            RG = self.regressor_dict.SM_scikit[self.common_dict_SM["base_estimator"]]
            print(RG)
            if "HPO_dict" in extra_parameters:
                hpr_HPO = RegressorChain(RG(**HPO_dict))
                randomsearch = opt(estimator=RegressorChain(base_estimator=RG()),
                                 **sett)
                BP=randomsearch.fit(input_data,output_data)
            else:
                hpr_HPO=RegressorChain(RG)
            #sett=extra_parameters["HPO_settings"]
                randomsearch=opt(estimator=RegressorChain(base_estimator=RG()),
                                 **sett)
                BP=randomsearch.fit(input_data,output_data)

            print("Best Parameters:")
            print(BP.best_params_)
            return BP.best_estimator_# retunring the regressor fitted with the best HP

        # default scikit learn regressors without support of MultiOutputRegressor
        elif regressor_name in self.regressor_dict.SM_scikit:
            SM=self.regressor_dict.SM_scikit[regressor_name]
            sett=extra_parameters["HPO_settings"]
            if "HPO_dict" in extra_parameters:
                hpr_HPO=SM(**HPO_dict)
            else:
                hpr_HPO=SM()
            print(hpr_HPO.get_params())
            randomsearch=opt(estimator=hpr_HPO,
                                 **sett)
            BP=randomsearch.fit(input_data,output_data)
            print("Best Parameters:")
            print(BP.best_params_)
            return BP.best_estimator_ # retunring the regressor fitted with the best HP
        # making regressor from gemseo compatible with scikit regressor we define a wrapper that makes gemseo act like scikit
        # Using Gemseo regressor wrapper for HPO
        elif regressor_name in self.regressor_dict.SM_gemseo:
            SM=self.regressor_dict.SM_gemseo[regressor_name]
            regressor_settings=self.regressor_dict.SM_gemseo_settings[regressor_name]
            settings=regressor_settings(**self.common_dict_SM)
            custom_regressor=GemesoRegressorWrapper(regressor=SM,settings_model=settings,regressor_name=regressor_name,**extra_parameters) #new regressor wrapper for gemseo
            randomsearch=opt(estimator=custom_regressor,
                                            **sett)
            BP=randomsearch.fit(input_data,output_data)
            print("Best Parameters:")
            print(BP.best_params_)
            return BP.best_estimator_