from typing import Any, Callable, Optional, Type
from gemseo.datasets.io_dataset import IODataset
from multiads.scenario import BaseVariable
import logging
from multiads.disciplines import MADSDiscipline
import numpy as np
from gemseo import create_design_space, create_surrogate, compute_doe
from numpy.typing import NDArray
from ..utils.data_utils import DataUtils
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import inspect

# Import GEMSEO regression algorithms and their settings
from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from gemseo.mlearning.regression.algos.linreg_settings import LinearRegressor_Settings
from gemseo.mlearning.regression.algos.moe import MOERegressor
from gemseo.mlearning.regression.algos.moe_settings import MOE_Settings
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from gemseo.mlearning.regression.algos.ot_gpr_settings import OTGaussianProcessRegressor_Settings
from gemseo.mlearning.regression.algos.polyreg import PolynomialRegressor
from gemseo.mlearning.regression.algos.polyreg_settings import PolynomialRegressor_Settings
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.mlearning.regression.algos.rbf_settings import RBFRegressor_Settings
from gemseo.mlearning.regression.algos.svm import SVMRegressor
from gemseo.mlearning.regression.algos.svm_settings import SVMRegressor_Settings
from gemseo.mlearning.regression.algos.thin_plate_spline import TPSRegressor
from gemseo.mlearning.regression.algos.thin_plate_spline_settings import TPSRegressor_Settings
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.io_dataset import IODataset

# Import scikit-learn regression algorithms and their settings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import inspect
from sklearn.gaussian_process.kernels import RBF, Kernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LinearRegression as LinearRegression_sci
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from ..utils.data_utils import DataUtils
from gemseo.datasets.io_dataset import IODataset
import numpy as np
import pandas as pd
import pyLOM, pyLOM.NN

class RegressorTrainingModelFactory:
    """
    A factory class for training various types of regressors from different libraries (GEMSEO, scikit-learn, pyLOM).

    This class provides methods to train regressors from different libraries, including GEMSEO, scikit-learn, and pyLOM.
    It handles the creation of datasets, model training, and evaluation for each type of regressor.
    """

    def __init__(self):
        """
        Initialize the RegressorTrainingModelFactory with dictionaries of available regressors and their settings.

        This method initializes the factory with dictionaries of regressors and their settings for GEMSEO, scikit-learn, and pyLOM.
        """
        # Dictionary for pyLOM surrogate models
        self.SM_pylom = {"MLP_pylom": pyLOM.NN.MLP, "KAN_pylom": pyLOM.NN.KAN}

        # Dictionary for GEMSEO surrogate models
        self.SM_gemseo = {
            "LinearRegressor_gemseo": LinearRegressor,
            "MOERegressor_gemseo": MOERegressor,
            "PCERegressor_gemseo": PCERegressor,
            "PolynomialRegressor_gemseo": PolynomialRegressor,
            "RBFRegressor_gemseo": RBFRegressor,
            "TPSRegressor_gemseo": TPSRegressor,
            "OTGaussianProcessRegressor_gemseo": OTGaussianProcessRegressor,
        }

        # Dictionary for GEMSEO surrogate model settings
        self.SM_gemseo_settings = {
            "LinearRegressor_gemseo": LinearRegressor_Settings,
            "PCERegressor_gemseo": PCERegressor_Settings,
            "PolynomialRegressor_gemseo": PolynomialRegressor_Settings,
            "RBFRegressor_gemseo": RBFRegressor_Settings,
            "TPSRegressor_gemseo": TPSRegressor_Settings,
            "MOERegressor_gemseo": MOE_Settings,
            "OTGaussianProcessRegressor_gemseo": OTGaussianProcessRegressor_Settings,
        }

        # Dictionary for scikit-learn surrogate models
        self.SM_scikit = {
            "GaussianProcessRegressor_scikit": GaussianProcessRegressor,
            "GradientBoosterRegressor_scikit": GradientBoostingRegressor,
            "RandomForestRegressor_scikit": RandomForestRegressor,
            "MLPRegressor_scikit": MLPRegressor,
            "SVMRegressor_scikit": SVR,
            "RegressorChain_scikit": RegressorChain,
            "LinearRegression_scikit": LinearRegression_sci,
            "PolynomialRegression_scikit": PolynomialFeatures,
            "PCA_scikit": PCA,
            "KPCA_scikit": KernelPCA,
            "PLSRegression_scikit": PLSRegression,
        }

        # Dictionary for pyLOM dimensionality reduction methods
        self.DR_methods_pylom = {"MLP_pylom": pyLOM.NN.MLP, "KAN_pylom": pyLOM.NN.KAN}
        self.common_dict_SM = {}

    def training_regressor(self, regressor_name: str, X_train: Any, Y_train: Any, inputs: Any, outputs: Any, extra_args: Any) -> Any:
        """
        Train a regressor based on the specified name and training data.

        This method selects the appropriate training method based on the regressor type and trains the model.

        Args:
            regressor_name: Name of the regressor to train.
            X_train: Training input data, typically a numpy array or pandas DataFrame.
            Y_train: Training output data, typically a numpy array or pandas DataFrame.
            inputs: Input variables for reference, typically a list of BaseVariable objects.
            outputs: Output variables for reference, typically a list of BaseVariable objects.
            extra_args: Additional arguments for the regressor, including model-specific settings.

        Returns:
            The trained surrogate model.

        Raises:
            ValueError: If the specified regressor is not found in the available dictionaries.

        Note:
            For pyLOM regressors, if optimization parameters are provided, the method skips training and returns None.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.dr_utils = DataUtils()

        # Check which type of regressor to train
        if regressor_name in self.SM_gemseo.keys():
            print("GEMSEO Regressor")
        elif regressor_name in self.SM_pylom.keys():
            print("PYLOM Surrogate Model")
        elif regressor_name in self.SM_scikit.keys():
            print("Scikit Regressor")
        else:
            raise ValueError("Regressor not Found")

        # Train the specified regressor
        if regressor_name in self.SM_gemseo.keys():
            # Find regressor settings for GEMSEO
            SM_settings = self.SM_gemseo_settings[regressor_name]
            # Train the regressor
            self.surrogate_train = self.train_surrogate(
                inputs=X_train,
                outputs=Y_train,
                regressor_name=regressor_name,
                regressor_settings=SM_settings,
                **extra_args
            )

        elif regressor_name in self.SM_pylom.keys():
            if "optimization_params" in extra_args:  # i.e., we need to do HPO, skip training
                return None
            else:
                SM = self.DR_methods_pylom  # Training of the SM. Different approach compared to scikit or gemseo
                self.surrogate_train = self.train_surrogate_pylom(
                    inputs=X_train,
                    outputs=Y_train,
                    regressor_name=regressor_name,
                    regressor_settings=SM,
                    **extra_args
                )

        elif regressor_name in self.SM_scikit.keys():
            # Find regressor from the dictionary
            SM = self.SM_scikit[regressor_name]
            # Train the regressor
            self.surrogate_train = self.train_surrogate(
                inputs=X_train,
                outputs=Y_train,
                regressor_name=regressor_name,
                regressor_settings=SM,
                **extra_args
            )
        return self.surrogate_train

    def train_surrogate(self, inputs: dict[str, NDArray], outputs: dict[str, NDArray], regressor_name: str, regressor_settings: Optional[Any], **regressor_args: Any) -> Any:
        """
        Train a surrogate model based on the specified name and settings.

        This method handles the training of both GEMSEO and scikit-learn regressors.
        It creates an IODataset for GEMSEO regressors and handles special cases like PCE and multi-output regressors.

        Args:
            inputs: Dictionary of input variables and their values as numpy arrays.
            outputs: Dictionary of output variables and their values as numpy arrays.
            regressor_name: Name of the regressor to train.
            regressor_settings: Settings for the regressor.
            **regressor_args: Additional arguments for the regressor, including model-specific settings.

        Returns:
            The trained surrogate model.

        Note:
            For GEMSEO regressors, the method creates an IODataset and uses the specified settings.
            For scikit-learn regressors, it handles special cases like MultiOutputRegressor and RegressorChain.
        """
        # Temporary naming for training if the size of inputs is different from the original size of inputs (i.e., when DR is applied)
        if np.size(self.inputs) != inputs.shape[1]:
            columns = (inputs.shape[1])
            my_string = [f"inp{j}" for j in range(1, columns + 1)]
            inputs = pd.DataFrame(inputs.values, columns=my_string)

        # If the size of outputs is different from the original size of outputs
        if np.size(self.outputs) != outputs.shape[1]:
            columns = (outputs.shape[1])
            my_string = [f"out{j}" for j in range(1, columns + 1)]
            outputs = pd.DataFrame(outputs.values, columns=my_string)

        # Create an IODataset, compulsory for GEMSEO regressor
        data = IODataset()
        for n, v in inputs.items():
            data.add_input_variable(n, v)
        for n, v in outputs.items():
            data.add_output_variable(n, v)

        self.dr_utils = DataUtils()

        # Get common arguments for the regressor
        self.common_dict_SM = self.dr_utils.common_arguments_dict(regressor_settings, regressor_args)

        print(f"\n Values for parameters given: {self.common_dict_SM} \n")

        # Train the specified regressor
        if regressor_name in self.SM_gemseo.keys():
            if regressor_name == "PCERegressor_gemseo":
                parameter_space = ParameterSpace()
                if "input_distribution" in regressor_args:
                    PDFdist = (regressor_args["input_distribution"])  # Compulsory input to be given when using PCE
                    if len(PDFdist) == inputs.shape[1]:
                        for i in range(inputs.shape[1]):
                            inp = (inputs.columns[i])
                            parameter_space.add_random_variable(str(inp), PDFdist[i - 1])  # Assigning the input_distribution to the inputs
                    else:
                        print("Incorect size of input_distribution")
                        ValueError
                elif "parameter_space" in regressor_args:
                    parameter_space=(regressor_args["parameter_space"]) 
                else: 
                    print("Need parameter_space in input arguments")
                print(parameter_space)
                settings = PCERegressor_Settings(probability_space=parameter_space, **self.common_dict_SM)
                self.surrogate_gemseo = PCERegressor(data=data, settings_model=settings)
                self.surrogate_gemseo.learn()
            # Default implementation of SM from GEMSEO, common for all regressors from GEMSEO
            elif regressor_name in self.SM_gemseo.keys():
                data = data.droplevel(level=[1], axis=1)
                # Default GEMSEO implementation for SM
                SM = self.SM_gemseo[regressor_name]
                settings = regressor_settings(**self.common_dict_SM)
                self.surrogate_gemseo = SM(data=data, settings_model=settings)
                self.surrogate_gemseo.learn()

        # Default approach for most of the regressors from scikit-learn
        if regressor_name == "GaussianProcessRegressor_scikit" or regressor_name == "RandomForestRegressor_scikit" or regressor_name == "MLPRegressor_scikit" or regressor_name == "LinearRegression_scikit" or regressor_name == "PolynomialRegression_scikit":
            GP = regressor_settings(**self.common_dict_SM)
            self.surrogate_gemseo = GP.fit(inputs.values, outputs.values)
        # Linking multiple features for regressors supporting only training for 1 output
        elif regressor_name == "GradientBoosterRegressor_scikit" or regressor_name == "SVMRegressor_scikit":
            GBR = regressor_settings(**self.common_dict_SM)
            multi_GBR = MultiOutputRegressor(GBR)  # Using MultiOutputRegressor as linking
            self.surrogate_gemseo = multi_GBR.fit(inputs.values, outputs.values)
        # Regressor chain trains 1 output dimension at a time and is useful when dependence in the outputs
        elif regressor_name == "RegressorChain_scikit":
            if "base_estimator" not in regressor_args:
                RG = GaussianProcessRegressor  # Default regressor when not given
            else:
                RG = self.SM_scikit[self.common_dict_SM["base_estimator"]]
            RC = RegressorChain(RG())
            self.surrogate_gemseo = RC.fit(inputs.values, outputs.values)

        return self.surrogate_gemseo

    def train_surrogate_pylom(self, inputs: dict[str, NDArray], outputs: dict[str, NDArray], regressor_name: str, regressor_settings: Optional[Any], **regressor_args: Any) -> Any:
        """
        Train a surrogate model using pyLOM.

        This method handles the training of pyLOM regressors, including data preparation, model training, and evaluation.

        Args:
            inputs: Dictionary of input variables and their values as numpy arrays.
            outputs: Dictionary of output variables and their values as numpy arrays.
            regressor_name: Name of the regressor to train.
            regressor_settings: Settings for the regressor.
            **regressor_args: Additional arguments for the regressor, including model and training parameters.

        Returns:
            The trained model.

        Note:
            The method uses the DataUtils class to prepare the dataset for pyLOM.
            It also handles the scaling of inputs and outputs and evaluates the model using a regression evaluator.
        """
        model = regressor_args["model"]  # Defining the model
        training_params = regressor_args["training_params"]  # Defining the training parameters
        pylom_args = self.dr_utils.dataset_for_pylom(inputs.values, outputs.values)
        self.inputs_scaler_pylom = pylom_args["inputs_scaler"]
        self.outputs_scaler_pylom = pylom_args["outputs_scaler"]

        dataset = pyLOM.NN.Dataset(**pylom_args)  # Creating dataset of tensors
        td_train, td_test = dataset.get_splits_by_parameters([0.8, 0.2])  # Test train split used for training

        pipeline = pyLOM.NN.Pipeline(train_dataset=td_train, test_dataset=td_test, valid_dataset=None,
                                     model=model, training_params=training_params)  # The pipeline to be used
        training_logs = pipeline.run()
        preds = model.predict(td_test)
        scaled_preds = self.outputs_scaler_pylom.inverse_transform([preds])  # Inverse transform as pyLOM uses MinMaxScaler as default
        scaled_y = self.outputs_scaler_pylom.inverse_transform([td_test[:][1]])
        evaluator = pyLOM.NN.RegressionEvaluator(tolerance=1e-10)
        evaluator(scaled_preds, scaled_y)  # Validation of the SM
        evaluator.print_metrics()
        return model