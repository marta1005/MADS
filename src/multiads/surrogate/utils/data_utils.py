# surrogate_library/utils/data_utils.py
from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from gemseo.datasets.io_dataset import IODataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import inspect
from sklearn.base import BaseEstimator, RegressorMixin
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    import pyLOM, pyLOM.NN
except:
    import pyLOM
import matplotlib.pyplot as plt



from gemseo.mlearning.transformers.dimension_reduction.pca import PCA
from gemseo.mlearning.transformers.dimension_reduction.kpca import KPCA

from gemseo.algos.parameter_space import ParameterSpace

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

class DataUtils:
    """
    Utility class for data handling and preprocessing, including dataset preparation for pyLOM,
    and common argument extraction for various functions.

    This class provides methods to prepare datasets for training and prediction with pyLOM,
    and to find common arguments between function signatures and provided keyword arguments.
    """

    def dataset_for_pylom(self, x_input: dict[str, NDArray], y_target: dict[str, NDArray]) -> dict:
        """
        Prepare dataset for pyLOM training.

        This method converts input and target data to the format required by pyLOM,
        including scaling and dummy geometry creation.

        Args:
            x_input: Dictionary of input variables and their values as numpy arrays.
            y_target: Dictionary of target variables and their values as numpy arrays.

        Returns:
            Dictionary containing the prepared dataset for pyLOM, including:
            - variables_out: Transposed target data
            - variables_in: Dummy geometry
            - parameters: Transposed input data
            - inputs_scaler: MinMaxScaler for input data
            - outputs_scaler: MinMaxScaler for output data
            - snapshots_by_column: Boolean flag for snapshot organization

        Note:
            The dummy geometry is created to indicate that the geometry has no influence on the outputs.
            The data is converted to float32 as recommended by pyLOM.
        """
        X = x_input.astype(np.float32)
        Y = y_target.astype(np.float32)
        Y = Y.T  # As recommended by pyLOM
        params_list = [row for row in X.T]
        n_outputs = np.shape(y_target)[1]
        dummy_geo = np.linspace(-1.0, 1.0, n_outputs).reshape(-1, 1).astype(np.float32)  # Dummy geometry
        self.input_scaler = pyLOM.NN.MinMaxScaler()
        self.outputs_scaler = pyLOM.NN.MinMaxScaler()
        return {
            "variables_out": (Y,),  # Outputs
            "variables_in": dummy_geo,  # Dummy variables as mesh
            "parameters": params_list,  # Input parameters
            "inputs_scaler": self.input_scaler,  # MinMaxScaler
            "outputs_scaler": self.outputs_scaler,  # MinMaxScaler
            "snapshots_by_column": True
        }

    def prediction_dataset_for_pylom(self, x_input: dict[str, NDArray], input_scaler: Any, n_output_features: int) -> pyLOM.NN.Dataset:
        """
        Prepare dataset for pyLOM prediction.

        This method converts input data to the format required by pyLOM for prediction,
        using the same scaler as during training.

        Args:
            x_input: Dictionary of input variables and their values as numpy arrays.
                     If an IODataset is provided, its values are used.
            input_scaler: Input scaler used during training.
            n_output_features: Number of output features.

        Returns:
            pyLOM.NN.Dataset object containing the prepared dataset for pyLOM prediction.

        Note:
            A dummy geometry is created to match the format used during training.
            The data is converted to float32 as recommended by pyLOM.
        """
        if isinstance(x_input, np.ndarray):
            pass
        else:
            x_input = x_input.values  # In case the inputs are in IODataset
        params_transposed = x_input.T.astype(np.float32)

        params_list = [row for row in params_transposed]

        dummy_geo = np.linspace(-1.0, 1.0, n_output_features).reshape(-1, 1).astype(np.float32)  # Dummy geometry
        n_samples = x_input.shape[0]
        dummy_output = np.zeros((n_output_features, n_samples), dtype=np.float32)
        pred = pyLOM.NN.Dataset(
            variables_out=(dummy_output,),
            variables_in=dummy_geo,
            parameters=params_list,
            inputs_scaler=input_scaler,  # Same scaler as used in training
            outputs_scaler=None,
            snapshots_by_column=True
        )
        return pred
    
    @staticmethod
    def ploting_data(plot_input:str,plot_output:str,training_data:Any,validation_data:Any,prediction_data:Any)-> None:

        input_training_plotting=(getattr(training_data.inputs,plot_input)).values
        output_training_plotting=(getattr(training_data.outputs,plot_output)).values
        
        input_validation_plotting=(getattr(validation_data.inputs,plot_input)).values
        output_validation_plotting=(getattr(validation_data.outputs,plot_output)).values
        
        input_prediction_plotting=(getattr(prediction_data.inputs,plot_input)).values
        output_prediction_plotting=(getattr(prediction_data.outputs,plot_output)).values
        

        fig,axes=plt.subplots(3,2,figsize=(12,6))
        
        axes[0,0].scatter(output_validation_plotting,output_prediction_plotting,alpha=0.7,color="teal")
        max_val=max(np.max(output_validation_plotting),np.max(output_prediction_plotting))
        min_val=min(np.min(output_validation_plotting),np.min(output_prediction_plotting))
        axes[0,0].plot([min_val,max_val],[min_val,max_val],color="red",alpha=0.5,lw=2,label="Perfect Fit")
        axes[0,0].set_title("Prediction vs actual values")
        axes[0,0].set_xlabel(f"Prediction Values for {plot_output}")
        axes[0,0].set_ylabel(f"Actual Values for {plot_output}")
        axes[0,0].legend()
        

        axes[0,1].scatter(input_training_plotting,output_training_plotting,color="blue",alpha=0.5,label="Training Dataset")
        axes[0,1].scatter(input_prediction_plotting,output_prediction_plotting,color="red",marker="x",label="Prediction Dataset")
        axes[0,1].set_title("Training vs Prediction values")
        axes[0,1].set_xlabel(f"Values for {plot_input}")
        axes[0,1].set_ylabel(f"Values for {plot_output}")
        axes[0,1].legend()
        
        axes[1,0].scatter(input_validation_plotting,output_validation_plotting,color="blue",alpha=0.5,label="Validation Dataset")
        axes[1,0].scatter(input_prediction_plotting,output_prediction_plotting,color="red",marker="x",label="Prediction Dataset")
        axes[1,0].set_title("Validation vs Prediction values")
        axes[1,0].set_xlabel(f"Values for {plot_input}")
        axes[1,0].set_ylabel(f"Values for {plot_output}")
        axes[1,0].legend()
        
        residuals=abs(output_validation_plotting.flatten())-abs(output_prediction_plotting.flatten())
        
        min_val=np.min(residuals)
        
        axes[1,1].hist(residuals.flatten(),bins=30,color="blue",edgecolor="black",alpha=0.7,density=True)
        axes[1,1].axvline(0,color="red",linestyle="--")
        axes[1,1].set_title("Distribution of Errors")
        axes[1,1].set_xlabel("Error Amount")
        axes[1,1].set_ylabel("Frequency")

        axes[2,1].scatter(output_validation_plotting,residuals,alpha=0.7)
        axes[2,1].axhline(0,color="red",linestyle="--")
        axes[2,1].set_title(f"Error vs {plot_output}")
        axes[2,1].set_xlabel(f"Value of {plot_output}")
        axes[2,1].set_ylabel("Error Amount")
        
        
        divider=make_axes_locatable(axes[2,1])
        ax_hist=divider.append_axes("right",size="20%",pad=0.1,sharey=axes[2,1])
        ax_hist.hist(residuals,bins=30,orientation="horizontal",color="teal")
        
        
        axes[2,0].scatter(input_validation_plotting,residuals,alpha=0.7)
        axes[2,0].axhline(0,color="red",linestyle="--")
        axes[2,0].set_title(f"Errors vs {plot_input}")
        axes[2,0].set_xlabel(f"Value of {plot_input}")
        axes[2,0].set_ylabel("Error Amount")
        
        divider=make_axes_locatable(axes[2,0])
        ax_hist=divider.append_axes("right",size="20%",pad=0.1,sharey=axes[2,0])
        ax_hist.hist(residuals,bins=30,orientation="horizontal",color="teal")
        

        
        plt.tight_layout()
        plt.show()
               

        
    @staticmethod
    def common_arguments_dict(function: Callable, kwargs: dict) -> dict:
        """
        Find common arguments between the function's signature and the provided kwargs.

        This method helps to identify which keyword arguments provided to a wrapper
        are also accepted by the underlying function.

        Args:
            function: The function to inspect for its signature.
            kwargs: Dictionary of keyword arguments provided to the wrapper.

        Returns:
            Dictionary of common arguments and their values.

        Example:
            >>> def my_function(a, b, c=3):
            ...     pass
            >>> common_args = DataUtils.common_arguments_dict(my_function, {'a': 1, 'b': 2, 'd': 4})
            >>> print(common_args)
            {'a': 1, 'b': 2}
        """
        signature = inspect.signature(function)  # Input arguments accepted by method
        signature_needed = signature.parameters
        needed_para = (signature_needed.keys())
        names_variables = ", ".join(list(needed_para))
        print(f"\n Input arguments for {function} can be: {names_variables}")  # Printing input arguments accepted by method

        # Input arguments given to wrapper
        all_para = (kwargs.keys())

        # Common arguments between the 2
        final_para = all_para & needed_para  # Checking the intersection of extra arguments to the SM wrapper and input arguments accepted by method
        # Define common dict of the arguments
        common_dict_DR = {}

        for key in final_para:
            if key in all_para:
                common_dict_DR[key] = kwargs[key]  # Assigning value to the common parameter

        return common_dict_DR

class GemesoRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for GEMSEO regressors to make them compatible with scikit-learn for hyperparameter optimization.

    This class wraps GEMSEO regressors to provide a scikit-learn compatible interface,
    allowing for hyperparameter optimization and other scikit-learn features.
    """

    def __init__(self, **kwargs):
        """
        Initialize the wrapper with hyperparameters and model settings.

        Args:
            **kwargs: Hyperparameters and settings for the regressor.
                     Must include 'regressor_name' to specify the regressor to use.

        Raises:
            KeyError: If 'regressor_name' is not provided in kwargs.
        """
        if 'regressor_name' not in kwargs:
            raise KeyError("'regressor_name' must be provided in kwargs")

        self.hyperparams = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.model_ = None
        self.SM_gemseo = {
            "LinearRegressor_gemseo": LinearRegressor,
            "MOERegressor_gemseo": MOERegressor,
            "PCERegressor_gemseo": PCERegressor,
            "PolynomialRegressor_gemseo": PolynomialRegressor,
            "RBFRegressor_gemseo": RBFRegressor,
            "TPSRegressor_gemseo": TPSRegressor
        }
        self.SM_gemseo_settings = {
            "LinearRegressor_gemseo": LinearRegressor_Settings,
            "PCERegressor_gemseo": PCERegressor_Settings,
            "PolynomialRegressor_gemseo": PolynomialRegressor_Settings,
            "RBFRegressor_gemseo": RBFRegressor_Settings,
            "TPSRegressor_gemseo": TPSRegressor_Settings,
            "MOERegressor_gemseo": MOE_Settings
        }
        self.regressor_name = kwargs["regressor_name"]
        self.sett = self.SM_gemseo_settings[self.regressor_name]

    def fit(self, X, y) -> Any:
        """
        Fit the surrogate model with the given data.

        Args:
            X: Input data, typically a numpy array or pandas DataFrame.
            y: Output data, typically a numpy array or pandas DataFrame.

        Returns:
            The fitted model.

        Raises:
            ValueError: If the regressor name is not supported.
        """
        if self.regressor_name not in self.SM_gemseo:
            raise ValueError(f"Unsupported regressor name: {self.regressor_name}")

        self.model_ = self.train_surrogate(
            inputs=X,
            outputs=y,
            regressor_settings=self.sett,
            **self.hyperparams,
            reg_name=self.regressor_name
        )
        return self.model_

    def predict(self, X) -> Any:
        """
        Predict using the fitted surrogate model.

        Args:
            X: Input data for prediction, typically a numpy array or pandas DataFrame.

        Returns:
            Predicted output.

        Raises:
            ValueError: If the model is not trained yet.
        """
        if self.model_ is None:
            raise ValueError("Surrogate model is not trained yet.")
        pred = []

        pr = self.model_.predict(X)
        pred.append(pr)

        return pr

    def get_params(self, deep: bool = True) -> dict:
        """
        Get the hyperparameters of the model.

        Args:
            deep: Whether to return deep parameters. If True, return the parameters
                  of the wrapped model as well.

        Returns:
            Dictionary of hyperparameters.
        """
        return self.hyperparams.copy()

    def set_params(self, **params) -> Any:
        """
        Set the hyperparameters of the model.

        Args:
            **params: Hyperparameters to set.

        Returns:
            The model instance.
        """
        if not params:
            return self

        for key, value in params.items():
            setattr(self, key, value)
            self.hyperparams[key] = value
        return self

    def train_surrogate(
        self,
        inputs: dict[str, NDArray],
        outputs: dict[str, NDArray],
        regressor_name: Any,
        regressor_settings: Optional[Any],
        **regressor_args: Any,
    ) -> Any:
        """
        Train the surrogate model.

        This method creates an IODataset from the inputs and outputs,
        finds common arguments to be passed to the regressor, and trains the model.

        Args:
            inputs: Dictionary of input variables and their values as numpy arrays.
            outputs: Dictionary of output variables and their values as numpy arrays.
            regressor_name: Name of the regressor to use.
            regressor_settings: Settings for the regressor.
            **regressor_args: Additional arguments for the regressor.

        Returns:
            The trained model.

        Raises:
            ValueError: If the regressor name is not supported.
        """
        regressor_name = regressor_args["reg_name"]
        data = IODataset()
        for n, v in inputs.items():
            data.add_input_variable(n, v)
        for n, v in outputs.items():
            data.add_output_variable(n, v)

        # Finding common arguments to be passed to regressor
        signature = inspect.signature(regressor_settings)
        signature_needed = signature.parameters
        self.needed_para_SM = (signature_needed.keys())
        names_variables = ", ".join(list(self.needed_para_SM))
        print(f"\n Input arguments for Surrogate Model can be: {names_variables}")

        # Input arguments given to class
        signature = inspect.signature(self.__init__)
        signature_given = signature.parameters
        kargs = (regressor_args.keys())
        all_para_wo_k = (signature_given.keys())
        all_para = kargs

        # Common arguments between the 2
        final_para = all_para & self.needed_para_SM
        self.common_dict_SM = {}
        for key in final_para:
            if key in all_para_wo_k:
                self.common_dict_SM[key] = regressor_args[key]
            elif key in regressor_args:
                self.common_dict_SM[key] = regressor_args[key]
        print(self.common_dict_SM)

        # Special case because PCE requires distribution as inputs for the choice of basis
        if regressor_name == "PCERegressor_gemseo":
            parameter_space = ParameterSpace()
            PDFdist = (regressor_args["input_distribution"])
            if len(PDFdist) == inputs.shape[1]:
                for i in range(inputs.shape[1]):
                    inp = (inputs.columns[i])
                    parameter_space.add_random_variable(str(inp), PDFdist[i - 1])
            settings = PCERegressor_Settings(probability_space=parameter_space, **self.common_dict_SM)
            self.model_ = PCERegressor(data=data, settings_model=settings)
            self.model_.learn()
        # Default gemseo implementation for SM
        elif regressor_name in self.SM_gemseo.keys():
            SM = self.SM_gemseo[regressor_name]
            settings = regressor_settings(**self.common_dict_SM)
            self.model_ = SM(data=data, settings_model=settings)
            self.model_.learn()
            print(self.model_)
        else:
            raise ValueError(f"Unsupported regressor name: {regressor_name}")

        return self.model_
