from typing import Any, Callable, Optional, Type
import inspect

import logging
import os, sys


import numpy as np

# pyLOM.NN import only in linux
try:
 import pyLOM, pyLOM.NN
except:
 import pyLOM

import optuna
# import torch

#sys.path.insert(
#        0,
#        os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")),
#    )

from gemseo.datasets.io_dataset import IODataset
#from gemseo.mlearning.core.algos.ml_algo import BaseAlgoSettings
from sklearn.model_selection import train_test_split

from scipy.stats import norm

from numpy.typing import NDArray
import numpy as np
from numpy.typing import NDArray

from multiads.disciplines import MADSDiscipline
from multiads.scenario import BaseVariable
from multiads.scenario import MADSScenario
from gemseo.disciplines.surrogate import SurrogateDiscipline as GemseoSurrogate
from gemseo.core.discipline.discipline import Discipline

# MDF settings
from gemseo.formulations.mdf import MDF_Settings
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

# Disciplinary OPt

# DOE Setting- https://gemseo.readthedocs.io/en/6.0.0/algorithms/doe_algos.html
from gemseo.settings.doe import LHS_Settings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

import logging

import pandas as pd

import openturns as ot

from ..models.doe_models import DoEModelFactory
from ..models.dimensionality_reduction_models import DimensionalityReductionModelFactory
from ..models.regressor_training_models import RegressorTrainingModelFactory
from ..models.hyperparameter_optimization import HPOModelFactory
from ..utils.data_utils import DataUtils
from ..utils import analysis_utils
# GemseoSurrogate Discipline
class SurrogateDiscipline(Discipline):
    """
    A class for creating and managing surrogate models for disciplines.

    This class provides functionality for creating surrogate models using various
    algorithms from different libraries (GEMSEO, scikit-learn, pyLOM), with options
    for data scaling, dimension reduction and hyperparameter optimization. It also handles the prediction
    and internal logic for post processing, including uncertainty quantification, gradient calculation,
    sensitivity analysis, and model validation.
    """

    def __init__(
        self,
        discipline: list[MADSDiscipline],
        regressor_name: str,
        num_samples: int,
        inputs: Optional[list[BaseVariable]] = None,
        outputs: Optional[list[BaseVariable]] = None,
        normalize: Optional[bool] = False,
        normalize_method: Optional[str] = "MinMaxScaler",
        train: Optional[bool] = True,
        new_doe: Optional[bool] = True,
        save_doe: Optional[bool] = False,
        doe_method: Optional[str] = "LHS",
        dimension_reduction: Optional[bool] = False,
        dimension_reduction_space: Optional[str] = "outputs",
        dimension_reduction_method: Optional[str] = "PCA_scikit",
        HPO: Optional[bool] = False,
        HPO_space: Optional[Any] = None,
        test_train_split: Optional[bool] = True,
        test_size: Optional[float]=0.2,
        **extra_args: Any,
    ):
        
        super().__init__(name="Surrogate Discipline")
        self.input_grammar.update_from_data({i.name: i.value for i in inputs})
        self.output_grammar.update_from_data({o.name: o.value for o in outputs})
        
        """
        Initialize the SurrogateDiscipline.

        This method initializes the surrogate discipline with the specified settings and prepares the data for model training.

        Args:
            discipline: List of disciplines to create the surrogate for.
            regressor_name: Name of the regressor to use.
            num_samples: Number of samples for DOE.
            inputs: List of input variables. Default is None.
            outputs: List of output variables. Default is None.
            normalize: Whether to normalize the data. Default is False.
            normalize_method: Method for normalization. Default is "MinMaxScaler".
            train: Whether to train the model. Default is True.
            new_doe: Whether to create a new DOE. Default is True.
            doe_method: DOE method to use. Default is "LHS".
            dimension_reduction: Whether to apply dimension reduction. Default is False.
            dimension_reduction_space: Space for dimension reduction (inputs/outputs). Default is "outputs".
            dimension_reduction_method: Method for dimension reduction. Default is "PCA_scikit".
            HPO: Whether to perform hyperparameter optimization. Default is False.
            HPO_space: Hyperparameter space for optimization. Default is None.
            test_train_split: Whether to split the data into training and validation sets. Default is True.
            test_size: Splitting percentage between training and testing samples
            **extra_args: Additional arguments for the regressor.

        Note:
            For pyLOM regressors, normalization is disabled as it is handled internally.
        """
        # Initialize instance variables

        
        self.normalize = normalize
        self.dimension_reduction = dimension_reduction
        self.normalize_method = normalize_method
        self.dimension_reduction = dimension_reduction
        self.dimension_reduction_space = dimension_reduction_space
        self.dimension_reduction_method = dimension_reduction_method
        self.regressor_name = regressor_name
        self.inputs = inputs
        self.outputs = outputs
        self.HPO = HPO
        self.test_train_split = test_train_split
        self.discipline = discipline
        self.extraargs = extra_args
        self.save_doe=save_doe
        # Initialize factories for different functionalities
        self.doe_factory = DoEModelFactory()
        self.DR_factory = DimensionalityReductionModelFactory()
        self.Regressor_training_factory = RegressorTrainingModelFactory()
        self.HPO_factory = HPOModelFactory()
        self.dr_utils = DataUtils()

        # Dictionary for surrogate models (SM) from gemseo
        self.SM_gemseo = self.Regressor_training_factory.SM_gemseo
        self.SM_gemseo_settings = self.Regressor_training_factory.SM_gemseo_settings

        # Dictionary for SM from scikit-learn
        self.SM_scikit = self.Regressor_training_factory.SM_scikit

        # Dictionary for SM from pylom
        self.SM_pylom = self.Regressor_training_factory.DR_methods_pylom

        # Dictionary for dimensionality reduction (DR) methods from scikit-learn
        self.DR_methods_scikit = self.DR_factory.DR_methods_scikit

        # Dictionary for normalization methods
        Normalize_methods = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler, "Normalizer": Normalizer}

        # MinMaxScaler normalization takes place in pylom SM, so disable normalization for pylom
        if self.regressor_name in self.SM_pylom:
            normalize = False
            self.normalize = False

        # Create discipline if only one discipline is provided and training is enabled
        if train == True:
            

                # Generate dataset for training
                data = self.doe_factory.dataset_for_training(
                            new_doe=new_doe,
                            save_doe=self.save_doe,
                            doe_method= doe_method,
                            num_samples=num_samples,
                            discipline=self.discipline,
                            inputs=self.inputs,
                            outputs=self.outputs)
            
                # Store dataset to used for reference in validation and prediction
                self.dataset = data
                data = (data.droplevel(level=[2], axis=1))

                # Split data into training and validation sets if test_train_split is True
                if test_train_split == True:
                    data, self.data_validation = (train_test_split(data, test_size=test_size))

                # Normalize the data if enabled
                if normalize == True:
                    # finding the scaling method to be used
                    Normalize = Normalize_methods[normalize_method]

                    self.scaler_X_start = Normalize()
                    self.scaler_Y_start = Normalize()

                    data.inputs = self.scaler_X_start.fit_transform(data.inputs.values)
                    data.outputs = self.scaler_Y_start.fit_transform(data.outputs.values)

                else:
                    pass

                # Prepare data for surrogate model training
                if dimension_reduction == False:
                    # X_train, Y_train are common outputs after the DR step is done.
                    # X_train, Y_train is accepted by Surrogate model defining
                    X_train = data.inputs
                    Y_train = data.outputs

                elif dimension_reduction == True:
                    # Apply dimensionality reduction
                    X_train, Y_train, self.pca = self.DR_factory.StaticDimensionalityReductionFactory(
                                        dimension_reduction_method=self.dimension_reduction_method,
                                        dimension_reduction_space=self.dimension_reduction_space,
                                        extra_args=extra_args,
                                        data=data)
                    #renormalizing if dimension reduction is applied

                    if normalize == True:
                        Normalize = Normalize_methods[normalize_method]
                        # Save scalers for transforming back
                        self.scaler_X_DR = Normalize()
                        self.scaler_Y_DR = Normalize()

                        column_inp_names = X_train.columns.tolist()
                        column_op_names = Y_train.columns.tolist()

                        X_train = self.scaler_X_DR.fit_transform(X_train.values)
                        Y_train = self.scaler_Y_DR.fit_transform(Y_train.values)
                        X_train = pd.DataFrame(X_train, columns=column_inp_names)
                        Y_train = pd.DataFrame(Y_train, columns=column_op_names)

#-----------------------Surrogate model--------------------------------------------
                # Train surrogate model
                self.surrogate_train = self.Regressor_training_factory.training_regressor(
                           regressor_name=self.regressor_name,
                           X_train=X_train,
                           Y_train=Y_train,
                           inputs=self.inputs,  # for reference for labels
                           outputs=self.outputs,  # for reference for labels
                           extra_args=extra_args)

                # Perform hyperparameter optimization if enabled
                if HPO == True or "optimization_params" in extra_args:  # checking if HPO is true, for pylom regressor having optimization_params in arguments also works
                    self.surrogate_train = self.HPO_factory.HyperParameterOptimizationFactory(
                        regressor_name=regressor_name,
                        X_train=X_train,
                        Y_train=Y_train,
                        HPO_space=HPO_space,
                        **extra_args)
        else:
            pass

    def prediction_internal(self, input_data: dict[str, NDArray]) -> NDArray:
        """
        Perform internal predictions that return the raw output as a numpy array.

        This method handles the preprocessing of input data, including normalization and dimensionality reduction,
        and returns the raw output from the surrogate model.

        Args:
            input_data: Dictionary of input variables and their values as numpy arrays.

        Returns:
            NDArray of output variables and their values as numpy arrays.

        Note:
            The method handles different types of regressors (GEMSEO, scikit-learn, pyLOM) and applies the necessary transformations.
        """
        data = np.atleast_2d(input_data)
        if self.normalize == True:
            if data.ndim == 2:  # reshaping if data has 1 feature or data has 1 sample
                pass
            elif data.shape[0] == self.scaler_X_start.n_features_in_:  # 1 feature
                data = data.reshape(1, -1)
            else:  # 1 sample
                data = data.reshape(-1, 1)
            data = self.scaler_X_start.transform(data)  # normalizing the inputs
        if self.dimension_reduction == True:  # if DR applied
                if "inputs" in self.dimension_reduction_space:
                    if self.dimension_reduction_method == "PCA_scikit" or self.dimension_reduction_method == "KPCA_scikit" or self.dimension_reduction_method == "PLSRegression_scikit":
                        data = self.pca.transform(data)
                    elif self.dimension_reduction_method == "POD_pylom":
                        data = self.PSI * data
                    if self.normalize == True:
                        data = self.scaler_X_DR.transform(data)
        if self.regressor_name in self.SM_gemseo or self.regressor_name in self.SM_scikit:  # same logic for all regressors except pylom
            out = self.surrogate_train.predict(data)
        elif self.regressor_name in self.SM_pylom:
            if self.HPO == True or "optimization_params" in self.extraargs:# if hyperparameter optimzation is done calling the scalers from it
                input_scaler = self.HPO_factory.inputs_scaler_pylom
                outputs_scaler = self.HPO_factory.outputs_scaler_pylom
            else: # if training is done calling the scalers from it
                input_scaler = self.Regressor_training_factory.inputs_scaler_pylom
                outputs_scaler = self.Regressor_training_factory.outputs_scaler_pylom
            input_tensor = self.dr_utils.prediction_dataset_for_pylom(input_data, input_scaler, np.size(self.outputs)) #creating a pytorch dataset for pylom
            out = self.surrogate_train.predict(input_tensor)
            out = np.array(outputs_scaler.inverse_transform([out]))
            flat_data = out
            if hasattr(flat_data, "numpy"):
                flat_data = flat_data.numpy()
            flat_data = flat_data.flatten()
            out = flat_data.reshape(np.shape(input_data)[0], np.shape(self.dataset.outputs.values)[1])  # restructuring the prediction from pylom in the format expected

        if self.dimension_reduction == True:
            if "outputs" in self.dimension_reduction_space:
                if self.normalize == True:
                    if out.ndim == 2:
                        pass
                    elif out.shape[0] == self.scaler_Y_start.n_features_in_:
                        out = out.reshape(1, -1)
                    else:
                        out = out.reshape(-1, 1)
                    out = self.scaler_Y_DR.inverse_transform(out)
                if self.dimension_reduction_method == "PCA_scikit" or self.dimension_reduction_method == "KPCA_scikit" or self.dimension_reduction_method == "PLSRegression_scikit":
                    out = self.pca.inverse_transform(out)
                elif self.dimension_reduction_method == "POD_pylom":
                    out = pyLOM.POD.reconstruct(out, self.S, self.V)
        if out.ndim == 2:
                pass
        elif out.shape[0] == self.scaler_Y_start.n_features_in_:
            out = out.reshape(1, -1)
        else:
            out = out.reshape(-1, 1)
        if self.normalize == True:  # denormalizing the predictions of the SM
            
            out = self.scaler_Y_start.inverse_transform(out)

        return out
    
    
    def prediction(self, input_data: dict[str, NDArray]) -> dict[str, NDArray]:
        """
        Perform predictions that return an IODataset with both inputs and outputs. Final prediction from the class can be called with this method.
        
        Uses prediction_internal for the output prediction and then creates an IODataset.

        This method handles the preprocessing of input data, including normalization and dimensionality reduction,
        and returns the predictions as an IODataset.

        Args:
            input_data: Dictionary of input variables and their values as numpy arrays.

        Returns:
            IODataset containing the input variables and their values, as well as the output variables and their values.

        Note:
            The method handles different types of regressors (GEMSEO, scikit-learn, pyLOM) and applies the necessary transformations.
            It also restructures the output to match the format of the training dataset.
        """

        out=self.prediction_internal(input_data=input_data)

        self.names_outputs = []
        self.names_inputs = []
        # Saving the names of the inputs and outputs variables
        for v in self.outputs:
            self.names_outputs.append(v.name)
        for v in self.inputs:
            self.names_inputs.append(v.name)
        dataio = IODataset()  # IODataset to be given out
        if isinstance(input_data, np.ndarray):
            pass
        else:
            input_data = input_data.values
        input_size=np.size(self.inputs)
        input_data=np.atleast_2d(input_data)
        for k in range(input_size):
            dataio.add_input_variable(self.names_inputs[k], input_data[:, k])
        
        if np.shape(out)[1] == np.shape(self.names_outputs):
            for k in range(np.shape(out)[1]):
                dataio.add_output_variable(self.names_outputs[k], out[:, k])
        else:  # incase DR is applied and the names have mismatch, using the names from training dataset
            for k in enumerate(np.shape(self.dataset.outputs.columns)):
                variable_name = (self.dataset.outputs.columns)
                df = pd.DataFrame(out, columns=variable_name)
                new_multiindex = [("outputs", *t) for t in variable_name]
                new_index = pd.MultiIndex.from_tuples(new_multiindex, names=["GROUP", "VARIABLE", "COMPONENT"])
                df = pd.DataFrame(out, columns=new_index)
                dataio = pd.concat([dataio, df], axis=1)
                dataio = IODataset(dataio)
        return dataio

    def uncertainty(self, input_data: dict[str, NDArray]) -> dict[str, NDArray]:
        """
        Calculate the uncertainty of the predictions.

        This method is applicable only when using Gaussian Process Regressor.
        It returns the standard deviation of the predictions as a measure of uncertainty.

        Args:
            input_data: Dictionary of input variables and their values as numpy arrays.

        Returns:
            Dictionary of output variables and their standard deviations as numpy arrays.

        Note:
            The method handles normalization and dimensionality reduction if applied.
        """
        if self.dimension_reduction == False:
            self.pca = None
        pred_std = analysis_utils.uncertainty(
            normalize=self.normalize,
            normalize_method=self.normalize_method,
            surrogate_train=self.surrogate_train,
            dimension_reduction=self.dimension_reduction,
            regressor_name=self.regressor_name,
            scaler_X_start=self.scaler_X_start,
            scaler_Y_start=self.scaler_Y_start,
            input_data=input_data,
            pca=self.pca)
        return pred_std

    def get_uncertainty_bounds(self, input_data: dict[str, NDArray], confidence_precentage: Optional[float] = 95) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        """
        Calculate the uncertainty bounds of the predictions.

        This method calculates the lower and upper bounds of the predictions based on the specified confidence percentage.

        Args:
            input_data: Dictionary of input variables and their values as numpy arrays.
            confidence_precentage: Confidence percentage for the uncertainty bounds. Default is 95.

        Returns:
            A tuple containing two dictionaries:
            - Lower bounds of the output variables as numpy arrays
            - Upper bounds of the output variables as numpy arrays

        Note:
            The method handles normalization and dimensionality reduction if applied.
        """
        if self.dimension_reduction == False:
            self.pca = None
        pred_std = self.uncertainty(input_data=input_data)

        low_b, upp_b = analysis_utils.get_uncertainty_bounds(
            input_data=input_data,
            prediction_internal=self.prediction_internal,
            uncertainty=pred_std,
            confidence_percentage=confidence_precentage)
        return low_b, upp_b

    def gradients(self, method: str, n: Optional[NDArray] = None, input_data: Optional[NDArray] = None, delta_x: Optional[float] = 1e-3) -> dict[str, NDArray]:
        """
        Calculate the gradients of the outputs with respect to the inputs.

        This method calculates the gradients using the specified method and returns them as a dictionary.

        Args:
            method: The method to use for calculating gradients (e.g., "FD" for finite differences).
            n: If input_data is not given, splits the design space into n parts. Default is None, which uses 50 parts.
            input_data: Optional input data for which to calculate gradients. If None, generates input data.
            delta_x: For finite differences, specifies the jump for the finite difference method. Default is 1e-3.

        Returns:
            Dictionary of output variables and their gradients with respect to the inputs as numpy arrays.

        Note:
            The method handles normalization and dimensionality reduction if applied.
        """
        if input_data is None:
            print(f"Generating input data")
            if n is None:
                n = 50

        grad_model = analysis_utils.gradients(
            method=method,
            delta_x=delta_x,  # for finite difference, specifies the jump for FD
            n=n,  # if input_data is not given, splits the design space in 50 parts
            inputs=self.inputs,
            prediction_internal=self.prediction_internal,
            dimension_reduction=self.dimension_reduction,
            normalize=self.normalize,
            scaler_X_start=self.scaler_X_start,
            scaler_Y_start=self.scaler_Y_start,
            surrogate_train=self.surrogate_train,
            input_data=input_data,
        )
        return grad_model

    def sensitivity(self, method: str, n: Optional[int] = 100000) -> dict[str, NDArray]:
        """
        Calculate the sensitivity of the outputs with respect to the inputs.

        This method calculates the sensitivity matrix using the specified method and returns it as a dictionary.

        Args:
            method: The method to use for calculating sensitivity (e.g., "sobol" or "DGSM").
            n: The number of samples to use for sensitivity analysis. Default is 100000, higher is better.

        Returns:
            Dictionary of output variables and their sensitivity matrices with respect to the inputs as numpy arrays.

        Note:
            The method uses the gradients calculated by the gradients method.
            Higher values of n lead to more accurate results.
        """
        gradients = self.gradients(method="FD", n=n)
        sensi_matrix = analysis_utils.sensitivity(
            method=method,  # sobol or DGSM
            outputs=self.outputs,
            dimension_reduction=self.dimension_reduction,
            regressor_name=self.regressor_name,
            surrogate_train=self.surrogate_train,
            inputs=self.inputs,
            prediction_internal=self.prediction_internal,
            gradients=gradients,
            n=n)  # higher the better
        return sensi_matrix

    def validation(self, num_samples: Optional[int] = 20, 
                   new_doe: Optional[bool] = True, 
                   generate_new_data:Optional[bool] = False,
                   plot:Optional[bool] = False,
                   plot_input=Optional[str],
                   plot_output=Optional[str],
                   doe_method: Optional[str] = "LHS", 
                   individual_error:Optional[bool] = False,
                    **extra_args: Any) -> dict[str, float]:
        """
        Validate the surrogate model using a separate dataset.

        This method validates the surrogate model using either a separate validation dataset or the validation set from the training split.
        It calculates and returns the validation metrics.

        Args:
            num_samples: The number of samples to use for validation. Default is 20.
            new_doe: Whether to create a new DOE for validation. Default is True.
            doe_method: The DOE method to use for validation. Default is "LHS".
            **extra_args: Additional arguments for the validation process.

        Returns:
            Dictionary containing the validation metrics:
            - R2 score
            - Root Mean Squared Error (RMSE)
            - Mean Absolute Percentage Error (MAPE)

        Note:
            If test_train_split is False, the method generates a new dataset for validation.
            If test_train_split is True, it uses the validation data from the training split.
        """
        if self.test_train_split == False or generate_new_data==True:
            # Generate dataset for validation if test_train_split is False
            data = self.doe_factory.dataset_for_validation(
                new_doe=new_doe,
                doe_method=doe_method,
                num_samples=num_samples,
                discipline=self.discipline,
                inputs=self.inputs,
                outputs=self.outputs)
            self.data_validation = data
            #logging.info(f"training_dataset_validate_{self.discipline[0].name}.csv")
            #from pandas import read_csv

#            dataframe = read_csv(
#                f"training_dataset_validate_{self.discipline[0].name}.csv", delimiter=",", header=[0, 1, 2], index_col=None)
#
 #           dataframe.columns = dataframe.columns.set_levels(dataframe.columns.levels[2], level=2)
 #           data = IODataset(dataframe)
            
            pred_dataset = self.prediction(data.inputs)
            pred = pred_dataset.outputs.values


        if self.test_train_split == True:
            # Use the validation data if test_train_split is True
            data = self.data_validation
            pred_dataset = self.prediction(data.inputs)
            pred = pred_dataset.outputs
            
        # Calculate validation metrics
        learning_r2 = r2_score(pred, data.outputs)
        learning_mse = root_mean_squared_error(pred, data.outputs)  # RMSE
        learning_mape = mean_absolute_percentage_error(pred, data.outputs)
        print("Validation Error on unseen data")
        print({"R2": learning_r2, "RMSE": learning_mse, "MAPE": learning_mape})
        
        
        if individual_error==True:
            output_size=np.shape(data.outputs)[1]
            nrmse_dict={}
            rmse_dict={}
            nrmse=[]
            rmse_list=[]
            for iter in range(output_size):
                true_col=data.outputs.values[:,iter] 
                pred_col=pred.values[:,iter] 
                rmse=root_mean_squared_error(true_col ,pred_col)
                val_rang=np.ptp(true_col)
                output_columns=[(col[1],col[2]) for col in pred_dataset.get_columns(as_tuple=True) if col[0]==pred_dataset.OUTPUT_GROUP]
                nrmse.append(rmse/val_rang)
                rmse_list.append(rmse)
            nrmse_dict=dict(zip(output_columns,nrmse))
            rmse_dict=dict(zip(output_columns,rmse_list))
            print({"RMSE":rmse_dict})
            #print({"Normalized RMSE":nrmse_dict})

            
        if plot==True:
            self.dr_utils.ploting_data(plot_input=plot_input,plot_output=plot_output,training_data=self.dataset,validation_data=data,prediction_data=pred_dataset)
        return {"R2": learning_r2, "RMSE": learning_mse, "MAPE": learning_mape}
    
    def _run(self,input_data):
        """
       Internal _run gets called when we execute the discipline or auto called when we run a scenario

        Args:
            input_data: An input and output dictionary for 1 sample and n feature

        Returns:
           results: output dictionary   
           
        Note: depending on the grammar of the ouputs may need to change if output is array or number

        """
        
        input_features={}

        input_features={k:input_data[k] for k in self.names_inputs if k in input_data} #creating a dict for the passed variables
        
        input_features=np.array(list(input_features.values()))# array for prediction_internal

        val=self.prediction_internal(input_features.reshape(1,-1))# reshaping because .execute always passes: 1 sample n features
        val=(val.flatten())        
        
        outputs={name:np.atleast_1d(value).flatten().astype(float) for name, value in zip(self.names_outputs,list(val))}
        return outputs #outputs #results
    
    def _compute_jacobian(self,inputs:None,outputs:None):
        """
        Internal _compute_jacobian gets called when we do .linearize
        
        Args:
            None, gets the input from the input dictionary

        Returns:
           None. jacobians are defined in self.jac, gemseo handles the output
           
        
        Note: for Finite differnce it is better to use the implemenatation form gemseo. Should be called outside with surrogatediscipline.set_jacobain_approximation(method="finite_difference")
        """
        
        input_data=self.get_input_data()# gets the input dictionary

        self._init_jacobian(inputs,outputs) # initialize with the labels
        data=(np.array(list(input_data.values())))
        gradients=self.gradients(method="gemseo",input_data=np.atleast_2d(data)) #getting the anaytical gradient if avaibale
        print(gradients)
        if gradients is None:# if not avaibale doing a finite difference
            gradients=self.gradients(method="FD",input_data=np.atleast_2d(data))

        gradients=np.atleast_2d(np.squeeze(gradients))
        for row_idx, out_names in enumerate(self.names_outputs): # assigning the inputs and outputs according to gemseo grammar
            for col_idx, in_names in  enumerate(self.names_inputs):
                derivative=gradients[row_idx,col_idx]
                self.jac[out_names][in_names]=np.atleast_2d(derivative) # defining the dictionary for jacobian

