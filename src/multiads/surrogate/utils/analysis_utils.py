from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from gemseo.datasets.io_dataset import IODataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import inspect
from sklearn.base import BaseEstimator, RegressorMixin
import pyLOM, pyLOM.NN
from scipy.stats import norm
import openturns as ot

def uncertainty(normalize: bool,
                normalize_method: str,
                surrogate_train: Any,
                dimension_reduction: bool,
                regressor_name: str,
                scaler_X_start: Any,
                scaler_Y_start: Any,
                input_data: dict[str, NDArray],
                pca: Optional[Any] = None,
                ) -> Optional[NDArray]:
    """
    Calculate uncertainty for the given input data.

    This function calculates the uncertainty of predictions for a Gaussian Process Regressor.
    If dimension reduction was applied, it adjusts the uncertainty accordingly.

    Args:
        normalize: Whether to normalize the input data before calculating uncertainty.
        normalize_method: Method used for normalization (e.g., "MinMaxScaler", "StandardScaler").
        surrogate_train: Trained surrogate model that supports uncertainty estimation.
        dimension_reduction: Whether dimension reduction was applied to the data.
        regressor_name: Name of the regressor. Only "GaussianProcessRegressor_scikit" supports uncertainty.
        scaler_X_start: Scaler for input data.
        scaler_Y_start: Scaler for output data.
        input_data: Input data for which uncertainty is to be calculated.
        pca: PCA transformer if dimension reduction was applied.

    Returns:
        Array containing the uncertainty for each output, or None if the regressor does not support uncertainty.

    Raises:
        ValueError: If the regressor does not support uncertainty.

    Note:
        The uncertainty is adjusted based on the normalization method used.
        For dimension reduction, the uncertainty is adjusted using the PCA components.
    """
    data = input_data
    if normalize == True:  # Normalize the input data if required
        data = scaler_X_start.transform(data)

    if regressor_name == "GaussianProcessRegressor_scikit":
        out, pred_std = surrogate_train.predict(data, return_std=True)  # return_std gives the uncertainty

        if dimension_reduction == True:
            variance = np.dot(pred_std**2, pca.components.T**2)  # Dot product of the uncertainty and PCA components
            pred_std = np.sqrt(variance)

        if normalize_method == "MinMaxScaler":  # Re-scaling the std as per the data scaling applied
            scale = scaler_Y_start.data_max_ - scaler_Y_start.data_min_
            pred_std = (pred_std * scale)  # sigma*(max-min)
        elif normalize_method == "StandardScaler":
            scale = np.sqrt(scaler_Y_start.var_)
            pred_std = (pred_std * scale)  # sigma*(sqrt(var(output)))
        elif normalize_method == "RobustScaler":
            scale = scaler_Y_start.scale_
            pred_std = (pred_std * scale)  # sigma*scale

        return pred_std
    else:
        raise ValueError(f"Regressor {regressor_name} does not support uncertainty")

def get_uncertainty_bounds(input_data: dict[str, NDArray],
                           prediction_internal: Callable,
                           uncertainty: NDArray,
                           confidence_percentage: int,
                           ) -> Tuple[NDArray, NDArray]:
    """
    Calculate uncertainty bounds for the given input data.

    This function calculates the lower and upper bounds for the predictions
    based on the given uncertainty and confidence percentage.

    Args:
        input_data: Input data for which uncertainty bounds are to be calculated.
        prediction_internal: Function to get predictions for the input data.
        uncertainty: Uncertainty for the input data.
        confidence_percentage: Confidence percentage for the bounds (e.g., 95 for 95% confidence).

    Returns:
        Tuple containing the lower and upper bounds for each output.

    Note:
        The bounds are calculated using the z-factor from the standard normal distribution.
        The function assumes that the uncertainty is already adjusted for normalization.
    """
    p = confidence_percentage / 100.0

    alpha_tail = (1.0 - p) / 2  # Getting the tail of the distribution

    factor = norm.ppf(1.0 - alpha_tail)  # z factor of CI

    pred = prediction_internal(input_data)  # Get predictions for the input data

    low_b = pred - factor * uncertainty  # Lower bound
    upp_b = pred + factor * uncertainty  # Upper bound

    return low_b, upp_b

def gradients(method: str,
              delta_x: float,
              inputs: Any,
              prediction_internal: Callable,
              dimension_reduction: bool,
              normalize: bool,
              scaler_X_start: Any,
              scaler_Y_start: Any,
              surrogate_train: Any,
              input_data: Optional[np.array] = None,
              n: Optional[int] = 50,
              ) -> NDArray:
    """
    Calculate gradients for the given input data.

    This function calculates the gradients of the model outputs with respect to the inputs.
    It supports both finite difference (FD) and analytical gradients from GEMSEO.

    Args:
        method: Method for calculating gradients ("FD" for finite difference, "gemseo" for analytical gradients).
        delta_x: Step size for finite difference.
        inputs: Input variables, typically a list of BaseVariable objects.
        prediction_internal: Function to get predictions for the input data.
        dimension_reduction: Whether dimension reduction was applied to the data.
        normalize: Whether to normalize the input data before calculating gradients.
        scaler_X_start: Scaler for input data.
        scaler_Y_start: Scaler for output data.
        surrogate_train: Trained surrogate model that supports gradient calculation.
        input_data: Input data for which gradients are to be calculated.
                     If None, the design space is split into n parts for gradient evaluation.
        n: Number of points to evaluate gradients if input_data is not provided.

    Returns:
        Array containing the gradients for each output with respect to each input.

    Note:
        If dimension reduction was applied, the finite difference method is used.
        For analytical gradients, the function adjusts the gradients to the original physical space.
    """
    x = input_data
    if x is None:  # If input_data is not given, split the design space in n parts for gradient evaluation
        x = []
        for v in inputs:
            if not isinstance(v, np.ndarray):
                v_array = np.atleast_1d(v.value)
            else:
                v_array = v.value

            size = v_array.shape[0]
            # Getting the bounds and splitting the design space in n parts
            lb = v.lb
            ub = v.ub
            val = v.value
            lb = np.where(np.isneginf(lb), -3 * val, lb)
            ub = np.where(np.isposinf(ub), 3 * val, ub)

            # Manage the case val=0
            equal_bounds = (lb == ub)
            lb = np.where(equal_bounds, -1.0, lb)
            ub = np.where(equal_bounds, 1.0, ub)

            input_variable = np.linspace(lb, ub, n)
            x.append(input_variable)

        x = np.column_stack(x)

    Y = prediction_internal(x)

    if method == "FD" or dimension_reduction == True:  # Finite difference method
        dim = len(x)
        grad_model = np.zeros(np.shape(x))
        n_sample = np.shape(x)[0]  # Number of samples
        n_inputs = np.shape(x)[1]  # Number of input features
        n_outputs = np.shape(Y)[1]  # Number of output features
        grad_model = np.zeros((n_sample, n_outputs, n_inputs))

        for i in range(n_inputs):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[:, i] = x_plus[:, i] + delta_x
            x_minus[:, i] = x_minus[:, i] - delta_x

            y_plus = prediction_internal(x_plus)
            y_minus = prediction_internal(x_minus)

            gradient = (y_plus - y_minus) / (2 * delta_x)  # Central difference formula
            grad_model[:, :, i] = gradient

    elif method == "gemseo":  # Analytical gradients from gemseo
        if normalize == True:  # Normalize the inputs if required
            x = scaler_X_start.transform(x)
        try:
            grad_model = surrogate_train.predict_jacobian(x)  # Gets the Jacobian if avaibale
        except:
            print("Regressor has no anaytical gradient")
            return None # returns None
        if normalize == True:
            n_inputs = scaler_X_start.n_features_in_  # Re-scaling the Jacobian to be in original physical space
            x_norm_0 = np.zeros((1, n_inputs))
            x_norm_1 = np.ones((1, n_inputs))

            x_phys_0 = scaler_X_start.inverse_transform(x_norm_0)  # Lower bound for MinMax, mean for standard scaler
            x_phys_1 = scaler_X_start.inverse_transform(x_norm_1)  # Upper bound for MinMax, std for standard scaler

            factor_x = (x_phys_1 - x_phys_0).flatten()  # Default approach for all normalization methods used

            try:
                n_outputs = scaler_Y_start.n_features_in_
                y_norm_0 = np.zeros((1, n_outputs))
                y_norm_1 = np.ones((1, n_outputs))

                y_phys_0 = scaler_Y_start.inverse_transform(y_norm_0)
                y_phys_1 = scaler_Y_start.inverse_transform(y_norm_1)

                factor_y = (y_phys_1 - y_phys_0).flatten()

            except AttributeError:
                factor_y = np.array([1.0])

            grad_model = grad_model * (factor_y[:, None] / factor_x[None, :])  # Converting the gradients to the physical space

    return grad_model

def sensitivity(method: str,
                outputs: Any,
                dimension_reduction: bool,
                regressor_name: str,
                surrogate_train: Any,
                inputs: Any,
                prediction_internal: Callable,
                gradients: Optional[Callable] = None,
                n: Optional[int] = 10000,
                ) -> Dict[str, Dict[str, Any]]:
    """
    Calculate sensitivity indices for the given input data.

    This function calculates sensitivity indices using either Sobol or DGSM methods.
    For Sobol, it uses either analytical indices from PCE or Monte Carlo sampling.
    For DGSM, it uses the gradients to calculate Morris sensitivity indices.

    Args:
        method: Method for calculating sensitivity indices ("Sobol" or "DGSM").
        outputs: Output variables, typically a list of BaseVariable objects.
        dimension_reduction: Whether dimension reduction was applied to the data.
        regressor_name: Name of the regressor.
        surrogate_train: Trained surrogate model.
        inputs: Input variables, typically a list of BaseVariable objects.
        prediction_internal: Function to get predictions for the input data.
        gradients: Function to calculate gradients, required for DGSM method.
        n: Number of samples for sensitivity analysis.

    Returns:
        Dictionary containing the sensitivity indices for each output.
        For Sobol, it includes first-order and total-order indices.
        For DGSM, it includes Morris sensitivity indices.

    Note:
        For Sobol, if the regressor is PCE and dimension reduction is not applied,
        it uses analytical indices. Otherwise, it uses Monte Carlo sampling.
        For DGSM, it uses the finite difference method to calculate gradients.
    """
    if method == "Sobol":  # Sobol sensitivity analysis
        names = []
        input_names=[]
        sensi_matrix = {}
        for v in outputs:
            names.append(v.name)
            
        for v in inputs:
            input_names.append(v.name)

        if regressor_name == "PCERegressor_gemseo" and dimension_reduction == False:  # Analytical sensitivity for PCE
            for i, name in enumerate(names):
                key = i
                sensi_matrix_first_order = surrogate_train.first_sobol_indices  # Gemseo approach for Sobol indices
                sensi_matrix_second_order = surrogate_train.total_sobol_indices
                s1_val = sensi_matrix_first_order[key]
                st_val = sensi_matrix_second_order[key]

                sensi_matrix[name] = {"S1 Main Effect": s1_val,
                                      "ST Total Effect": st_val}
            return sensi_matrix
        else:  # Monte Carlo sampling for Sobol indices
            marginals = []
            for v in inputs:  # Getting the bounds of the function to be evaluated
                if not isinstance(v, np.ndarray):
                    v_array = np.atleast_1d(v.value)
                else:
                    v_array = v.value

                size = v_array.shape[0]

                lb = v.lb
                ub = v.ub
                val = v.value
                lb = np.where(np.isneginf(lb), -3 * val, lb)
                ub = np.where(np.isposinf(ub), 3 * val, ub)

                # Manage the case val=0
                equal_bounds = (lb == ub)
                lb = np.where(equal_bounds, -1.0, lb)
                ub = np.where(equal_bounds, 1.0, ub)

                x = ot.Uniform(int(lb), int(ub))
                marginals.append(x)

            input_dist = ot.ComposedDistribution(marginals)
            sie = ot.SobolIndicesExperiment(input_dist, n)
            input = sie.generate()
            input = np.array(input)
            pred = prediction_internal(input)

            sensi_matrix_first_order = {}  # Creating a dict for first order
            sensi_matrix_second_order = {}  # Creating a dict for total order

            for i in range(np.shape(pred)[1]):
                Y_col = pred[:, i].reshape(-1, 1)
                Y_col = ot.Sample(Y_col)
                sensi = ot.SaltelliSensitivityAlgorithm(input, Y_col, n)
                first_order = sensi.getFirstOrderIndices()
                first_order = list(first_order)
                second_order = sensi.getTotalOrderIndices()
                second_order = list(second_order)

                sensi_matrix_first_order[f"{i + 1}"] = [first_order]
                sensi_matrix_second_order[f"{i + 1}"] = [second_order]

            for i, name in enumerate(names):
                key = str(i + 1)
                s1_val = sensi_matrix_first_order[key][0]
                st_val = sensi_matrix_second_order[key][0]

                sensi_matrix[name] = {"S1 Main Effect": s1_val,
                                      "ST Total Effect": st_val}

            return sensi_matrix

    elif method == "DGSM":  # Morris sensitivity analysis based on gradients
        if gradients is None:
            raise ValueError("Gradients function is required for DGSM method")

        grad = gradients(method="FD", n=n)
        morris_sensitivity = np.mean((grad)**2, axis=0)
        return morris_sensitivity