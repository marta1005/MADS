from typing import Any, Callable, Optional, Type
import matplotlib.cm as cm

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
import matplotlib.pyplot as plt
import pyLOM
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from ..models.regressor_training_models import RegressorTrainingModelFactory

class DimensionalityReductionModelFactory:
    """
    A factory class for dimensionality reduction methods from different libraries (scikit-learn, pyLOM).

    This class provides methods to perform both static and dynamic dimensionality reduction
    on input and output data using various algorithms from scikit-learn and pyLOM.
    """

    def __init__(self):
        """
        Initialize the DimensionalityReductionModelFactory with dictionaries of available dimensionality reduction methods.

        The factory is initialized with a dictionary of scikit-learn dimensionality reduction methods.
        """
        self.DR_methods_scikit = {
            "PCA_scikit": PCA,
            "KPCA_scikit": KernelPCA,
            "PLSRegression_scikit": PLSRegression
        }

    def StaticDimensionalityReductionFactory(self, dimension_reduction_method: str, dimension_reduction_space: str, extra_args: dict, data: IODataset = IODataset()) -> tuple[Any, Any, Any]:
        """
        Perform static dimensionality reduction on the specified data.

        This method applies dimensionality reduction to either the input or output space
        of the provided dataset using the specified method and additional arguments.

        Args:
            dimension_reduction_method: Name of the dimensionality reduction method to use.
                                        Must be one of the keys in `DR_methods_scikit`.
            dimension_reduction_space: Space to apply dimensionality reduction ("inputs" or "outputs").
            extra_args: Dictionary of additional arguments for the dimensionality reduction method.
            data: Input data for dimensionality reduction, typically an IODataset.

        Returns:
            A tuple containing:
            - Reduced input data (IODataset)
            - Reduced output data (IODataset)
            - The dimensionality reduction transformer (e.g., PCA, KernelPCA, PLSRegression)

        Raises:
            KeyError: If the specified dimensionality reduction method is not supported.

        Note:
            The method creates temporary variable names for the reduced dimensions.
            The dimensionality reduction transformer is stored as an instance variable for later use.
        """
        if dimension_reduction_method not in self.DR_methods_scikit.keys():
            raise KeyError(
                f"Unknown dimensionality reduction method: '{dimension_reduction_method}'."
                f"Must be one of: {list(self.DR_methods_scikit.keys())}"
            )

        self.dr_utils = DataUtils()

        self.common_dict = self.dr_utils.common_arguments_dict(self.DR_methods_scikit[dimension_reduction_method], extra_args)

        print(f"\n Values for parameters given: {self.common_dict} \n")

        # DR in output or input space
        if "outputs" in dimension_reduction_space:
            # Saving the transformation for reconstruction
            [data_reduced_out, self.pca] = self.dim_red_tranform(
                DR_method=dimension_reduction_method,
                fulldata=data.outputs.to_numpy(),
                input_data=data.inputs.to_numpy(),
                DR_methods_scikit=self.DR_methods_scikit
            )

            columns = data_reduced_out.shape[1]
            my_string = [f"out{j}" for j in range(1, columns + 1)]  # Creating temporary variable names for reduced dimensions
            u_T = pd.DataFrame(data_reduced_out, columns=my_string)
            u_T = IODataset(u_T)

            # New reduced dataset split
            X_train = data.inputs
            Y_train = u_T
            return X_train, Y_train, self.pca

        elif "inputs" in dimension_reduction_space:
            [data_reduced_in, self.pca] = self.dim_red_tranform(
                DR_method=dimension_reduction_method,
                fulldata=data.inputs.to_numpy(),
                input_data=data.outputs.to_numpy(),  # Switched to be comparable to the "outputs"
                DR_methods_scikit=self.DR_methods_scikit
            )

            # Reshaping
            columns = data_reduced_in.shape[1]
            my_string = [f"inp{j}" for j in range(1, columns + 1)]  # Creating temporary variable names for reduced dimensions
            u_T = pd.DataFrame(data_reduced_in, columns=my_string)
            u_T = IODataset(u_T)
            X_train = u_T
            Y_train = data.outputs
            return X_train, Y_train, self.pca

    def dim_red_tranform(self, DR_method: str, fulldata: NDArray, input_data: NDArray, **extra_parameters: Any) -> tuple[NDArray, Any]:
        """
        Transform data using the specified dimensionality reduction method.

        This method applies the specified dimensionality reduction method to the provided data
        and returns the transformed data along with the transformer.

        Args:
            DR_method: Name of the dimensionality reduction method to use.
            fulldata: Data to transform, typically a numpy array.
            input_data: Additional input data for some methods (e.g., PLSRegression).
            **extra_parameters: Additional parameters for the dimensionality reduction method.

        Returns:
            A tuple containing:
            - Transformed data (numpy array)
            - The dimensionality reduction transformer (e.g., PCA, KernelPCA, PLSRegression)

        Note:
            The method uses the common arguments dictionary to initialize the transformer.
            For PLSRegression, both input and output data are used for transformation.
        """
        u_t_red = IODataset()

        print(f"DR dictionary: {self.common_dict}")

        # Matching the strings
        match DR_method:
            case "KPCA_scikit" | "PCA_scikit":  # Same implementation
                # Find the transformer class being used
                transformer_class = self.DR_methods_scikit[DR_method]
                # Passing the extra arguments being used
                pca = transformer_class(**self.common_dict)
                # Data transformed in reduced dimension
                u_t_red = pca.fit_transform(fulldata)

                return u_t_red, pca  # Return both the reduced data and the transformer

            case "PLSRegression_scikit":  # Needs both inputs and outputs
                pca = PLSRegression(**self.common_dict)
                # Sequence can change if the inputs or outputs are to be reduced
                u_t_red, U = pca.fit_transform(fulldata, input_data)
                return u_t_red, pca  # Return both the reduced data and the transformer

    def dim_red_tranform_dynamic(self, DR_method: str, output_data: dict[str, NDArray], input_data: NDArray, DR_methods_pylom: Optional[dict[str, Any]], name: str = "", scale: Optional[bool] = False, fit_inverse_tranform: bool = True, num_samples: int = int, T: int = int, N_snapshots: int = int, SM: Any = Any, common_dict_DR: dict = dict, normalize: bool = bool, normalize_method: str = str, **extra_parameters: Any) -> tuple[Any, ...]:
        """
        Perform dynamic dimensionality reduction on the specified data.

        This method applies dynamic dimensionality reduction methods such as POD, DMD, and SPOD
        to the provided data and returns the reduced data along with the necessary transformers.

        Args:
            DR_method: Name of the dynamic dimensionality reduction method to use.
                       Must be one of the keys in `DR_methods_pylom`.
            output_data: Output data for dimensionality reduction, typically a dictionary of numpy arrays.
            input_data: Input data for dimensionality reduction, typically a numpy array.
            DR_methods_pylom: Dictionary of pyLOM dimensionality reduction methods.
            name: Name for the dimensionality reduction method (optional).
            scale: Whether to scale the data (default: False).
            fit_inverse_tranform: Whether to fit the inverse transform (default: True).
            num_samples: Number of samples (default: int).
            T: Time steps (default: int).
            N_snapshots: Number of snapshots (default: int).
            SM: Surrogate model to use (default: Any).
            common_dict_DR: Common arguments for the dimensionality reduction method (default: dict).
            normalize: Whether to normalize the data (default: bool).
            normalize_method: Method to use for normalization (default: str).
            **extra_parameters: Additional parameters for the dimensionality reduction method.

        Returns:
            A tuple containing the reduced data and the necessary transformers, which may include:
            - Surrogate model for A (surr_A)
            - Singular values (S_out)
            - Modes (PSI_out)
            - Scalers for input data, A vectors, and initial conditions (if normalization is applied)

        Note:
            The method reshapes the output data and applies the specified dynamic dimensionality reduction method.
            For POD and DMD, it also fits surrogate models to the reduced data.
            The method plots the reconstructed data for visualization.
        """
        # Defining method with arguments provided
        # Matching cases for implementation
        self.Regressor_training_factory = RegressorTrainingModelFactory()
        r = extra_parameters["extra_parameters"]["n_components"]

        output_data = list(output_data.values())
        output_data = np.vstack(output_data)
        output_data = output_data.reshape(N_snapshots * num_samples, -1)
        output_data = output_data.T

        SM_gemseo = self.Regressor_training_factory.SM_gemseo
        SM_gemseo_settings = self.Regressor_training_factory.SM_gemseo_settings

        # Dictionary for SM scikit
        SM_scikit = self.Regressor_training_factory.SM_scikit

        # Dictionary for SM pylom
        SM_pylom = self.Regressor_training_factory.DR_methods_pylom

        print(DR_method)
        if DR_method == "POD_pylom":
            transformer_class = DR_methods_pylom[DR_method]
            PSI, S, V = transformer_class.run(output_data, remove_mean=False, **common_dict_DR)
            PSI_out, S_out, V_out = transformer_class.truncate(PSI, S, V, r=r)
            V_out = V_out.T

            X_train = []
            for i in range(num_samples):
                t_col = T.reshape(-1, 1)
                p_current = input_data[i]
                p_repeated = np.tile(p_current, (len(T), 1))
                block = np.hstack((t_col, p_repeated))
                X_train.append(block)

            X_train = np.vstack(X_train)
            SM = SM_scikit[SM]
            SM_A = SM()
            surr_A = SM_A.fit(X_train, V_out)

            recon = pyLOM.POD.reconstruct(PSI_out, S_out, V_out.T)
            return surr_A, S_out, PSI_out

        if DR_method == "DMD_parameterized" or DR_method == "DMD_pylom":
            PSI, S, V = pyLOM.POD.run((output_data), remove_mean=False)
            PSI_out, S_out, V_out = pyLOM.POD.truncate(PSI, S, V, r)

            S_mat = np.diag(S_out)
            C = S_mat @ V_out

            all_C_matrix = []
            for k in range(num_samples):
                X_k = output_data[:, k * len(T):k * len(T) + len(T)]
                C_k = PSI_out.T @ X_k
                all_C_matrix.append(C_k)

            all_C_matrix = np.array(all_C_matrix)
            A_vectors = []
            initial_conds = []

            for k in range(num_samples):
                C = all_C_matrix[k, :, :]
                C1 = C[:, :-1]
                C2 = C[:, 1:]
                A_tilde = C2 @ np.linalg.pinv(C1)
                A_flat = A_tilde.flatten()
                A_vectors.append(A_flat)
                initial_conds.append(C[:, 0])
            A_vectors = np.array(A_vectors)
            initial_conds = np.array(initial_conds)

            if normalize:
                Normalize_methods = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler, "Normalizer": Normalizer}
                Normalize = Normalize_methods[normalize_method]
                # Save the scalers for transforming back
                self.scaler_input_data = Normalize()
                self.scaler_A_vectors = Normalize()
                self.scaler_initial_conds = Normalize()
                input_data = self.scaler_input_data.fit_transform(input_data)
                A_vectors = self.scaler_A_vectors.fit_transform(A_vectors)
                initial_conds = self.scaler_initial_conds.fit_transform(initial_conds)

            if SM in SM_scikit.keys():
                SoMo_A = SM_scikit[SM]
                SM_A = SoMo_A()
                surr_A = SM_A.fit(input_data, A_vectors)
                SoMo_B = SM_scikit[SM]
                SM_B = SoMo_B()
                surr_B = SM_B.fit(input_data, initial_conds)

            elif SM in SM_gemseo.keys():
                SoMo_A = SM_gemseo[SM]
                SoMo_B = SM_gemseo[SM]
                data_ip = IODataset(input_data, A_vectors)
                data_op = IODataset(input_data, initial_conds)
                surr_A = SoMo_A(data_ip)
                surr_B = SoMo_B(data_op)


            return surr_A, surr_B, PSI_out, self.scaler_input_data, self.scaler_A_vectors, self.scaler_initial_conds

        if DR_method == "SPOD_pylom":
            transformer_class = DR_methods_pylom[DR_method]
            L, P, f = transformer_class.run(output_data, **self.common_dict_DR)
            return L, P, f