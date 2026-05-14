from typing import Any, Callable, Optional, Type
import inspect
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pyLOM
# gemseo 6
from gemseo import create_design_space, create_surrogate, compute_doe

from gemseo.datasets.io_dataset import IODataset
# gemseo 6
#from gemseo.mlearning.core.algos.ml_algo import BaseAlgoSettings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from numpy.typing import NDArray
import numpy as np

from multiads.disciplines import MADSDiscipline
from multiads.scenario import BaseVariable
from multiads.scenario import MADSScenario

from gemseo.disciplines.surrogate import SurrogateDiscipline as GemseoSurrogate
# MDF settings
from gemseo.formulations.mdf import MDF_Settings

# Disciplinary OPt

# DOE Settings - https://gemseo.readthedocs.io/en/6.0.0/algorithms/doe_algos.html
from gemseo.settings.doe import LHS_Settings

import logging

from ..models.doe_models import DoEModelFactory
from ..models.dimensionality_reduction_models import DimensionalityReductionModelFactory
from ..models.regressor_training_models import RegressorTrainingModelFactory
from ..models.hyperparameter_optimization import HPOModelFactory
from ..utils.data_utils import DataUtils
from ..utils import analysis_utils

import pandas as pd

class SurrogateDiscipline(GemseoSurrogate):
    def __init__(
        self,
        discipline: list[MADSDiscipline],
        regressor_name: str,
        num_samples: int = 0,
        inputs: list[BaseVariable] = None,
        T: Any = None,
        outputs: list[BaseVariable] = None,
        normalize: Optional[bool] = False,
        normalize_method: Optional[str] = "MinMaxScaler",
        train: Optional[bool] = True,
        new_doe: Optional[bool] = True,
        save_doe:Optional[bool]=False,
        dimension_reduction: Optional[bool] = True,
        dimension_reduction_space: Optional[Any] = "outputs",
        dimension_reduction_method: Optional[str] = "DMD_pylom",
        doe_method: Optional[str] = "LHS",
        HPO: Optional[bool] = False,
        HPO_dict: Optional[dict]  = None,
        HPO_space: Optional[dict]  = None,
        **extra_args: Any,
    ):
        # Initialize instance variables
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.dimension_reduction = dimension_reduction
        self.dimension_reduction_space = dimension_reduction_space
        self.dimension_reduction_method = dimension_reduction_method
        self.regressor_name = regressor_name
        self.inputs = inputs
        self.outputs = outputs

        # Initialize factories for different functionalities
        self.doe_factory = DoEModelFactory()
        self.DR_factory = DimensionalityReductionModelFactory()
        self.Regressor_training_factory = RegressorTrainingModelFactory()
        self.HPO_factory = HPOModelFactory()
        self.dr_utils = DataUtils()

        # Dictionary for SM gemseo
        self.SM_gemseo = self.Regressor_training_factory.SM_gemseo
        self.SM_gemseo_settings = self.Regressor_training_factory.SM_gemseo_settings

        # Dictionary for SM scikit
        self.SM_scikit = self.Regressor_training_factory.SM_scikit

        # Dictionary for SM pylom
        self.SM_pylom = self.Regressor_training_factory.DR_methods_pylom

        # Dictionary for DR scikit
        self.DR_methods_scikit = self.DR_factory.DR_methods_scikit

        if len(discipline) == 1:
            self.discipline = discipline[0]

        if train == True:
            if len(discipline) == 1:
                # Generate dataset for training dynamic surrogate
                X_inp_s, outputs_doe = self.doe_factory.dataset_for_training_dynamic(
                    new_doe=new_doe,
                    doe_method=doe_method,
                    num_samples=num_samples,
                    discipline=self.discipline,
                    inputs=self.inputs,
                    save_doe=save_doe
                )

                N_snapshots = len(T)

                # Dictionary for DR methods from pylom
                self.DR_methods_pylom = {"DMD_parameterized": pyLOM.DMD, "DMD_pylom": pyLOM.DMD, "POD_pylom": pyLOM.POD, "SPOD_pylom": pyLOM.SPOD}

                if dimension_reduction == False:
                    # Test train split
                    pass
                elif dimension_reduction == True:
                    # Apply dimensionality reduction
                    if dimension_reduction_method in self.DR_methods_pylom.keys():
                        DR = self.DR_methods_pylom[dimension_reduction_method]

                        # Get common arguments for DR method
                        self.common_dict_DR = self.dr_utils.common_arguments_dict(DR.run, extra_args)

                        print(f"\n Values for parameters given: {self.common_dict_DR} \n")

                        if "outputs" in dimension_reduction_space:
                            if dimension_reduction_method == "DMD_parameterized" or dimension_reduction_method == "DMD_pylom":
                                # Apply DMD for dimensionality reduction
                                [self.surr_A, self.surr_B, self.PSI, self.scaler_input_data, self.scaler_A_vectors, self.scaler_inital_conds] = self.DR_factory.dim_red_tranform_dynamic(
                                    DR_method=dimension_reduction_method,
                                    output_data=outputs_doe,
                                    input_data=X_inp_s,
                                    extra_parameters=extra_args,
                                    DR_methods_scikit=self.DR_methods_scikit,
                                    DR_methods_pylom=self.DR_methods_pylom,
                                    num_samples=num_samples,
                                    T=T,
                                    N_snapshots=N_snapshots,
                                    SM=regressor_name,
                                    normalize_method=normalize_method,
                                    normalize=self.normalize)

                            if dimension_reduction_method == "POD_pylom":
                                # Apply POD for dimensionality reduction
                                [self.surr_A, self.surr_B, self.PSI, self.scaler_input_data, self.scaler_A_vectors, self.scaler_inital_conds] = self.DR_factory.dim_red_tranform_dynamic(
                                    DR_method=dimension_reduction_method,
                                    output_data=outputs_doe,
                                    input_data=X_inp_s,
                                    extra_parameters=extra_args,
                                    DR_methods_scikit=self.DR_methods_scikit,
                                    DR_methods_pylom=self.DR_methods_pylom,
                                    num_samples=num_samples,
                                    T=T,
                                    N_snapshots=N_snapshots,
                                    SM=regressor_name,
                                    normalize_method=normalize_method)
            else:
                # Process Design of Experiment for multiple disciplines
                print(f"\n----- Processing Design Of Experiment for {discipline[0].name} discipline... -----\n")

                # Create scenario for DOE training
                mads_scenario = MADSScenario()
                mads_scenario.fill_parameter_space(inputs)
                mads_scenario.create_scenario(
                    disciplines=discipline,
                    formulation=MDF_Settings(main_mda_name="MDAGaussSeidel"),  # old #'DisciplinaryOpt',#'BiLevel', #'MDF',
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
                dataset.add_output_group(data.output_dataset, [out.name for out in outputs], {out.name: len(out.value) for out in outputs})

                # Filter out zeros by replacing them with the average
                data_filter = dataset.get_columns(variable_names=outputs.name)
                non_zero_elements = data_filter[data_filter != 0]
                average = np.mean(data_filter)
                data_filter = np.where(data_filter == 0, average, data_filter)
                dataset.update_data(data_filter, variable_names=outputs.name)

                # Save the dataset to a CSV file
                with open(f"training_dataset_{discipline[0].name}.csv", "w") as f:
                    dataset.to_csv(f, index=False)

                # Initialize the surrogate model with the dataset
                super().__init__(regressor_name, data=dataset, disc_name=discipline[0].name + " [Surrogate]")
        else:
            pass

    def prediction(self,
                input:dict[str, NDArray],
                t=np.array,
                )-> dict[str, NDArray]:
        num_iter=((np.shape(input)[0]))
        output=[]
        
        match self.dimension_reduction_method:
            case "POD_pylom":
                for i in range(num_iter):
                    pred=input[i,:]
                    if (len(np.shape(pred)))==1:
                        t=t.reshape(-1,1)
                        pred=np.tile(pred,(len(t),1))
                        pred=np.array(pred)

                        block=np.hstack((t,pred))
                        
                        if self.normalize==True:
                            pred=self.scaler_input_data.transform(pred)
                            pred=np.array(pred).reshape(1,-1) 
                    elif ((np.shape(pred)))==1:
                        pred=np.array(pred).reshape(-1,1) 
                        if self.normalize==True:
                            pred=self.scaler_input_data.transform(pred)
                            pred=np.array(pred).reshape(-1,1)   

                        
                    A_pred=(self.surr_A.predict(block))
   
                    
                    recon=pyLOM.POD.reconstruct(self.PSI,self.surr_B,A_pred.T)
                        
                    return recon
                    
                    
            case "DMD_pylom":
                pass         
            case "SPOD_pylom":
                pass
            case "DMD_parameterized":
                for i in range(num_iter):
            
                    pred=input[i,:]
            
                    if (len(np.shape(pred)))==1:
                        pred=np.array(pred).reshape(1,-1) 
                        if self.normalize==True:
                            pred=self.scaler_input_data.transform(pred)
                            pred=np.array(pred).reshape(1,-1) 

                    elif ((np.shape(pred)))==1:
                        pred=np.array(pred).reshape(-1,1) 
                        if self.normalize==True:
                            pred=self.scaler_input_data.transform(pred)
                            pred=np.array(pred).reshape(-1,1)   
                    #else:
                    #    pred=np.array(pred)
                    A_pred=(self.surr_A.predict(pred))

                    B_pred=(self.surr_B.predict(pred)).flatten()
                    
                    if self.normalize==True:
                        A_pred =self.scaler_A_vectors.inverse_transform(A_pred)
                        B_pred=B_pred.reshape(1, -1)
                        B_pred =self.scaler_inital_conds.inverse_transform(B_pred)
                        B_pred=B_pred.flatten()
         
            
                    r=np.shape(self.PSI)[1]
                    A_pred=A_pred.reshape(r,r)             
                    C_hist=[B_pred]
                    current_state=B_pred
                
                    for i in range(len(t)-1):
                        next_state=A_pred@current_state
                    
                        C_hist.append(next_state)
                        current_state=next_state
                
                    C_new_predicted=np.array(C_hist).T
                
                    X_final=self.PSI@C_new_predicted
                    output.append(X_final)
                output=np.vstack(output)
                return output
 
