#
# library to update databases in GEMSEO

from numpy import array
import numpy as np

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo import read_design_space
from gemseo import write_design_space
import copy
from gemseo import import_database


class mads_post_database:
	"""
	class to postpone the database FROM mads
	"""

	# database.last_item
	# database.n_iterations
	# database.get_x_vect_history()

	def __init__(self, database) -> None:
		"""_summary_

		Args:
			database (_type_): _description_
		"""
		self.database

	def get_history(self, var=""):
		"""recover optimisation history from the database

		Args:
			var (str, optional): _description_. Defaults to ''.

		Returns:
			_type_: _description_
		"""
		outputs, inputs = self.database.get_history()
		jacobian = self.database.get_gradient_history(var)
		# return the history of the optimisation
		return outputs, inputs, jacobian

	def write_history_to_file(self, history_file):
		"""_summary_"""
		f = open(history_file, "w")
		for i, (input, output) in enumerate(self.database.items()):
			f.write(f"iter {i} --> input = {input} / output = {output}")
		f.close()

	def get_iteration(self, x_vect=()):
		"""_summary_

		Args:
			x_vect (tuple, optional): _description_. Defaults to ().

		Returns:
			_type_: _description_
		"""
		# e.g. init_point = array([5., 0.])
		iter = self.database.get_iteration(x_vect)
		return iter

	def export_to_hdf(self, file_name=""):
		"""_summary_

		Args:
			file_name (str, optional): _description_. Defaults to "".
		"""
		self.database.to_hdf(file_name)

	def write_to_cpacs(self, input_file, output_file, variables):
		"""_summary_"""
		cpacs = CPACSStructureData(input_file)
		for name_var, ov in variables.items():
			if "CPACS" in ov.options:
				cpacs = CPACSStructureData(
					ov.options.get("CPACS", {}).get(input_file, {})
				)
				cpacs.set_value(
					ov.options.get("CPACS", {}).get("cpacs_marker", {}), value=ov.value
				)
		# write to new file
		cpacs.write_xml(output_file)


class MadsExportHistory:
	
	def __init__(
		self,
		mads_scenario,
		constraints,
		var_name_list,
		out_file,
		scenario_type="DOE",  # "MDO" , "DOE"
		) -> None:
		"""_summary_

		Args:
			mads_scenario (_type_): scenario from MADS
		"""
		# initialise Variables and setup modus
		self.mads_scenario = mads_scenario
		self.var_name_list = var_name_list
		self.out_file = out_file
		self.constraints = constraints
		self.scenario_type = scenario_type

	def export_feasible_design(self):

		# recover database
		# database = self.mads_scenario.scenario.formulation.optimization_problem
		database = import_database(self.out_file)

		# recover design space
		design_space = self.mads_scenario.parameter_space.to_design_space()

		# save optimisation
		self.mads_scenario.scenario.save_optimization_history(self.out_file)

		# define desing spcae to be psotrpocessed
		to_be_postprocess = OptimizationProblem(design_space)

		# load optimisation problem
		opt_problem = OptimizationProblem.from_hdf(self.out_file)

		# write desing space
		write_design_space(design_space, "design_space.csv")

		# OptimizationHistory(constraints, database, design_space)[source]
		from gemseo.algos.optimization_history import OptimizationHistory

		opt_history = OptimizationHistory(self.constraints, database, design_space)

		# get history
		points = opt_history.get_data_by_names(
			self.var_name_list, as_dict=True, filter_non_feasible=False
		)
  
		# retrieve feasible points
		# feasible_var, feas_objective = opt_history.feasible_points()
		feasible_point = opt_history.feasible_points
		feas_objective = feasible_point[0]
		feasible_var = feasible_point[1]
		
		# debug - print feasible
		# print(dir(opt_history))
		# print(feasible_var)
		
		# initialize container
		data_array = np.zeros((int(len(feas_objective)),int(len(self.var_name_list))))
		# pull feasbile variables
		for nn in range(0, len(feas_objective)):
			for v, iv in enumerate(self.var_name_list):
				data_array[:,v] = feasible_var[nn].get(iv)

		# Convert the list to a numpy array
	   # print(data_to_save)
	   # data_array = np.array(data_to_save)

		# print feasible design
		with open("feasible_var_and_objective.txt", "a") as txt_file:
			np.savetxt(txt_file, data_array, newline="\n")

		if self.scenario_type == "MDO":
			opt_result = self.mads_scenario.scenario.get_optimum()
			# print optimum
			with open("optimum.txt", "a") as myfile:
				myfile.write(str(opt_result.f_opt).replace(".", ",") + ";")
				for i in range(0, len(opt_result.x_opt)):
					myfile.write(str(opt_result.x_opt[i]).replace(".", ",") + ";")
