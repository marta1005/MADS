from typing import Any

from scenario import Variable
from solvers import SolverFactory


@SolverFactory.register("weight_and_balance", "mass_fuselage_uav_gundlach")
class mass_fuselage_uav_gundlach():

    def __init__(self) -> None:
#        # parameter dictionary
#        self.parameters = None

        # components

        # state of the variables to be observed
        # self.observable_space = {}

        #inputs
        self.F_mg = None
        self.F_ng = None
        self.F_press = None
        self.F_vt = None
        self.F_mat = None
        self.length = None
        self.W_carried = None
        self.loadFactorMax = None
        self.Veq_max = None

        # outputs
        self.output = None

    def set_options(self, parameters) -> None:
        self.F_mg = parameters["F_mg"]
        self.F_ng = parameters["F_ng"]
        self.F_press = parameters["F_press"]
        self.F_vt = parameters["F_vt"]
        self.F_mat = parameters["F_mat"]
        self.length = parameters["length"]
        self.W_carried = parameters["W_carried"]
        self.loadFactorMax = parameters["loadFactorMax"]
        self.Veq_max = parameters["Veq_max"]

    def run(self) -> None:

        # calculation
        m_fuselage = calculateMassFuselageUAVGundlach(self.F_mg, self.F_ng, self.F_press, self.F_vt, self.F_mat,
                                                      self.length, self.W_carried, self.loadFactorMax, self.Veq_max)

        self.output = {"m_fuselage": m_fuselage}

    def compute_output(
        self,
        requested_outputs: list[str],
        mapped_outputs: list[Variable],
    ) -> dict[str, Any]:
        """_summary_: post-process data if needed and collect results"""
        # outputs = {}
        # for out in requested_outputs:
        #     if output_var := next((o for o in mapped_outputs if o.name == out), False):
        #         # Callback options
        #         out_options = output_var.options.get("dust", {})
        #         out_type = output_var.output_type

        #         # Callback
        #         try:
        #             out_function = DUST.implemented_outputs[out_type]
        #             outputs[out] = out_function(out, self.driver, **out_options)
        #         except KeyError:
        #             raise ValueError(
        #                 f"'{type(self).__name__}' cannot compute '{out_type}"
        #             )

        # return outputs
        return self.output
