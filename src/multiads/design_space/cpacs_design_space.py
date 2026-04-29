"""Design Space based on CPACS definition."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from gemseo.algos.design_space import DesignSpace

if TYPE_CHECKING:
    from multiads.design_space.cpacs_structure_data import CPACSStructureData


class CPACSDesignSpace(DesignSpace):
    """Design space based on CPACS data."""

    def __init__(
        self,
        cpacs_structure_data: CPACSStructureData | None = None,
    ) -> None:
        """
        Args:
            cpacs_structure_data: The CPACS structure data.

        """  # noqa: D205, D212
        super().__init__()
        self._cpacs_structure = cpacs_structure_data
        self.__set_design_variable()

    @property
    def cpacs_data(self) -> CPACSStructureData:
        """Return the CPACS structure data."""
        return self._cpacs_structure

    @cpacs_data.setter
    def cpacs_data(self, data: CPACSStructureData) -> None:
        """Set the CPACS structure data."""
        self._cpacs_structure = data

    def __set_design_variable(self):  # type: (...) -> None
        """Set variable to the design space from the CPACS structure data."""
        if not self._cpacs_structure:
            return

        for variable, value in self._cpacs_structure:
            size = self._cpacs_structure.get_variable_size(variable)
            bounds = self._cpacs_structure.get_bounds(variable)
            self.add_variable(
                variable,
                size=size,
                value=value,
                lb=bounds[0],
                ub=bounds[1],
            )

    def _build_for_deepcopy(self) -> CPACSDesignSpace:
        """Build an instance when doing a deepcopy."""
        return type(self)(self._cpacs_structure)

    def __deepcopy__(self, memo=None) -> CPACSDesignSpace:
        """This function is called when doing deepcopy.

        This function is overridden because
        we don't want _cpacs_structure attribute to be deepcopied
        when doing deepcopy of the whole object.
        Indeed, _cpacs_structure is a reference toward the inputs structure
        so we want to keep that reference when executing :func:`set_cpacs_value`.
        This means that if a design space is deepcopied,
        executing :func:`set_cpacs_value` on each other will override
        the data.
        Design space deepcopy should be used in order to filter design variables
        such as two spaces should involve different variables.

        Returns:
            A new design space instance.

        """
        if memo is None:
            memo = {}
        new_obj = self._build_for_deepcopy()

        for attr_name, attr_value in self.__dict__.items():
            if attr_name != "_cpacs_structure":
                new_obj.__dict__[attr_name] = deepcopy(attr_value)

        return new_obj

    def set_cpacs_values(self) -> None:
        """Set the current design space values into the CPACS structure."""
        # First set values of design variables
        for var in self.variable_names:
            if var in self._cpacs_structure.design_variables:
                self._cpacs_structure.set_value(
                    var,
                    value=self.get_current_value([var]),
                )

        # Then set dependent variables (processed variables)
        all_values = self.get_current_value(as_dict=True)
        self._cpacs_structure.set_processed_value_auto(**all_values)

    def set_new_values_from_cpacs(self, other_cpacs: CPACSStructureData) -> None:
        """Set new values to the design space from another CPACS structure."""
        for var in self.variables_names:
            self.set_current_variable(var, other_cpacs.get_value(var))
        self.set_cpacs_values()
