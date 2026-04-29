"""Mapping to CPACS elements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from multiads.design_space.cpacs_variable_set import CPACSVariableSet

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET
    from collections.abc import Mapping


class CPACSMapping:
    """A structure that maps variable names to its CPACS content."""

    def __init__(self) -> None:
        # associate either variable name or group name to a variable set
        self.__variable_sets = {}
        # associates the sub-var name to the group of variable it belongs to
        self.__single_var_to_group_bijection = {}

    def add_xml_element(
        self,
        var_name: str,
        element: ET,
        group_name: str | None = None,
    ) -> None:
        """Add a xml element.

        Args:
            var_name: The name of the variable.
            element: The xml element.
            group_name: The name of the group of variables.
                If ``None``,
                the variable will not be associated to any group.

        """
        if group_name is None:
            variable_set = CPACSVariableSet()
            variable_set.add_element(element, var_name)
            self.__variable_sets.update({var_name: variable_set})
        else:
            if group_name not in self.__variable_sets:
                self.__variable_sets.update({group_name: CPACSVariableSet()})
            self.__variable_sets[group_name].add_element(element, var_name)
            self.__single_var_to_group_bijection.update({var_name: group_name})

    def get_variable_index_in_group(self, group: str, variable: str) -> int:
        """Get a variable index into a group.

        Args:
            group: The name of the group.
            variable: The name of the variable.

        Return:
            The index of the variable into the group.

        """
        return self.__variable_sets[group].get_index(variable)

    def get_variable_set(self, name_set: str) -> CPACSVariableSet:
        """Get a variable set.

        Args:
            name_set: The name of variable set.

        """
        return self.__variable_sets[name_set]

    def add_variable_set(self, name_set: str, variable_set: CPACSVariableSet) -> None:
        """Add a new set of variables.

        Args:
            name_set: The name of the set of variables.
            variable_set: The variable set.

        """
        self.__variable_sets.update({name_set: variable_set})

    def __getitem__(self, name: str) -> np.ndarray:
        """Get the values of either a single variable or a set.

        In case of a set, all values of sub-variables are concatenated.

        Args:
            name: The name of the variable or the variable set.

        Returns:
            The values of the variable.

        """
        if name in self.__variable_sets:
            return self.__variable_sets[name].get_values()

        if name not in self.__single_var_to_group_bijection:
            msg = f"Required variable {name} is neither a variable nor a set."
            raise KeyError(msg)

        main_var = self.__single_var_to_group_bijection[name]
        return self.__variable_sets[main_var].get_values([name])

    def __setitem__(self, name: str, value: float | np.ndarray) -> None:
        """Set a value to a variable.

        Args:
            name: The name of the variable.
            value: The value.

        """
        if isinstance(value, (float, int)):
            value = np.array([value])

        if self.get_variable_size(name) != len(value):
            msg = (
                f"Variable {name}, prescribed "
                f"value has the wrong size: "
                f"{len(value)} instead of {self.get_variable_size(name)}."
            )
            raise ValueError(msg)

        if (
            name not in self.__variable_sets
            and name not in self.__single_var_to_group_bijection
        ):
            msg = f"Variable {name} is neither an existing variable nor a group name"
            raise NameError(msg)

        if name in self.__variable_sets:
            if self.__is_group(name):
                self.__variable_sets[name].set_values(value)
            else:
                self.__variable_sets[name].set_values(value, name)
        else:
            group = self.__single_var_to_group_bijection[name]
            self.__variable_sets[group].set_values(value, name)

    def get_variable_size(self, name: str) -> int:
        """Get the size of either a single variable or a set.

        Args:
            name: The name of the variable or the set.

        Returns:
            The size.

        """
        return len(self.__getitem__(name))

    def dump_as_dict(self) -> dict[str, np.ndarray]:
        """Get a dictionary of variables that maps to array of values."""
        return {name: self[name] for name in self.__variable_sets}

    def keys(self) -> list[str]:
        """Get mapping keys."""
        return list(self.__variable_sets.keys())

    def __is_group(self, name: str) -> bool:
        """Whether a name corresponds to a group.

        Args:
            name: The name.

        Returns:
            True if it is a group,
            False otherwise.

        """
        return bool(
            name in self.__variable_sets and len(self.__variable_sets[name]) > 1,
        )

    def split_set_into_single_variable(
        self,
        data: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Split all values of a set into a single variable.

        The function operates over all couples key/value found into data.
        If a couple key/value corresponds to a single variable,
        the new data does not change.
        However, if a couple key/value corresponds to a set of variables,
        the new returned data is split into several couples key/value for all
        single variables involved into the set.

        Args:
            data: The data associating variables or groups to values.

        Returns:
            Split data including only single variables.

        """
        new_data = {}
        for name, values in data.items():
            if self.__is_group(name):
                new_data.update(self.__variable_sets[name].split_values(values))
            else:
                new_data.update({name: values})

        return new_data

    def resize(
        self,
        name: str,
        size: int | None = None,
        values: float | np.ndarray | None = None,
    ) -> None:
        """Resize a variable (not group).

        This function allows to resize the number of values into a CPACS variable.

        Args:
            name: The name of the variable.
            size: The size of the variable.
                If ``None``,
                values must have a length and this latter one is used as size.
            values: The new values (all values).
                If ``None``,
                the default value 0 is used with the size prescribed.

        """
        if self.__is_group(name):
            msg = "Only single variables can be resized, not group."
            raise ValueError(msg)

        if name in self.__variable_sets:
            self.__variable_sets[name].resize(name, size, values)
        else:
            name_group = self.__single_var_to_group_bijection[name]
            self.__variable_sets[name_group].resize(name, size, values)
