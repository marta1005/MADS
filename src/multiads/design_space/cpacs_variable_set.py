"""Variable set definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET
    from collections.abc import Iterable


class CPACSVariableSet:
    """Define a set of CPACS variable."""

    def __init__(self) -> None:
        # associate variable to xml element
        self.__elements = {}

    def add_element(self, element: ET, name: str) -> None:
        """Add xml element to the set.

        Args:
            element: An xml element.
            name: The name of the variable.

        """
        self.__elements.update({name: element})

    def get_index(self, variable: str) -> int:
        """Get the index of the variable in the set."""
        return list(self.__elements.keys()).index(variable)

    def get_values(self, variables: Iterable[str] | None = None) -> np.ndarray:
        """Get values.

        Args:
            variables: The variables whose values must be returned.
                If variables is ``None``,
                the values of all variables in the set are returned.

        Returns:
            The values of variables concatenated into a single array.

        """
        result = np.array([])
        if variables is not None:
            for var in variables:
                result = np.hstack(
                    [
                        result,
                        self.__get_xml_element_value(self.__elements[var]),
                    ],
                )
        else:
            for elm in self.__elements.values():
                result = np.hstack([result, self.__get_xml_element_value(elm)])
        return result

    @staticmethod
    def __get_xml_element_value(element: ET) -> np.ndarray:
        """Get the values of an xml element.

        Args:
            element: The xml element.

        Returns:
            The values of the element.

        """
        return np.array([float(x) for x in element.text.split(";")])

    def __len__(self) -> int:
        """Get the length of the variable set.

        Returns:
            The number of elements.

        """
        return len(self.__elements)

    def set_values(self, values: np.ndarray, name_var: str | None = None) -> None:
        """Set new values to an xml element.

        Args:
            values: The values.
            name_var: The name of the variable.
                If ``None``,
                the values must correspond to all variables and are split.

        """
        if name_var is None:
            split_data = self.split_values(values)
            for name_var, value in split_data.items():
                self.__elements[name_var].text = self.__build_text_value(value)
        else:
            self.__elements[name_var].text = self.__build_text_value(values)

    def split_values(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Split values with respect to the variables with their size in the set.

        Args:
            values: The values to be split.

        Return:
            The split values associated to the variables

        """
        beg = 0
        end = 0
        split_data = {}
        for name_var, elt in self.__elements.items():
            end = beg + len(self.__get_xml_element_value(elt))
            split_data.update({name_var: values[beg:end]})
            beg = end
        return split_data

    @staticmethod
    def __build_text_value(
        values: np.ndarray,
    ) -> str:
        """Build the content text value for the CPACS file.

        Args:
            values: The values

        Returns:
            The content for the CPACS tag.

        """
        return ";".join([str(x) for x in values])

    def resize(
        self,
        name: str,
        size: int | None = None,
        values: float | np.ndarray | None = None,
    ) -> None:
        """Resize a variable.

        Args:
            name: The name of the variable.
            size: The size of the variable.
                If ``None``,
                values must have a length and this latter one is used as size.
            values: The new values (all values).
                If ``None``, the default value 0 is used with
                the size prescribed.

        """
        if size is None and values is None:
            msg = "Either 'size' or 'value' must be prescribed."
            raise ValueError(msg)

        if values is None:
            values = np.zeros(size)
        elif isinstance(values, float):
            values = np.full(size, values)

        self.set_values(values, name)
