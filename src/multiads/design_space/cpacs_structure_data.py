"""CPACS structure data."""

from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy import inf

from multiads.design_space.cpacs_mapping import CPACSMapping
from multiads.design_space.cpacs_path_template import CPACSPathTemplate

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class CPACSStructureData:
    """The main class to get CPACS data from a xml file."""

    _PROFILE_CST_LOWER = CPACSPathTemplate(
        "./vehicles/profiles/wingAirfoils/"
        "wingAirfoil[@uID='{airfoil_ID}']/cst2D/lowerB",
        ["airfoil_ID"],
    )

    _PROFILE_CST_UPPER = CPACSPathTemplate(
        "./vehicles/profiles/wingAirfoils/"
        "wingAirfoil[@uID='{airfoil_ID}']/cst2D/upperB",
        ["airfoil_ID"],
    )

    _SWEEP_ANGLE = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "positionings/positioning[@uID='{position_ID}']/sweepAngle",
        ["wing_ID", "position_ID"],
    )

    _TWIST_ANGLE = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "sections/section[@uID='{section_ID}']/transformation/rotation/y",
        ["wing_ID", "section_ID"],
    )

    _WING_SEG = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/segments/segment",
        ["wing_ID"],
    )

    def __init__(
        self,
        file_name: str,
        path: Path | None = None,
        load_tigl: bool = False,
    ) -> None:
        """
        Args:
            file_name: The name of the CPACS file.
            path: The path where the CPACS file is located.
                If ``None``,
                the path is the current directory.
            load_tigl: Whether Tigl is loaded.

        """  # noqa: D205, D212
        self.__cpacs_mapping = CPACSMapping()
        self.__design_variables = []
        self.__bounds = {}
        self.__processed_variables = {}
        self.__processed_args = {}
        self.__path = path or Path()
        self.__load_tigl = load_tigl

        file_name = self.__path / file_name
        self.__tree = ET.parse(file_name.as_posix())

        # if self.__load_tigl:
        #     self.__tigl = CPACSTigl(str(file_name))

    @property
    def path(self) -> Path:
        """Get the path to the xml file."""
        return self.__path

    @property
    def variables(self) -> Iterable[str]:
        """Get all variable names selected into the CPACS."""
        return self.__cpacs_mapping.keys()

    @property
    def design_variables(self) -> list[str]:
        """Get the design variable names only."""
        return self.__design_variables

    @property
    def processed_variables(self) -> Iterable[str]:
        """Get the processed variable names only."""
        return self.__processed_variables.keys()

    # @property
    # def tigl(self):  # type: (...) -> CPACSTigl
    #     """Return the TiGL wrapper."""
    #     return self.__tigl

    def get_process_args_values(
        self,
        name: str,
        **extra_args_values: Any,
    ) -> dict[str, float]:
        """Get the values of the arguments of a process function.

        Args:
            name: The name of the processed variable.
            extra_args_values: Any extra argument values which are needed
               to the processed variable,
               but which does not belong to the CPACS data.

        Returns:
            The values of the arguments of the process function.

        """
        args = self.get_process_arguments(name)
        if name in self.__processed_args:
            pseudo = self.__processed_args[name]
            map_names = dict(zip(args, pseudo))
            try:
                values_args = {arg: self.get_value(map_names[arg]) for arg in args}
            except KeyError as e:
                msg = f"Arguments {e.args} not found for processed variable {name}."
                raise KeyError(msg) from e
        else:
            values_args = {}
            for arg in args:
                try:
                    value = self.get_value(arg)
                except KeyError:
                    value = extra_args_values[arg]
                values_args[arg] = value
        return values_args

    def get_process_arguments(self, name: str) -> list[str]:
        """Get the arguments of the process function related to a variable.

        Args:
            name: The name of the processed variable.

        Returns:
            The arguments of the process function.

        """
        return list(
            inspect.signature(self.__processed_variables[name]).parameters.keys(),
        )

    def select_variable_from_xpath(
        self,
        xpath: str,
        name: str,
        is_design_variable: bool = False,
        lower_bound: float | np.ndarray | None = -inf,
        upper_bound: float | np.ndarray | None = inf,
        process: Callable | None = None,
        group_name: str | None = None,
        process_args_aliases: list[str] | None = None,
    ) -> None:
        """Select a variable into the xml tree from a XPath.

        Args:
            xpath: The xml XPath of the selected variable.
            name: The name of the variable.
            is_design_variable: Whether the variable is a design variable or not.
                If it is a design variable,
                then the variable can be used into a design space.
            lower_bound: The lower bounds of values.
                Only for design variables.
            upper_bound: The upper bounds of values.
                Only for design variables.
            process: The process that is performed when a processed variable is set with
                new values.
                A processed variable is a variable that depends on other
                variables defined into the CPACS structure.
                In such case,
                setting new values imply computing the process from values of
                these latter primary variables.
                If a process is prescribed, the variable is automatically
                a processed variable.
                A process variable cannot be a designed variable.
            group_name: The name of a group.
                Groups (or sets) enable to concatenate variables as vector.
                A group can be manipulated as a single variable.
                In such case,
                the group will be the concatenation of all underlying single variables.
                A group can only involve design variables and not processed variables.
            process_args_aliases: The name of the variables that should be
                used instead of arguments for the processed function.
                These names will be used to retrieve the values of the arguments
                during the auto-set procedure.

        """
        element = self.__get_xml_element(xpath)
        self.__cpacs_mapping.add_xml_element(name, element, group_name)

        self.__set_variable_properties(
            name,
            is_design_variable,
            lower_bound,
            upper_bound,
            process,
            group_name=group_name,
            process_args_aliases=process_args_aliases,
        )

    def __set_variable_properties(
        self,
        name: str,
        is_design_variable: bool = False,
        lower_bound: float | np.ndarray | None = None,
        upper_bound: float | np.ndarray | None = None,
        process: Callable | None = None,
        group_name: str | None = None,
        process_args_aliases: list[str] | None = None,
    ) -> None:
        """Set properties of variables.

        Args:
            name: The name of the variable.
            is_design_variable: Whether the variable is a design variable.
                See func::select_variable_from_xpath.
            lower_bound: The lower bound value.
                Only for design variables.
            upper_bound: The upper bound value.
                Only for design variables.
            process: The process executed when the variable is set with new values.
                See func::select_variable_from_xpath.
            process_args_aliases: The name of variables used to set values
                to the arguments of the processed function.

        """
        if process:
            if is_design_variable:
                msg = (
                    "CPACS variable cannot be a design variable"
                    "and a processed variable in the same time."
                )
                raise ValueError(msg)
            if group_name is not None:
                msg = "CPACS process variable cannot be associated to a group name."
                raise ValueError(msg)

            self.__processed_variables.update({name: process})
            if process_args_aliases:
                self.__processed_args.update({name: process_args_aliases})

        if is_design_variable:
            if group_name is None:
                self.__add_variable(name)
                lower_bound, upper_bound = self.__adapt_bounds(
                    name,
                    lower_bound,
                    upper_bound,
                )
                self.__bounds[name] = [lower_bound, upper_bound]
            else:
                self.__add_variable(group_name)
                lower_bound, upper_bound = self.__adapt_bounds(
                    name,
                    lower_bound,
                    upper_bound,
                )
                self.__extend_bounds(group_name, lower_bound, upper_bound)

    def __add_variable(self, variable: str):
        """Add a new variable."""
        if variable not in self.__design_variables:
            self.__design_variables.append(variable)

    def __extend_bounds(
        self,
        name: str,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        """Extend and concatenate bounds in case of set of variables.

        Args:
            name: The name of variable.
            lower_bounds: Lower bound values.
            upper_bounds: Upper bound values.

        """
        if name in self.__bounds:
            self.__bounds[name][0] = np.hstack([self.__bounds[name][0], lower_bounds])
            self.__bounds[name][1] = np.hstack([self.__bounds[name][1], upper_bounds])
        else:
            self.__bounds[name] = [lower_bounds, upper_bounds]

    def __adapt_bounds(
        self,
        name: str,
        lower_bounds: float | np.ndarray | None = None,
        upper_bounds: float | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adapt the format of bounds depending on the size and the type.

        Args:
            name: The name of the variable.
            lower_bounds: The lower bound values.
            upper_bounds: The upper bound values.

        Returns:
            Adapted values of lower bounds and upper bounds

        """
        size = self.__cpacs_mapping.get_variable_size(name)

        if isinstance(lower_bounds, (float, int)):
            if size > 1:
                lower_bounds = np.full(size, lower_bounds)
            else:
                lower_bounds = np.array([lower_bounds])

        if isinstance(upper_bounds, (float, int)):
            if size > 1:
                upper_bounds = np.full(size, upper_bounds)
            else:
                upper_bounds = np.array([upper_bounds])

        if lower_bounds is None:
            lower_bounds = np.array([None] * size)
        if upper_bounds is None:
            upper_bounds = np.array([None] * size)

        return lower_bounds, upper_bounds

    def __get_xml_element(self, xpath: str) -> ET:
        """Get the xml element corresponding XPath.

        Args:
            xpath: An XPath

        Returns:
            An element of the xml tree.

        """
        elements = self.__xml_find_xpath(xpath)
        # check that xpath contains exactly 1 element
        if len(elements) == 0:
            msg = f"None element found corresponding to the current XPath {xpath}."
            raise ValueError(msg)
        if len(elements) > 1:
            msg = "Current XPath has more than one element."
            raise ValueError(msg)
        return elements[0]

    def __get_nb_elements(self, xpath: str) -> int:
        """Get the number of elements corresponding to an XPath.

        Args:
            xpath: A XPath.

        Returns:
            The number of elements

        """
        return len(self.__xml_find_xpath(xpath))

    def __xml_find_xpath(self, xpath: str) -> list[ET]:
        """Find XPath in xml tree.

        Args:
            xpath: An XPath.

        Returns:
            All xml elements found.

        """
        return self.__tree.findall(xpath)

    def write_xml(self, name_file: str, auto_process: bool = True) -> None:
        """Write an .xml file from data.

        Args:
            name_file: The file name.
            auto_process: Whether the processes values are automatically
                set before writing.

        """
        if auto_process:
            self.set_processed_value_auto()
        self.__tree.write(name_file)

    def __str__(self) -> str:
        """Convert object to string.

        Returns:
            The variable as a string.

        """
        return f"CPACS VARIABLES =\n{self.__cpacs_mapping!s}\n"

    def get_value(self, name: str) -> np.ndarray:
        """Return the value corresponding to the required input variable.

        Args:
            name: The name of variable.

        Returns:
            The value.

        """
        return self.__cpacs_mapping[name]

    def set_value(self, name: str, **kwargs: float | np.ndarray) -> None:
        """Set new values to the variable.

        Args:
            name: The name of the variable.
            kwargs: The values of arguments.
                If the variable is a processed variable,
                kwargs must include all keyword arguments needed in the
                processed function.
                Otherwise, only keyword argument ``value`` should be provided.

        """
        process = self.__processed_variables.get(name)
        if process:
            self.__cpacs_mapping[name] = process(**kwargs)
        else:
            self.__cpacs_mapping[name] = kwargs["value"]

    def set_processed_value_auto(
        self,
        **extra_values: Mapping[str, np.ndarray],
    ) -> None:
        """Automatically set processed values from current parameters value.

        Args:
            extra_values: Any extra values which do not belong to the CPACS data,
                and which is needed to process the variables.

        """
        for name in self.__processed_variables:
            values_args = self.get_process_args_values(name, **extra_values)
            self.set_value(name, **values_args)

        # if self.__load_tigl:
        #     tmp_xml = self.__path / Path("tmp.xml")
        #     self.write_xml(str(tmp_xml), auto_process=False)
        #     self.__tigl = CPACSTigl(str(tmp_xml))
        #     tmp_xml.unlink()

    def get_xml_element(self, name: str) -> ET:
        """Get the xml element of the variable.

        Args:
            name: The name of the variable.

        Returns:
            The xml element tree.

        """
        return self.__cpacs_mapping.get_xml_element(name)

    def get_variable_size(self, name: str) -> int:
        """Get the size of the variable.

        Args:
            name: The nme of the variable.

        Returns:
            The size of the variable.

        """
        return self.__cpacs_mapping.get_variable_size(name)

    def get_bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounds of the design variable.

        Args:
            name: The name of the design variable.

        Returns:
            The values of bounds.

        """
        return self.__bounds[name][0], self.__bounds[name][1]

    def get_sub_mapping(self, variables: Iterable[str]) -> CPACSMapping:
        """Get a sub-mapping.

        This function enables to extract a subset of CPACS variable.

        Args:
            variables: The name of variables of the subset.

        Returns:
            A new mapping involving only the selected subset.

        """
        sub_mapping = CPACSMapping()
        for variable in variables:
            var_set = self.__cpacs_mapping.get_variable_set(variable)
            sub_mapping.add_variable_set(variable, var_set)
        return sub_mapping

    def __iter__(self):
        """Iterate over design variables.

        This function enables to iterate over the set
        of design variables (only, not all selected variables),
        returning its name and value.
        Design variables are variables selected with ``is_design_variable=True``.

        Examples:
            >>> inputs = CPACSStructureData(file_name)
            >>> # select any variables...
            >>> for var, val in inputs:
            >>>    print(var, val)

        """
        for var in self.__design_variables:
            yield var, self.get_value(var)

    def split_set_into_single_variable(
        self,
        data: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Split group of variables into single variable.

        This function takes any group of variables or single variable
        and return the same values but split into single variables only,
        i.e. all group values are split into single variables.

        Args:
            data: The input data

        Returns:
            New split values.

        """
        return self.__cpacs_mapping.split_set_into_single_variable(data)

    def resize(
        self,
        name: str,
        size: int | None = None,
        values: float | np.ndarray | None = None,
    ) -> None:
        """Resize a variable (not group).

        Args:
            name: The name of variable.
            size: The size of the variable.
                If ``None``,
                values must have a length and this latter one is used as size.
            values: The new values (all values).
                If ``None``,
                the default value 0 is used with the size prescribed.

        """
        self.__cpacs_mapping.resize(name, size, values)

    def offset(self, name: str, value: float) -> None:
        """Offset a variables, value and bounds.

        Args:
            name: The name of the variable.
            value: The offset value.

        """
        current_value = self.get_value(name)
        self.set_value(name, value=current_value + value)
        self.__bounds[name][0] += value
        self.__bounds[name][1] += value

    def get_number_of_wing_segments(self, wing_id: str) -> int:
        """Get the number of segment of a wing.

        Args:
            wing_id: The uID of the wing.

        Returns:
            The number of segment in the wing.

        """
        xpath = self._WING_SEG.get_xpath([wing_id])
        return self.__get_nb_elements(xpath)

    def select_twist_angle(
        self,
        name: str,
        wing_id: str,
        section_id: str,
        is_design_variable: bool = False,
        lower_bound: float | np.ndarray | None = None,
        upper_bound: float | np.ndarray | None = None,
        process: Callable | None = None,
        group_name: str | None = None,
        process_args_aliases: list[str] | None = None,
    ) -> None:
        """Select the twist angle.

        Args:
            name: The name of the variable.
            wing_id: The wing ID.
            section_id: The section ID.
            is_design_variable: See func::select_variable_from_xpath.
            lower_bound: See func::select_variable_from_xpath.
            upper_bound: See func::select_variable_from_xpath.
            group_name: See func::select_variable_from_xpath.
            process: See func::select_variable_from_xpath.
            process_args_aliases: See func::select_variable_from_xpath.

        """
        xpath = self._TWIST_ANGLE.get_xpath([wing_id, section_id])
        self.select_variable_from_xpath(
            xpath,
            name,
            is_design_variable,
            lower_bound,
            upper_bound,
            group_name=group_name,
            process=process,
            process_args_aliases=process_args_aliases,
        )

    def select_lower_profile_variable(
        self,
        name: str,
        airfoil_id: str,
        is_design_variable: bool = False,
        lower_bound: float | np.ndarray | None = None,
        upper_bound: float | np.ndarray | None = None,
        group_name: str | None = None,
    ) -> None:
        """Add a variable that corresponds to lower profile airfoil.

        Args:
            name: The name of variable.
            airfoil_id: The name of airfoil ID (@uID xml attribute).
            is_design_variable: See func::select_variable_from_xpath.
            lower_bound: See func::select_variable_from_xpath.
            upper_bound: See func::select_variable_from_xpath.
            group_name: See func::select_variable_from_xpath.

        """
        xpath = self._PROFILE_CST_LOWER.get_xpath([airfoil_id])
        self.select_variable_from_xpath(
            xpath,
            name,
            is_design_variable,
            lower_bound,
            upper_bound,
            group_name=group_name,
        )

    def select_upper_profile_variable(
        self,
        name: str,
        airfoil_id: str,
        is_design_variable: bool = False,
        lower_bound: float | np.ndarray | None = None,
        upper_bound: float | np.ndarray | None = None,
        group_name: str | None = None,
    ) -> None:
        """Add a variable that corresponds to upper profile airfoil.

        Args:
            name: The name of variable.
            airfoil_id: The name of airfoil ID (@uID xml attribute)
            is_design_variable: See func::select_variable_from_xpath.
            lower_bound: See func::select_variable_from_xpath.
            upper_bound: See func::select_variable_from_xpath.
            group_name: See func::select_variable_from_xpath.

        """
        xpath = self._PROFILE_CST_UPPER.get_xpath([airfoil_id])
        self.select_variable_from_xpath(
            xpath,
            name,
            is_design_variable,
            lower_bound,
            upper_bound,
            group_name=group_name,
        )

    def select_sweep_angle_variable(
        self,
        name: str,
        wing_id: str,
        position_id: str,
        is_design_variable: bool = False,
        lower_bound: float | np.ndarray | None = None,
        upper_bound: float | np.ndarray | None = None,
        process: Callable | None = None,
        group_name: str | None = None,
    ) -> None:
        """Add variable that corresponds to wing section sweep angle.

        Args:
            name: The name of variable.
            wing_id: The name of a wing ID (@uID xml attribute).
            position_id: The name of a position ID (@uID xml attribute).
            is_design_variable: See func::select_variable_from_xpath.
            lower_bound: See func::select_variable_from_xpath.
            upper_bound: See func::select_variable_from_xpath.
            group_name: See func::select_variable_from_xpath.
            process: See func::select_variable_from_xpath.

        """
        xpath = self._SWEEP_ANGLE.get_xpath([wing_id, position_id])
        self.select_variable_from_xpath(
            xpath,
            name,
            is_design_variable,
            lower_bound,
            upper_bound,
            group_name=group_name,
            process=process,
        )
