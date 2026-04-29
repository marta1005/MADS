"""Template for CPACS XPath."""

from collections.abc import Sequence


class CPACSPathTemplate:
    """Define CPACS variable that corresponds to a xml XPath."""

    def __init__(self, xpath_template: str, template_variables: Sequence[str]) -> None:
        """
        Args:
            xpath_template: The XPath template.
            template_variables: The variables that will be replaced by values.

        """  # noqa: D205, D212
        self.__xpath_template = xpath_template
        self.__template_variables = template_variables

    @property
    def xpath_template(self) -> str:
        """Get the XPath template."""
        return self.__xpath_template

    @property
    def template_variables(self) -> list[str]:
        """Get template variables."""
        return list(self.__template_variables)

    def get_xpath(self, template_values: Sequence[str]) -> str:
        """Get the real xpath, replacing template variable with values.

        Args:
            template_values: The values for template variables.

        Returns:
            The real xpath.

        """
        if len(template_values) != len(self.__template_variables):
            msg = (
                f"The number of values in should be "
                f"{len(self.__template_variables)} while it "
                f"is {len(template_values)}."
            )
            raise ValueError(msg)

        return self.__xpath_template.format(
            **dict(zip(self.__template_variables, template_values)),
        )
