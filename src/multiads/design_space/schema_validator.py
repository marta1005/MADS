"""Schema validation for CPACS files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_VERSION = "3.5"
SCHEMA_BASE_URL = "https://github.com/DLR-SL/CPACS/raw/v{version}/CPACS.xsd"


class SchemaValidator:
    """Validates CPACS XML files against XSD schema.

    Args:
        schema_path: Path to the XSD schema file.
            If None, uses the bundled schema.
        validate: Whether to perform validation.
        cache_dir: Directory for caching schema files.
    """

    def __init__(
        self,
        schema_path: Path | None = None,
        validate: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self._schema_path = schema_path
        self._validate = validate
        self._cache_dir = cache_dir or Path.home() / ".multiads" / "schema"
        self._schema: etree.XMLSchema | None = None

        if self._validate:
            self._load_schema()

    def _load_schema(self) -> None:
        """Load and parse the XSD schema."""
        if self._schema_path is None:
            self._schema_path = self._get_bundled_schema_path()

        if self._schema_path is None or not self._schema_path.exists():
            logger.warning(
                f"Schema file not found at {self._schema_path}. "
                "Validation disabled. Please provide a valid schema file."
            )
            self._validate = False
            return

        try:
            with open(self._schema_path, "rb") as f:
                schema_doc = etree.parse(f)
            self._schema = etree.XMLSchema(schema_doc)
            logger.info(f"Loaded CPACS schema from {self._schema_path}")
        except etree.XMLSchemaParseError as e:
            logger.warning(f"Failed to parse schema: {e}. Validation disabled.")
            self._validate = False
            self._schema = None

    def _get_bundled_schema_path(self) -> Path | None:
        """Get path to bundled schema file."""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "schema" / "cpacs_3_5.xsd",
            Path(__file__).parent.parent.parent.parent / "schema" / "CPACS.xsd",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return None

    def validate(self, xml_tree: ET | etree.Element) -> bool:
        """Validate an XML tree against the schema.

        Args:
            xml_tree: The XML element tree to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not self._validate or self._schema is None:
            return True

        try:
            if isinstance(xml_tree, etree.Element):
                self._schema.assertValid(xml_tree)
            else:
                root = etree.fromstring(etree.tostring(xml_tree.getroot()))
                self._schema.assertValid(root)
            logger.debug("XML validation successful")
            return True
        except etree.DocumentInvalid as e:
            logger.error(f"XML validation failed: {e}")
            return False

    def get_validation_errors(self, xml_tree: ET | etree.Element) -> list[str]:
        """Get validation errors without raising exceptions.

        Args:
            xml_tree: The XML element tree to validate.

        Returns:
            List of validation error messages.
        """
        if not self._validate or self._schema is None:
            return []

        errors: list[str] = []

        if isinstance(xml_tree, etree.Element):
            element = xml_tree
        else:
            element = etree.fromstring(etree.tostring(xml_tree.getroot()))

        for error in self._schema.error_log:
            errors.append(f"Line {error.line}: {error.message}")

        return errors

    @property
    def is_validating(self) -> bool:
        """Return whether validation is enabled."""
        return self._validate

    @property
    def schema_path(self) -> Path | None:
        """Return the schema file path."""
        return self._schema_path
