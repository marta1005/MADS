"""Descartes solver for geometry generation.

This module provides the Descartes solver for MultiADS, which generates
aircraft geometry from CPACS files using the Descartes Python interface.

The solver:
1. Loads CPACS as the primary geometric source
2. Updates CPACS with current Assembly state for compatible parameters
3. Uses Descartes Python interface to generate/modify geometry
4. Exposes results as InnerVariables for downstream analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from multiads.assembly import (
    Aircraft,
    Environment,
    Fuselage,
    MADSComponent,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.design_space import CPACSImporter
from multiads.scenario import BaseVariable, InnerVariable, InnerVariableFloat
from multiads.solvers import BaseSolver

from multiads.solvers.synthesis.descartes_lib import (
    DESCARTES_AVAILABLE,
    GeometryOutput,
    GeometryParameters,
    Options,
    check_descartes_available,
    require_descartes,
)

try:
    from multiads.solvers.synthesis.build123d_generator import BUILD123D_AVAILABLE
except ImportError:
    BUILD123D_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def geometry_wing_reference_area(name: str, output: GeometryOutput, **kwargs: Any) -> NDArray[np.float64]:
    """Extract wing reference area from geometry output."""
    return np.asarray([output.wing_reference_area])


def geometry_wing_span(name: str, output: GeometryOutput, **kwargs: Any) -> NDArray[np.float64]:
    """Extract wing span from geometry output."""
    return np.asarray([output.wing_span])


def geometry_wing_mgc(name: str, output: GeometryOutput, **kwargs: Any) -> NDArray[np.float64]:
    """Extract wing mean geometric chord from geometry output."""
    return np.asarray([output.wing_mean_geometric_chord])


def geometry_wing_ar(name: str, output: GeometryOutput, **kwargs: Any) -> NDArray[np.float64]:
    """Extract wing aspect ratio from geometry output."""
    return np.asarray([output.wing_aspect_ratio])


def geometry_fuselage_wetted_area(name: str, output: GeometryOutput, **kwargs: Any) -> NDArray[np.float64]:
    """Extract fuselage wetted area from geometry output."""
    return np.asarray([output.fuselage_wetted_area])


def geometry_fuselage_volume(name: str, output: GeometryOutput, **kwargs: Any) -> NDArray[np.float64]:
    """Extract fuselage volume from geometry output."""
    return np.asarray([output.fuselage_volume])


from enum import Enum


class GeometryBackend(Enum):
    """Available geometry backends."""
    DESCARTES = "descartes"
    PYOCCT = "pyocct"
    BUILD123D = "build123d"


IMPLEMENTED_OUTPUTS = {
    "geometry_wing_reference_area": geometry_wing_reference_area,
    "geometry_wing_span": geometry_wing_span,
    "geometry_wing_mgc": geometry_wing_mgc,
    "geometry_wing_ar": geometry_wing_ar,
    "geometry_fuselage_wetted_area": geometry_fuselage_wetted_area,
    "geometry_fuselage_volume": geometry_fuselage_volume,
}


class DescartesSolver(BaseSolver):
    """Descartes solver for geometry generation.

    This solver reads geometry from CPACS, optionally updates with MultiADS
    Assembly state, and uses Descartes to generate/modify aircraft geometry.

    The solver requires:
    - An Aircraft component (to determine CPACS source and naming)
    - Optionally Wing/Fuselage components for geometry parameters
    - Optionally Environment for flight conditions

    Attributes:
        options: Descartes solver options.
        cpacs_importer: CPACSImporter for handling CPACS files.
        cpacs_path: Path to CPACS input file.
    """

    def __init__(
        self,
        options: Options | None = None,
        cpacs_path: str | Path | None = None,
        backend: str = "descartes",
    ) -> None:
        """Initialize Descartes solver.

        Args:
            options: Descartes solver options.
            cpacs_path: Path to CPACS input file (optional, can also be set via Aircraft).
            backend: Geometry backend to use ("descartes" or "pyocct").
        """
        super().__init__()

        self.options: Options = options or Options()
        self.cpacs_path: Path | None = Path(cpacs_path) if cpacs_path else None
        self.cpacs_importer: CPACSImporter | None = None
        self.aircraft: Aircraft | None = None
        self.wings: list[Wing] | None = None
        self.fuselages: list[Fuselage] | None = None
        self.environment: Environment | None = None
        self.inputs_map: dict[str, BaseVariable] | None = None
        self.outputs_map: dict[str, InnerVariable] | None = None
        self._geometry_output: GeometryOutput | None = None

        # Backend selection
        try:
            self.backend = GeometryBackend(backend.lower())
        except ValueError:
            valid = [b.value for b in GeometryBackend]
            raise ValueError(
                f"Invalid backend '{backend}'. Valid options: {valid}"
            ) from None

    def _get_cpacs_path_from_aircraft(self, aircraft: Aircraft) -> Path | None:
        """Get CPACS path from Aircraft component.

        The CPACS path may be stored in aircraft.variables or in aircraft.metadata.

        Args:
            aircraft: Aircraft component.

        Returns:
            Path to CPACS file if found, None otherwise.
        """
        if hasattr(aircraft, "variables"):
            cpacs_path_var = aircraft.variables.get("cpacs_path")
            if cpacs_path_var is not None:
                return Path(cpacs_path_var.value)

        if hasattr(aircraft, "metadata"):
            cpacs_path = aircraft.metadata.get("cpacs_path")
            if cpacs_path:
                return Path(cpacs_path)

        return self.cpacs_path

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,
    ) -> Sequence[MADSComponent]:
        """Parse MultiADS components and prepare Descartes inputs.

        Extracts Aircraft, Wing, Fuselage, and Environment components,
        then sets up CPACS importer and input variables.

        Args:
            components: Sequence of MultiADS components.

        Returns:
            List of components for MultiADS tracking.
        """
        _components = copy_components(components)
        components_flat = flatten_components(_components)

        try:
            ac = (c for c in components_flat if isinstance(c, Aircraft))
            self.aircraft = next(ac)
        except StopIteration:
            msg = f"An aircraft must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        self.wings = [c for c in components_flat if isinstance(c, Wing)]

        self.fuselages = [c for c in components_flat if isinstance(c, Fuselage)]

        try:
            env = (c for c in components_flat if isinstance(c, Environment))
            self.environment = next(env)
        except StopIteration:
            raise ValueError(
                f"An environment must be provided to solver '{type(self).__name__}'."
            ) from None

        self.cpacs_path = self._get_cpacs_path_from_aircraft(self.aircraft)

        if self.cpacs_path is None:
            if hasattr(self.aircraft, "variables"):
                cpacs_var = self.aircraft.variables.get("cpacs_path")
                if cpacs_var:
                    self.cpacs_path = Path(cpacs_var.value)

        if self.cpacs_path is not None and self.cpacs_path.exists():
            try:
                self.cpacs_importer = CPACSImporter(
                    file_name=self.cpacs_path.name,
                    path=self.cpacs_path.parent,
                    validate=False,
                )
            except Exception:
                self.cpacs_importer = None
        else:
            self.cpacs_importer = None

        inputs: list[BaseVariable] = []

        if self.environment:
            env_dvars = ["height", "speed", "alpha", "beta", "gamma"]
            inputs.extend(
                v for k, v in self.environment.variables.items() if k in env_dvars
            )

        if self.aircraft:
            ac_dvars = ["mass", "global_pos"]
            inputs.extend(
                v for k, v in self.aircraft.variables.items() if k in ac_dvars
            )

        outputs: list[InnerVariable] = []

        if self.aircraft:
            prefix = self.aircraft.name
            for out_name in IMPLEMENTED_OUTPUTS.keys():
                full_name = f"{prefix}.{out_name}"
                outputs.append(InnerVariableFloat(full_name, 0.0))

        self.inputs = inputs
        self.outputs = outputs
        self.inputs_map = {v.name: v for v in self.inputs}
        self.outputs_map = {v.name: v for v in self.outputs}

        return [self.environment, self.aircraft, *self.wings, *self.fuselages]

    def _sync_assembly_to_cpacs(self) -> None:
        """Synchronize current Assembly state to CPACS.

        Updates CPACS XML tree with current values from Wing/Section/Span components.

        Updates:
        - Section chord → CPACS element scaling/x
        - Section twist → CPACS section transformation/rotation/y
        - Section position (y) → CPACS section transformation/translation/y
        - Segment sweep → CPACS segment sweepAngle
        - Segment dihedral → CPACS segment dihedralAngle
        """
        if self.cpacs_importer is None or self.cpacs_importer._CPACSStructureData__tree is None:
            return

        tree = self.cpacs_importer._CPACSStructureData__tree

        # Sync Wing components
        if self.wings:
            cumulative_y = 0.0

            for wing in self.wings:
                wing_uid = wing.name if hasattr(wing, "name") else "wing_main"

                # Sync sections (chord, twist, position)
                if hasattr(wing, "sections") and wing.sections:
                    for i, section in enumerate(wing.sections):
                        section_uid = section.name if hasattr(section, "name") else f"{wing_uid}_section_{i}"

                        # Update chord in CPACS: section/elements/element/transformation/scaling/x
                        chord_xpath = f".//section[@uID='{section_uid}']/elements/element/transformation/scaling/x"
                        chord_elem = tree.find(chord_xpath)
                        if chord_elem is not None and hasattr(section, "chord"):
                            chord_val = float(section.chord) if section.chord is not None else 1.0
                            chord_elem.text = str(chord_val)

                        # Update twist in CPACS: section/transformation/rotation/y
                        twist_xpath = f".//section[@uID='{section_uid}']/transformation/rotation/y"
                        twist_elem = tree.find(twist_xpath)
                        if twist_elem is not None and hasattr(section, "twist"):
                            twist_val = float(section.twist) if section.twist is not None else 0.0
                            twist_elem.text = str(twist_val)

                # Sync spans (length → position update, sweep, dihedral)
                if hasattr(wing, "spans") and wing.spans:
                    cumulative_y = 0.0

                    for i, span in enumerate(wing.spans):
                        span_uid = span.name if hasattr(span, "name") else f"{wing_uid}_segment_{i}"

                        # Update section position based on span length
                        # (Each span's toElement position = cumulative_y + span.length)
                        if i < len(wing.sections) - 1:
                            next_section = wing.sections[i + 1]
                            next_section_uid = next_section.name if hasattr(next_section, "name") else f"{wing_uid}_section_{i+1}"

                            cumulative_y += float(span.length) if span.length is not None else 10.0

                            # Update translation/y for next section
                            trans_xpath = f".//section[@uID='{next_section_uid}']/transformation/translation"
                            trans_elem = tree.find(trans_xpath)
                            if trans_elem is not None:
                                y_elem = trans_elem.find("y")
                                if y_elem is not None:
                                    y_elem.text = str(cumulative_y)

                        # CPACS segments don't have sweepAngle/dihedralAngle by default
                        # We can add them, or let build123d handle via overrides
                        # For now, we'll rely on build123d overrides

        # Sync Fuselage components
        if self.fuselages:
            for fuse in self.fuselages:
                fuse_uid = fuse.name if hasattr(fuse, "name") else "fuselage_1"

                # Update length
                length_xpath = f".//fuselage[@uID='{fuse_uid}']/geometry/length"
                length_elem = tree.find(length_xpath)
                if length_elem is not None and hasattr(fuse, "length"):
                    length_val = float(fuse.length) if fuse.length is not None else 30.0
                    length_elem.text = str(length_val)

                # Update width
                width_xpath = f".//fuselage[@uID='{fuse_uid}']/geometry/width"
                width_elem = tree.find(width_xpath)
                if width_elem is not None and hasattr(fuse, "maximum_width"):
                    width_val = float(fuse.maximum_width) if fuse.maximum_width is not None else 4.0
                    width_elem.text = str(width_val)

                # Update height
                height_xpath = f".//fuselage[@uID='{fuse_uid}']/geometry/height"
                height_elem = tree.find(height_xpath)
                if height_elem is not None and hasattr(fuse, "maximum_height"):
                    height_val = float(fuse.maximum_height) if fuse.maximum_height is not None else 4.0
                    height_elem.text = str(height_val)

        # Write updated CPACS to file (in memory, for next read)
        if hasattr(self.cpacs_importer, "sync_to_cpacs"):
            self.cpacs_importer.sync_to_cpacs()

    def run(self, components: Sequence[MADSComponent]) -> None:
        """Run geometry generation with selected backend."""
        self.set_state(components)
        super().run(components)

    def _run(self) -> None:
        """Abstract method implementation - routes to appropriate backend."""
        if self.aircraft is None or self.environment is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        if self.backend == GeometryBackend.BUILD123D and BUILD123D_AVAILABLE:
            return self._build123d_run()
        elif DESCARTES_AVAILABLE:
            return self._descartes_run()
        else:
            return self._fallback_run()


    def _build123d_run(self) -> None:
        """Run geometry generation using build123d."""
        try:
            from multiads.solvers.synthesis.build123d_generator import (
                Build123DGenerator,
            )

            # Step 1: Sync MultiADS Assembly state to CPACS FIRST
            self._sync_assembly_to_cpacs()

            # Step 2: Create generator with updated CPACS
            generator = Build123DGenerator()

            if self.cpacs_path:
                generator.cpacs_path = self.cpacs_path
                # Use the same CPACSImporter (with updated tree)
                if self.cpacs_importer:
                    generator._importer = self.cpacs_importer
                    generator.load_cpacs()

            # Step 3: Generate complete aircraft geometry
            print("Generating complete aircraft...")
            aircraft = generator.generate_aircraft(
                symmetry=True,
                fillet_radius=0.3,
            )

            if aircraft is None:
                self._geometry_output = GeometryOutput(
                    success=False,
                    message="Aircraft generation failed",
                )
                return

            # Step 4: Export to STEP
            output_dir = Path("/tmp/multiads_geometry")
            output_dir.mkdir(exist_ok=True)

            aircraft_step = output_dir / "aircraft.step"
            generator.export_step(aircraft, aircraft_step)
            print(f"Aircraft exported to: {aircraft_step}")

            # Step 5: Get metrics
            metrics = generator.get_geometry_metrics(aircraft)

            # Step 6: Return output
            self._geometry_output = GeometryOutput(
                success=True,
                message="build123d aircraft geometry generated",
                wing_span=metrics.get("span", 0.0),
                wing_reference_area=metrics.get("surface_area", 0.0),
                wing_mean_geometric_chord=metrics.get("max_chord", 0.0),
                wing_aspect_ratio=(
                    metrics.get("span", 0.0) ** 2
                    / metrics.get("surface_area", 1.0)
                    if metrics.get("surface_area", 0.0) > 0
                    else 0.0
                ),
                fuselage_wetted_area=0.0,  # Combined aircraft
                fuselage_volume=metrics.get("volume", 0.0),
            )

        except Exception as e:
            self._geometry_output = GeometryOutput(
                success=False,
                message=f"build123d geometry generation failed: {e}",
            )
            import traceback
            traceback.print_exc()

    def _descartes_run(self) -> None:
        """Run Descartes geometry generation."""
        if self.aircraft is None or self.environment is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        if self.cpacs_path is None:
            msg = f"CPACS path not set in solver '{type(self).__name__}'."
            raise RuntimeError(msg)

        from multiads.solvers.synthesis.descartes_lib import DescartesDriver

        params = GeometryParameters(
            wing_span_distribution=self.options.geometry_type,
            n_span_panels=20,
            n_chord_panels=5,
        )

        try:
            driver = DescartesDriver(
                cpacs_path=self.cpacs_path,
                options=self.options,
                params=params,
            )
            self._geometry_output = driver.run()

        except ImportError:
            self._geometry_output = GeometryOutput(
                success=False,
                message="Descartes not available - install Descartes to enable geometry generation",
            )
        except Exception as e:
            self._geometry_output = GeometryOutput(
                success=False,
                message=f"Geometry generation failed: {e}",
            )

    def _fallback_run(self) -> None:
        """Fallback when no geometry backend is available."""
        self._geometry_output = GeometryOutput(
            success=False,
            message="No geometry backend available (Descartes or pyOCCT)",
        )

        try:
            driver = DescartesDriver(
                cpacs_path=self.cpacs_path,
                options=self.options,
                params=params,
            )

            self._geometry_output = driver.run()

        except ImportError:
            self._geometry_output = GeometryOutput(
                success=False,
                message="Descartes not available - install Descartes to enable geometry generation",
            )
        except Exception as e:
            self._geometry_output = GeometryOutput(
                success=False,
                message=f"Geometry generation failed: {e}",
            )

    def compute_output(self) -> None:
        """Compute output variables from Descartes results."""
        if self._geometry_output is None:
            msg = f"The geometry output of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The outputs of solver '{type(self).__name__}' are not initialized"
            raise RuntimeError(msg)

        if self.aircraft is None:
            return

        for out in self.outputs:
            out_name = out.name
            if "." in out_name:
                base_name = out_name.split(".", 1)[1]
            else:
                base_name = out_name

            if base_name in IMPLEMENTED_OUTPUTS:
                out_func = IMPLEMENTED_OUTPUTS[base_name]
                result = out_func(out_name, self._geometry_output)
                # Handle numpy arrays (convert to scalar if 0-dimensional)
                if hasattr(result, 'ndim') and result.ndim == 0:
                    out.value = float(result)
                elif hasattr(result, 'item'):
                    out.value = result.item()
                else:
                    out.value = float(result)

    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[BaseVariable],
        output_names: Sequence[str],
        outputs: Sequence[BaseVariable],
    ) -> Mapping[str, NDArray]:
        """Compute sensitivities from Descartes.

        Note: Descartes sensitivity support is limited. This returns
        an empty Jacobian for most cases.

        Args:
            input_names: Names of input variables.
            inputs: Input variables.
            output_names: Names of output variables.
            outputs: Output variables.

        Returns:
            Jacobian matrix mapping inputs to outputs.
        """
        return {}