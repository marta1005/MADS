# ruff: noqa: N806

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from multiads.assembly import (
    AirfoilNACA4,
    ComponentOptions,
    Environment,
    MADSComponent,
    PointMass,
    Propeller,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario import (
    BaseVariable,
    InnerVariable,
    InnerVariableFloat,
)
from multiads.scenario.mass_properties import MassPropertiesVariable
from multiads.scenario.span_loads import SpanLoadsGroupVariable
from multiads.scenario.tensors import MatrixVariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.aerodynamics.loads_aggregator import SpanloadsOptions
from multiads.solvers.structure.ampet.structural_material import StructuralMaterial
from multiads.solvers.structure.ampet.structural_profile import StructuralProfile
from multiads.solvers.structure.ampet.structural_wing import StructuralWing

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class Options(ComponentOptions):
    def __init__(
        self,
        ribs_max_separation: float = 0.5,
        panel_max_length: float = 0.1,
        reserve_factor: float = 1.5,
        load_factor: float = 1.5,
        extra_sections: int = 0,
    ) -> None:
        self.ribs_max_separation = ribs_max_separation
        self.panel_max_length = panel_max_length
        self.reserve_factor = reserve_factor
        self.load_factor = load_factor
        self.extra_sections = extra_sections


class AMPET(BaseSolver):
    def __init__(self) -> None:
        super().__init__()
        self.structuralWing: StructuralWing | None = None
        self.a_cap_root: float = 0.0
        self.a_stringer_root: float = 0.0
        self.inputs_map: Mapping[str, BaseVariable] | None = None
        self.outputs_map: Mapping[str, InnerVariable] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        # Parse components and create local wing
        environment, wing, propellers, point_masses = self._parse_components(components)

        self.structuralWing = self._make_wing(
            environment,
            wing,
            propellers,
            point_masses,
        )

        # Inputs and outputs
        self.inputs, self.outputs = self._make_interface(
            environment,
            wing,
            propellers,
            point_masses,
        )
        self.inputs_map = {v.name: v for v in self.inputs}
        self.outputs_map = {v.name: v for v in self.outputs}

        return [environment, wing, *propellers, *point_masses]

    def _parse_components(
        self,
        components: Sequence[MADSComponent],
    ) -> tuple[Environment, Wing, Sequence[Propeller], Sequence[PointMass]]:
        # Filter components
        _components = copy_components(components)
        components_flat = flatten_components(_components)

        try:
            environment = next(c for c in components_flat if type(c) is Environment)
        except StopIteration:
            msg = f"No environment provided to solver '{type(self).__name__}'."
            raise RuntimeError(msg) from None

        try:
            wing = next(c for c in components_flat if type(c) is Wing)
        except StopIteration:
            msg = f"No wing provided to solver '{type(self).__name__}'."
            raise RuntimeError(msg) from None

        propellers = [c for c in components_flat if type(c) is Propeller]
        point_masses = [c for c in components_flat if type(c) is PointMass]

        return environment, wing, propellers, point_masses

    def _make_wing(
        self,
        environment: Environment,
        wing: Wing,
        propellers: Sequence[Propeller],
        point_masses: Sequence[PointMass],
    ) -> StructuralWing:
        profile = StructuralProfile(
            airfoil=wing.sections[0].airfoil.airfoil_name,
            frontSparPosition=0.15,
            rearSparPosition=0.55,
        )

        material = StructuralMaterial(
            rho=2700.0,
            E=70.0e9,
            nu=0.3,
            thicknessMin=1.0e-3,
            thicknessStep=0.2e-3,
            allowableMaxRoot=450.0e6,
            allowableMaxTip=250.0e6,
            allowableShear=250.0e6,
            allowableElastic=415.0e6,
            metal=True,
        )

        x0 = wing.global_pos[0]
        y0 = wing.global_pos[1]
        z0 = wing.global_pos[2]

        nSpans = len(wing.spans)
        nSections = nSpans + 1

        stSpans = [0.0 for _ in range(nSpans)]
        stDihed = [0.0 for _ in range(nSpans)]
        stSweep = [0.0 for _ in range(nSpans)]

        for iSpan in range(nSpans):
            stSpans[iSpan] = wing.spans[iSpan].length
            stDihed[iSpan] = wing.spans[iSpan].dihed
            stSweep[iSpan] = wing.spans[iSpan].sweep * np.pi / 180.0

        stChords = [0.0 for _ in range(nSections)]
        stThicne = [0.0 for _ in range(nSections)]
        stTorsio = [0.0 for _ in range(nSections)]
        stOTorsi = [0.0 for _ in range(nSections)]

        for iSection in range(nSections):
            stChords[iSection] = wing.sections[iSection].chord
            stThicne[iSection] = wing.sections[iSection].thickness_to_chord_ratio
            stTorsio[iSection] = wing.sections[iSection].twist * np.pi / 180.0
            stOTorsi[iSection] = wing.xc_ref

        try:
            opts: Options = next(o for o in wing.options if type(o) is Options)
        except StopIteration:
            msg = f"No AMPET options in component '{wing.name}'"
            raise RuntimeError(msg) from None

        masses = [*propellers, *point_masses]
        yEngines = np.zeros((len(masses), 3))
        yEngines[:, 1] = [m.global_pos[1] for m in masses]
        yEngines = yEngines.flatten().tolist()
        mEngines = [m.mass for m in masses]
        pSupport = [0.0, wing.y_wing_fuselage_interface, 0.0]

        return StructuralWing(
            name=wing.name,
            baseProfile=profile,
            material=material,
            reserveFactor=opts.reserve_factor,
            vCruise=environment.speed,
            xRootPosition=x0,
            yRootPosition=y0,
            zRootPosition=z0,
            sectionSpans=stSpans,
            dihedralDistribution=stDihed,
            sweepDistribution=stSweep,
            chordDistribution=stChords,
            thicknessDistribution=stThicne,
            torsionDistribution=stTorsio,
            percentTorsionOriginDistribution=stOTorsi,
            maxRibsSeparation=opts.ribs_max_separation,
            panelLength=opts.panel_max_length,
            loadFactor=opts.load_factor,
            posEngines=yEngines,
            massEngines=mEngines,
            posSupport=pSupport,
        )

    def _make_interface(
        self,
        environment: Environment,
        wing: Wing,
        propellers: Sequence[Propeller],
        point_masses: Sequence[PointMass],
    ) -> tuple[Sequence[BaseVariable], Sequence[InnerVariable]]:
        inputs = self._make_design_inputs(environment, wing, propellers, point_masses)
        inputs.extend(self._make_inner_inputs(wing))
        outputs = self._make_outputs(wing)

        return inputs, outputs

    def _make_design_inputs(
        self,
        env: Environment,
        wing: Wing,
        propellers: Sequence[Propeller],
        point_masses: Sequence[PointMass],
    ) -> list[BaseVariable]:
        # Environment inputs
        inputs: list[BaseVariable] = []
        if v := env.variables.get("speed"):
            inputs.append(v)

        # Wing inputs
        wing_vars = [
            "pos",
            "xc_ref",
            "a_cap_root",
            "a_stringer_root",
            "y_wing_fuselage_interface",
        ]
        inputs.extend(v for k, v in wing.variables.items() if k in wing_vars)

        # Spans inputs
        span_vars = ["length", "sweep", "dihed"]
        for span in wing.spans:
            inputs.extend(v for k, v in span.variables.items() if k in span_vars)

        # Sections inputs
        sec_vars = ["chord", "twist"]
        for sec in wing.sections:
            inputs.extend(v for k, v in sec.variables.items() if k in sec_vars)

        # Airfoils inputs
        foil_vars = ["thickness_factor", "camber_factor"]
        naca_vars = ["m", "p", "t"]
        for sec in wing.sections:
            inputs.extend(v for k, v in sec.airfoil.variables.items() if k in foil_vars)
            if type(airfoil := sec.airfoil) is AirfoilNACA4:
                inputs.extend(v for k, v in airfoil.variables.items() if k in naca_vars)

        # Propeller inputs
        prop_vars = ["global_pos", "mass"]
        for prop in propellers:
            if prop.name in wing.children:
                inputs.extend(v for k, v in prop.variables.items() if k in prop_vars)

        # Point mass inputs
        mass_vars = ["global_pos", "mass"]
        for mass in point_masses:
            if mass.name in wing.children:
                inputs.extend(v for k, v in mass.variables.items() if k in mass_vars)

        return inputs

    def _make_inner_inputs(self, wing: Wing) -> list[SpanLoadsGroupVariable]:
        try:
            opts = next(o for o in wing.options if type(o) is SpanloadsOptions)
            return [
                SpanLoadsGroupVariable.from_sizes(
                    name=f"{wing.name}.span_loads_group",
                    loads_names=list(opts.loads.keys()),
                    loads_sizes=list(opts.loads.values()),
                ),
            ]
        except StopIteration:
            msg = f"'{wing.name}' does not have '{SpanloadsOptions.__name__}'."
            raise RuntimeError(msg) from None

    def _make_outputs(self, wing: Wing) -> list[InnerVariable]:
        if self.structuralWing is None:
            msg = f"Solver '{type(self).__name__}' was not initialized."
            raise RuntimeError(msg)

        prefix = self.structuralWing.name
        n_struct_sections = self.structuralWing.numberOfSections

        # Since AMPET may change the number of sections, add a buffer if requested
        if opts := next((o for o in wing.options if type(o) is Options), None):
            n_struct_sections += opts.extra_sections

        return [
            InnerVariableFloat(f"{prefix}.mass", 0.0),
            InnerVariableFloat(f"{prefix}.min_rfc", 1.0),
            InnerVariableFloat(f"{prefix}.min_rft", 1.0),
            MassPropertiesVariable.from_num_elements(
                f"{prefix}.mass_properties",
                n_struct_sections + 1,
            ),
            MatrixVariableFloat.zeros(
                f"{prefix}.stiffness_matrix",
                n_struct_sections,
                21,
            ),
        ]

    def set_state(self, components: Sequence[MADSComponent]) -> None:
        if self.inputs_map is None:
            msg = f"Solver {type(self).__name__} was not initialized."
            raise RuntimeError(msg)

        env = next(c for c in components if type(c) is Environment)
        wing = next(c for c in components if type(c) is Wing)
        propellers = [c for c in components if type(c) is Propeller]
        masses = [c for c in components if type(c) is PointMass]

        # TODO @Andres: update instead
        self.structuralWing = self._make_wing(env, wing, propellers, masses)

        self.a_cap_root = wing.a_cap_root
        self.a_stringer_root = wing.a_stringer_root

        aeroLoads = self.inputs_map[f"{self.structuralWing.name}.span_loads_group"]
        self.structuralWing.setAeroLoads(aeroLoads)  # type: ignore[invalid-argument-type]

    def _run(self) -> None:
        if self.structuralWing is None or self.outputs_map is None:
            msg = f"Solver {type(self).__name__} was not initialized."
            raise RuntimeError(msg)

        ACapRoot = self.a_cap_root
        AStringerRoot = self.a_stringer_root

        mass, massProperties, stiffness, minRFC, minRFT = (
            self.structuralWing.wingSizing2Vars(ACapRoot, AStringerRoot)
        )
        massProperties[:, 1:] /= massProperties[:, 0:1]  # AMPET uses "mass" variables

        name = self.structuralWing.name
        self.outputs_map[f"{name}.mass"].value = mass
        self.outputs_map[f"{name}.min_rfc"].value = minRFC
        self.outputs_map[f"{name}.min_rft"].value = minRFT

        var = self.outputs_map[f"{name}.mass_properties"]
        var.num_elements = self.structuralWing.numberOfSections + 1  # type: ignore[unresolved-attribute]
        var.matrix = massProperties  # type: ignore[unresolved-attribute]

        n = self.structuralWing.numberOfSections
        var = self.outputs_map[f"{name}.stiffness_matrix"]
        var.matrix[n:, :] = 0.0  # type: ignore[unresolved-attribute]
        var.matrix[:n, :] = stiffness  # type: ignore[unresolved-attribute]

    def compute_output(
        self,
    ) -> None:
        # All outputs were computed in `_run`
        if self.outputs is None:
            msg = f"The outputs of solver '{type(self).__name__}' are not initialized"
            raise RuntimeError(msg)

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
