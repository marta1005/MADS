import copy
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from warnings import warn

import numpy as np
from numpy.testing import assert_allclose

from multiads.assembly import AirfoilNACA4, Environment
from multiads.assembly import Section as AssemblySection
from multiads.assembly import Span as AssemblySpan
from multiads.assembly import Wing as AssemblyWing
from multiads.disciplines.aerodynamics import Aerodynamics
from multiads.scenario import MADSScenario, VariableFloat
from multiads.solvers.aerodynamics.dust import DUST
from multiads.solvers.aerodynamics.dust_lib import (
    Driver,
    OutputOptions,
    PostLoads,
    PostSpanwiseLoads,
    Section,
    SectionOptions,
    Span,
    SpanOptions,
    SpanPanelType,
    Wing,
    WingMethod,
    WingOptions,
    WingPanelType,
)
from multiads.solvers.aerodynamics.dust_lib import Options as DustOptions
from multiads.solvers.aerodynamics.neuralfoil import Neuralfoil
from multiads.solvers.aerodynamics.neuralfoil import Options as NFOptions


class TestDustDriver:
    def test_component_initialization(self) -> None:
        """Test the initialization and parsing of components."""
        # Create test environment
        env = Environment(
            name="test_env",
            height=0.0,
            speed=100.0,
        )

        # Create test wing
        wing = AssemblyWing(
            name="test_wing",
            sections=[
                AssemblySection(
                    name="sec1",
                    airfoil=AirfoilNACA4("NACA2412", 2, 4, 12),
                    chord=1.0,
                    twist=0.0,
                ),
                AssemblySection(
                    name="sec2",
                    airfoil=AirfoilNACA4("NACA2412", 1, 2, 12),
                    chord=1.0,
                    twist=0.0,
                ),
            ],
            spans=[
                AssemblySpan(
                    name="span1",
                    length=10.0,
                    sweep=5.0,
                    dihed=2.0,
                    options=[
                        SpanOptions(panel_type=SpanPanelType.UNIFORM, panel_density=1),
                    ],
                ),
            ],
            options=[
                WingOptions(
                    discretization_method=WingMethod.LIFTING_LINE,
                    panel_type=WingPanelType.UNIFORM,
                    num_panels=10,
                ),
            ],
        )

        # Create solver with options
        options = DustOptions(
            dust_pre="dust_pre",
            dust="dust",
            dust_post="dust_post",
            work_dir="test_work_dir",
            output_options=OutputOptions(compute_loads=True),
        )
        solver = DUST(options)

        # Test component parsing
        parsed_components = solver.parse_variables([env, wing])
        assert len(parsed_components) == 2
        assert isinstance(parsed_components[0], Environment)
        assert isinstance(parsed_components[1], AssemblyWing)

        # Verify internal state
        assert isinstance(env, Environment)
        assert isinstance(solver.wings, Sequence)
        assert len(solver.wings) == 1
        assert isinstance(solver.wings[0], Wing)
        assert solver.wings[0].name == "test_wing"

    def test_state_management(self) -> None:
        """Test the state management functionality."""
        # Create test components
        env = Environment(
            height=0.0,
            name="test_env",
            speed=100.0,
        )

        wing = AssemblyWing(
            name="test_wing",
            sections=[
                AssemblySection(
                    name="sec1",
                    airfoil=AirfoilNACA4("NACA2412", 2, 4, 12),
                    chord=1.0,
                    twist=0.0,
                ),
                AssemblySection(
                    name="sec2",
                    airfoil=AirfoilNACA4("NACA2412", 1, 2, 12),
                    chord=1.0,
                    twist=0.0,
                ),
            ],
            spans=[
                AssemblySpan(
                    name="span1",
                    length=10.0,
                    sweep=5.0,
                    dihed=2.0,
                    options=[
                        SpanOptions(panel_type=SpanPanelType.EQUALAREA, num_panels=15),
                    ],
                ),
            ],
            options=[
                WingOptions(
                    discretization_method=WingMethod.LIFTING_LINE,
                    panel_type=WingPanelType.UNIFORM,
                    num_panels=10,
                ),
            ],
        )

        # Create solver
        options = DustOptions(
            dust_pre="dust_pre",
            dust="dust",
            dust_post="dust_post",
            work_dir="test_work_dir",
            output_options=OutputOptions(compute_loads=True),
        )
        solver = DUST(options)
        solver.parse_variables([env, wing])

        # Test state setting
        new_wing = copy.deepcopy(wing)
        new_wing.sections[0].chord = 1.2
        solver.set_state([env, new_wing])

        # Verify the state was updated
        assert isinstance(solver.wings, Sequence)
        assert solver.wings[0].sections[0].chord == 1.2

        shutil.rmtree("test_work_dir")


class TestDustLib:
    def test_wing_creation(self) -> None:
        """Test the creation and serialization of Wing components."""
        # Create sections
        section1 = Section(
            name="sec1",
            airfoil="NACA2412",
            chord=1.0,
            twist=0.0,
            polar=True,
            polar_length=100,
        )

        section2 = Section(
            name="sec2",
            airfoil="NACA0012",
            chord=0.8,
            twist=-2.0,
            polar=False,
            polar_length=0,
        )

        # Create spans
        span1 = Span(
            length=5.0,
            sweep=2.0,
            dihed=1.0,
            panel_type=SpanPanelType.UNIFORM,
            num_panels=10,
        )

        # Create wing
        wing = Wing(
            name="test_wing",
            sections=[section1, section2],
            spans=[span1],
            method=WingMethod.LIFTING_LINE,
            panel_type=WingPanelType.UNIFORM,
            num_panels=5,
            pos=np.array([0.0, 0.0, 0.0]),
            alpha=3.0,
            beta=0.0,
            roll=0.0,
            xc_ref=0.25,
        )

        assert wing.name == "test_wing"
        assert len(wing.sections) == 2
        assert len(wing.spans) == 1
        assert wing.method == WingMethod.LIFTING_LINE

        # Test file writing
        test_file = Path("test_wing.in")
        wing.write_file(test_file)
        assert test_file.exists()

        # Clean up
        test_file.unlink()

    def test_component_conversion(self) -> None:
        """Test the conversion from assembly components to DUST components."""
        # Create assembly wing
        assembly_wing = AssemblyWing(
            name="test_wing",
            sections=[
                AssemblySection(
                    name="sec1",
                    airfoil=AirfoilNACA4("NACA2412", 2, 4, 12),
                    chord=1.0,
                    twist=0.0,
                    options=[SectionOptions(polar=True, polar_length=100)],
                ),
            ],
            spans=[
                AssemblySpan(
                    name="span1",
                    length=10.0,
                    sweep=5.0,
                    dihed=2.0,
                    options=[
                        SpanOptions(
                            panel_type=SpanPanelType.UNIFORM,
                            num_panels=10,
                        ),
                    ],
                ),
            ],
            options=[
                WingOptions(
                    discretization_method=WingMethod.LIFTING_LINE,
                    panel_type=WingPanelType.UNIFORM,
                    num_panels=10,
                ),
            ],
        )

        # Convert to DUST wing
        dust_wing = Wing.from_component(assembly_wing)
        assert dust_wing.name == "test_wing"
        assert len(dust_wing.sections) == 1
        assert len(dust_wing.spans) == 1
        assert dust_wing.sections[0].polar
        assert dust_wing.sections[0].polar_length == 100

    def test_driver_functionality(self) -> None:
        """Test the basic functionality of the Driver class."""
        # Create minimal environment
        env = Environment(
            name="test_env",
            height=0.0,
            speed=100.0,
        )

        # Create minimal wing
        wing = Wing(
            name="test_wing",
            sections=[
                Section(
                    name="sec1",
                    airfoil="NACA2412",
                    chord=1.0,
                    twist=0.0,
                    polar=False,
                    polar_length=0,
                ),
                Section(
                    name="sec2",
                    airfoil="NACA1212",
                    chord=0.5,
                    twist=0.0,
                    polar=False,
                    polar_length=0,
                ),
            ],
            spans=[
                Span(
                    length=10.0,
                    sweep=0.0,
                    dihed=0.0,
                    panel_type=SpanPanelType.UNIFORM,
                    num_panels=5,
                ),
            ],
            method=WingMethod.VORTEX_LATTICE,
            panel_type=WingPanelType.UNIFORM,
            num_panels=5,
        )

        # Create options
        options = DustOptions(
            dust_pre="dust_pre",
            dust="dust",
            dust_post="dust_post",
            work_dir="test_driver",
            t_end=0.1,
            dt=0.01,
            output_options=OutputOptions(compute_loads=True),
        )

        # Create driver
        driver = Driver(
            environment=env,
            options=options,
            wings=[wing],
        )

        main_dir = Path.cwd()
        test_dir = main_dir / "test_driver"
        test_dir.mkdir(exist_ok=True)
        os.chdir(test_dir)

        # Test preprocessing
        # TODO @Andres: Skip tests if DUST is not installed (TEMPORARY!!!!!)
        try:
            driver.preprocess()
        except FileNotFoundError:
            warn("Could not complete the test as DUST is not installed.", stacklevel=2)
            return

        assert Path("output").exists()
        assert Path("post").exists()

        # Test postprocessing setup
        post_loads = PostLoads(
            name="test_loads",
            start_res=1,
            end_res=2,
            components=["test_wing"],
        )

        post_spanwise = PostSpanwiseLoads(
            name="test_spanwise",
            start_res=1,
            end_res=2,
            components=["test_wing"],
        )

        driver.run()
        driver.postprocess([post_loads, post_spanwise])

        assert (Path("post") / "dust_test_loads.dat").exists()
        assert (Path("post") / "dust_test_spanwise_Fx.dat").exists()
        assert (Path("post") / "dust_test_spanwise_Fy.dat").exists()
        assert (Path("post") / "dust_test_spanwise_Fz.dat").exists()
        assert (Path("post") / "dust_test_spanwise_Mo.dat").exists()

        # Clean up
        os.chdir(main_dir)
        shutil.rmtree(test_dir)


class TestDust:
    def test_dust_mdo(self) -> None:
        # Variables
        alpha = VariableFloat(name="alpha", value=0.0, lb=-5.0, ub=10.0)
        chord_2 = VariableFloat(name="chord_2", value=0.7, lb=0.3, ub=1.0)

        # Components
        env = Environment(name="env", height=1000.0, speed=40.0, alpha=alpha)

        wing = AssemblyWing(
            name="wing",
            sections=[
                AssemblySection(
                    name="sec_1",
                    airfoil=AirfoilNACA4("airfoil_1", 1, 2, 24),
                    chord=1.0,
                    twist=0.0,
                    options=[SectionOptions(polar=True)],
                ),
                AssemblySection(
                    name="sec_2",
                    airfoil=AirfoilNACA4("airfoil_2", 2, 2, 12),
                    chord=chord_2,
                    twist=-1.0,
                    options=[SectionOptions(polar=True)],
                ),
            ],
            spans=[
                AssemblySpan(
                    name="span_1",
                    length=10.0,
                    sweep=5.0,
                    dihed=2.0,
                    options=[
                        SpanOptions(panel_type=SpanPanelType.UNIFORM, panel_density=1),
                    ],
                ),
            ],
            symmetry=True,
            options=[
                WingOptions(discretization_method=WingMethod.LIFTING_LINE),
            ],
        )

        # Disciplines
        nf_solver = Neuralfoil(options=NFOptions())
        polars = Aerodynamics(name="Polars", components=[env, wing], solver=nf_solver)

        dust_solver = DUST(
            options=DustOptions(
                work_dir="test_dust",
                t_end=0.1,
                dt=0.01,
                particles_box_min=np.array([-1.0, -12.0, 1.0]),
                particles_box_max=np.array([11.0, 12.0, -3.0]),
                output_options=OutputOptions(
                    compute_loads=True,
                    loads_start=1,
                    loads_end=10,
                    loads_avg=True,
                ),
            ),
        )
        dust = Aerodynamics(name="DUST", components=[env, wing], solver=dust_solver)

        # Scenario
        scenario = MADSScenario()
        scenario.fill_parameter_space([alpha, chord_2])
        scenario.create_scenario(
            disciplines=[polars, dust],
            formulation="DisciplinaryOpt",
            objective_name="efficiency",
            scenario_type="DOE",
        )

        # TODO @Andres: Skip tests if DUST is not installed (TEMPORARY!!!!!)
        try:
            scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=3**2)
            shutil.rmtree("test_dust")
        except FileNotFoundError:
            warn("Could not complete the test as DUST is not installed.", stacklevel=2)
            return

        # Post process
        data = scenario.to_dataset()
        data_a = data["designs"]["alpha"][0]
        data_c2 = data["designs"]["chord_2"][0]
        data_eff = data["functions"]["efficiency"][0]

        assert_allclose(data_a, [-5.0, 2.5, 10.0, -5.0, 2.5, 10.0, -5.0, 2.5, 10.0])
        assert_allclose(data_c2, [0.30, 0.30, 0.30, 0.65, 0.65, 0.65, 1.00, 1.00, 1.00])
        assert_allclose(
            data_eff,
            [
                -34.983095,
                34.601156,
                31.484918,
                -32.449206,
                32.793661,
                28.191246,
                -29.530808,
                31.004558,
                24.752899,
            ],
        )
