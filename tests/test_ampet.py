import copy
import shutil
from warnings import warn

import numpy as np
from numpy.testing import assert_allclose

import multiads.utilities.units as unit
from multiads.assembly import (
    AirfoilFile,
    Environment,
    PointMass,
    Section,
    Span,
    Wing,
)
from multiads.disciplines import UserDefined
from multiads.disciplines.aerodynamics import Aerodynamics
from multiads.disciplines.weight_and_balance import WeightAndBalance
from multiads.scenario import (
    Constraint,
    ConstraintType,
    MADSScenario,
    VariableFloat,
    VariableFloatNP,
)
from multiads.solvers.aerodynamics import dust_lib
from multiads.solvers.aerodynamics.dust import DUST
from multiads.solvers.aerodynamics.loads_aggregator import (
    LoadsAggregator,
    SpanloadsOptions,
)
from multiads.solvers.structure.ampet import AMPET
from multiads.solvers.structure.ampet import Options as AMPETOptions


def test_ampet_mdo() -> None:  # noqa: PLR0915
    # Parameters
    xc_ref = 0.25
    chord_root = 3.0
    ribs_separation = 0.475
    panel_length = 0.1
    y_wing_fus_interface = 1.1
    wing_pos = np.zeros(3)
    wing_span = 13.5
    wing_profile = "assets/naca653218.dat"

    # Dust Variables
    n_panels_inner = 10
    n_panels_outer = 10
    n_panels_chord = 8
    dust_delta_t = 0.005
    dust_n_temp_steps = 35
    dust_n_step_0_calc = 30
    dust_tf = dust_delta_t * dust_n_temp_steps

    sfc = 0.5 / (3600.0 * 9.81)  # 0.5 lb/(h*lbf)

    mtow = 32000.0
    zfw_wing_excluded = 24000.0

    wing_surf_plus = wing_span * chord_root
    wing_surf_minus = 8.8 * (chord_root - 1.5)
    wing_surf = 2.0 * (wing_surf_plus - wing_surf_minus)

    # Variables
    alpha = VariableFloat(name="alpha", value=3.0, lb=-5.0, ub=10.0)
    span_0 = VariableFloat(name="span_0", value=4.7, lb=2.0, ub=10.0)
    angle_tip = VariableFloat(name="angle_tip", value=0.0, lb=-10.0, ub=30.0)
    a_cap = VariableFloat(name="ACapRoot", value=1e-2, lb=1.0e-5, ub=2e-2)
    a_stringer = VariableFloat(name="AStringerRoot", value=1e-3, lb=1.0e-5, ub=2e-2)

    # Coupling Variables
    span_1 = VariableFloat(name="span_1", value=8.8)
    chord_tip = VariableFloat(name="chord_tip", value=1.5)
    engine_pos = VariableFloatNP(name="engine_pos", value=np.array([0.0, 3.5, 0.0]))
    efficiency = VariableFloat(name="efficiency", value=1.0)
    lift_const = VariableFloat(name="lift_const", value=1.0)
    distance = VariableFloat(name="range", value=50.0)

    variables = [alpha, chord_tip, span_0, angle_tip, a_cap, a_stringer]

    # Components
    env = Environment(
        name="env",
        height=25000.0 * unit.ft,
        speed=300.0 * unit.kt,
        alpha=alpha,
    )

    engine_mass = PointMass(name="engine_mass", mass=988.650, global_pos=engine_pos)

    wing = Wing(
        name="wing",
        sections=[
            Section(
                "root_sec",
                airfoil=AirfoilFile("root_foil", filename=wing_profile),
                chord=chord_root,
                twist=0.0,
            ),
            Section(
                "kink_sec",
                airfoil=AirfoilFile("kink_foil", filename=wing_profile),
                chord=chord_root,
                twist=0.0,
            ),
            Section(
                "tip_sec",
                airfoil=AirfoilFile("tip_foil", filename=wing_profile),
                chord=chord_tip,
                twist=angle_tip,
            ),
        ],
        spans=[
            Span(
                "span_0",
                length=span_0,
                sweep=0.0,
                dihed=0.0,
                options=[
                    dust_lib.SpanOptions(
                        panel_type=dust_lib.SpanPanelType.UNIFORM,
                        num_panels=n_panels_inner,
                    ),
                ],
            ),
            Span(
                "span_1",
                length=span_1,
                sweep=0.0,
                dihed=0.0,
                options=[
                    dust_lib.SpanOptions(
                        panel_type=dust_lib.SpanPanelType.UNIFORM,
                        num_panels=n_panels_outer,
                    ),
                ],
            ),
        ],
        xc_ref=xc_ref,
        symmetry=True,
        global_pos=wing_pos,
        a_cap_root=a_cap,
        a_stringer_root=a_stringer,
        y_wing_fuselage_interface=y_wing_fus_interface,
        children=[engine_mass],
        options=[
            dust_lib.WingOptions(
                discretization_method=dust_lib.WingMethod.VORTEX_LATTICE,
                num_panels=n_panels_chord,
                panel_type=dust_lib.WingPanelType.UNIFORM,
                output_options=dust_lib.OutputOptions(
                    compute_loads=True,
                    compute_spanwise=True,
                    loads_start=dust_n_step_0_calc,
                    loads_end=dust_n_temp_steps,
                    spanwise_start=dust_n_step_0_calc,
                    spanwise_end=dust_n_temp_steps,
                ),
            ),
            SpanloadsOptions(),
            AMPETOptions(
                ribs_max_separation=ribs_separation,
                panel_max_length=panel_length,
                extra_sections=1,
            ),
        ],
    )

    # Disciplines
    # Aerodynamics
    disc_aero = Aerodynamics(
        name="Aerodynamics",
        components=[env, wing, engine_mass],
        solver=DUST(
            dust_lib.Options(
                work_dir="test_dust",
                t_end=dust_tf,
                dt=dust_delta_t,
                n_threads=30,
                n_wake_particles=100000,
                particles_box_min=np.array([-4.0, -15.0, -5.0]),
                particles_box_max=np.array([6.0, 15.0, 5.0]),
            ),
        ),
    )

    # Spanloads aggregator
    disc_spanloads = Aerodynamics(
        name="Spanloads",
        components=[wing],
        solver=LoadsAggregator(),
    )

    # Mass estimation
    disc_mass = WeightAndBalance(
        name="AMPET",
        components=[env, wing, engine_mass],
        solver=AMPET(),
    )

    # Geometry
    def geometry_function(span0):  # noqa: ANN001, ANN202
        span_1 = wing_span - span0[0]
        chord_tip = chord_root - wing_surf_minus / span_1
        return ([span_1], [chord_tip], span0 - 0.75)

    disc_geometry = UserDefined(
        name="Geometry",
        inputs=[span_0],
        outputs=[span_1, chord_tip, engine_pos],
        expression=geometry_function,
    )

    # Objective
    lift = copy.deepcopy(
        next(o for o in disc_aero.solver.outputs if o.name == f"{wing.name}.lift"),  # type: ignore[not-iterable]
    )
    drag = copy.deepcopy(
        next(o for o in disc_aero.solver.outputs if o.name == f"{wing.name}.drag"),  # type: ignore[not-iterable]
    )
    mass = copy.deepcopy(
        next(o for o in disc_mass.solver.outputs if o.name == f"{wing.name}.mass"),  # type: ignore[not-iterable]
    )

    def objective_function(lift, drag, m_wing):  # noqa: ANN001, ANN202
        drag_fr = 0.5 * env.density * env.speed**2 * wing_surf * 0.022
        eff = lift / (drag + drag_fr)

        trimming = (lift - 9.81 * mtow) / (9.81 * mtow)

        zfw = zfw_wing_excluded + m_wing
        distance = env.speed * eff / (9.81 * sfc) * np.log(mtow / zfw)

        return eff, trimming, distance

    disc_objective = UserDefined(
        name="Objective",
        inputs=[lift, drag, mass],
        outputs=[efficiency, lift_const, distance],
        expression=objective_function,
    )

    # Design space
    mads_scenario = MADSScenario()
    mads_scenario.fill_parameter_space(variables)
    mads_scenario.create_scenario(
        disciplines=[
            disc_geometry,
            disc_aero,
            disc_spanloads,
            disc_mass,
            disc_objective,
        ],
        formulation="DisciplinaryOpt",
        objective_name=distance.name,
        maximize_objective=True,
    )
    mads_scenario.set_differentiation_method("finite_differences", step=1e-3)

    # Constraints
    min_rfc = copy.deepcopy(
        next(o for o in disc_mass.solver.outputs if o.name == f"{wing.name}.min_rfc"),  # type: ignore[not-iterable]
    )
    min_rft = copy.deepcopy(
        next(o for o in disc_mass.solver.outputs if o.name == f"{wing.name}.min_rft"),  # type: ignore[not-iterable]
    )
    mads_scenario.add_constraints(
        [
            Constraint(
                output_name=min_rfc.name,
                constraint_type=ConstraintType.INEQ,
                value=1.5,
                positive=True,
            ),
            Constraint(
                output_name=min_rft.name,
                constraint_type=ConstraintType.INEQ,
                value=1.5,
                positive=True,
            ),
            Constraint(
                output_name=lift_const.name,
                constraint_type=ConstraintType.EQ,
            ),
        ],
    )

    # Observables
    mads_scenario.add_observables([mass.name, efficiency.name])

    # Run the scenario
    # TODO @Andres: Skip tests if DUST is not installed (TEMPORARY!!!!!)
    try:
        mads_scenario.execute(
            algo_name="COBYQA",
            max_iter=4,
            eq_tolerance=0.1,
            ineq_tolerance=0.1,
        )
        shutil.rmtree("test_dust")
    except FileNotFoundError:
        warn("Could not complete the test as DUST is not installed.", stacklevel=2)
        return

    # Postprocess
    data = mads_scenario.to_dataset()
    data_alpha = data["designs"][alpha.name][0]
    data_span = data["designs"][span_0.name][0]
    data_angle = data["designs"][angle_tip.name][0]
    data_mass = data["functions"][mass.name][0]
    data_lift = data["functions"][lift_const.name][0]
    data_eff = data["functions"][efficiency.name][0]

    assert_allclose(data_alpha, [3.0, 2.5, 10.0, 2.5])
    assert_allclose(data_span, [4.7, 6.0, 6.0, 10.0])
    assert_allclose(data_angle, [0.0, -10.0, -10.0, -10.0])
    assert_allclose(data_mass, [5030.401100, 3687.670344, 4433.595523, 5526.299731])
    assert_allclose(data_lift, [-0.323366, -0.606120, 0.568626, -0.373597], rtol=1e-6)
    assert_allclose(data_eff, [22.370911, 13.334772, 28.678773, 20.934993])
