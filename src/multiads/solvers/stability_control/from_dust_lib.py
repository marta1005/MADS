from __future__ import annotations

from enum import Enum

from multiads.solvers import SolverOptions


class SnCAnalysisType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    CONTROL = "control"
    ZERO_ATTITUDE = "zero_attitude"


class SnCDerivativeType(Enum):
    ALPHA = "alpha"
    BETA = "beta"
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"
    PLUNGE = "plunge"
    LATERAL = "lateral"
    PHUGOID = "phugoid"
    LATERAL_PHUGOID = "lateral_phugoid"


class Options(SolverOptions):
    def __init__(
        self,
        analysis_type: SnCAnalysisType | None = None,
        derivative_type: SnCDerivativeType | None = None,
        d_aoa: float = 0.0,
        d_aob: float = 0.0,
        derivative_ampl: float = 0.0,
        derivative_omega: float = 0.0,
        derivative_n_periods: int = 1,
        derivative_n_start: int = 1,
    ) -> None:
        self.snc_analysis_type = analysis_type
        self.snc_derivative_type = derivative_type
        self.d_aoa = d_aoa
        self.d_aob = d_aob
        self.derivative_ampl = derivative_ampl
        self.derivative_omega = derivative_omega
        self.derivative_n_periods = derivative_n_periods
        self.derivative_n_start = derivative_n_start


class DUSTAeroDerivatives:
    """Extract aerodynamic derivatives for S&C.

    Perform differential or oscillatory simulations needed to extract the values of the
    aerodynamic stability and control derivatives.
    """

    def run(self) -> None:
        # select analysis type
        analysis_type = self.options.snc_analysis_type

        if analysis_type == SnCAnalysisType.static:
            # Compute current values
            self.options.output_dir = (
                self.options.work_dir / self.options.output_dir / "output_1"
            )
            self.compute_dust()

            # Apply change of attitude (through velocity direction)
            if self.options.snc_derivative_type == SnCDerivativeType.alpha:
                d_aoa = self.options.d_aoa
                self.environment.alpha += d_aoa
            elif self.options.snc_derivative_type == SnCDerivativeType.beta:
                d_aob = self.options.d_aob
                self.environment.beta += d_aob

            # Perform perturbed simulation
            self.options.output_dir = (
                self.options.work_dir / self.options.output_dir / "output_2"
            )
            self.compute_dust()

            # restore attitude
            if self.options.snc_derivative_type == SnCDerivativeType.alpha:
                self.environment.alpha -= d_aoa
            elif self.options.snc_derivative_type == SnCDerivativeType.beta:
                self.environment.beta -= d_aob

        if analysis_type == SnCAnalysisType.control:
            # Compute current values
            self.options.output_dir = (
                self.options.work_dir / self.options.output_dir / "output_1"
            )
            self.compute_dust()

            # cycle on all wings
            for wing in self.wings:
                wing_name = wing.name
                movable_surfaces = wing.movable_surfaces

                # cycle on all movable_surfaces of the current wing
                for i, h in enumerate(movable_surfaces):
                    if h.derivative:
                        dDelta = h.dDelta

                        # apply change of control surface deflection
                        movable_surfaces[i].ampl += dDelta

                        # Perform perturbed simulation
                        self.options.output_dir = (
                            self.options.work_dir
                            / self.options.output_dir
                            / f"output_2_{wing_name}_movable_surface{i}"
                        )
                        self.compute_dust()

                        # reset control surface deflection
                        movable_surfaces[i].ampl -= dDelta

        if analysis_type == SnCAnalysisType.dynamic:
            # Perform single dynamic computation
            self.compute_dust()

        if analysis_type == SnCAnalysisType.zero_attitude:
            # set zero-attitude condition
            self.environment.alpha = 0
            self.environment.beta = 0

            # Perform single static computation
            self.compute_dust()

        else:
            msg = "SnC analysis not specified for stability computation."
            raise ValueError(msg)

    def compute_dust(self) -> None:
        self.preprocess()
        self.run()


def _get_motion_params(  # noqa: PLR0915
    self,
) -> tuple[
    NDArray[np.float64],
    float,
    float,
    float,
    NDArray[np.float64],
    float,
    float,
    float,
]:
    rotation_dir = np.array([1.0, 0.0, 0.0])
    rotation_ampl = 1.0
    rotation_omega = 1.0
    rotation_phase = 0.0
    pole_dir = np.array([1.0, 0.0, 0.0])
    pole_ampl = 1.0
    pole_omega = 1.0
    pole_phase = 0.0

    if self.options.snc_derivative_type == SnCDerivativeType.roll:
        rotation_dir = np.array([1.0, 0.0, 0.0])
        rotation_ampl = self.options.derivative_ampl
        rotation_omega = self.options.derivative_omega

    elif self.options.snc_derivative_type == SnCDerivativeType.pitch:
        rotation_dir = np.array([0.0, 1.0, 0.0])
        rotation_ampl = self.options.derivative_ampl
        rotation_omega = self.options.derivative_omega

    elif self.options.snc_derivative_type == SnCDerivativeType.yaw:
        rotation_dir = np.array([0.0, 0.0, 1.0])
        rotation_ampl = self.options.derivative_ampl
        rotation_omega = self.options.derivative_omega

    elif self.options.snc_derivative_type == SnCDerivativeType.plunge:
        pole_dir = np.array([0.0, 0.0, 1.0])
        pole_ampl = self.options.derivative_ampl
        pole_omega = self.options.derivative_omega
        pole_phase = -np.pi / 2
        rotation_dir = np.array([0.0, 0.0, 1.0])
        rotation_ampl = 0.0
        rotation_omega = 0.0

    elif self.options.snc_derivative_type == SnCDerivativeType.lateral:
        pole_dir = np.array([0.0, 1.0, 0.0])
        pole_ampl = self.options.derivative_ampl
        pole_omega = self.options.derivative_omega
        pole_phase = np.pi / 2
        rotation_dir = np.array([0.0, 0.0, 1.0])
        rotation_ampl = 0.0
        rotation_omega = 0.0

    elif self.options.snc_derivative_type == SnCDerivativeType.phugoid:
        pole_dir = np.array([0.0, 0.0, 1.0])
        rotation_dir = np.array([0.0, 1.0, 0.0])
        pole_ampl = self.options.derivative_ampl
        rotation_ampl = (
            self.options.derivative_ampl
            * self.options.derivative_omega
            / self.environment.speed
        )
        pole_omega = self.options.derivative_omega
        rotation_omega = self.options.derivative_omega
        pole_phase = np.pi / 2
        rotation_phase = 0.0

    elif self.options.snc_derivative_type == SnCDerivativeType.lateral_phugoid:
        pole_dir = np.array([0.0, 1.0, 0.0])
        rotation_dir = np.array([0.0, 0.0, 1.0])
        pole_ampl = -self.options.derivative_ampl
        rotation_ampl = (
            -self.options.derivative_ampl
            * self.options.derivative_omega
            / self.environment.speed
        )
        pole_omega = self.options.derivative_omega
        rotation_omega = -self.options.derivative_omega
        pole_phase = np.pi / 2
        rotation_phase = 0.0

    return (
        pole_dir,
        pole_ampl,
        pole_omega,
        pole_phase,
        rotation_dir,
        rotation_ampl,
        rotation_omega,
        rotation_phase,
    )
