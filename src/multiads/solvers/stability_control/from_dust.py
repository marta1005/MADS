def compute_derivative(name, driver, **kwargs) -> NDArray[np.float64]:
    """Compute aerodynamic force and moment derivatives as a 6-component vector."""
    if (
        driver.options.analysis_type == "static"
        or driver.options.analysis_type == "control"
    ):
        # Save derivatives inputs from kwargs
        A_ref = kwargs.get("A_ref", 1.0)
        l_ref = kwargs.get("l_ref", 1.0)
        wing_name = kwargs.get("wing_name", 1.0)
        movable_surface_index = kwargs.get("movable_surface_index", 1)
        exclude_keys = {"A_ref", "l_ref", "wing_name", "movable_surface_index"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}

        # save external directory path
        ext_output_dir = driver.options.output_dir

        # run dust postpro for the first point of finite difference
        driver.options.output_dir = (
            driver.options.work_dir / ext_output_dir / "output_1"
        )
        forces_1 = dl.DUSTpost_loads(name, average=True, **filtered_kwargs)
        driver.postprocess([forces_1])
        m1, f1 = forces_1.m[-1], forces_1.f[-1]

        # enter the selected type of analysis
        if driver.options.analysis_type == "static":
            if driver.options.derivative_type == "alpha":
                delta = driver.options.dAoA
                output_2 = "output_2"
            elif driver.options.derivative_type == "beta":
                delta = driver.options.dAoB
                output_2 = "output_2"
        elif driver.options.analysis_type == "control":
            wing = next((w for w in driver.wings if w.name == wing_name), None)

            # check the consistency of nomenclature
            if wing is None:
                raise ValueError(f"Wing '{wing_name}' not found in driver.wings")

            delta = wing.movable_surfaces[movable_surface_index].dDelta
            output_2 = f"output_2_{wing_name}_movable_surface{movable_surface_index}"

        # run dust postpro for the second point of finite difference
        driver.options.output_dir = driver.options.work_dir / ext_output_dir / output_2
        forces_2 = dl.DUSTpost_loads(name, average=True, **filtered_kwargs)
        driver.postprocess([forces_2])
        m2, f2 = forces_2.m[-1], forces_2.f[-1]

        # restore external output directory name
        driver.options.output_dir = ext_output_dir

        # Compute finite difference
        df = (np.asarray(f2) - np.asarray(f1)) / (delta * np.pi / 180)
        dm = (np.asarray(m2) - np.asarray(m1)) / (delta * np.pi / 180)

        # Compute denominator
        q = (
            0.5
            * driver.environment.density
            * (np.linalg.norm(driver.environment.velocity)) ** 2
        )
        f_denominator = q * A_ref
        m_denominator = q * A_ref * l_ref

        # Compute non-dimensionalized derivatives
        f_derivatives = df / f_denominator
        m_derivatives = dm / m_denominator

    if driver.options.analysis_type == "dynamic":
        # Save derivatives inputs from kwargs
        A_ref = kwargs.get("A_ref", 1.0)
        l_ref = kwargs.get("l_ref", 1.0)
        exclude_keys = {"A_ref", "l_ref", "wing_name", "movable_surface_index"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}

        # extract dynamic motion data
        omega = driver.options.derivative_omega
        A = driver.options.derivative_ampl
        f = omega / (2 * np.pi)
        k = l_ref * omega / np.linalg.norm(driver.environment.velocity)
        T = 1 / f
        N_start = driver.options.derivative_N_start
        N_per = driver.options.derivative_N_periods
        dt = driver.options.dt

        # run postpro
        forces = dl.DUSTpost_loads(name, average=False, **filtered_kwargs)
        driver.postprocess([forces])
        t, f, m = forces.t, forces.f, forces.m

        # Compute denominator
        q = (
            0.5
            * driver.environment.density
            * (np.linalg.norm(driver.environment.velocity)) ** 2
        )
        f_denominator = q * A_ref
        m_denominator = q * A_ref * l_ref

        # Compute non-dimensionalized coefficients
        f_coefficient = f / f_denominator
        m_coefficient = m / m_denominator

        # Compute truncated time histories
        i_start = int(np.ceil(N_start * (T / dt))) - 1
        i_end = int(np.ceil(N_start * (T / dt))) + int(np.ceil(N_per * (T / dt)))
        t = np.array(t)
        t = t[i_start:i_end]
        f_coefficient = f_coefficient[i_start:i_end]
        m_coefficient = m_coefficient[i_start:i_end]

        # Compute derivative with the single point method
        f_derivatives = (
            f_coefficient[-1] - f_coefficient[int(np.ceil(T / dt) // 2)]
        ) / (2 * k * A)
        m_derivatives = (
            m_coefficient[-1] - m_coefficient[int(np.ceil(T / dt) // 2)]
        ) / (2 * k * A)

        if (
            driver.options.derivative_type == "plunge"
            or driver.options.derivative_type == "lateral"
            or driver.options.derivative_type == "phugoid"
            or driver.options.derivative_type == "lateral_phugoid"
        ):
            f_derivatives = (
                f_derivatives * (np.linalg.norm(driver.environment.velocity)) / omega
            )
            m_derivatives = (
                m_derivatives * (np.linalg.norm(driver.environment.velocity)) / omega
            )

        # Debug (Fourier coefficient method)
        # f_derivatives_fc = (2 / (k * A * N_per * T)) * np.trapz(f_coefficient * np.cos(omega * t)[:, np.newaxis], t, axis=0)
        # m_derivatives_fc = (2 / (k * A * N_per * T)) * np.trapz(m_coefficient * np.cos(omega * t)[:, np.newaxis], t, axis=0)
        # print(i_end)
        # print(i_start)
        # print(f_derivatives_fc)
        # print(m_derivatives_fc)
        # print(f"Derivatives: {f_derivatives}, {m_derivatives}")

    if driver.options.analysis_type == "zero_attitude":
        # Save derivatives inputs from kwargs
        A_ref = kwargs.get("A_ref", 1.0)
        l_ref = kwargs.get("l_ref", 1.0)
        exclude_keys = {"A_ref", "l_ref", "wing_name", "movable_surface_index"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}

        # run dust postpro
        forces = dl.DUSTpost_loads(name, average=True, **filtered_kwargs)
        driver.postprocess([forces])
        m, f = forces.m[-1], forces.f[-1]

        # Compute denominator
        q = (
            0.5
            * driver.environment.density
            * (np.linalg.norm(driver.environment.velocity)) ** 2
        )
        f_denominator = q * A_ref
        m_denominator = q * A_ref * l_ref

        # Compute non-dimensionalized coefficients at zero-attitude
        f_derivatives = f / f_denominator
        m_derivatives = m / m_denominator

    # return concatenated value
    return np.concatenate((f_derivatives, m_derivatives))


def dCx(name, driver, **kwargs):
    return np.array([compute_derivative(name, driver, **kwargs)[0]])


def dCy(name, driver, **kwargs):
    return np.array([compute_derivative(name, driver, **kwargs)[1]])


def dCz(name, driver, **kwargs):
    return np.array([compute_derivative(name, driver, **kwargs)[2]])


def dCl(name, driver, **kwargs):
    return np.array([compute_derivative(name, driver, **kwargs)[3]])


def dCm(name, driver, **kwargs):
    return np.array([compute_derivative(name, driver, **kwargs)[4]])


def dCn(name, driver, **kwargs):
    return np.array([compute_derivative(name, driver, **kwargs)[5]])


def aero_derivatives(name, driver, **kwargs):
    return np.concatenate(
        [
            dCx(name, driver, **kwargs),
            dCy(name, driver, **kwargs),
            dCz(name, driver, **kwargs),
            dCl(name, driver, **kwargs),
            dCm(name, driver, **kwargs),
            dCn(name, driver, **kwargs),
        ],
    )


def NP(name, driver, **kwargs) -> NDArray[np.float64]:
    # by definition, to compute the neutral point, this analysis is necessary:
    if (
        driver.options.analysis_type == "static"
        and driver.options.derivative_type == "alpha"
    ):
        # compute neutral point from derivatives (moment transport formula) --> """ control fixed neutral point """
        l_ref = kwargs.get("l_ref", 1.0)
        neutral_point = -(l_ref * dCm(name, driver, **kwargs)) / dCz(
            name,
            driver,
            **kwargs,
        )

        return neutral_point


class AeroDerivatives:
    implemented_outputs = {
        "dCx": dCx,
        "dCy": dCy,
        "dCz": dCz,
        "dCl": dCl,
        "dCm": dCm,
        "dCn": dCn,
        "AeroDerivatives": aero_derivatives,
        "NP": NP,
    }

    def _run(self) -> None:
        analysis_type = self.options.analysis_type

        if (
            analysis_type == "static"
            or analysis_type == "dynamic"
            or analysis_type == "zero_attitude"
            or analysis_type == "control"
        ):
            # Build the driver for the derivatives
            self.driver = dl.DUSTAeroDerivatives(
                environment=self.environment,
                propellers=self.propellers,
                wings=self.wings,
                fuselage=self.fuselage,
                options=self.options,
            )
            # Execute DUST driver
            self.driver()

    def compute_output(self) -> None:
        if analysis_type == "control":
            # extract the selected movable_surface on the selected wing:
            wing_name = out_options.get("wing_name")
            movable_surface_index = out_options.get("movable_surface_index")
            output_key = (
                f"{out_type}_{wing_name}_movable_surface{movable_surface_index}"
            )

            # Callback
            try:
                outputs[output_key] = out_function(
                    out,
                    self.driver,
                    **out_options,
                    **out_parameters,
                )
            except KeyError:
                raise ValueError(
                    f"'{type(self).__name__}' cannot compute '{out_type}",
                )

            # Debug
            print(f"{output_key} = {outputs[output_key]}")
