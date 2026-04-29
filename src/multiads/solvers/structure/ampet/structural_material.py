# ruff: noqa: N802, N803, N806

import math


class StructuralMaterial:
    def __init__(
        self,
        rho: float,
        E: float,
        nu: float,
        thicknessMin: float,
        thicknessStep: float,
        allowableMaxRoot: float,
        allowableMaxTip: float,
        allowableShear: float,
        allowableElastic: float,
        metal: bool = True,
    ) -> None:
        self.rho = rho
        self.E = E
        self.nu = nu
        self.thicknessMin = thicknessMin
        self.thicknessStep = thicknessStep
        self.G = E / (2.0 * (1.0 + nu))

        if metal:
            self.sigmaMaxRoot = allowableMaxRoot
            self.sigmaMaxTip = allowableMaxTip
            self.tauMax = allowableShear
            self.sigma02 = allowableElastic
        else:
            self.sigmaMaxRoot = allowableMaxRoot * E
            self.sigmaMaxTip = allowableMaxTip * E
            self.tauMax = allowableShear * E
            self.sigma02 = allowableElastic * E

    def sigmaAllowable(self, position: float, length: float) -> float:
        u = position / length
        return self.sigmaMaxRoot * (1.0 - u) + self.sigmaMaxTip * u

    def adjustThickness(self, t0: float) -> float:
        t = math.ceil(t0 / self.thicknessStep) * self.thicknessStep
        return max(t, self.thicknessMin)

    def birdStrikeThickness(
        self,
        leadingEdgeRadius: float,
        leadingEdgeAngle: float,
        vCruise: float,
    ) -> float:
        massBird = 8.0 * 0.4536  # 8 pounds
        vImpact = vCruise

        cosThetaImpact = math.cos(leadingEdgeAngle)

        # FROM NT-LA-ADS-94004
        factorMaterial = (
            self.E / 1e7 * (self.rho * self.rho / 1e6)
        )  # E in hbar and rho in kg/dm3
        factorAngle = cosThetaImpact ** (-4.0 / 3.0)
        radius = leadingEdgeRadius * 1000.0  # radius in mm
        factorRadius = math.exp(1700.0 / (radius * radius + 30.0 * radius + 1000.0))

        UImpact = massBird * vImpact * vImpact / 2.0
        UAbsorption_t2 = (
            27.9
            * factorMaterial
            * (massBird ** (1.0 / 3.0))
            * factorAngle
            * factorRadius
            / 1000.0
        )

        thickness = (
            math.sqrt(UImpact / UAbsorption_t2) / 1000.0
        )  # this formula is for t in mm

        return self.adjustThickness(thickness)
