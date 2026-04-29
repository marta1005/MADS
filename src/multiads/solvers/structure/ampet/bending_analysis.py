# ruff: noqa: N802, N803, N806

from collections.abc import Sequence

from multiads.solvers.structure.ampet.structural_material import StructuralMaterial


class BendingAnalysis:
    def __init__(
        self,
        Sy: float,
        Mx: float,
        Mz: float,
        inertiaVector: list[float],
        material: StructuralMaterial,
    ) -> None:
        self.E = material.E

        xCdA = inertiaVector[1] / inertiaVector[0]
        zCdA = inertiaVector[2] / inertiaVector[0]

        Ixx = inertiaVector[5] - zCdA * zCdA * inertiaVector[0]
        Pxz = inertiaVector[4] - xCdA * zCdA * inertiaVector[0]
        Izz = inertiaVector[3] - xCdA * xCdA * inertiaVector[0]

        Mx = Mx + Sy * zCdA
        Mz = Mz - Sy * xCdA

        inertiaFactor = 1.0 / (self.E * (Ixx * Izz - Pxz * Pxz))

        self.sRho = inertiaFactor * (Pxz * Mx + Ixx * Mz)
        self.cRho = inertiaFactor * (-Izz * Mx - Pxz * Mz)

        self.xCdA = xCdA
        self.zCdA = zCdA

        self.sigmaBase = Sy / inertiaVector[0]

    def sigma(
        self,
        x: float,
        z: float,
    ) -> float:
        eps = (x - self.xCdA) * self.sRho + (z - self.zCdA) * self.cRho

        return self.sigmaBase + self.E * eps

    def brazierFV(
        self,
        x: float,
        z: float,
    ) -> Sequence[float]:
        sigma = self.sigma(x, z)

        return [sigma * self.sRho, -sigma * self.cRho]
