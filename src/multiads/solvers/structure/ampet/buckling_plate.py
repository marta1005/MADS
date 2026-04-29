# ruff: noqa: N802, N803, N806

from collections.abc import Sequence

import numpy as np
import scipy as sp

from multiads.solvers.structure.ampet.structural_material import StructuralMaterial


class BucklingPlate:
    def __init__(
        self,
        lengthX: float,
        lengthY: float,
        material: StructuralMaterial,
        reserveFactor: float = 1.5,
        n: int = 3,
        m: int = 3,
        alpha: float = 0.7,
    ) -> None:
        self.nMax = n
        self.mMax = m

        self.CompressMatrix()
        self.BendingMatrix()
        self.ShearMatrix()

        self.material = material

        structuralCoef = (
            np.pi
            * np.pi
            * material.E
            / (12.0 * alpha * alpha * (1.0 - material.nu * material.nu) * reserveFactor)
        )

        beta = lengthY / lengthX

        self.shearCoef = self.ShearCoef(beta) * structuralCoef
        if beta >= 1.0:
            self.shearCoef = self.shearCoef / (lengthY * lengthY)
        else:
            self.shearCoef = self.shearCoef / (lengthX * lengthX)

        self.compressFCoefX = self.CompressCoef(beta) * structuralCoef / lengthY
        self.compressSCoefX = self.compressFCoefX / lengthY
        self.bendingCoefX = (
            self.BendingCoef(beta) * structuralCoef / (lengthY * lengthY)
        )

        beta = 1.0 / beta

        self.compressFCoefY = self.CompressCoef(beta) * structuralCoef / lengthX
        self.compressSCoefY = self.compressFCoefY / lengthX
        self.bendingCoefY = (
            self.BendingCoef(beta) * structuralCoef / (lengthX * lengthX)
        )

    def CreateMatrixS(
        self,
        beta: float,
    ) -> Sequence[Sequence[float]]:
        matrixS = [
            [0.0 for _ in range(self.nMax * self.mMax)]
            for _ in range(self.nMax * self.mMax)
        ]
        for iRow in range(self.nMax * self.mMax):
            n = (iRow % self.nMax) + 1
            m = np.floor(iRow / self.nMax) + 1
            matrixS[iRow][iRow] = (m * m + n * n * beta * beta) * (
                m * m + n * n * beta * beta
            )

        return matrixS

    def CompressMatrix(self) -> None:
        self.matrixKCompress = [
            [0.0 for _ in range(self.nMax * self.mMax)]
            for _ in range(self.nMax * self.mMax)
        ]

        for iRow in range(self.nMax * self.mMax):
            m = np.floor(iRow / self.nMax) + 1

            self.matrixKCompress[iRow][iRow] = m * m

    def BendingMatrix(self) -> None:
        self.matrixKBending = [
            [0.0 for _ in range(self.nMax * self.mMax)]
            for _ in range(self.nMax * self.mMax)
        ]

        for iRow in range(self.nMax * self.mMax):
            n = (iRow % self.nMax) + 1
            m = np.floor(iRow / self.nMax) + 1
            for iCol in range(self.nMax * self.mMax):
                q = (iCol % self.nMax) + 1
                p = np.floor(iCol / self.nMax) + 1

                if (m == p) and (abs(n - q) % 2 == 1):
                    self.matrixKBending[iRow][iCol] = (
                        16.0 * m * m * n * q / ((n * n - q * q) * (n * n - q * q))
                    ) / (np.pi * np.pi)

    def ShearMatrix(self) -> None:
        self.matrixKShear = [
            [0.0 for _ in range(self.nMax * self.mMax)]
            for _ in range(self.nMax * self.mMax)
        ]

        for iRow in range(self.nMax * self.mMax):
            n = (iRow % self.nMax) + 1
            m = np.floor(iRow / self.nMax) + 1
            for iCol in range(self.nMax * self.mMax):
                q = (iCol % self.nMax) + 1
                p = np.floor(iCol / self.nMax) + 1

                if (abs(m - p) % 2 == 1) and (abs(n - q) % 2 == 1):
                    self.matrixKShear[iRow][iCol] = (
                        32.0 * m * n * p * q / ((m * m - p * p) * (q * q - n * n))
                    ) / (np.pi * np.pi)

    def CompressCoef(self, beta: float) -> float:
        matrixS = self.CreateMatrixS(beta)

        kVector = sp.linalg.eig(np.array(matrixS), np.array(self.matrixKCompress))
        return min(abs(kVector[0])) / (beta * beta)

    def BendingCoef(self, beta: float) -> float:
        matrixS = self.CreateMatrixS(beta)

        kVector = sp.linalg.eig(np.array(matrixS), np.array(self.matrixKBending))
        return min(abs(kVector[0])) / (beta * beta)

    def ShearCoef(self, beta: float) -> float:
        if beta < 1.0:
            beta = 1.0 / beta

        matrixS = self.CreateMatrixS(beta)

        kVector = sp.linalg.eig(np.array(matrixS), np.array(self.matrixKShear))
        return min(abs(kVector[0])) / (beta * beta * beta)

    def CalculateSafety(self, iThickness: float) -> Sequence[float]:
        safety = (
            self.gammaCompressF * (iThickness * iThickness * iThickness)
            + self.gammaCompressS * (iThickness * iThickness)
            + self.gammaShear
            * self.gammaShear
            * (
                iThickness
                * iThickness
                * iThickness
                * iThickness
                * iThickness
                * iThickness
            )
            + self.gammaBending
            * self.gammaBending
            * (iThickness * iThickness * iThickness * iThickness)
        )

        dsafetydt = (
            3.0 * self.gammaCompressF * (iThickness * iThickness)
            + 2.0 * self.gammaCompressS * iThickness
            + 6.0
            * self.gammaShear
            * self.gammaShear
            * (iThickness * iThickness * iThickness * iThickness * iThickness)
            + 4.0
            * self.gammaBending
            * self.gammaBending
            * (iThickness * iThickness * iThickness * iThickness)
        )

        return [safety, dsafetydt]

    def CalculateThickness(
        self,
        q: float,
        compressionX: float,
        sigma0X: float,
        sigmaFX: float,
        compressionY: float,
        sigma0Y: float,
        sigmaFY: float,
    ) -> float:
        self.gammaShear = q / self.shearCoef

        self.gammaCompressF = (
            compressionX / self.compressFCoefX + compressionY / self.compressFCoefY
        )

        sigmaCX = (sigma0X + sigmaFX) / 2.0
        sigmaBX = abs(sigma0X - sigmaFX) / 2.0

        sigmaCY = (sigma0Y + sigmaFY) / 2.0
        sigmaBY = abs(sigma0Y - sigmaFY) / 2.0

        self.gammaCompressS = (
            sigmaCX / self.compressSCoefX + sigmaCY / self.compressSCoefY
        )
        self.gammaBending = sigmaBX / self.bendingCoefX + sigmaBY / self.bendingCoefY

        iThickness = 1.0 / self.material.thicknessMin

        [safety, dsafetydt] = self.CalculateSafety(iThickness)

        while safety > 1.0:
            while abs(safety - 1.0) > 1.0e-6:  # noqa: PLR2004
                diT = (1.0 - safety) / dsafetydt
                iThickness += diT
                [safety, dsafetydt] = self.CalculateSafety(iThickness)

            thickness = 1.0 / iThickness + self.material.thicknessStep
            iThickness = 1.0 / thickness
            [safety, dsafetydt] = self.CalculateSafety(iThickness)

        return self.material.adjustThickness(1.0 / iThickness)
