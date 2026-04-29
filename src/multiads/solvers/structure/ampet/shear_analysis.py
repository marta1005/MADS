# ruff: noqa: N802, N803, N806

import numpy as np

from multiads.solvers.structure.ampet.structural_profile import StructuralProfile


class ShearAnalysis:
    def __init__(self, profile: StructuralProfile) -> None:
        self.nWebs = profile.nStringerLower + profile.nStringerUpper + 4

        self.x = profile.xCaps
        self.z = profile.yCaps

        self.LBox = [0.0 for _ in range(self.nWebs)]
        self.sBox = [0.0 for _ in range(self.nWebs)]

        self.LBoxTotal = 0.0

        self.VTorsion = [0.0 for _ in range(3)]

        for iStep in range(profile.nStringerLower + 1):
            self.LBoxTotal -= profile.lowerBoxBroken[iStep].shearProperties[0]
            self.LBox[iStep] = -profile.lowerBoxBroken[iStep].shearProperties[0]
            self.sBox[iStep] = profile.lowerBoxBroken[iStep].length

        self.iRearSpar = profile.nStringerLower + 1
        self.LBoxTotal -= profile.rearSpar.shearProperties[0]
        self.LBox[self.iRearSpar] = -profile.rearSpar.shearProperties[0]
        self.sBox[self.iRearSpar] = profile.rearSpar.length

        for iStep in range(profile.nStringerUpper + 1):
            self.LBoxTotal -= profile.upperBoxBroken[iStep].shearProperties[0]
            self.LBox[self.iRearSpar + 1 + iStep] = -profile.upperBoxBroken[
                iStep
            ].shearProperties[0]
            self.sBox[self.iRearSpar + 1 + iStep] = profile.upperBoxBroken[iStep].length

        self.iFrontSpar = profile.nStringerLower + profile.nStringerUpper + 3
        self.LBoxTotal -= profile.frontSpar.shearProperties[0]
        self.LBox[self.iFrontSpar] = -profile.frontSpar.shearProperties[0]
        self.sBox[self.iFrontSpar] = profile.frontSpar.length

        self.LLeadingEdge = -profile.leadingEdge.shearProperties[0]
        self.sLeadingEdge = profile.leadingEdge.length
        self.LUpperLeadingEdge = -profile.upperTrailingEdge.shearProperties[0]
        self.sUpperLeadingEdge = profile.upperTrailingEdge.length
        self.LLowerLeadingEdge = -profile.lowerTrailingEdge.shearProperties[0]
        self.sLowerLeadingEdge = -profile.lowerTrailingEdge.length

        self.VTorsion[0] = self.LLeadingEdge - self.LBox[self.iFrontSpar]
        self.VTorsion[1] = self.LBoxTotal
        self.VTorsion[2] = (
            self.LUpperLeadingEdge + self.LLowerLeadingEdge - self.LBox[self.iRearSpar]
        )

    def solveShearForce(
        self,
        ACaps: list[float],
        inertiaFactor: float,
        xCdA: float,
        zCdA: float,
        xFactor: float,
        zFactor: float,
    ) -> tuple[list[float], float]:
        qBasic = [0.0 for _ in range(self.nWebs)]
        qBasic[0] = 0.0
        q0 = 0.0

        for iWall in range(1, self.nWebs):
            qBasic[iWall] = qBasic[iWall - 1] + ACaps[iWall] * inertiaFactor * (
                (self.x[iWall] - xCdA) * xFactor + (self.z[iWall] - zCdA) * zFactor
            )
            q0 += qBasic[iWall] * self.LBox[iWall]

        q0 = -q0 / self.LBoxTotal

        return qBasic, q0

    def prepareBasicSolution(
        self,
        inertiaVector: list[float],
        ACaps: list[float],
    ) -> None:
        xCdA = inertiaVector[1] / inertiaVector[0]
        zCdA = inertiaVector[2] / inertiaVector[0]

        Ixx = inertiaVector[5] - zCdA * zCdA * inertiaVector[0]
        Pxz = inertiaVector[4] - xCdA * zCdA * inertiaVector[0]
        Izz = inertiaVector[3] - xCdA * xCdA * inertiaVector[0]

        inertiaFactor = 1.0 / (Ixx * Izz - Pxz * Pxz)

        self.qShearX = self.solveShearForce(ACaps, inertiaFactor, xCdA, zCdA, -Ixx, Pxz)
        self.qShearZ = self.solveShearForce(ACaps, inertiaFactor, xCdA, zCdA, Pxz, -Izz)
        self.qShearY = [(1.0 / self.LBoxTotal) for _ in range(self.nWebs)]

    def calculateQBox(self, Sx: float, Sz: float, My: float) -> list[float]:
        solution = [0.0 for _ in range(self.nWebs)]

        for iWall in range(self.nWebs):
            solution[iWall] = (
                (self.qShearX[0][iWall] + self.qShearX[1]) * Sx
                + (self.qShearZ[0][iWall] + self.qShearZ[1]) * Sz
                + self.qShearY[iWall] * My
            )

        return solution

    def calculateQProfile(
        self,
        Sx: float,
        Sz: float,
        My: float,
        tWebsBox: list[float],
        tLeadingEdge: float,
        tUpperTrailingEdge: float,
        tLowerTrailingEdge: float,
    ) -> list[float]:
        JTorsion = [0.0 for _ in range(3)]
        UTorsion = [0.0 for _ in range(3)]
        HTorsion = [0.0 for _ in range(3)]

        Q = -(
            self.qShearX[0][self.iFrontSpar] * Sx
            + self.qShearZ[0][self.iFrontSpar] * Sz
            + self.qShearY[self.iFrontSpar] * My
        )
        JTorsion[0] = (
            self.sLeadingEdge / tLeadingEdge
            - self.sBox[self.iFrontSpar] / tWebsBox[self.iFrontSpar]
        )
        UTorsion[0] = Q * self.LBox[self.iFrontSpar]
        HTorsion[0] = Q * self.sBox[self.iFrontSpar] / tWebsBox[self.iFrontSpar]

        for iWall in range(self.nWebs):
            Q = (
                self.qShearX[0][iWall] * Sx
                + self.qShearZ[0][iWall] * Sz
                + self.qShearY[iWall] * My
            ) * self.LBox[iWall]
            JTorsion[1] += self.sBox[iWall] / tWebsBox[iWall]
            UTorsion[1] += Q * self.LBox[iWall]
            HTorsion[1] += Q * self.sBox[iWall] / tWebsBox[iWall]

        Q = -(
            self.qShearX[0][self.iRearSpar] * Sx
            + self.qShearZ[0][self.iRearSpar] * Sz
            + self.qShearY[self.iRearSpar] * My
        )
        JTorsion[2] = (
            self.sUpperLeadingEdge / tUpperTrailingEdge
            + self.sLowerLeadingEdge / tLowerTrailingEdge
            - self.sBox[self.iRearSpar] / tWebsBox[self.iRearSpar]
        )
        UTorsion[2] = Q * self.LBox[self.iRearSpar]
        HTorsion[2] = Q * self.sBox[self.iRearSpar] / tWebsBox[self.iRearSpar]

        matrix = np.array(
            [
                [JTorsion[0], -JTorsion[1], 0.0],
                [JTorsion[0], 0.0, -JTorsion[2]],
                [self.VTorsion[0], self.VTorsion[1], self.VTorsion[2]],
            ],
        )
        vector = np.array(
            [
                HTorsion[1] - HTorsion[0],
                HTorsion[2] - HTorsion[0],
                My - UTorsion[0] - UTorsion[1] - UTorsion[2],
            ],
        )

        q0 = np.linalg.solve(matrix, vector)

        return [q0[0].item(), q0[2].item()]

    def CalculateShearParameters(
        self,
        tWebsBox: list[float],
    ) -> list[float]:
        boxArea = self.LBoxTotal

        intJ = 0.0

        intS10 = 0.0
        intS01 = 0.0
        intS11 = 0.0

        thetaS10 = 0.0
        thetaS01 = 0.0
        thetaS11 = 0.0

        for iWall in range(self.nWebs):
            intJ += self.sBox[iWall] / tWebsBox[iWall]
            q10 = self.qShearZ[0][iWall] + self.qShearZ[1]
            q01 = self.qShearX[0][iWall] + self.qShearX[1]
            q11 = q10 + q01
            sPt = self.sBox[iWall] / tWebsBox[iWall]
            intS10 += q10 * q10 * sPt
            intS01 += q01 * q01 * sPt
            intS11 += q11 * q11 * sPt
            thetaS10 += q10 * sPt
            thetaS01 += q01 * sPt
            thetaS11 += q11 * sPt

        J = 4.0 * boxArea * boxArea / intJ
        thetaS10 /= 2.0 * boxArea
        thetaS01 /= 2.0 * boxArea
        thetaS11 /= 2.0 * boxArea

        intS10 -= thetaS10 * thetaS10 * J
        intS01 -= thetaS01 * thetaS01 * J
        intS11 -= thetaS11 * thetaS11 * J

        s11 = intS10
        s22 = intS01
        s12 = (intS11 - intS10 - intS01) / 2.0

        beta = np.arctan2(2 * s12, (s22 - s11))
        cBeta = np.cos(beta)
        sBeta = np.sin(beta)

        matrix = np.array(
            [
                [cBeta * cBeta, sBeta * sBeta],
                [sBeta * sBeta, cBeta * cBeta],
            ],
        )
        vector = np.array([s11, s22])
        vA = np.linalg.solve(matrix, vector)

        matrix = np.array(
            [
                [sBeta * vA[1], -cBeta * vA[0]],
                [cBeta * vA[1], sBeta * vA[0]],
            ],
        )
        vector = np.array([thetaS10, thetaS01])
        vX = np.linalg.solve(matrix, vector)

        Az = vA[0]
        Ax = vA[1]

        zSC = vX[0]
        xSC = vX[1]

        return [xSC, zSC, J, Ax, Az, sBeta, cBeta]
