# ruff: noqa: N802, N803, N806

from __future__ import annotations

import numpy as np

from multiads.solvers.structure.ampet.poly_line_2d import PolyLine2D


class StructuralProfile:
    def __init__(
        self,
        airfoil: str | None = None,
        frontSparPosition: float = 0.25,
        rearSparPosition: float = 0.75,
    ) -> None:
        if airfoil is None:
            aPoly = PolyLine2D(
                [[0.75, 0.5], [-0.15, 0.5], [-0.25, 0.0], [-0.15, -0.5], [0.75, -0.5]],
            )
        else:
            airfoilCoordinates = np.loadtxt(airfoil, skiprows=1)
            aPoly = PolyLine2D(airfoilCoordinates.tolist())
            factorX = 1.0 / (aPoly.maxX - aPoly.minX)
            factorY = 1.0 / (aPoly.maxY - aPoly.minY)
            aPoly = aPoly.rescaleAndMove(factorX, factorY, 0.25, 0.0)

        self.frontSparPosition = frontSparPosition - 0.25
        self.rearSparPosition = rearSparPosition - 0.25

        aPoly1 = aPoly.breakByXPosition(self.rearSparPosition)
        aPoly2 = aPoly1[1].breakByXPosition(self.frontSparPosition)
        aPoly3 = aPoly2[1].reverse().breakByXPosition(self.frontSparPosition)
        aPoly3[0] = aPoly3[0].reverse()
        aPoly3[1] = aPoly3[1].reverse()
        aPoly4 = aPoly3[0].breakByXPosition(self.rearSparPosition)

        if aPoly1[1].y[0] > 0.0:
            self.upperTrailingEdge = aPoly1[0]
            self.upperBox = aPoly2[0]
            self.leadingEdge = aPoly3[1]
            self.lowerBox = aPoly4[0]
            self.lowerTrailingEdge = aPoly4[1]
        else:
            self.lowerTrailingEdge = aPoly1[0].reverse()
            self.lowerBox = aPoly2[0].reverse()
            self.leadingEdge = aPoly3[1].reverse()
            self.upperBox = aPoly4[0].reverse()
            self.upperTrailingEdge = aPoly4[1].reverse()

        self.frontSpar = PolyLine2D(
            [
                [self.leadingEdge.x[0], self.leadingEdge.y[0]],
                [self.lowerBox.x[0], self.lowerBox.y[0]],
            ],
        )
        self.rearSpar = PolyLine2D(
            [
                [self.lowerTrailingEdge.x[0], self.lowerTrailingEdge.y[0]],
                [self.upperBox.x[0], self.upperBox.y[0]],
            ],
        )

        self.leadingEdgePointPosition = int(np.argmin(self.leadingEdge.x))
        self.leadingRadius = self.leadingEdge.radiusInPosition(
            self.leadingEdgePointPosition,
        )

        self.leadingArea = self.leadingEdge.area
        self.boxArea = self.upperBox.area + self.lowerBox.area
        self.trailingArea = self.upperTrailingEdge.area + self.lowerTrailingEdge.area

    def rescale(self, chord: float, relativeThickness: float) -> StructuralProfile:
        solution = StructuralProfile()

        solution.frontSparPosition = self.frontSparPosition * chord
        solution.rearSparPosition = self.rearSparPosition * chord

        solution.leadingEdgePointPosition = self.leadingEdgePointPosition
        solution.leadingEdge = self.leadingEdge.rescaleAndMove(
            chord,
            chord * relativeThickness,
        )
        solution.leadingRadius = solution.leadingEdge.radiusInPosition(
            solution.leadingEdgePointPosition,
        )

        solution.frontSpar = self.frontSpar.rescaleAndMove(
            chord,
            chord * relativeThickness,
        )
        solution.rearSpar = self.rearSpar.rescaleAndMove(
            chord,
            chord * relativeThickness,
        )

        solution.upperBox = self.upperBox.rescaleAndMove(
            chord,
            chord * relativeThickness,
        )
        solution.upperTrailingEdge = self.upperTrailingEdge.rescaleAndMove(
            chord,
            chord * relativeThickness,
        )

        solution.lowerBox = self.lowerBox.rescaleAndMove(
            chord,
            chord * relativeThickness,
        )
        solution.lowerTrailingEdge = self.lowerTrailingEdge.rescaleAndMove(
            chord,
            chord * relativeThickness,
        )

        solution.leadingArea = solution.leadingEdge.area
        solution.boxArea = solution.upperBox.area + solution.lowerBox.area
        solution.trailingArea = (
            solution.upperTrailingEdge.area + solution.lowerTrailingEdge.area
        )

        return solution

    def prepareStructuralSurface(
        self,
        nStringerLower: int = 0,
        nStringerUpper: int = 0,
    ) -> None:
        self.xMainCaps = [
            self.lowerBox.x[0],
            self.rearSpar.x[0],
            self.upperBox.x[0],
            self.frontSpar.x[0],
        ]
        self.yMainCaps = [
            self.lowerBox.y[0],
            self.rearSpar.y[0],
            self.upperBox.y[0],
            self.frontSpar.y[0],
        ]

        self.nStringerLower = nStringerLower
        self.lowerBoxBroken = self.lowerBox.breakNParts(nStringerLower + 1)

        self.xCaps = [0.0 for _ in range(nStringerLower + nStringerUpper + 4)]
        self.yCaps = [0.0 for _ in range(nStringerLower + nStringerUpper + 4)]
        self.lowerTan = [[0.0, 0.0] for _ in range(nStringerLower + 2)]
        deltaL = self.lowerBox.length / (nStringerLower + 1)
        for iStringer in range(nStringerLower + 1):
            self.xCaps[iStringer] = self.lowerBoxBroken[iStringer].x[0]
            self.yCaps[iStringer] = self.lowerBoxBroken[iStringer].y[0]
            self.lowerTan[iStringer] = self.lowerBox.tangentInL(deltaL * iStringer)
        self.xCaps[nStringerLower + 1] = self.rearSpar.x[0]
        self.yCaps[nStringerLower + 1] = self.rearSpar.y[0]
        self.lowerTan[nStringerLower + 1] = self.lowerBox.tangentInL(
            self.lowerBox.length,
        )

        self.nStringerUpper = nStringerUpper
        self.upperBoxBroken = self.upperBox.breakNParts(nStringerUpper + 1)
        self.upperTan = [[0.0, 0.0] for _ in range(nStringerUpper + 2)]
        deltaL = self.upperBox.length / (nStringerUpper + 1)
        for iStringer in range(nStringerUpper + 1):
            iStringer2 = iStringer + nStringerLower + 2
            self.xCaps[iStringer2] = self.upperBoxBroken[iStringer].x[0]
            self.yCaps[iStringer2] = self.upperBoxBroken[iStringer].y[0]
            self.upperTan[iStringer] = self.upperBox.tangentInL(deltaL * iStringer)
        self.xCaps[nStringerLower + nStringerUpper + 3] = self.frontSpar.x[0]
        self.yCaps[nStringerLower + nStringerUpper + 3] = self.frontSpar.y[0]

        self.upperTan[nStringerUpper + 1] = self.upperBox.tangentInL(
            self.upperBox.length,
        )

    def getAMainCaps1Var(self, ACap: float) -> list[float]:
        return [ACap for _ in range(self.nStringerLower + self.nStringerUpper + 4)]

    def getInertiaVectorMain(self, ACap: list[float]) -> list[float]:
        solution = [0.0 for _ in range(6)]

        for iCap in range(4):
            solution[0] += ACap[iCap]
            solution[1] += ACap[iCap] * self.xMainCaps[iCap]
            solution[2] += ACap[iCap] * self.yMainCaps[iCap]
            solution[3] += ACap[iCap] * self.xMainCaps[iCap] * self.xMainCaps[iCap]
            solution[4] += ACap[iCap] * self.xMainCaps[iCap] * self.yMainCaps[iCap]
            solution[5] += ACap[iCap] * self.yMainCaps[iCap] * self.yMainCaps[iCap]

        return solution

    def getACaps2Vars(self, ACap: float, AStringers: float) -> list[float]:
        ACaps = [0.0 for _ in range(self.nStringerLower + self.nStringerUpper + 4)]

        ACaps[0] = ACap

        for iStringer in range(1, self.nStringerLower + 1):
            ACaps[iStringer] = AStringers

        ACaps[self.nStringerLower + 1] = ACap
        ACaps[self.nStringerLower + 2] = ACap

        for iStringer in range(1, self.nStringerUpper + 1):
            ACaps[iStringer + self.nStringerLower + 2] = AStringers

        ACaps[self.nStringerLower + self.nStringerUpper + 3] = ACap

        return ACaps

    def getInertiaVector(self, ACaps: list[float]) -> list[float]:
        solution = [0.0 for _ in range(6)]

        for iStringer in range(self.nStringerLower + self.nStringerUpper + 4):
            solution[0] += ACaps[iStringer]
            solution[1] += ACaps[iStringer] * self.xCaps[iStringer]
            solution[2] += ACaps[iStringer] * self.yCaps[iStringer]
            solution[3] += (
                ACaps[iStringer] * self.xCaps[iStringer] * self.xCaps[iStringer]
            )
            solution[4] += (
                ACaps[iStringer] * self.xCaps[iStringer] * self.yCaps[iStringer]
            )
            solution[5] += (
                ACaps[iStringer] * self.yCaps[iStringer] * self.yCaps[iStringer]
            )

        return solution

    def getInertiaVectorProfile(
        self,
        ACaps: list[float],
        tBox: list[float],
        tLeading: float,
        tTrailingUpper: float,
        tTrailingLower: float,
    ) -> list[float]:
        solution = self.getInertiaVector(ACaps)

        for iCap in range(self.nStringerLower + 1):
            inertiaVectorT = self.lowerBoxBroken[iCap].getInertiaVector(tBox[iCap])
            for iV in range(6):
                solution[iV] += inertiaVectorT[iV]

        iCap = self.nStringerLower + 1
        inertiaVectorT = self.rearSpar.getInertiaVector(tBox[iCap])
        for iV in range(6):
            solution[iV] += inertiaVectorT[iV]

        for iCap2 in range(self.nStringerUpper + 1):
            iCap = iCap2 + self.nStringerLower + 2
            inertiaVectorT = self.upperBoxBroken[iCap2].getInertiaVector(tBox[iCap])
            for iV in range(6):
                solution[iV] += inertiaVectorT[iV]

        iCap = self.nStringerLower + self.nStringerUpper + 3
        inertiaVectorT = self.frontSpar.getInertiaVector(tBox[iCap])
        for iV in range(6):
            solution[iV] += inertiaVectorT[iV]

        inertiaVectorT = self.leadingEdge.getInertiaVector(tLeading)
        for iV in range(6):
            solution[iV] += inertiaVectorT[iV]

        inertiaVectorT = self.upperTrailingEdge.getInertiaVector(tTrailingUpper)
        for iV in range(6):
            solution[iV] += inertiaVectorT[iV]

        inertiaVectorT = self.lowerTrailingEdge.getInertiaVector(tTrailingLower)
        for iV in range(6):
            solution[iV] += inertiaVectorT[iV]

        return solution
