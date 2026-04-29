# ruff: noqa: N802, N803, N806

import math

import numpy as np

from multiads.solvers.structure.ampet.bending_analysis import BendingAnalysis
from multiads.solvers.structure.ampet.buckling_plate import BucklingPlate
from multiads.solvers.structure.ampet.shear_analysis import ShearAnalysis
from multiads.solvers.structure.ampet.structural_material import StructuralMaterial
from multiads.solvers.structure.ampet.structural_profile import StructuralProfile


class StructuralSection:
    def __init__(
        self,
        unitProfile: StructuralProfile,
        material: StructuralMaterial,
        vCruise: float,
        angleLeadingEdge: float,
        torsion: float,
        xReference: float,
        yReference: float,
        zReference: float,
        spanSection: float,
        chord: float,
        relativeThickness: float,
        panelLength: float,
        sigmaAllowable: float,
        reserveFactor: float,
    ) -> None:
        self.material = material
        self.vCruise = vCruise
        self.xReference = xReference
        self.yReference = yReference
        self.zReference = zReference
        self.spanSection = spanSection
        self.sectionProfile = unitProfile.rescale(chord, relativeThickness)

        self.chord = chord
        self.relativeThickness = relativeThickness

        nStringerUpper = (
            math.ceil(self.sectionProfile.upperBox.length / panelLength) - 1
        )
        nStringerLower = (
            math.ceil(self.sectionProfile.lowerBox.length / panelLength) - 1
        )

        self.sectionProfile.prepareStructuralSurface(nStringerLower, nStringerUpper)
        self.shearAnalysis = ShearAnalysis(self.sectionProfile)

        self.nStringerLower = nStringerLower
        self.nStringerUpper = nStringerUpper
        self.nWebs = self.shearAnalysis.nWebs

        self.lowerPlate = BucklingPlate(
            self.sectionProfile.lowerBoxBroken[0].length,
            spanSection,
            material,
            reserveFactor,
        )
        self.rearPlate = BucklingPlate(
            self.sectionProfile.rearSpar.length,
            spanSection,
            material,
            reserveFactor,
        )
        self.upperPlate = BucklingPlate(
            self.sectionProfile.upperBoxBroken[0].length,
            spanSection,
            material,
            reserveFactor,
        )
        self.frontPlate = BucklingPlate(
            self.sectionProfile.frontSpar.length,
            spanSection,
            material,
            reserveFactor,
        )

        self.angleLeadingEdge = angleLeadingEdge
        self.torsion = torsion

        self.sigmaAllowable = sigmaAllowable

    def prepareSection2Vars(self, ACap: float, AStringers: float) -> None:
        self.mainCaps = self.sectionProfile.getAMainCaps1Var(ACap)
        self.inertiaVectorMain = self.sectionProfile.getInertiaVectorMain(self.mainCaps)
        self.ACaps = self.sectionProfile.getACaps2Vars(ACap, AStringers)
        self.inertiaVector = self.sectionProfile.getInertiaVector(self.ACaps)
        self.shearAnalysis.prepareBasicSolution(self.inertiaVector, self.ACaps)

    def calculateSectionMassProperties(self) -> None:
        cos = math.cos(self.torsion)
        sin = math.sin(self.torsion)

        self.mass = self.inertiaVectorP[0] * self.material.rho * self.spanSection
        massXTemp = self.inertiaVectorP[1] * self.material.rho * self.spanSection
        massYTemp = self.inertiaVectorP[2] * self.material.rho * self.spanSection
        self.massX = self.xReference * self.mass + massXTemp * cos + massYTemp * sin
        self.massY = self.yReference * self.mass
        self.massZ = self.zReference * self.mass - massXTemp * sin + massYTemp * cos

        h1 = self.yReference - self.spanSection / 4.0
        h2 = self.yReference + self.spanSection / 4.0
        self.massYHalf1 = h1 * self.mass / 2.0
        self.massYHalf2 = h2 * self.mass / 2.0
        massXXTemp = self.inertiaVector[3] * self.material.rho * self.spanSection
        massXZTemp = self.inertiaVector[4] * self.material.rho * self.spanSection
        massZZTemp = self.inertiaVector[5] * self.material.rho * self.spanSection
        self.massXX = (
            massXXTemp * cos * cos
            + 2.0 * massXZTemp * cos * sin
            + massZZTemp * sin * sin
        )
        self.massXZ = (
            -massXXTemp * sin * cos
            + massXZTemp * (cos * cos - sin * sin)
            + massZZTemp * sin * cos
        )
        self.massZZ = (
            massXXTemp * sin * sin
            - 2.0 * massXZTemp * cos * sin
            + massZZTemp * cos * cos
        )
        self.massYYHalf = 1.0 / 96.0 * self.spanSection * self.spanSection * self.mass
        self.massXYHalf1 = self.massX * h1
        self.massXYHalf2 = self.massX * h2
        self.massYZHalf1 = self.massZ * h1
        self.massYZHalf2 = self.massZ * h2

    def calculateSectionStiffness(self) -> None:
        cos = math.cos(self.torsion)
        sin = math.sin(self.torsion)

        E = self.material.E
        self.bendingStiffnessMatrix = np.zeros(6)
        self.bendingStiffnessMatrix[0] = E * self.inertiaVectorP[0]
        self.bendingStiffnessMatrix[1] = E * (
            self.inertiaVectorP[2] * cos - self.inertiaVectorP[1] * sin
        )
        self.bendingStiffnessMatrix[2] = -E * (
            self.inertiaVectorP[1] * cos + self.inertiaVectorP[2] * sin
        )
        self.bendingStiffnessMatrix[3] = E * (
            +self.inertiaVectorP[3] * sin * sin
            - self.inertiaVectorP[4] * 2.0 * sin * cos
            + self.inertiaVectorP[5] * cos * cos
        )
        self.bendingStiffnessMatrix[4] = -E * (
            -self.inertiaVectorP[3] * sin * cos
            + self.inertiaVectorP[4] * (cos * cos - sin * sin)
            + self.inertiaVectorP[5] * sin * cos
        )
        self.bendingStiffnessMatrix[5] = E * (
            +self.inertiaVectorP[3] * cos * cos
            + self.inertiaVectorP[4] * 2.0 * sin * cos
            + self.inertiaVectorP[5] * sin * sin
        )

        [xSC, zSC, J, SAx, SAz, sBeta, cBeta] = (
            self.shearAnalysis.CalculateShearParameters(self.tWebsBox)
        )

        sAlpha = sin
        cAlpha = cos
        cos = cAlpha * cBeta - sAlpha * sBeta
        sin = cAlpha * sBeta + sAlpha * cBeta

        G = self.material.G
        self.shearStiffnessMatrix = np.zeros(6)
        self.shearStiffnessMatrix[0] = G * (SAz * cos * cos + SAx * sin * sin)
        self.shearStiffnessMatrix[1] = G * (SAx - SAz) * sin * cos
        self.shearStiffnessMatrix[2] = G * (zSC * SAz * sin - xSC * SAz * cos)
        self.shearStiffnessMatrix[3] = G * (SAx * cos * cos + SAz * sin * sin)
        self.shearStiffnessMatrix[4] = G * (zSC * SAx * cos + xSC * SAz * sin)
        self.shearStiffnessMatrix[5] = G * (J + zSC * zSC * SAx + xSC * xSC * SAz)

    def calculateBoxTensions(
        self,
        Sx: float,
        Sy: float,
        Sz: float,
        Mx: float,
        My: float,
        Mz: float,
    ) -> None:
        self.bendingAnalysis = BendingAnalysis(
            Sy,
            Mx,
            Mz,
            self.inertiaVector,
            self.material,
        )
        self.bendingAnalysisMain = BendingAnalysis(
            Sy,
            Mx,
            Mz,
            self.inertiaVectorMain,
            self.material,
        )

        self.webQ = self.shearAnalysis.calculateQBox(Sx, Sz, My)

        self.sigmas = [0.0 for _ in range(self.nWebs)]
        self.sigmasMain = [0.0 for _ in range(4)]

        self.webCompression = [0.0 for _ in range(self.nWebs)]

        capBrazierLoads = [[0.0, 0.0] for _ in range(self.nWebs)]
        capBrazierLoadsMain = [[0.0, 0.0] for _ in range(4)]

        sigmaMinCompression = 1.0e20
        sigmaMaxCompression = 0.0
        sigmaMinTraction = 1.0e20
        sigmaMaxTraction = 0.0

        for iCap in range(4):
            self.sigmasMain[iCap] = self.bendingAnalysisMain.sigma(
                self.sectionProfile.xMainCaps[iCap],
                self.sectionProfile.yMainCaps[iCap],
            )
            if self.sigmasMain[iCap] < 0.0:
                sigmaMinCompression = min(sigmaMinCompression, -self.sigmasMain[iCap])
                sigmaMaxCompression = max(sigmaMaxCompression, -self.sigmasMain[iCap])
            else:
                sigmaMinTraction = min(sigmaMinTraction, self.sigmasMain[iCap])
                sigmaMaxTraction = max(sigmaMaxTraction, self.sigmasMain[iCap])

            brazierLoadsV = self.bendingAnalysis.brazierFV(
                self.sectionProfile.xMainCaps[iCap],
                self.sectionProfile.yMainCaps[iCap],
            )
            vCap = self.mainCaps[iCap] * self.spanSection
            capBrazierLoadsMain[iCap] = [
                brazierLoadsV[0] * vCap,
                brazierLoadsV[1] * vCap,
            ]

        for iCap in range(self.nWebs):
            self.sigmas[iCap] = self.bendingAnalysis.sigma(
                self.sectionProfile.xCaps[iCap],
                self.sectionProfile.yCaps[iCap],
            )
            if self.sigmas[iCap] < 0.0:
                sigmaMinCompression = min(sigmaMinCompression, -self.sigmas[iCap])
                sigmaMaxCompression = max(sigmaMaxCompression, -self.sigmas[iCap])
            else:
                sigmaMinTraction = min(sigmaMinTraction, self.sigmas[iCap])
                sigmaMaxTraction = max(sigmaMaxTraction, self.sigmas[iCap])

            brazierLoadsV = self.bendingAnalysis.brazierFV(
                self.sectionProfile.xCaps[iCap],
                self.sectionProfile.yCaps[iCap],
            )
            vCap = self.ACaps[iCap] * self.spanSection
            capBrazierLoads[iCap] = [brazierLoadsV[0] * vCap, brazierLoadsV[1] * vCap]

        self.sigmaMinCompression = sigmaMinCompression
        self.sigmaMaxCompression = sigmaMaxCompression

        self.sigmaMinTraction = sigmaMinTraction
        self.sigmaMaxTraction = sigmaMaxTraction

        fAccumulatedLower = 0.0
        for iCap in range(self.nStringerLower + 2):
            fAccumulatedLower += (
                capBrazierLoads[iCap][0] / self.sectionProfile.lowerTan[iCap][0]
            )
        qLowerBrazier = -fAccumulatedLower / self.sectionProfile.lowerBox.length
        fAccumulatedLower = 0.0
        for iCap in range(self.nStringerLower + 1):
            fAccumulatedLower += (
                capBrazierLoads[iCap][0] / self.sectionProfile.lowerTan[iCap][0]
            )
            self.webCompression[iCap] = fAccumulatedLower + max(
                0,
                qLowerBrazier * self.sectionProfile.lowerBoxBroken[iCap].length,
            )
            fAccumulatedLower += (
                qLowerBrazier * self.sectionProfile.lowerBoxBroken[iCap].length
            )
            self.webQ[iCap] = self.webQ[iCap] + qLowerBrazier

        iCap = self.nStringerLower + 1
        fAccumulatedRearSpar = (
            capBrazierLoadsMain[1][1]
            - capBrazierLoadsMain[1][0]
            * self.sectionProfile.lowerTan[iCap][1]
            / self.sectionProfile.lowerTan[iCap][0]
        )
        qRearBrazier = (
            fAccumulatedRearSpar
            - capBrazierLoadsMain[2][1]
            + capBrazierLoadsMain[2][0]
            * self.sectionProfile.upperTan[0][1]
            / self.sectionProfile.upperTan[0][0]
        )
        qRearBrazier = -qRearBrazier / self.sectionProfile.rearSpar.length
        self.webCompression[iCap] = fAccumulatedRearSpar + max(
            0,
            qRearBrazier * self.sectionProfile.rearSpar.length,
        )
        self.webQ[iCap] = self.webQ[iCap] + qRearBrazier

        fAccumulatedUpper = 0.0
        for iCap in range(self.nStringerUpper + 2):
            iCap2 = self.nStringerLower + 2 + iCap
            fAccumulatedUpper += (
                capBrazierLoads[iCap2][0] / self.sectionProfile.upperTan[iCap][0]
            )
        qUpperBrazier = -fAccumulatedUpper / self.sectionProfile.upperBox.length
        fAccumulatedUpper = 0.0
        for iCap in range(self.nStringerUpper + 1):
            fAccumulatedUpper += (
                capBrazierLoads[iCap2][0] / self.sectionProfile.upperTan[iCap][0]
            )
            self.webCompression[iCap2] = fAccumulatedUpper + max(
                0,
                qUpperBrazier * self.sectionProfile.upperBoxBroken[iCap].length,
            )
            fAccumulatedUpper += (
                qUpperBrazier * self.sectionProfile.upperBoxBroken[iCap].length
            )
            self.webQ[iCap2] = self.webQ[iCap2] + qUpperBrazier

        iCap = self.nStringerLower + self.nStringerUpper + 3
        iCap2 = self.nStringerUpper + 1
        fAccumulatedFrontSpar = (
            -capBrazierLoadsMain[3][1]
            + capBrazierLoadsMain[3][0]
            * self.sectionProfile.upperTan[iCap2][1]
            / self.sectionProfile.upperTan[iCap2][0]
        )
        qFrontBrazier = (
            fAccumulatedFrontSpar
            + capBrazierLoadsMain[0][1]
            - capBrazierLoadsMain[0][0]
            * self.sectionProfile.upperTan[0][1]
            / self.sectionProfile.upperTan[0][0]
        )
        qFrontBrazier = -qFrontBrazier / self.sectionProfile.frontSpar.length
        self.webCompression[iCap] = fAccumulatedFrontSpar + max(
            0,
            qFrontBrazier * self.sectionProfile.frontSpar.length,
        )
        self.webQ[iCap] = self.webQ[iCap] + qFrontBrazier

        if abs(self.sigmaMaxCompression) > 0:
            self.minRFCompression = self.sigmaAllowable / self.sigmaMaxCompression
        else:
            self.minRFCompression = 10.0

        if abs(self.sigmaMinCompression) > 0:
            self.maxRFCompression = self.sigmaAllowable / self.sigmaMinCompression
        else:
            self.maxRFCompression = 10.0

        if abs(self.sigmaMaxTraction) > 0:
            self.minRFTraction = self.sigmaAllowable / self.sigmaMaxTraction
        else:
            self.minRFTraction = 10.0

        if abs(self.sigmaMinTraction) > 0:
            self.maxRFTraction = self.sigmaAllowable / self.sigmaMinTraction
        else:
            self.maxRFTraction = 10.0

    def calculateBoxThickness(
        self,
        Sx: float,
        Sy: float,
        Sz: float,
        Mx: float,
        My: float,
        Mz: float,
    ) -> None:
        self.calculateBoxTensions(Sx, Sy, Sz, Mx, My, Mz)

        self.tWebsBox = [0.0 for _ in range(self.nWebs)]

        for iCap in range(self.nStringerLower + 1):
            self.tWebsBox[iCap] = self.lowerPlate.CalculateThickness(
                self.webQ[iCap],
                self.webCompression[iCap],
                0.0,
                0.0,
                0.0,
                self.sigmas[iCap],
                self.sigmas[iCap + 1],
            )

        iCap = self.nStringerLower + 1

        self.tWebsBox[iCap] = self.rearPlate.CalculateThickness(
            self.webQ[iCap],
            self.webCompression[iCap],
            0.0,
            0.0,
            0.0,
            self.sigmasMain[1],
            self.sigmasMain[2],
        )

        for iCap2 in range(self.nStringerUpper + 1):
            iCap = iCap2 + self.nStringerLower + 2
            self.tWebsBox[iCap] = self.upperPlate.CalculateThickness(
                self.webQ[iCap],
                self.webCompression[iCap],
                0.0,
                0.0,
                0.0,
                self.sigmas[iCap],
                self.sigmas[iCap + 1],
            )

        iCap = self.nStringerLower + self.nStringerUpper + 3
        self.tWebsBox[iCap] = self.frontPlate.CalculateThickness(
            self.webQ[iCap],
            self.webCompression[iCap],
            0.0,
            0.0,
            0.0,
            self.sigmasMain[3],
            self.sigmasMain[0],
        )

    def calculateEdgesStresses(self, Sx: float, Sz: float, My: float) -> None:
        self.edgesQ = self.shearAnalysis.calculateQProfile(
            Sx,
            Sz,
            My,
            self.tWebsBox,
            self.tLeadingEdge,
            self.tUpperTrailingEdge,
            self.tLowerTrailingEdge,
        )

    def calculateEdgesThickness(self, Sx: float, Sz: float, My: float) -> None:
        tLeadingEdgeBS = self.material.birdStrikeThickness(
            self.sectionProfile.leadingRadius,
            self.angleLeadingEdge,
            self.vCruise,
        )
        self.tLeadingEdge = tLeadingEdgeBS
        self.tUpperTrailingEdge = self.material.thicknessMin
        self.tLowerTrailingEdge = self.material.thicknessMin

        self.calculateEdgesStresses(Sx, Sz, My)

        tLeadingEdge0 = self.tLeadingEdge
        tUpperTrailingEdge0 = self.tUpperTrailingEdge
        tLowerTrailingEdge0 = self.tLowerTrailingEdge

        self.tLeadingEdge = max(
            tLeadingEdgeBS,
            self.material.adjustThickness(abs(self.edgesQ[0] / self.material.tauMax)),
        )
        self.tUpperTrailingEdge = self.material.adjustThickness(
            abs(self.edgesQ[1] / self.material.tauMax),
        )
        self.tLowerTrailingEdge = self.material.adjustThickness(
            abs(self.edgesQ[1] / self.material.tauMax),
        )

        while (
            (self.tLeadingEdge != tLeadingEdge0)
            or (self.tUpperTrailingEdge != tUpperTrailingEdge0)
            or (self.tLowerTrailingEdge != tLowerTrailingEdge0)
        ):
            self.calculateEdgesStresses(Sx, Sz, My)

            tLeadingEdge0 = self.tLeadingEdge
            tUpperTrailingEdge0 = self.tUpperTrailingEdge
            tLowerTrailingEdge0 = self.tLowerTrailingEdge

            tLeadingEdge1 = max(
                tLeadingEdgeBS,
                self.material.adjustThickness(
                    abs(self.edgesQ[0] / self.material.tauMax),
                ),
            )
            tUpperTrailingEdge1 = self.material.adjustThickness(
                abs(self.edgesQ[1] / self.material.tauMax),
            )
            tLowerTrailingEdge1 = self.material.adjustThickness(
                abs(self.edgesQ[1] / self.material.tauMax),
            )

            if (tLeadingEdge0 > tLeadingEdgeBS) and (tLeadingEdge1 == tLeadingEdgeBS):
                self.tLeadingEdge = 0.5 * (tLeadingEdge0 + tLeadingEdge1)

            if (tUpperTrailingEdge0 > self.material.thicknessMin) and (
                tUpperTrailingEdge1 == self.material.thicknessMin
            ):
                self.tUpperTrailingEdge = 0.5 * (
                    tUpperTrailingEdge0 + tUpperTrailingEdge1
                )

            if (tLowerTrailingEdge0 > self.material.thicknessMin) and (
                tLowerTrailingEdge1 == self.material.thicknessMin
            ):
                self.tLowerTrailingEdge = 0.5 * (
                    tLowerTrailingEdge0 + tLowerTrailingEdge1
                )

    def calculateSectionBraziersForces(self, Sy: float, Mx: float, Mz: float) -> None:
        self.inertiaVectorP = self.sectionProfile.getInertiaVectorProfile(
            self.ACaps,
            self.tWebsBox,
            self.tLeadingEdge,
            self.tUpperTrailingEdge,
            self.tLowerTrailingEdge,
        )

        self.bendingAnalysisP = BendingAnalysis(
            Sy,
            Mx,
            Mz,
            self.inertiaVectorP,
            self.material,
        )

        point = self.sectionProfile.leadingEdge.getPoint(
            self.sectionProfile.leadingEdge.length / 4.0,
        )
        brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
        vPanel = (
            self.spanSection
            * self.sectionProfile.leadingEdge.length
            * self.tLeadingEdge
            / 2.0
        )
        uForce = brazierLoadsV[1] * vPanel

        point = self.sectionProfile.leadingEdge.getPoint(
            3.0 * self.sectionProfile.leadingEdge.length / 4.0,
        )
        brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
        lForce = brazierLoadsV[1] * vPanel

        self.leadingBrazierZ = max(abs(uForce), abs(lForce))

        lForce = 0.0
        for iCap in range(self.nStringerLower + 1):
            point = self.sectionProfile.lowerBoxBroken[iCap].getPoint(
                self.sectionProfile.lowerBoxBroken[iCap].length / 2.0,
            )
            brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
            vPanel = (
                self.spanSection
                * self.sectionProfile.lowerBoxBroken[iCap].length
                * self.tWebsBox[iCap]
            )
            lForce += brazierLoadsV[1] * vPanel
            if iCap > 0:
                brazierLoadsV = self.bendingAnalysis.brazierFV(
                    self.sectionProfile.xCaps[iCap],
                    self.sectionProfile.yCaps[iCap],
                )
                vCap = self.ACaps[iCap] * self.spanSection
                lForce += brazierLoadsV[1] * vCap

        uForce = 0.0
        for iCap2 in range(self.nStringerUpper + 1):
            iCap = iCap2 + self.nStringerLower + 2
            point = self.sectionProfile.upperBoxBroken[iCap2].getPoint(
                self.sectionProfile.upperBoxBroken[iCap2].length / 2.0,
            )
            brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
            vPanel = (
                self.spanSection
                * self.sectionProfile.upperBoxBroken[iCap2].length
                * self.tWebsBox[iCap]
            )
            uForce += brazierLoadsV[1] * vPanel
            if iCap2 > 0:
                brazierLoadsV = self.bendingAnalysis.brazierFV(
                    self.sectionProfile.xCaps[iCap],
                    self.sectionProfile.yCaps[iCap],
                )
                vCap = self.ACaps[iCap] * self.spanSection
                uForce += brazierLoadsV[1] * vCap

        self.boxBrazierZ = max(abs(uForce), abs(lForce))

        point = self.sectionProfile.frontSpar.getPoint(
            self.sectionProfile.frontSpar.length / 2.0,
        )
        brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
        vPanel = (
            self.spanSection
            * self.sectionProfile.frontSpar.length
            * self.tWebsBox[self.nStringerLower + self.nStringerUpper + 3]
        )
        uForce = brazierLoadsV[0] * vPanel

        point = self.sectionProfile.rearSpar.getPoint(
            self.sectionProfile.rearSpar.length / 2.0,
        )
        brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
        vPanel = (
            self.spanSection
            * self.sectionProfile.rearSpar.length
            * self.tWebsBox[self.nStringerLower + 1]
        )
        lForce = brazierLoadsV[0] * vPanel

        self.boxBrazierX = max(abs(uForce), abs(lForce))

        point = self.sectionProfile.upperTrailingEdge.getPoint(
            self.sectionProfile.upperTrailingEdge.length / 2.0,
        )
        brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
        vPanel = (
            self.spanSection
            * self.sectionProfile.upperTrailingEdge.length
            * self.tUpperTrailingEdge
        )
        uForce = brazierLoadsV[1] * vPanel

        point = self.sectionProfile.lowerTrailingEdge.getPoint(
            self.sectionProfile.lowerTrailingEdge.length / 2.0,
        )
        brazierLoadsV = self.bendingAnalysis.brazierFV(point[0], point[1])
        vPanel = (
            self.spanSection
            * self.sectionProfile.lowerTrailingEdge.length
            * self.tLowerTrailingEdge
        )
        lForce = brazierLoadsV[1] * vPanel

        self.trailingBrazierZ = max(abs(uForce), abs(lForce))
