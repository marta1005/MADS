# ruff: noqa: N802, N803, N806

import math

import numpy as np
from numpy.typing import NDArray

from multiads.scenario.span_loads import SpanLoadsGroupVariable, SpanLoadsVariable
from multiads.solvers.structure.ampet.structural_material import StructuralMaterial
from multiads.solvers.structure.ampet.structural_profile import StructuralProfile
from multiads.solvers.structure.ampet.structural_rib import StructuralRib
from multiads.solvers.structure.ampet.structural_section import StructuralSection


class StructuralWing:
    def __init__(
        self,
        name: str,
        baseProfile: StructuralProfile,
        material: StructuralMaterial,
        reserveFactor: float,
        vCruise: float,
        xRootPosition: float,  # 25% referenced
        yRootPosition: float,
        zRootPosition: float,
        sectionSpans: list[float],
        dihedralDistribution: list[float],
        sweepDistribution: list[float],
        chordDistribution: list[float],
        thicknessDistribution: list[float],
        torsionDistribution: list[float],
        percentTorsionOriginDistribution: list[float],
        maxRibsSeparation: float,
        panelLength: float,
        loadFactor: float,
        posEngines: list[float],
        massEngines: list[float],
        posSupport: list[float],
    ) -> None:
        self.name = name
        self.nWingSections = len(sectionSpans)

        # One element per section

        self.sectionSpans = sectionSpans
        self.dihedralDistribution = dihedralDistribution
        self.sweepDistribution = sweepDistribution

        self.nSectionDistribution = [0 for _ in range(self.nWingSections)]

        # One element per section + 1

        self.xPositions = [0.0 for _ in range(self.nWingSections + 1)]
        self.yPositions = [0.0 for _ in range(self.nWingSections + 1)]
        self.zPositions = [0.0 for _ in range(self.nWingSections + 1)]

        self.accumulatedSpan = [0.0 for _ in range(self.nWingSections + 1)]

        self.chordDistribution = chordDistribution
        self.thicknessDistribution = thicknessDistribution
        self.torsionDistribution = torsionDistribution
        self.percentTorsionOriginDistribution = percentTorsionOriginDistribution

        self.xPositions[0] = xRootPosition
        self.yPositions[0] = yRootPosition
        self.zPositions[0] = zRootPosition
        self.accumulatedSpan[0] = 0.0
        numberOfSections = 0

        for iStep in range(self.nWingSections):
            aSpan = self.accumulatedSpan[iStep] + sectionSpans[iStep]
            self.accumulatedSpan[iStep + 1] = aSpan

            [chord, thickness, angleLE, torsion, _, x, y, z] = self.profileDimensions(
                aSpan,
            )

            self.xPositions[iStep + 1] = x
            self.yPositions[iStep + 1] = y
            self.zPositions[iStep + 1] = z

            nStructuralSections = math.ceil(sectionSpans[iStep] / maxRibsSeparation)

            self.nSectionDistribution[iStep] = nStructuralSections

            numberOfSections += nStructuralSections

        self.numberOfSections = numberOfSections

        numberOfRibs = numberOfSections + 1
        self.numberOfRibs = numberOfRibs

        wingSpan = self.accumulatedSpan[self.nWingSections]

        self.externalForcesInRibs = [[0.0, 0.0, 0.0] for _ in range(numberOfRibs)]
        self.externalMomentsInRibs = [[0.0, 0.0, 0.0] for _ in range(numberOfRibs)]

        self.aeroloadsForcesInSections = [
            [0.0, 0.0, 0.0] for _ in range(numberOfSections)
        ]
        self.aeroloadsMomentsInSections = [
            [0.0, 0.0, 0.0] for _ in range(numberOfSections)
        ]

        self.structuralSections = [
            StructuralSection(
                baseProfile,
                material,
                vCruise,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.1,
                0.1,
                0.0,
                reserveFactor,
            )
            for _ in range(numberOfSections)
        ]
        self.structuralRibs = [
            StructuralRib(baseProfile, material, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0)
            for _ in range(numberOfRibs)
        ]

        span = 0.0

        jRib = 0
        [chord, thickness, angleLE, torsion, _, x, y, z] = self.profileDimensions(span)
        self.structuralRibs[jRib] = StructuralRib(
            baseProfile,
            material,
            chord,
            thickness,
            x,
            y,
            z,
            torsion,
        )

        jSection = -1

        for iStep in range(self.nWingSections):
            nSectionsInSpan = self.nSectionDistribution[iStep]
            deltaSpan = sectionSpans[iStep] / nSectionsInSpan
            deltaY = deltaSpan * math.cos(dihedralDistribution[iStep])
            for _ in range(nSectionsInSpan):
                jSection += 1
                span += deltaSpan / 2.0
                allowableSigma = material.sigmaAllowable(span, wingSpan)
                [chord, thickness, angleLE, torsion, _, x, y, z] = (
                    self.profileDimensions(span)
                )
                self.structuralSections[jSection] = StructuralSection(
                    baseProfile,
                    material,
                    vCruise,
                    angleLE,
                    torsion,
                    x,
                    y,
                    z,
                    deltaY,
                    chord,
                    thickness,
                    panelLength,
                    allowableSigma,
                    reserveFactor,
                )

                jRib += 1
                span += deltaSpan / 2.0
                [chord, thickness, angleLE, torsion, _, x, y, z] = (
                    self.profileDimensions(span)
                )
                self.structuralRibs[jRib] = StructuralRib(
                    baseProfile,
                    material,
                    chord,
                    thickness,
                    x,
                    y,
                    z,
                    torsion,
                )

        self.loadFactor = loadFactor

        self.posEngines = posEngines
        self.mssEngines = massEngines
        self.posSupport = posSupport

    def profileDimensions(self, span: float) -> list[float]:
        for iPosition in range(self.nWingSections):
            if (
                ((iPosition == 0) and (span == self.accumulatedSpan[iPosition]))
                or (
                    (span > self.accumulatedSpan[iPosition])
                    and (span <= self.accumulatedSpan[iPosition + 1])
                )
                or (iPosition == self.nWingSections - 1)
            ):
                sweep = self.sweepDistribution[iPosition]
                dihedral = self.dihedralDistribution[iPosition]

                sectionSpan = (
                    self.accumulatedSpan[iPosition + 1]
                    - self.accumulatedSpan[iPosition]
                )

                localSpan = span - self.accumulatedSpan[iPosition]
                spanFactor0 = (self.accumulatedSpan[iPosition + 1] - span) / sectionSpan
                spanFactor1 = (span - self.accumulatedSpan[iPosition]) / sectionSpan

                chord0 = self.chordDistribution[iPosition]
                percentTorsionOrigin0 = self.percentTorsionOriginDistribution[iPosition]
                torsion0 = self.torsionDistribution[iPosition]

                lRotationBase0 = chord0 * (percentTorsionOrigin0 - 0.25)

                xRotationBase0 = self.xPositions[iPosition] + lRotationBase0 * math.cos(
                    torsion0,
                )
                zRotationBase0 = self.zPositions[iPosition] - lRotationBase0 * math.sin(
                    torsion0,
                )

                xLeadingEdge0 = xRotationBase0 - lRotationBase0 - chord0 * 0.25
                xLeadingEdgef = xLeadingEdge0 + sectionSpan * math.sin(sweep)

                chord = (
                    self.chordDistribution[iPosition] * spanFactor0
                    + self.chordDistribution[iPosition + 1] * spanFactor1
                )
                thickness = (
                    self.thicknessDistribution[iPosition] * spanFactor0
                    + self.thicknessDistribution[iPosition + 1] * spanFactor1
                )
                percentTorsionOrigin = (
                    self.percentTorsionOriginDistribution[iPosition] * spanFactor0
                    + self.percentTorsionOriginDistribution[iPosition + 1] * spanFactor1
                )
                torsion = (
                    self.torsionDistribution[iPosition] * spanFactor0
                    + self.torsionDistribution[iPosition + 1] * spanFactor1
                )

                lRotationBasef = chord * (percentTorsionOrigin - 0.25)
                xRotationBasef = xLeadingEdgef + lRotationBasef
                zRotationBasef = zRotationBase0 + localSpan * math.sin(dihedral)

                xProfile = xRotationBasef - lRotationBasef * math.cos(torsion)
                yProfile = self.yPositions[iPosition] + localSpan * math.cos(dihedral)
                zProfile = zRotationBasef + lRotationBasef * math.sin(torsion)

                return [
                    chord,
                    thickness,
                    sweep,
                    torsion,
                    dihedral,
                    xProfile,
                    yProfile,
                    zProfile,
                ]

        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def setAeroLoads(self, loadsGroups: SpanLoadsGroupVariable) -> None:
        self.loadsGroup = loadsGroups

    def externalForcesAndMomentsInSection(
        self,
        iSection: int,
        iLoad: int,
    ) -> list[float]:
        x = self.structuralSections[iSection].xReference
        y = (
            self.structuralSections[iSection].yReference
            + self.structuralSections[iSection].spanSection / 2.0
        )
        z = self.structuralSections[iSection].zReference

        nMassPoints = self.numberOfSections
        massVector = [0 for _ in range(nMassPoints * 4)]

        for iSectionTemp in range(iSection + 1, self.numberOfSections):
            mass = self.structuralSections[iSectionTemp].mass
            massVector[iSectionTemp * 4] = mass
            massVector[iSectionTemp * 4 + 1] = (
                self.structuralSections[iSectionTemp].massX / mass
            )
            massVector[iSectionTemp * 4 + 2] = (
                self.structuralSections[iSectionTemp].massY / mass
            )
            massVector[iSectionTemp * 4 + 3] = (
                self.structuralSections[iSectionTemp].massZ / mass
            )

        [S, M] = self.get_accumulated_load(
            self.loadsGroup.span_loads[iLoad],
            [x, y, z],
            self.posEngines,
            self.mssEngines,
            self.posSupport,
            massVector,
            self.loadFactor,
        )

        torsion = self.structuralSections[iSection].torsion

        SxT = S[0] * math.cos(torsion) - S[2] * math.sin(torsion)
        SzT = S[2] * math.cos(torsion) + S[0] * math.sin(torsion)
        MxT = M[0] * math.cos(torsion) - M[2] * math.sin(torsion)
        MzT = M[2] * math.cos(torsion) + M[0] * math.sin(torsion)

        return [SxT, S[1], SzT, MxT, M[1], MzT]

    def externalforcesInRibs(self, iRib: int, iLoad: int) -> list[float]:
        if iRib == 0:
            y0 = -self.structuralSections[iRib].yReference
            yf = self.structuralSections[iRib].yReference
        elif iRib == self.numberOfRibs - 1:
            y0 = self.structuralSections[iRib - 1].yReference
            yf = y0 + self.structuralSections[iRib - 1].spanSection
        else:
            y0 = self.structuralSections[iRib - 1].yReference
            yf = self.structuralSections[iRib].yReference

        F = self.get_rib_load(
            self.loadsGroup.span_loads[iLoad],
            y0,
            yf,
            self.posEngines,
            self.mssEngines,
            self.posSupport,
            self.loadFactor,
        )

        return [F[0], F[1]]

    def internalForcesInRibs(self, iRib: int) -> list[float]:
        if iRib == 0:
            leadingBrazierZ = self.structuralSections[iRib].leadingBrazierZ
            boxBrazierX = self.structuralSections[iRib].boxBrazierX
            boxBrazierZ = self.structuralSections[iRib].boxBrazierZ
            trailingBrazierZ = self.structuralSections[iRib].trailingBrazierZ
        elif iRib == self.numberOfRibs - 1:
            leadingBrazierZ = self.structuralSections[iRib - 1].leadingBrazierZ / 2.0
            boxBrazierX = self.structuralSections[iRib - 1].boxBrazierX / 2.0
            boxBrazierZ = self.structuralSections[iRib - 1].boxBrazierZ / 2.0
            trailingBrazierZ = self.structuralSections[iRib - 1].trailingBrazierZ / 2.0
        else:
            leadingBrazierZ = (
                self.structuralSections[iRib - 1].leadingBrazierZ
                + self.structuralSections[iRib].leadingBrazierZ
            ) / 2.0
            boxBrazierX = (
                self.structuralSections[iRib - 1].boxBrazierX
                + self.structuralSections[iRib].boxBrazierX
            ) / 2.0
            boxBrazierZ = (
                self.structuralSections[iRib - 1].boxBrazierZ
                + self.structuralSections[iRib].boxBrazierZ
            ) / 2.0
            trailingBrazierZ = (
                self.structuralSections[iRib - 1].trailingBrazierZ
                + self.structuralSections[iRib].trailingBrazierZ
            ) / 2.0

        return [leadingBrazierZ, boxBrazierX, boxBrazierZ, trailingBrazierZ]

    def get_accumulated_load(
        self,
        load: SpanLoadsVariable,
        point: list[float],
        pos_engines: list[float],
        mass_engines: list[float],
        pos_supports: list[float],
        mass_points: list[float],
        load_factor: float = 1.0,
    ) -> list[list[float]]:
        # Use numpy as AMPET is implemented using lists now
        l_point = np.array(point)
        l_pos_engines = np.array(pos_engines).reshape((-1, 3))
        l_mass_engines = np.array(mass_engines)
        l_pos_supports = np.array(pos_supports).reshape((-1, 3))
        l_mass_points = np.array(mass_points).reshape((-1, 4))

        s = np.zeros(3)
        m = np.zeros(3)

        self._station_contribution(load, l_point, s, m)
        self._engine_contribution(load, l_point, l_pos_engines, l_mass_engines, s, m)
        self._support_contribution(load, l_point, l_pos_supports, s, m)
        self._mass_contribution(load, l_point, l_mass_points, s, m)

        m *= load_factor
        s *= load_factor

        return [m.tolist(), s.tolist()]

    def _station_contribution(
        self,
        load: SpanLoadsVariable,
        point: NDArray[np.float64],
        s: NDArray[np.float64],
        m: NDArray[np.float64],
    ) -> None:
        h = point[1] - load.origin[1]

        for i in range(load.n_stations):
            h_delta = load.spans[i]
            h_application = load.centers[i]
            h0 = h_application - h_delta / 2.0
            hf = h_application + h_delta / 2.0

            if h <= h0:
                h_delta = 0.0
            elif h <= hf:
                h_delta = hf - h
                h_application = h0 + h_delta / 2.0

            si = (
                load.fx[i] * h_delta * load.u
                + load.fy[i] * h_delta * load.v
                + load.fz[i] * h_delta * load.w
            )
            s += si

            i_point = load.origin + np.array([0.0, h_application, 0.0])
            m += load.m[i] * h_delta * load.v + np.cross(i_point - point, si)
            m += np.cross(i_point - point, si)

    def _engine_contribution(
        self,
        load: SpanLoadsVariable,
        point: NDArray[np.float64],
        pos_engines: NDArray[np.float64],
        mass_engines: NDArray[np.float64],
        s: NDArray[np.float64],
        m: NDArray[np.float64],
    ) -> None:
        h = point[1] - load.origin[1]
        n_engines = pos_engines.shape[0]
        load_engines = -load.thrust / np.sum(mass_engines)

        for i in range(n_engines):
            si = np.array([load_engines * mass_engines[i], 0.0, 0.0])
            si += mass_engines[i] * 9.81 * np.array([load.u[2], load.v[2], load.w[2]])

            i_point = pos_engines[i, :]
            mi = np.cross(i_point - point, si)

            h_point = load.v @ i_point
            if h >= h_point:
                s += si
                m += mi

    def _support_contribution(
        self,
        load: SpanLoadsVariable,
        point: NDArray[np.float64],
        pos_supports: NDArray[np.float64],
        s: NDArray[np.float64],
        m: NDArray[np.float64],
    ) -> None:
        h = point[1] - load.origin[1]
        lift = load.spans @ load.fz
        n_supports = pos_supports.shape[0]
        load_supports = -lift / n_supports

        for i in range(n_supports):
            si = load_supports * np.array([load.u[2], load.v[2], load.w[2]])

            i_point = pos_supports[i, :]
            mi = np.cross(i_point - point, si)

            h_point = load.v @ i_point
            if h >= h_point:
                s += si
                m += mi

    def _mass_contribution(
        self,
        load: SpanLoadsVariable,
        point: NDArray[np.float64],
        mass_points: NDArray[np.float64],
        s: NDArray[np.float64],
        m: NDArray[np.float64],
    ) -> None:
        h = point[1] - load.origin[1]
        n_mass_points = mass_points.shape[0]

        for i in range(n_mass_points):
            i_point = mass_points[i, 1:]
            rel_v = np.cross(load.gamma_p, i_point - load.origin)
            i_point_pp = load.origin_pp + np.cross(load.gamma_pp, i_point - load.origin)
            i_point_pp += np.cross(load.gamma_p, rel_v)
            i_point_pp -= 9.81 * np.array([load.u[2], load.v[2], load.w[2]])

            si = -mass_points[i, 0] * i_point_pp
            mi = np.cross(i_point - point, si)

            h_point = load.v @ i_point
            if h >= h_point:
                s += si
                m += mi

    def get_rib_load(
        self,
        load: SpanLoadsVariable,
        y0: float,
        yf: float,
        pos_engines: list[float],
        mass_engines: list[float],
        pos_support: list[float],
        load_factor: float = 1.0,
    ) -> list[float]:
        l_pos_engines = np.array(pos_engines).reshape((-1, 3))
        l_pos_support = np.array(pos_support).reshape((-1, 3))
        n_engines = l_pos_engines.shape[0]
        n_supports = l_pos_support.shape[0]

        lift = np.dot(load.spans, load.fz)
        load_support = lift / n_supports
        load_engines = load.thrust / np.sum(mass_engines)

        fx = 0.0
        fz = 0.0

        for i_engine in range(n_engines):
            if (y0 <= l_pos_engines[i_engine, 1]) and (
                yf >= l_pos_engines[i_engine, 1]
            ):
                fx += load_engines * mass_engines[i_engine]
                fz += mass_engines[i_engine] * 9.81

        for i_support in range(n_supports):
            if (y0 <= l_pos_support[i_support, 1]) and (
                yf >= l_pos_support[i_support, 1]
            ):
                fz += load_support

        fx *= load_factor
        fz *= load_factor

        return [fx, fz]

    def wingCaps2Vars(self, ACapRoot: float, AStringerRoot: float) -> None:
        [chord, thickness, _, _, _, _, _, _] = self.profileDimensions(0.0)
        areaFactor0 = 1.0 / (chord * chord * thickness)

        for iSection in range(self.numberOfSections):
            chord = self.structuralSections[iSection].chord
            thickness = self.structuralSections[iSection].relativeThickness
            areaFactor = chord * chord * thickness * areaFactor0
            ACap = ACapRoot * areaFactor
            AStringer = AStringerRoot * areaFactor
            self.structuralSections[iSection].prepareSection2Vars(ACap, AStringer)

    def wingSizing2Vars(
        self,
        ACapRoot: float,
        AStringerRoot: float,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float, float]:
        self.wingCaps2Vars(ACapRoot, AStringerRoot)

        mass = 0.0
        massProperties = np.zeros((self.numberOfSections + 1, 10))
        stiffnessMatrix = np.zeros((self.numberOfSections, 21))  # Symmetric matrix
        minRFCompression = 1000.0
        minRFTraction = 1000.0
        matrixQ = np.eye(3)

        for iSection in reversed(range(self.numberOfSections)):
            tWebsBox = [0.0 for _ in range(self.structuralSections[iSection].nWebs)]

            for iLoad in range(self.loadsGroup.n_loads):
                [Sx, Sy, Sz, Mx, My, Mz] = self.externalForcesAndMomentsInSection(
                    iSection,
                    iLoad,
                )

                self.structuralSections[iSection].calculateBoxThickness(
                    Sx,
                    Sy,
                    Sz,
                    Mx,
                    My,
                    Mz,
                )

                for iWeb in range(self.structuralSections[iSection].nWebs):
                    tWebsBox[iWeb] = max(
                        tWebsBox[iWeb],
                        self.structuralSections[iSection].tWebsBox[iWeb],
                    )

                minRFCompression = min(
                    minRFCompression,
                    self.structuralSections[iSection].minRFCompression,
                )
                minRFTraction = min(
                    minRFTraction,
                    self.structuralSections[iSection].minRFTraction,
                )

            for iWeb in range(self.structuralSections[iSection].nWebs):
                self.structuralSections[iSection].tWebsBox[iWeb] = tWebsBox[iWeb]

            tLeadingEdge = 0.0
            tUpperTrailingEdge = 0.0
            tLowerTrailingEdge = 0.0

            for iLoad in range(self.loadsGroup.n_loads):
                [Sx, Sy, Sz, Mx, My, Mz] = self.externalForcesAndMomentsInSection(
                    iSection,
                    iLoad,
                )

                self.structuralSections[iSection].calculateEdgesThickness(Sx, Sz, My)

                tLeadingEdge = max(
                    tLeadingEdge,
                    self.structuralSections[iSection].tLeadingEdge,
                )

                tUpperTrailingEdge = max(
                    tUpperTrailingEdge,
                    self.structuralSections[iSection].tUpperTrailingEdge,
                )

                tLowerTrailingEdge = max(
                    tLowerTrailingEdge,
                    self.structuralSections[iSection].tLowerTrailingEdge,
                )

            self.structuralSections[iSection].tLeadingEdge = tLeadingEdge
            self.structuralSections[iSection].tUpperTrailingEdge = tUpperTrailingEdge
            self.structuralSections[iSection].tLowerTrailingEdge = tLowerTrailingEdge

            leadingBrazierZ = 0.0
            boxBrazierZ = 0.0
            boxBrazierX = 0.0
            trailingBrazierZ = 0.0

            for iLoad in range(self.loadsGroup.n_loads):
                [Sx, Sy, Sz, Mx, My, Mz] = self.externalForcesAndMomentsInSection(
                    iSection,
                    iLoad,
                )

                self.structuralSections[iSection].calculateSectionBraziersForces(
                    Sy,
                    Mx,
                    Mz,
                )

                leadingBrazierZ = max(
                    leadingBrazierZ,
                    self.structuralSections[iSection].leadingBrazierZ,
                )

                boxBrazierZ = max(
                    boxBrazierZ,
                    self.structuralSections[iSection].boxBrazierZ,
                )

                boxBrazierX = max(
                    boxBrazierX,
                    self.structuralSections[iSection].boxBrazierX,
                )

                trailingBrazierZ = max(
                    trailingBrazierZ,
                    self.structuralSections[iSection].trailingBrazierZ,
                )

            self.structuralSections[iSection].leadingBrazierZ = leadingBrazierZ
            self.structuralSections[iSection].boxBrazierZ = boxBrazierZ
            self.structuralSections[iSection].boxBrazierX = boxBrazierX
            self.structuralSections[iSection].trailingBrazierZ = trailingBrazierZ

            self.structuralSections[iSection].calculateSectionMassProperties()

            massProperties[iSection + 1, 0] += (
                self.structuralSections[iSection].mass / 2.0
            )
            massProperties[iSection + 1, 1] += (
                self.structuralSections[iSection].massX / 2.0
            )
            massProperties[iSection + 1, 2] += self.structuralSections[
                iSection
            ].massYHalf2
            massProperties[iSection + 1, 3] += (
                self.structuralSections[iSection].massZ / 2.0
            )
            massProperties[iSection + 1, 4] += (
                self.structuralSections[iSection].massXX / 2.0
            )
            massProperties[iSection + 1, 5] += self.structuralSections[
                iSection
            ].massYYHalf
            massProperties[iSection + 1, 6] += (
                self.structuralSections[iSection].massZZ / 2.0
            )
            massProperties[iSection + 1, 7] += self.structuralSections[
                iSection
            ].massXYHalf2
            massProperties[iSection + 1, 8] += (
                self.structuralSections[iSection].massXZ / 2.0
            )
            massProperties[iSection + 1, 9] += self.structuralSections[
                iSection
            ].massYZHalf2

            massProperties[iSection, 0] += self.structuralSections[iSection].mass / 2.0
            massProperties[iSection, 1] += self.structuralSections[iSection].massX / 2.0
            massProperties[iSection, 2] += self.structuralSections[iSection].massYHalf1
            massProperties[iSection, 3] += self.structuralSections[iSection].massZ / 2.0
            massProperties[iSection, 4] += (
                self.structuralSections[iSection].massXX / 2.0
            )
            massProperties[iSection, 5] += self.structuralSections[iSection].massYYHalf
            massProperties[iSection, 6] += (
                self.structuralSections[iSection].massZZ / 2.0
            )
            massProperties[iSection, 7] += self.structuralSections[iSection].massXYHalf1
            massProperties[iSection, 8] += (
                self.structuralSections[iSection].massXZ / 2.0
            )
            massProperties[iSection, 9] += self.structuralSections[iSection].massYZHalf1

            self.structuralSections[iSection].calculateSectionStiffness()

            s00 = self.structuralSections[iSection].shearStiffnessMatrix[3]
            s01 = self.structuralSections[iSection].shearStiffnessMatrix[1]
            s02 = self.structuralSections[iSection].shearStiffnessMatrix[4]
            s03 = self.structuralSections[iSection].bendingStiffnessMatrix[0]
            s04 = self.structuralSections[iSection].bendingStiffnessMatrix[2]
            s05 = self.structuralSections[iSection].bendingStiffnessMatrix[1]
            s06 = self.structuralSections[iSection].shearStiffnessMatrix[0]
            s07 = self.structuralSections[iSection].shearStiffnessMatrix[2]
            s08 = self.structuralSections[iSection].bendingStiffnessMatrix[5]
            s09 = self.structuralSections[iSection].bendingStiffnessMatrix[4]
            s10 = self.structuralSections[iSection].shearStiffnessMatrix[5]
            s11 = self.structuralSections[iSection].bendingStiffnessMatrix[3]

            stiffnessMatrix[iSection, 0] = s00
            stiffnessMatrix[iSection, 2] = s01
            stiffnessMatrix[iSection, 4] = s02
            stiffnessMatrix[iSection, 6] = s03
            stiffnessMatrix[iSection, 8] = s04
            stiffnessMatrix[iSection, 10] = s05
            stiffnessMatrix[iSection, 11] = s06
            stiffnessMatrix[iSection, 13] = s07
            stiffnessMatrix[iSection, 15] = s08
            stiffnessMatrix[iSection, 17] = s09
            stiffnessMatrix[iSection, 18] = s10
            stiffnessMatrix[iSection, 20] = s11

            mass += self.structuralSections[iSection].mass

        mass += self.computeRibs()

        for iSection in range(self.numberOfSections):
            if massProperties[iSection, 0] != 0.0:
                u = massProperties[iSection, 1] / massProperties[iSection, 0]
                v = massProperties[iSection, 2] / massProperties[iSection, 0]
                w = massProperties[iSection, 3] / massProperties[iSection, 0]
                muu = (
                    massProperties[iSection, 4]
                    - massProperties[iSection, 1]
                    * massProperties[iSection, 1]
                    * massProperties[iSection, 0]
                )
                mvv = (
                    massProperties[iSection, 5]
                    - massProperties[iSection, 2]
                    * massProperties[iSection, 2]
                    * massProperties[iSection, 0]
                )
                mww = (
                    massProperties[iSection, 6]
                    - massProperties[iSection, 3]
                    * massProperties[iSection, 3]
                    * massProperties[iSection, 0]
                )
                muv = (
                    massProperties[iSection, 7]
                    - massProperties[iSection, 1]
                    * massProperties[iSection, 2]
                    * massProperties[iSection, 0]
                )
                muw = (
                    massProperties[iSection, 8]
                    - massProperties[iSection, 1]
                    * massProperties[iSection, 3]
                    * massProperties[iSection, 0]
                )
                mvw = (
                    massProperties[iSection, 9]
                    - massProperties[iSection, 2]
                    * massProperties[iSection, 3]
                    * massProperties[iSection, 0]
                )

                Iuu = mvv + mww
                Ivv = muu + mww
                Iww = muu + mvv

                I = np.array([[Iuu, -muv, -muw], [-muv, Ivv, -mvw], [-muw, -mvw, Iww]])

                x = matrixQ.dot(np.array([u, v, w]))

                I = np.linalg.inv(matrixQ).dot(I.dot(matrixQ))

                massProperties[iSection, 1] = x[0]
                massProperties[iSection, 2] = x[1]
                massProperties[iSection, 3] = x[2]

                massProperties[iSection, 4] = I[0, 0]
                massProperties[iSection, 5] = I[1, 1]
                massProperties[iSection, 6] = I[2, 2]
                massProperties[iSection, 7] = -I[0, 1]
                massProperties[iSection, 8] = -I[0, 2]
                massProperties[iSection, 9] = -I[1, 2]

                s00 = stiffnessMatrix[iSection, 0]
                s01 = stiffnessMatrix[iSection, 2]
                s02 = stiffnessMatrix[iSection, 4]
                s03 = stiffnessMatrix[iSection, 6]
                s04 = stiffnessMatrix[iSection, 8]
                s05 = stiffnessMatrix[iSection, 10]
                s06 = stiffnessMatrix[iSection, 11]
                s07 = stiffnessMatrix[iSection, 13]
                s08 = stiffnessMatrix[iSection, 15]
                s09 = stiffnessMatrix[iSection, 17]
                s10 = stiffnessMatrix[iSection, 18]
                s11 = stiffnessMatrix[iSection, 20]

                sMatrix = np.array(
                    [
                        [s00, 0.0, s01, 0.0, s02, 0.0],
                        [0.0, s03, 0.0, s04, 0.0, s05],
                        [s01, 0.0, s06, 0.0, s07, 0.0],
                        [0.0, s04, 0.0, s08, 0.0, s09],
                        [s02, 0.0, s07, 0.0, s10, 0.0],
                        [0.0, s05, 0.0, s09, 0.0, s11],
                    ],
                )

                matrixQ2 = np.zeros((6, 6))
                matrixQ2[0:3, 0:3] = matrixQ
                matrixQ2[3:6, 3:6] = matrixQ

                sMatrix = np.linalg.inv(matrixQ2).dot(sMatrix.dot(matrixQ2))

                stiffnessMatrix[iSection, 0] = sMatrix[0, 0]
                stiffnessMatrix[iSection, 1] = sMatrix[0, 1]
                stiffnessMatrix[iSection, 2] = sMatrix[0, 2]
                stiffnessMatrix[iSection, 3] = sMatrix[0, 3]
                stiffnessMatrix[iSection, 4] = sMatrix[0, 4]
                stiffnessMatrix[iSection, 5] = sMatrix[0, 5]
                stiffnessMatrix[iSection, 6] = sMatrix[1, 1]
                stiffnessMatrix[iSection, 7] = sMatrix[1, 2]
                stiffnessMatrix[iSection, 8] = sMatrix[1, 3]
                stiffnessMatrix[iSection, 9] = sMatrix[1, 4]
                stiffnessMatrix[iSection, 10] = sMatrix[1, 5]
                stiffnessMatrix[iSection, 11] = sMatrix[2, 2]
                stiffnessMatrix[iSection, 12] = sMatrix[2, 3]
                stiffnessMatrix[iSection, 13] = sMatrix[2, 4]
                stiffnessMatrix[iSection, 14] = sMatrix[2, 5]
                stiffnessMatrix[iSection, 15] = sMatrix[3, 3]
                stiffnessMatrix[iSection, 16] = sMatrix[3, 4]
                stiffnessMatrix[iSection, 17] = sMatrix[3, 5]
                stiffnessMatrix[iSection, 18] = sMatrix[4, 4]
                stiffnessMatrix[iSection, 19] = sMatrix[4, 5]
                stiffnessMatrix[iSection, 20] = sMatrix[5, 5]

        return (
            mass * 2.0,
            massProperties,
            stiffnessMatrix,
            minRFCompression,
            minRFTraction,
        )

    def computeRibs(self) -> float:
        mass = 0.0

        for iRib in range(self.numberOfRibs):
            tLeading = 0.0
            tBox = 0.0
            tTrailing = 0.0

            torsion = self.structuralRibs[iRib].torsion

            [leadingBrazierZ, boxBrazierX, boxBrazierZ, trailingBrazierZ] = (
                self.internalForcesInRibs(iRib)
            )

            for iLoad in range(self.loadsGroup.n_loads):
                [Fx0, Fz0] = self.externalforcesInRibs(iRib, iLoad)

                Fx = Fx0 * math.cos(torsion) - Fz0 * math.sin(torsion)
                Fz = Fz0 * math.cos(torsion) + Fx0 * math.sin(torsion)

                self.structuralRibs[iRib].calculateRibThickness(
                    Fx,
                    Fz,
                    0.0,
                    leadingBrazierZ,
                    boxBrazierX,
                    boxBrazierZ,
                    0.0,
                    trailingBrazierZ,
                )

                tLeading = max(tLeading, self.structuralRibs[iRib].tLeading)
                tBox = max(tBox, self.structuralRibs[iRib].tBox)
                tTrailing = max(tTrailing, self.structuralRibs[iRib].tTrailing)

            self.structuralRibs[iRib].tLeading = tLeading
            self.structuralRibs[iRib].tBox = tBox
            self.structuralRibs[iRib].tTrailing = tTrailing

            self.structuralRibs[iRib].calculateRibMass()

            if iRib == 0:
                mass += self.structuralRibs[iRib].mass / 2.0
            else:
                mass += self.structuralRibs[iRib].mass

        return mass
