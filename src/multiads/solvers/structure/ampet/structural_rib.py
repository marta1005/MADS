# ruff: noqa: N802, N803, N806

from multiads.solvers.structure.ampet.buckling_plate import BucklingPlate
from multiads.solvers.structure.ampet.structural_material import StructuralMaterial
from multiads.solvers.structure.ampet.structural_profile import StructuralProfile


class StructuralRib:
    def __init__(
        self,
        unitProfile: StructuralProfile,
        material: StructuralMaterial,
        chord: float,
        relativeThickness: float,
        xReference: float,
        yReference: float,
        zReference: float,
        torsion: float,
    ) -> None:
        self.material = material
        self.wingProfile = unitProfile.rescale(chord, relativeThickness)

        self.leadingLengthX = self.wingProfile.frontSparPosition - 0.25 * chord
        self.leadingLengthZ = abs(self.wingProfile.leadingArea) / self.leadingLengthX
        self.leadingPlate = BucklingPlate(
            self.leadingLengthX,
            self.leadingLengthZ,
            material,
        )

        self.boxLengthX = (
            self.wingProfile.rearSparPosition - self.wingProfile.frontSparPosition
        )
        self.boxLengthZ = abs(self.wingProfile.boxArea) / self.boxLengthX
        self.boxPlate = BucklingPlate(self.boxLengthX, self.boxLengthZ, material)

        self.trailingLengthX = 0.75 * chord - self.wingProfile.rearSparPosition
        self.trailingLengthZ = abs(self.wingProfile.trailingArea) / self.trailingLengthX
        self.trailingPlate = BucklingPlate(
            self.trailingLengthX,
            self.trailingLengthZ,
            material,
        )

        self.xReference = xReference
        self.yReference = yReference
        self.zReference = zReference

        self.torsion = torsion

    def calculateRibThickness(
        self,
        loadX: float,
        loadZ: float,
        leadingFX: float,
        leadingFZ: float,
        boxFX: float,
        boxFZ: float,
        trailingFX: float,
        trailingFZ: float,
    ) -> None:
        boxShear = abs(loadX / self.boxLengthX) + abs(loadZ / self.boxLengthZ)

        self.tLeading = self.leadingPlate.CalculateThickness(
            0.0,
            leadingFX,
            0.0,
            0.0,
            leadingFZ,
            0.0,
            0.0,
        )
        self.tBox = self.boxPlate.CalculateThickness(
            boxShear,
            boxFX,
            0.0,
            0.0,
            boxFZ,
            0.0,
            0.0,
        )
        self.tTrailing = self.trailingPlate.CalculateThickness(
            0.0,
            trailingFX,
            0.0,
            0.0,
            trailingFZ,
            0.0,
            0.0,
        )

    def calculateRibMass(self) -> None:
        mass = self.tLeading * abs(self.wingProfile.leadingArea)
        mass += self.tBox * abs(self.wingProfile.boxArea)
        mass += self.tTrailing * abs(self.wingProfile.trailingArea)

        self.mass = mass * self.material.rho
