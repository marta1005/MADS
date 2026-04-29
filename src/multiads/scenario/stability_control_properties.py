from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from multiads.scenario import InnerVariableFloatNP

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self


class StabilityControlVariable(InnerVariableFloatNP):
    """Coordinates of the neutral point, and global forces and moments."""

    @classmethod
    def empty(cls, name: str) -> Self:
        value = np.empty(9)
        return cls(name, value)

    @classmethod
    def zeros(cls, name: str) -> Self:
        value = np.zeros(9)
        return cls(name, value)

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.reshape(self.value_np, (3, 3))

    @matrix.setter
    def matrix(self, value: NDArray[np.float64]) -> None:
        self.value_np[:] = np.ravel(value)

    @property
    def neutral_point(self) -> NDArray[np.float64]:
        return self.matrix[0]

    @neutral_point.setter
    def neutral_point(self, value: NDArray[np.float64]) -> None:
        self.matrix[0] = value

    @property
    def lift(self) -> int:
        return self.matrix[1, 2]

    @lift.setter
    def lift(self, value: float) -> None:
        self.matrix[1, 2] = value

    @property
    def drag(self) -> int:
        return self.matrix[1, 0]

    @drag.setter
    def drag(self, value: float) -> None:
        self.matrix[1, 0] = value

    @property
    def forces(self) -> NDArray[np.float64]:
        return self.matrix[1]

    @forces.setter
    def forces(self, value: NDArray[np.float64]) -> None:
        self.matrix[1] = value

    @property
    def moments(self) -> NDArray[np.float64]:
        return self.matrix[2]

    @moments.setter
    def moments(self, value: NDArray[np.float64]) -> None:
        self.matrix[2] = value
