from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from multiads.scenario import InnerVariableFloatNP

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self


class AeroDerivativesVariable(InnerVariableFloatNP):
    """Aerodynamic derivatives of forces and moments.

    0  - Alpha
    1  - Beta
    2  - p
    3  - q
    4  - r
    5  - Alpha dot
    6  - Beta dot
    7  - Pitching
    8  - Yawing
    9  - Delta aileron
    10 - Delta elevator
    11 - Delta rudder
    """

    @classmethod
    def empty(cls, name: str) -> Self:
        value = np.empty(12 * 6)
        return cls(name, value)

    @classmethod
    def zeros(cls, name: str) -> Self:
        value = np.zeros(12 * 6)
        return cls(name, value)

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.reshape(self.value_np, (12, 6))

    @matrix.setter
    def matrix(self, value: NDArray[np.float64]) -> None:
        self.value_np[:] = np.ravel(value)

    @property
    def fx(self) -> NDArray[np.float64]:
        return self.matrix[:, 0]

    @fx.setter
    def fx(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 0] = value

    @property
    def fy(self) -> NDArray[np.float64]:
        return self.matrix[:, 1]

    @fy.setter
    def fy(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 1] = value

    @property
    def fz(self) -> NDArray[np.float64]:
        return self.matrix[:, 2]

    @fz.setter
    def fz(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 2] = value

    @property
    def mx(self) -> NDArray[np.float64]:
        return self.matrix[:, 3]

    @mx.setter
    def mx(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 3] = value

    @property
    def my(self) -> NDArray[np.float64]:
        return self.matrix[:, 4]

    @my.setter
    def my(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 4] = value

    @property
    def mz(self) -> NDArray[np.float64]:
        return self.matrix[:, 5]

    @mz.setter
    def mz(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 5] = value
