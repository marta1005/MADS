from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from multiads.scenario import InnerVariable

N = TypeVar("N", NDArray[np.int32], NDArray[np.float64])


class MatrixVariable(InnerVariable[N, N]):
    @classmethod
    def empty(cls, name: str, rows: int, cols: int) -> Self:
        value = np.empty(1 + rows * cols)
        value[0] = rows
        return cls(name, value)

    @classmethod
    def zeros(cls, name: str, rows: int, cols: int) -> Self:
        value = np.zeros(1 + rows * cols)
        value[0] = rows
        return cls(name, value)

    @classmethod
    def ones(cls, name: str, rows: int, cols: int) -> Self:
        value = np.ones(1 + rows * cols)
        value[0] = rows
        return cls(name, value)

    @property
    def value(self) -> N:
        return self.value_np[1:]  # type: ignore[invalid-return-type]

    @value.setter
    def value(self, value: N) -> None:
        self.value_np[1:] = value

    @property
    def matrix(self) -> N:
        n = int(self.value_np[0])
        return self.value_np[1:].reshape((n, -1))  # type: ignore[invalid-return-type]

    @matrix.setter
    def matrix(self, value: N) -> None:
        self.value = np.ravel(value)


MatrixVariableInt = MatrixVariable[NDArray[np.int32]]
MatrixVariableFloat = MatrixVariable[NDArray[np.float64]]
