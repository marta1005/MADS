from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from multiads.scenario import InnerVariableFloatNP

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self


# HACK @Andres: Should this be here?
POLAR_DEFAULT_AOA = np.array(
    [
        -180.0,
        -135.0,
        -90.0,
        -45.0,
        -20.0,
        -15.0,
        -12.5,
        -10.0,
        -8.0,
        -5.0,
        0.0,
        5.0,
        8.0,
        10.0,
        12.5,
        15.0,
        20.0,
        45.0,
        90.0,
        135.0,
        180.0,
    ],
)
POLAR_DEFAULT_SIZE = len(POLAR_DEFAULT_AOA)


class PolarVariable(InnerVariableFloatNP):
    @classmethod
    def from_num_points(
        cls,
        name: str,
        num_points: int = POLAR_DEFAULT_SIZE,
        num_polars: int = 1,
    ) -> Self:
        arr_size = 1 + num_polars * (2 + 4 * num_points)
        value = np.empty(arr_size)
        value[0] = num_polars
        return cls(name, value)

    def _set_chunk(self, i0: int, i1: int, value: NDArray[np.float64]) -> None:
        self.value_np[i0:i1] = value

    def _get_chunk(self, i0: int, i1: int) -> NDArray[np.float64]:
        return self.value_np[i0:i1]

    @property
    def num_polars(self) -> int:
        return int(self.value_np[0])

    @property
    def num_points(self) -> int:
        block_size = (len(self.value_np) - 1) // self.num_polars
        return (block_size - 2) // 4

    @property
    def mach(self) -> NDArray[np.float64]:
        i0 = 1
        i1 = i0 + self.num_polars
        return self._get_chunk(i0, i1)

    @mach.setter
    def mach(self, value: NDArray[np.float64]) -> None:
        i0 = 1
        i1 = i0 + self.num_polars
        self._set_chunk(i0, i1, value)

    @property
    def reynolds(self) -> NDArray[np.float64]:
        i0 = 1 + self.num_polars
        i1 = i0 + self.num_polars
        return self._get_chunk(i0, i1)

    @reynolds.setter
    def reynolds(self, value: NDArray[np.float64]) -> None:
        i0 = 1 + self.num_polars
        i1 = i0 + self.num_polars
        self._set_chunk(i0, i1, value)

    @property
    def aoa(self) -> NDArray[np.float64]:
        i0 = 1 + 2 * self.num_polars
        i1 = i0 + self.num_points * self.num_polars
        return self._get_chunk(i0, i1).reshape((self.num_polars, self.num_points))

    @aoa.setter
    def aoa(self, value: NDArray[np.float64]) -> None:
        i0 = 1 + 2 * self.num_polars
        i1 = i0 + self.num_points * self.num_polars
        self._set_chunk(i0, i1, value.flatten())

    @property
    def cl(self) -> NDArray[np.float64]:
        n = self.num_points * self.num_polars
        i0 = 1 + 2 * self.num_polars + n
        i1 = i0 + n
        return self._get_chunk(i0, i1).reshape((self.num_polars, self.num_points))

    @cl.setter
    def cl(self, value: NDArray[np.float64]) -> None:
        n = self.num_points * self.num_polars
        i0 = 1 + 2 * self.num_polars + n
        i1 = i0 + n
        self._set_chunk(i0, i1, value.flatten())

    @property
    def cd(self) -> NDArray[np.float64]:
        n = self.num_points * self.num_polars
        i0 = 1 + 2 * self.num_polars + 2 * n
        i1 = i0 + n
        return self._get_chunk(i0, i1).reshape((self.num_polars, self.num_points))

    @cd.setter
    def cd(self, value: NDArray[np.float64]) -> None:
        n = self.num_points * self.num_polars
        i0 = 1 + 2 * self.num_polars + 2 * n
        i1 = i0 + n
        self._set_chunk(i0, i1, value.flatten())

    @property
    def cm(self) -> NDArray[np.float64]:
        n = self.num_points * self.num_polars
        i0 = 1 + 2 * self.num_polars + 3 * n
        i1 = i0 + n
        return self._get_chunk(i0, i1).reshape((self.num_polars, self.num_points))

    @cm.setter
    def cm(self, value: NDArray[np.float64]) -> None:
        n = self.num_points * self.num_polars
        i0 = 1 + 2 * self.num_polars + 3 * n
        i1 = i0 + n
        self._set_chunk(i0, i1, value.flatten())
