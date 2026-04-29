import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from multiads.scenario import InnerVariableFloatNP


class MassPropertiesVariable(InnerVariableFloatNP):
    """Mass properties of `num_elements` elements.

    0    - Mass
    1:4  - CG coordinates
    4:10 - Inertia vector (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    """

    VECTOR_SIZE = 10

    @classmethod
    def from_num_elements(cls, name: str, num_elements: int) -> Self:
        value = np.zeros(1 + cls.VECTOR_SIZE * num_elements)
        value[0] = num_elements
        return cls(name, value)

    @property
    def max_num_elements(self) -> int:
        return (len(self.value_np) - 1) // self.VECTOR_SIZE

    @property
    def num_elements(self) -> int:
        return int(self.value_np[0])

    @num_elements.setter
    def num_elements(self, value: int) -> None:
        if value > self.max_num_elements:
            msg = (
                f"'{type(self).__name__}' with max size {self.max_num_elements} "
                f"cannot hold {value} element(s)."
            )
            raise ValueError(msg)

        self.value_np[0] = np.float64(value)
        self.value_np[self._data_size() :] = 0.0

    @property
    def value(self) -> NDArray[np.float64]:
        return self.value_np[1 : self._data_size()]

    @value.setter
    def value(self, value: NDArray[np.float64]) -> None:
        self.value_np[1 : self._data_size()] = value

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Return the mass matrix (num_elements, 10) of the system."""
        return np.reshape(self.value, (self.num_elements, self.VECTOR_SIZE))

    @matrix.setter
    def matrix(self, value: NDArray[np.float64]) -> None:
        self.value = np.ravel(value)

    @property
    def mass(self) -> NDArray[np.float64]:
        return self.matrix[:, 0]

    @mass.setter
    def mass(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 0] = value

    @property
    def cg(self) -> NDArray[np.float64]:
        return self.matrix[:, 1:4]

    @cg.setter
    def cg(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 1:4] = value

    @property
    def inertia(self) -> NDArray[np.float64]:
        inertia = np.empty((self.num_elements, 3, 3))
        l_matrix = self.matrix
        for i in range(inertia.shape[0]):
            inertia[i] = np.array(
                [
                    [l_matrix[i, 4], -l_matrix[i, 7], -l_matrix[i, 8]],
                    [-l_matrix[i, 7], l_matrix[i, 5], -l_matrix[i, 9]],
                    [-l_matrix[i, 8], -l_matrix[i, 9], l_matrix[i, 6]],
                ],
            )
        return inertia

    @inertia.setter
    def inertia(self, value: NDArray[np.float64]) -> None:
        l_matrix = self.matrix
        l_matrix[:, 4] = value[:, 0, 0]
        l_matrix[:, 5] = value[:, 1, 1]
        l_matrix[:, 6] = value[:, 2, 2]
        l_matrix[:, 7] = -value[:, 0, 1]
        l_matrix[:, 8] = -value[:, 0, 2]
        l_matrix[:, 9] = -value[:, 1, 2]

    @property
    def inertia_vector(self) -> NDArray[np.float64]:
        return self.matrix[:, 4:10]

    @inertia_vector.setter
    def inertia_vector(self, value: NDArray[np.float64]) -> None:
        self.matrix[:, 4:10] = value

    def join(self) -> "MassPropertiesVariable":
        """Return the joint mass properties of the system."""
        matrix = self.matrix
        r_val = np.zeros(10)

        for i in range(self.num_elements):
            r_val[0] += matrix[i, 0]  # mass
            r_val[1] += matrix[i, 1] * matrix[i, 0]  # mx
            r_val[2] += matrix[i, 2] * matrix[i, 0]  # my
            r_val[3] += matrix[i, 3] * matrix[i, 0]  # mz
            r_val[4] += (matrix[i, 5] + matrix[i, 6] - matrix[i, 4]) / 2.0 + (  # mxx
                matrix[i, 1] * matrix[i, 1] * matrix[i, 0]
            )
            r_val[5] += (matrix[i, 4] + matrix[i, 6] - matrix[i, 5]) / 2.0 + (  # myy
                matrix[i, 2] * matrix[i, 2] * matrix[i, 0]
            )
            r_val[6] += (matrix[i, 4] + matrix[i, 5] - matrix[i, 6]) / 2.0 + (  # mzz
                matrix[i, 3] * matrix[i, 3] * matrix[i, 0]
            )
            r_val[7] += matrix[i, 7] - matrix[i, 1] * matrix[i, 2] * matrix[i, 0]  # mxy
            r_val[8] += matrix[i, 8] - matrix[i, 1] * matrix[i, 3] * matrix[i, 0]  # mxz
            r_val[9] += matrix[i, 9] - matrix[i, 2] * matrix[i, 3] * matrix[i, 0]  # myz

        if r_val[0] != 0.0:
            r_val[1] = r_val[1] / r_val[0]  # xG
            r_val[2] = r_val[2] / r_val[0]  # yG
            r_val[3] = r_val[3] / r_val[0]  # zG

            mxx_g = r_val[4] - r_val[1] * r_val[1] * r_val[0]
            myy_g = r_val[5] - r_val[2] * r_val[2] * r_val[0]
            mzz_g = r_val[6] - r_val[3] * r_val[3] * r_val[0]
            r_val[4] = myy_g + mzz_g  # Ixx
            r_val[5] = mxx_g + mzz_g  # Iyy
            r_val[6] = mxx_g + myy_g  # Izz

            r_val[7] = r_val[7] + r_val[1] * r_val[2] * r_val[0]  # Pxy
            r_val[8] = r_val[8] + r_val[1] * r_val[3] * r_val[0]  # Pxz
            r_val[9] = r_val[9] + r_val[2] * r_val[3] * r_val[0]  # Pyz
        else:
            r_val = np.zeros(10)

        new_var = MassPropertiesVariable.from_num_elements(self.name + "_joint", 1)
        new_var.value = r_val
        return new_var

    def push_element(self, mass_vector: NDArray[np.float64]) -> None:
        """Add a new element to the mass vector."""
        self.value_np = np.hstack((self.value_np, mass_vector))
        self.value_np[0] += 1

    def _data_size(self) -> int:
        return 1 + self.num_elements * self.VECTOR_SIZE
