from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from multiads.scenario import InnerVariableFloatNP, ValueType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from typing_extensions import Self


SPANLOAD_SIZE = 6
SPANLOAD_DEFAULT_NUM_STATIONS = 10


class SpanLoadsVariable(InnerVariableFloatNP):
    pos_origin = 0
    pos_u = 3
    pos_v = 6
    pos_w = 9
    pos_origin_p = 12
    pos_gamma_p = 15
    pos_origin_pp = 18
    pos_gamma_pp = 21
    pos_thrust = 24
    pos_span_loads = 25
    non_span_data_size = 25

    @classmethod
    def from_loads(
        cls,
        name: str,
        loads: NDArray[np.float64],
        origin: NDArray[np.float64] | None = None,
        u: NDArray[np.float64] | None = None,
        v: NDArray[np.float64] | None = None,
        w: NDArray[np.float64] | None = None,
        origin_p: NDArray[np.float64] | None = None,
        gamma_p: NDArray[np.float64] | None = None,
        origin_pp: NDArray[np.float64] | None = None,
        gamma_pp: NDArray[np.float64] | None = None,
        thrust: float = 0.0,
    ) -> Self:
        origin = np.zeros(3) if origin is None else origin
        u = np.array([1.0, 0.0, 0.0]) if u is None else u
        v = np.array([0.0, 1.0, 0.0]) if v is None else v
        w = np.array([0.0, 0.0, 1.0]) if w is None else w
        origin_p = np.zeros(3) if origin_p is None else origin_p
        gamma_p = np.zeros(3) if gamma_p is None else gamma_p
        origin_pp = np.zeros(3) if origin_pp is None else origin_pp
        gamma_pp = np.zeros(3) if gamma_pp is None else gamma_pp

        if loads.size // SPANLOAD_SIZE * SPANLOAD_SIZE != loads.size:
            msg = f"Size of 'loads' is not divisible by {SPANLOAD_SIZE}"
            raise ValueError(msg)

        value = np.concatenate(
            (
                origin,
                u,
                v,
                w,
                origin_p,
                gamma_p,
                origin_pp,
                gamma_pp,
                [thrust],
                loads,
            ),
        )
        return cls(name, value)

    @classmethod
    def from_num_stations(
        cls,
        name: str,
        num_stations: int = SPANLOAD_DEFAULT_NUM_STATIONS,
        origin: NDArray[np.float64] | None = None,
        u: NDArray[np.float64] | None = None,
        v: NDArray[np.float64] | None = None,
        w: NDArray[np.float64] | None = None,
        origin_p: NDArray[np.float64] | None = None,
        gamma_p: NDArray[np.float64] | None = None,
        origin_pp: NDArray[np.float64] | None = None,
        gamma_pp: NDArray[np.float64] | None = None,
        thrust: float = 0.0,
    ) -> Self:
        loads = np.zeros(SPANLOAD_SIZE * num_stations)
        return cls.from_loads(
            name,
            loads,
            origin,
            u,
            v,
            w,
            origin_p,
            gamma_p,
            origin_pp,
            gamma_pp,
            thrust,
        )

    @property
    def loads(self) -> NDArray[np.float64]:
        i0 = self.pos_span_loads
        return self.value_np[i0:].reshape((SPANLOAD_SIZE, -1))

    @loads.setter
    def loads(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_span_loads
        self.value_np[i0:] = value

    @property
    def n_stations(self) -> int:
        return int(self.loads.shape[1])

    @property
    def centers(self) -> NDArray[np.float64]:
        return self.loads[0, :]

    @centers.setter
    def centers(self, value: NDArray[np.float64]) -> None:
        self.loads[0, :] = value

    @property
    def spans(self) -> NDArray[np.float64]:
        return self.loads[1, :]

    @spans.setter
    def spans(self, value: NDArray[np.float64]) -> None:
        self.loads[1, :] = value

    @property
    def fx(self) -> NDArray[np.float64]:
        return self.loads[2, :]

    @fx.setter
    def fx(self, value: NDArray[np.float64]) -> None:
        self.loads[2, :] = value

    @property
    def fy(self) -> NDArray[np.float64]:
        return self.loads[3, :]

    @fy.setter
    def fy(self, value: NDArray[np.float64]) -> None:
        self.loads[3, :] = value

    @property
    def fz(self) -> NDArray[np.float64]:
        return self.loads[4, :]

    @fz.setter
    def fz(self, value: NDArray[np.float64]) -> None:
        self.loads[4, :] = value

    @property
    def m(self) -> NDArray[np.float64]:
        return self.loads[5, :]

    @m.setter
    def m(self, value: NDArray[np.float64]) -> None:
        self.loads[5, :] = value

    @property
    def origin(self) -> NDArray[np.float64]:
        i0 = self.pos_origin
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @origin.setter
    def origin(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_origin
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def u(self) -> NDArray[np.float64]:
        i0 = self.pos_u
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @u.setter
    def u(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_u
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def v(self) -> NDArray[np.float64]:
        i0 = self.pos_v
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @v.setter
    def v(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_v
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def w(self) -> NDArray[np.float64]:
        i0 = self.pos_w
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @w.setter
    def w(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_w
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def origin_p(self) -> NDArray[np.float64]:
        i0 = self.pos_origin_p
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @origin_p.setter
    def origin_p(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_origin_p
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def gamma_p(self) -> NDArray[np.float64]:
        i0 = self.pos_gamma_p
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @gamma_p.setter
    def gamma_p(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_gamma_p
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def origin_pp(self) -> NDArray[np.float64]:
        i0 = self.pos_origin_pp
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @origin_pp.setter
    def origin_pp(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_origin_pp
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def gamma_pp(self) -> NDArray[np.float64]:
        i0 = self.pos_gamma_pp
        i1 = i0 + 3
        return self.value_np[i0:i1]

    @gamma_pp.setter
    def gamma_pp(self, value: NDArray[np.float64]) -> None:
        i0 = self.pos_gamma_pp
        i1 = i0 + 3
        self.value_np[i0:i1] = value

    @property
    def thrust(self) -> float:
        i = self.pos_thrust
        return self.value_np[i]

    @thrust.setter
    def thrust(self, value: float) -> None:
        i = self.pos_thrust
        self.value_np[i] = value


# NOTE @Andres: Do we need this?
class SpanLoadsGroupVariable(InnerVariableFloatNP):
    def __init__(
        self,
        name: str,
        value: ValueType,
        loads_names: Sequence[str],
    ) -> None:
        super().__init__(name, value)
        self.load_names = loads_names

    @classmethod
    def from_sizes(
        cls,
        name: str,
        loads_names: Sequence[str],
        loads_sizes: Sequence[int] | None = None,
    ) -> Self:
        if loads_sizes is not None:
            if any(s <= 0 for s in loads_sizes):
                msg = f"'{cls.__name__}' cannot have negative 'load_sizes'."
                raise ValueError(msg)
            if len(loads_names) != len(loads_sizes):
                msg = (
                    "'load_names' and 'loads_sizes' cannot have "
                    f"different lengths in '{cls.__name__}'"
                )
                raise ValueError(msg)
        else:
            loads_sizes = [SPANLOAD_DEFAULT_NUM_STATIONS] * len(loads_names)

        loads = [
            SpanLoadsVariable.from_num_stations(n, s)
            for n, s in zip(loads_names, loads_sizes, strict=True)
        ]

        return cls.from_loads(name, loads)

    @classmethod
    def from_loads(
        cls,
        name: str,
        span_loads: Sequence[SpanLoadsVariable],
    ) -> Self:
        n_span_loads = len(span_loads)
        names_span_loads = [sl.name for sl in span_loads]
        sizes_span_loads = np.array([float(sl.n_stations) for sl in span_loads])
        value_span_loads = np.concatenate([sl.value_np for sl in span_loads])

        value = np.concatenate(
            (
                [n_span_loads],
                sizes_span_loads,
                value_span_loads,
            ),
        )
        return cls(name, value, names_span_loads)

    @property
    def n_loads(self) -> int:
        return int(self.value_np[0])

    @property
    def n_stations(self) -> NDArray[np.int_]:
        return self.value_np[1 : self.n_loads + 1].astype(np.int_)

    @property
    def span_loads(self) -> Sequence[SpanLoadsVariable]:
        if self.n_loads < 1:
            return []

        loads: list[SpanLoadsVariable] = []
        sizes = self.n_stations * SPANLOAD_SIZE
        i0 = 1 + self.n_loads

        for i in range(self.n_loads):
            name = self.load_names[i]
            i1 = i0 + SpanLoadsVariable.non_span_data_size + sizes[i]
            value = self.value_np[i0:i1]
            loads.append(SpanLoadsVariable(name, value))
            i0 = i1

        return loads

    def push_loads(self, loads: SpanLoadsVariable) -> None:
        n_loads = self.n_stations + 1
        sizes = np.concatenate(
            (
                self.value_np[1 : self.n_loads + 1],
                [float(loads.n_stations)],
            ),
        )
        values = np.concatenate(
            (
                self.value_np[1 + self.n_loads :],
                loads.loads,
            ),
            axis=None,
        )
        self.value_np = np.concatenate(([float(n_loads)], sizes, values))
