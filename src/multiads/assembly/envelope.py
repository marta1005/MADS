from __future__ import annotations

import copy
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import numpy as np
import scipy.interpolate as sint  # type: ignore[import-untyped]
from numpy.typing import NDArray

from multiads.ambiance import Atmosphere
from multiads.scenario import ValueType, Variable


V = TypeVar("V")
N = TypeVar("N")
T = TypeVar("T")

Optimizable: TypeAlias = Variable[V, N] | T

OptimizableInt: TypeAlias = Optimizable[
    np.int32,
    NDArray[np.int32],
    int | np.int32,
]
OptimizableFloat: TypeAlias = Optimizable[
    np.float64,
    NDArray[np.float64],
    float | np.float64,
]
OptimizableIntNP: TypeAlias = Optimizable[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]
OptimizableFloatNP: TypeAlias = Optimizable[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


def flatten_segments(
    segments: Iterable[MADSPhase],
) -> Sequence[MADSPhase]:
    segments_flat: list[MADSPhase] = []
    _flatten_segments(segments, segments_flat)
    return segments_flat


def _flatten_segments(
    segments: Iterable[MADSPhase],
    flat_list: list[MADSPhase],
) -> None:
    for sg in segments:
        # Avoid repetitions
        if sg in flat_list:
            continue
        flat_list.append(sg)

        # Loop over attributes
        for att in vars(sg):
            next_sg = _get_comps_in_attribute(sg, att)
            _flatten_segments(next_sg, flat_list)

def _get_comps_in_attribute(
    parent: MADSPhase,
    att: str,
) -> Sequence[MADSPhase]:
    value = getattr(parent, att)
    next_obj = []
    if isinstance(value, Sequence):
        next_obj.extend([c for c in value if isinstance(c, MADSPhase)])
    elif isinstance(value, MADSPhase):
        next_obj.append(value)
    return next_obj

def update_segments(
    segments: Sequence[MADSPhase],
    variables: Mapping[str, NDArray],
) -> None:
    _segments = flatten_segments(segments)
    for seg in _segments:
        for att, var in seg.variables.items():
            if (v := variables.get(var.name)) is not None:
                var.value = v
            setattr(seg, att, var.value)



class PhaseOptions:
    pass

class MADSPhase:
    def __init__(
        self,
        name: str,
        type: str,
        options: Sequence[PhaseOptions] | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.variables: dict[str, PhaseOptions] = {}
        self.options: Sequence[PhaseOptions] = options or []

    def __setattr__(self, name: str, val: Optimizable) -> None:
        if isinstance(val, Variable):
            self.variables[name] = val
            val = val.value
        super().__setattr__(name, val)

    def remove_segment(self, attr: str, value: ValueType) -> None:
        try:
            self.segment.pop(attr)
        except KeyError:
            msg = f"Attribute '{attr}' of segment '{self.name}' is not a variable."
            raise RuntimeError(msg) from None
        setattr(self, attr, value)


class Segment(MADSPhase):
    def __init__(
        self,
        name: str,
        type: str,
        range: OptimizableFloat = 0.0,  # range of the segment [m]
        mass_start: OptimizableFloat = 0.0,  # mass at the start [kg]
        mass_end: OptimizableFloat = 0.0,  # mass at the end [kg]
        duration: OptimizableFloat = 0.0,  # duration of the segment [s]
        airspeed_start: OptimizableFloat = 0.0,  # airspeed at the start [m/s]
        airspeed_end: OptimizableFloat = 0.0,  # airspeed at the end [m/s]
        altitude_start: OptimizableFloat = 0.0,  # altitude at the start [m]
        altitude_end: OptimizableFloat = 0.0,  # altitude at the end [m]
        climb_angle: OptimizableFloat = 0.0,  # climb angle  [rad]
        angle_of_attack: OptimizableFloat = 0.0,  # angle of attack [deg]
        #
        #lift_drag_ratio: Optimizable[float],  # lift to drag ratio [-]
        #fuel_used: Optimizable[float],  # fuel consumed [kg]
        #battery_mass_used: Optimizable[float],  # battery mass required [kg]
        #h2o_used: Optimizable[float],  # H2 used [kg]
        #hybridization: Optimizable[
        #    float
        #],  # degree of hybridization. 1: first type, 2: second type. If only one type: DoH = 1 [-]
        #lift_coefficient: Optimizable[float] = 0.0,  # lift coefficient [-]
        #
        # maybe each mission segment schould have an individual environment attribute storing an individual Environment object
    ) -> None:
        super().__init__(name,type)
        self.range = range
        self.mass_start = mass_start
        self.mass_end = mass_end
        self.duration = duration
        self.airspeed_start = airspeed_start
        self.airspeed_end = airspeed_end
        self.altitude_start = altitude_start
        self.altitude_end = altitude_end
        self.climb_angle = climb_angle
        self.angle_of_attack = angle_of_attack
        #self.lift_drag_ratio: float = lift_drag_ratio
        #self.fuel_used: float = fuel_used
        #self.battery_mass_used: float = battery_mass_used
        #self.h2o_used: float = h2o_used
        #self.hybridization: float = hybridization
        #self.lift_coefficient: float = lift_coefficient
        # check if each mission segment should have its own environment


class SegmentEnvelope(MADSPhase):
    def __init__(
        self,
        name: str,
        mission_segments: Sequence[Segment],
    ) -> None:
        super().__init__(name)
        
        self.mission_segments = mission_segments