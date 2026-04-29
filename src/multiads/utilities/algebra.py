import numpy as _np
from typing import Tuple, Union

def softmax(
    *args: Union[float, _np.ndarray],
    softness: float = None,
    hardness: float = None,
) -> Union[float, _np.ndarray]:
    """
    An element-wise softmax between two or more arrays. Also referred to as the logsumexp() function.

    Useful for optimization because it's differentiable and preserves convexity!

    Great writeup by John D Cook here:
        https://www.johndcook.com/soft_maximum.pdf

    Notes: Can provide either `hardness` or `softness`, not both. These are the inverse of each other. If neither is
    provided, `hardness` is set to 1.

    Args:

        *args: Provide any number of arguments as values to take the softmax of.

        hardness: Hardness parameter. Higher values make this closer to max(x1, x2).

        softness: Softness parameter. (Inverse of hardness.) Lower values make this closer to max(x1, x2).

            - Setting `softness` is particularly useful, because it has the same units as each of the function's
            inputs. For example, if you're taking the softmax of two values that are lengths in units of meters,
            then `softness` is also in units of meters. In this case, `softness` has the rough meaning of "an amount
            of discrepancy between the input values that would be considered physically significant".

    Returns:
        Soft maximum of the supplied values.
    """
    ### Set defaults for hardness/softness
    n_specified_arguments = (hardness is not None) + (softness is not None)
    if n_specified_arguments == 0:
        softness = 1
    elif n_specified_arguments == 2:
        raise ValueError("You must provide exactly one of `hardness` or `softness`.")

    if hardness is not None:
        softness = 1 / hardness

    if _np.any(softness <= 0):
        if softness is not None:
            raise ValueError("The value of `softness` must be positive.")
        else:
            raise ValueError("The value of `hardness` must be positive.")

    if len(args) <= 1:
        raise ValueError(
            "You must call softmax with the value of two or more arrays that you'd like to take the "
            "element-wise softmax of."
        )

    ### Scale the args by softness
    args = [arg / softness for arg in args]

    ### Find the element-wise max and min of the arrays:
    min = args[0]
    max = args[0]
    for arg in args[1:]:
        min = _np.fmin(min, arg)
        max = _np.fmax(max, arg)

    out = max + _np.log(
        sum([_np.exp(_np.maximum(array - max, -500)) for array in args])
    )
    out = out * softness
    return out

import numpy as _onp
from numpy import pi as _pi

_deg2rad = 180.0 / _pi
_rad2deg = _pi / 180.0


def degrees(x):
    """Converts an input x from radians to degrees"""
    return x * _deg2rad


def radians(x):
    """Converts an input x from degrees to radians"""
    return x * _rad2deg


def sind(x):
    """Returns the sin of an angle x, given in degrees"""
    return _onp.sin(radians(x))


def cosd(x):
    """Returns the cos of an angle x, given in degrees"""
    return _onp.cos(radians(x))


def tand(x):
    """Returns the tangent of an angle x, given in degrees"""
    return _onp.tan(radians(x))


def arcsind(x):
    """Returns the arcsin of an x, in degrees"""
    return degrees(_onp.arcsin(x))


def arccosd(x):
    """Returns the arccos of an x, in degrees"""
    return degrees(_onp.arccos(x))


def arctan2d(y, x):
    """Returns the angle associated with arctan(y, x), in degrees"""
    return degrees(_onp.arctan2(y, x))

def arctan(x):
    """Returns the arctan of an x, in degrees"""
    return degrees(_onp.arctan(x))