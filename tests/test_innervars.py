import numpy as np

from multiads.scenario import InnerVariableFloat, InnerVariableFloatNP


def test_scalar() -> None:
    value = np.array([3.1415])
    var = InnerVariableFloat(name="test_var", value=value[0])
    assert var.value == value[0]
    assert all(var.value_np == value)
    assert not var.is_array


def test_array() -> None:
    value = np.array([3.1415, 42.0])
    var = InnerVariableFloatNP(name="test_var", value=value)
    assert all(var.value == value)
    assert all(var.value_np == value)
    assert var.is_array
