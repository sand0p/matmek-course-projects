from collections.abc import Callable

import numpy as np


def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    return f(t)


def func(t: float) -> float:
    return np.where((0<=t) & (t<=3),np.e**(-t),np.e**(-3*t))


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()
