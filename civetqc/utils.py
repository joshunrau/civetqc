from __future__ import annotations

import time

import numpy as np

from typing import Any, Callable


def use_timer(func: Callable) -> Callable:
    """used as a decorator to time function execution"""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        return_value = func(*args, **kwargs)
        print(f"Completed in {time.perf_counter() - start_time:.3f}s")
        return return_value

    return wrapper


def get_non_unique(arr: np.ndarray) -> np.ndarray:
    """return all non-unique values in array"""
    unique, counts = np.unique(arr, return_counts=True)
    return unique[counts > 1]


def check_types(*args: tuple[object, type]) -> None:
    """raise TypeError if any objects are not of correct type"""
    for o, t in args:
        if not isinstance(o, t):
            raise TypeError(f"Expected {t}, not {type(o)}")


def get_index(arr: np.ndarray, value: Any) -> int:
    """equivalent to index method on list"""
    if arr.ndim != 1:
        raise ValueError("Array must be one-dimensional")
    try:
        return np.where(arr == value)[0][0]
    except IndexError:
        raise ValueError(f"Array does not contain value: {value}")


def joint_sort(
    a: np.ndarray, *args: np.ndarray, axis: int = 0
) -> tuple[np.ndarray, ...]:
    """return sorted 1D array and arbitrary number of arrays sorted along axis by a1"""
    if a.ndim != 1:
        raise ValueError("Array given as first argument must be one-dimensional")
    for arr in args:
        if arr.ndim - 1 < axis:
            raise ValueError(
                f"Cannot sort array with {arr.ndim} dimensions by axis {axis}"
            )
        if a.size != arr.shape[axis]:
            raise ValueError(
                f"Size of axis '{axis}' of array given as argument '{args.index(arr) + 1}' must be equal to size '{a.size}' of first array, not '{arr.shape[axis]}'"
            )
    indices = np.argsort(a)
    return np.take(a, indices), *[np.take(arr, indices, axis) for arr in args]
