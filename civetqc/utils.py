from __future__ import annotations

import time

import numpy as np

from typing import Callable


def use_timer(func: Callable) -> Callable:
    """ used as a decorator to time function execution """

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        return_value = func(*args, **kwargs)
        print(f"Completed in {time.perf_counter() - start_time:.3f}s")
        return return_value

    return wrapper


def get_non_unique(arr: np.ndarray) -> np.ndarray:
    """ return all non-unique values in array """
    unique, counts = np.unique(arr, return_counts=True)
    return unique[counts > 1]


def check_types(*args: tuple[object, type]) -> None:
    """ raise TypeError if any objects are not of correct type """
    for o, t in args:
        if not isinstance(o, t):
            raise TypeError(f"Expected {t}, not {type(o)}")


def get_index(arr: np.ndarray, value: any) -> int:
    """ equivalent to index method on list """
    if arr.ndim != 1:
        raise ValueError("Array must be one-dimensional")
    try:
        return np.where(arr == value)[0][0]
    except IndexError:
        raise ValueError(f"Array does not contain value: {value}")


def joint_sort(a1: np.ndarray, a2: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """ return a sorted copy of a1 and a copy of a2 sorted by a1 """
    assert a1.ndim == 1 and a2.ndim == 2
    assert a1.size == a2.shape[axis], f"{a1.size} != {a2.shape[axis]}"
    indices = np.argsort(a1)
    return np.take(a1, indices), np.take(a2, indices, axis)
