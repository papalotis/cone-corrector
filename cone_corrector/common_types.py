"""Define common types used in the cone_corrector package."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BeforeValidator


def validate_number_of_dimensions(array: Any, number_of_dimensions: int) -> np.ndarray:  # noqa: ANN401
    """Check if the number of dimensions of an array is equal to the expected number of dimensions."""
    array = np.array(array)
    if array.ndim != number_of_dimensions:
        raise ValueError(f"Expected an array with {number_of_dimensions} dimensions, got {array.ndim} dimensions.")

    return array


def validate_mxn_dimensions(array: Any, m: int | None, n: int | None) -> np.ndarray:  # noqa: ANN401
    """Check if the number of dimensions of an array is equal to the expected number of dimensions."""
    array = np.array(array)

    if m is not None and array.shape[0] != m:
        raise ValueError(f"Expected an array with {m} rows, got {array.shape[0]} rows.")

    if n is not None and array.shape[1] != n:
        raise ValueError(f"Expected an array with {n} columns, got {array.shape[1]} columns.")

    return array


def validate_exact_shape(array: Any, shape: tuple[int]) -> np.ndarray:  # noqa: ANN401
    """Check if the shape of an array is equal to the expected shape."""
    array = np.array(array)

    if array.shape != shape:
        raise ValueError(f"Expected an array with shape {shape}, got shape {array.shape}.")

    return array


FloatArrayValidator = BeforeValidator(lambda x: np.array(x, dtype=np.float64))
IntArrayValidator = BeforeValidator(lambda x: np.array(x, dtype=np.int64))
Dim1Validator = BeforeValidator(lambda x: validate_number_of_dimensions(x, number_of_dimensions=1))
Nx2Validator = BeforeValidator(lambda x: validate_mxn_dimensions(x, m=None, n=2))
Nx3Validator = BeforeValidator(lambda x: validate_mxn_dimensions(x, m=None, n=3))
Vector2Validator = BeforeValidator(lambda x: validate_exact_shape(x, shape=(2,)))

FloatArrayNx2 = Annotated[np.ndarray[tuple[int, Literal[2]], np.float64], FloatArrayValidator, Nx2Validator]
FloatArrayNx3 = Annotated[np.ndarray[tuple[int, Literal[3]], np.float64], FloatArrayValidator, Nx3Validator]
FloatVector = Annotated[np.ndarray[tuple[int], np.float64], FloatArrayValidator, Dim1Validator]
IntVector = Annotated[np.ndarray[tuple[int], np.int64], IntArrayValidator, Dim1Validator]
FloatVector2 = Annotated[np.ndarray[tuple[int], np.float64], FloatArrayValidator, Vector2Validator]
