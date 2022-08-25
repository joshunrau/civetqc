from __future__ import annotations

from pathlib import Path
import numpy as np


class CivetQCError(Exception):
    pass


class ColumnNotFoundError(CivetQCError):
    """raised when a required column is not found when reading a CSV file"""

    def __init__(self, colname: str, filepath: Path | str) -> None:
        super().__init__(f"Required column '{colname}' not found in file: {filepath}")


class NonNumericValueError(CivetQCError):
    """raised when a value that cannot be converted to float is encountered while reading CSV file"""

    def __init__(self, colname: str, value: str, filepath: Path | str) -> None:
        super().__init__(
            f"Unexpected non-numeric value '{value}' for column '{colname}' in file: {filepath}"
        )


class NonUniqueIDsError(CivetQCError):
    """raised when instantiating CivetData with non-unique subject IDs"""

    def __init__(self, subject_ids: np.ndarray) -> None:
        super().__init__(f"Non-unique value for subject IDs: {subject_ids}")
