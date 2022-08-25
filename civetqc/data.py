from __future__ import annotations

import csv
import json
import os

from pathlib import Path

import numpy as np
import numpy.typing as npt

from .exceptions import ColumnNotFoundError, NonNumericValueError, NonUniqueIDsError
from .utils import check_types, get_non_unique, get_index, joint_sort


class CivetData:
    """represents outputs from CIVET to be predicted by the model"""

    feature_names = np.array(
        [
            "MASK_ERROR",
            "WM_PERCENT",
            "GM_PERCENT",
            "CSF_PERCENT",
            "SC_PERCENT",
            "BRAIN_VOL",
            "CEREBRUM_VOL",
            "CORTICAL_GM",
            "WHITE_VOL",
            "SUBGM_VOL",
            "SC_VOL",
            "CSF_VENT_VOL",
            "LEFT_WM_AREA",
            "LEFT_MID_AREA",
            "LEFT_GM_AREA",
            "RIGHT_WM_AREA",
            "RIGHT_MID_AREA",
            "RIGHT_GM_AREA",
            "GI_LEFT",
            "GI_RIGHT",
            "LEFT_INTER",
            "RIGHT_INTER",
            "LEFT_SURF_SURF",
            "RIGHT_SURF_SURF",
            "LAPLACIAN_MIN",
            "LAPLACIAN_MAX",
            "LAPLACIAN_MEAN",
            "GRAY_LEFT_RES",
            "GRAY_RIGHT_RES",
        ]
    )

    def __init__(self, subject_ids: np.ndarray, features: np.ndarray) -> None:
        check_types((subject_ids, np.ndarray), (features, np.ndarray))
        non_unique_ids = get_non_unique(subject_ids)
        if non_unique_ids.size != 0:
            raise NonUniqueIDsError(non_unique_ids)
        expected_shape = (len(subject_ids), len(self.feature_names))
        if features.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape of features {features.shape}, expected: {expected_shape}"
            )
        self._subject_ids, self._features = joint_sort(subject_ids, features, axis=0)

    def __array__(self) -> np.ndarray:
        return self.features

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def subject_ids(self) -> np.ndarray:
        return self._subject_ids

    @classmethod
    def from_output_files(
        cls,
        dir_path: Path | str,
        prefix: str = "",
        subset_subject_ids: list | None = None,
    ) -> CivetData:
        """create instance from raw QC files outputted by CIVET"""

        dir_path = Path(dir_path)

        target_file_suffix = "civet_qc.txt"
        subject_ids = []
        filepaths = []

        for filename in os.listdir(dir_path):
            if filename.endswith(target_file_suffix):
                subject_id = (
                    filename.replace(prefix, "")
                    .replace(target_file_suffix, "")
                    .strip("_")
                )
                if subset_subject_ids is None or subject_id in subset_subject_ids:
                    subject_ids.append(subject_id)
                    filepaths.append(dir_path.joinpath(filename))

        features: npt.NDArray[np.float64] = np.ndarray(
            shape=((len(filepaths), len(cls.feature_names))), dtype=np.float64
        )

        for row_index, subject_id in enumerate(subject_ids):
            with open(filepaths[row_index], "r") as file:
                lines = file.read().splitlines()
                for line in lines:
                    key, value = [s.strip() for s in line.split("=")]
                    if key in cls.feature_names:
                        column_index = get_index(cls.feature_names, key)
                        try:
                            column_value = float(value)
                        except ValueError as err:
                            raise NonNumericValueError(
                                key, value, filepaths[row_index]
                            ) from err
                        features.itemset((row_index, column_index), column_value)

        return cls(np.array(subject_ids), features)

    @classmethod
    def from_csv(cls, filepath: Path | str, idvar: str = "ID") -> CivetData:
        """create instance from aggregated QC file outputted by CIVET"""

        subject_ids = []
        features = []

        with open(filepath, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    subject_ids.append(row[idvar])
                except KeyError as err:
                    raise ColumnNotFoundError(idvar, filepath) from err
                values = []
                for feature_name in cls.feature_names:
                    try:
                        values.append(float(row[feature_name]))
                    except KeyError as err:
                        raise ColumnNotFoundError(feature_name, filepath) from err
                    except ValueError as err:
                        raise NonNumericValueError(
                            feature_name, row[feature_name], filepath
                        ) from err
                features.append(values)
        return cls(np.array(subject_ids), np.array(features, dtype=float))

    def to_output_files(self, dir_path: Path, prefix: str = "") -> None:
        """write features to files in the row format outputted by CIVET"""

        for index, subject_id in enumerate(self.subject_ids):
            filename = f"{subject_id}_civet_qc.txt"
            if prefix != "":
                filename = prefix + "_" + filename
            filepath = dir_path.joinpath(filename)
            with open(filepath, "w") as file:
                for key, value in zip(self.feature_names, self.features[index]):
                    file.write(f"{key}={value}\n")


class QCRatingsData:
    """contains subject IDs and QC ratings, either for development or to format predictions for user"""

    rating_labels = {0: "PASS", 1: "FAIL"}

    def __init__(
        self,
        subject_ids: np.ndarray,
        ratings: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> None:
        if probabilities is None:
            probabilities = np.full((len(subject_ids), 2), np.NaN)
        if not len(subject_ids) == len(ratings) == len(probabilities):
            raise AssertionError(
                f"False: {len(subject_ids)} == {len(ratings)} == {len(probabilities)}"
            )
        self._subject_ids, self._ratings, self._probabilities = joint_sort(
            subject_ids, ratings, probabilities
        )

    @property
    def subject_ids(self) -> np.ndarray:
        return self._subject_ids

    @property
    def ratings(self) -> np.ndarray:
        return self._ratings

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities

    def to_dict(self) -> dict:
        d = {}
        for index, subject_id in enumerate(self.subject_ids):
            d[subject_id] = {
                "rating": self.rating_labels[self.ratings[index]],
                "probabilities": {
                    self.rating_labels[0]: round(self.probabilities[index][0], 3),
                    self.rating_labels[1]: round(self.probabilities[index][1], 3),
                },
            }
        return d

    def to_csv(self, filepath: Path | str) -> None:
        with open(filepath, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "ID",
                    "RATING",
                    f"P({self.rating_labels[0]})",
                    f"P({self.rating_labels[1]})",
                ],
            )
            writer.writeheader()
            for index, (subject_id, qc_rating) in enumerate(
                zip(self.subject_ids, self.ratings)
            ):
                writer.writerow(
                    {
                        "ID": subject_id,
                        "RATING": self.rating_labels[qc_rating],
                        f"P({self.rating_labels[0]})": round(
                            self.probabilities[index][0], 3
                        ),
                        f"P({self.rating_labels[1]})": round(
                            self.probabilities[index][1], 3
                        ),
                    }
                )

    def to_json(self, filepath: Path | str) -> None:
        with open(filepath, "w") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def from_csv(
        cls,
        filepath: Path | str,
        idvar: str = "ID",
        qcvar: str = "QC",
        allow_non_numeric: bool = False,
    ) -> QCRatingsData:
        """create instance from a CSV file containing columns idvar and qcvar"""

        subject_ids = []
        ratings = []

        with open(filepath, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    subject_ids.append(row[idvar])
                except KeyError as err:
                    raise ColumnNotFoundError(idvar, filepath) from err
                try:
                    qc_rating = row[qcvar]
                except KeyError as err:
                    raise ColumnNotFoundError(qcvar, filepath) from err
                if not allow_non_numeric:
                    try:
                        qc_rating = float(qc_rating)
                    except ValueError as err:
                        raise NonNumericValueError(qcvar, row[qcvar], filepath) from err
                ratings.append(qc_rating)

        return cls(np.array(subject_ids), np.array(ratings))
