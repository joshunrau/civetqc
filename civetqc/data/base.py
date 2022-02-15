from abc import ABC
import os
import numpy as np
import pandas as pd

from . import RAW_DATA_DIRECTORY

class BaseData(ABC):
    idvar = "ID"
    def __init__(self, study_name, study_files) -> None:
        self.study_name = study_name
        self.study_files = study_files
        self.study_raw_data_dir = os.path.join(RAW_DATA_DIRECTORY, study_name, "raw")


class QCRatingsData(BaseData):
    qcvar = "QC"
    target_names = ["Acceptable", "Unacceptable"]

    def __init__(self, study_name, study_files) -> None:
        super().__init__(study_name, study_files)
        self.path_csv = os.path.join(
            RAW_DATA_DIRECTORY, self.study_name, f"{self.study_name}_QC.csv"
        )
        self.df = pd.read_csv(self.path_csv, dtype=str)
        self.df[self.qcvar] = self.df[self.qcvar].apply(pd.to_numeric, errors="coerce")
        self.df = self.df.dropna()
        if not all(self.df[self.qcvar] >= 0):
            raise ValueError(
                f"All QC ratings in file {self.path_csv} must greater than zero"
            )
        self.df[self.qcvar] = np.where(self.df[self.qcvar] < 1, 1, 0)
        self.df = self.df[[self.idvar, self.qcvar]]


class TabularCivetData(BaseData):

    feature_names = [
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

    file_suffixes = ["civet_qc.txt", "classify_qc.txt", "surface_qc.txt"]

    def __init__(self, study_name, study_files) -> None:
        super().__init__(study_name, study_files)
        self.df = {self.idvar: []}

        for patient_id in study_files:
            self.df[self.idvar].append(patient_id)
            civet_qc = study_files[patient_id]["civet_qc"]
            if civet_qc is None:
                continue
            with open(os.path.join(self.study_raw_data_dir, civet_qc), "r") as f:
                contents = f.read().strip().split("\n")
                for line in contents:
                    name, value = [x.strip() for x in line.split("=")]
                    try:
                        self.df[name].append(value)
                    except KeyError:
                        self.df[name] = [value]
        self.df = pd.DataFrame(self.df, dtype=str)
        self.df[self.feature_names] = self.df[self.feature_names].apply(
            pd.to_numeric, errors="coerce"
        )
        self.df = self.df[[self.idvar] + self.feature_names]


class ImageCivetData(BaseData):

    file_suffixes = [
        "angles.png",
        "atlas.png",
        "clasp.png",
        "converg.png",
        "gradient.png",
        "laplace.png",
        "surfsurf.png",
        "verify.png",
    ]

    def __init__(self, study_name, study_files) -> None:
        super().__init__(study_name, study_files)


class StudyData(BaseData):

    file_suffixes = TabularCivetData.file_suffixes + ImageCivetData.file_suffixes

    def __init__(self, study_name, id_prefix=None, id_len=None, includes=None) -> None:
        super().__init__(study_name, study_files={})
        self.id_prefix = id_prefix
        self.id_len = id_len
        self.includes = includes

        for filename in os.listdir(self.study_raw_data_dir):

            check_file_types = [
                filename.endswith(suffix) for suffix in self.file_suffixes
            ]
            if not any(check_file_types):
                continue

            file_type = self.file_suffixes[check_file_types.index(True)].split(".")[0]

            if self.includes is not None and self.includes not in filename:
                continue

            patient_id = filename.replace(file_type, "")
            if self.id_prefix is not None:
                patient_id = patient_id.replace(self.id_prefix, "")
            patient_id = patient_id.split(".")[0].strip("_")

            if self.id_len is not None:
                patient_id = patient_id[: self.id_len]

            if patient_id not in self.study_files:
                self.study_files[patient_id] = {
                    x.split(".")[0]: None for x in self.file_suffixes
                }
            self.study_files[patient_id][file_type] = filename

        n_files = max([len(x) for x in self.study_files.values()])
        for key, value in self.study_files.items():
            if len(value) != n_files:
                raise ValueError(
                    f"Missing files for subject with ID {key}, expected {n_files} got {len(value)}"
                )

        self.qc_ratings_data = QCRatingsData(self.study_name, self.study_files)
        self.tabular_civet_data = TabularCivetData(self.study_name, self.study_files)
        self.df = pd.merge(self.qc_ratings_data.df, self.tabular_civet_data.df, on="ID")
        self.df.dropna(axis=1, how="all", inplace=True)
        self.df["StudyName"] = self.study_name