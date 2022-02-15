import os
import pandas as pd

try:
    from base import BaseData, QCRatingsData, TabularCivetData, ImageCivetData
except ImportError:
    from .base import BaseData, QCRatingsData, TabularCivetData, ImageCivetData


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
