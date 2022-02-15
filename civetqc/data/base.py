from abc import ABC
import os
import numpy as np
import pandas as pd

class BaseData(ABC):
    idvar = "ID"
    data_dir = "/Users/joshua/Developer/civetqc/data"
    top_raw_data_dir = os.path.join(data_dir, "raw")
    def __init__(self, study_name, study_files) -> None:
        self.study_name = study_name
        self.study_files = study_files
        self.study_raw_data_dir = os.path.join(self.top_raw_data_dir, study_name, "raw")

class QCRatingsData(BaseData):
    qcvar = "QC"
    target_names = ["Acceptable", "Unacceptable"]

    def __init__(self, study_name, study_files) -> None:
        super().__init__(study_name, study_files)
        self.path_csv = os.path.join(self.top_raw_data_dir, self.study_name, f"{self.study_name}_QC.csv")
        self.df = pd.read_csv(self.path_csv, dtype=str)
        self.df[self.qcvar] = self.df[self.qcvar].apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna()
        if not all(self.df[self.qcvar] >= 0):
            raise ValueError(f"All QC ratings in file {self.path_csv} must greater than zero")
        self.df[self.qcvar] = np.where(self.df[self.qcvar] < 1, 1, 0)
        self.df = self.df[[self.idvar, self.qcvar]]

class TabularCivetData(BaseData):
    
    features = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
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
            with open(os.path.join(self.study_raw_data_dir, civet_qc), 'r') as f:
                contents = f.read().strip().split("\n")
                for line in contents:
                    name, value = [x.strip() for x in line.split("=")]
                    try:
                        self.df[name].append(value)
                    except KeyError:
                        self.df[name] = [value]
        self.df = pd.DataFrame(self.df, dtype=str)
        self.df[self.features] = self.df[self.features].apply(pd.to_numeric, errors='coerce')
        self.df = self.df[[self.idvar] + self.features]

class ImageCivetData(BaseData):
    
    file_suffixes = [
        "angles.png", "atlas.png",  "clasp.png",  "converg.png", 
        "gradient.png", "laplace.png", "surfsurf.png", "verify.png"
    ]

    def __init__(self, study_name, study_files) -> None:
        super().__init__(study_name, study_files)


