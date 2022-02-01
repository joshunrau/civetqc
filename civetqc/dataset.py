import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

class Study:
    
    root_data_dir = "/Users/joshua/Developer/civetqc/data"
    output_file_suffixes = [
        "angles.png", "atlas.png", "civet_qc.txt", "clasp.png", "classify_qc.txt", "converg.png", 
        "gradient.png", "laplace.png", "surface_qc.txt", "surfsurf.png", "verify.png"
    ]
    tabular_features = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
    ]

    def __init__(self, name: str, id_prefix: str, id_len: int | None = None, includes: str | None = None) -> None:
        self.name = name
        self.qc_data_path = os.path.join(self.root_data_dir, name, f"{self.name}_QC.csv")
        self.raw_data_dir = os.path.join(self.root_data_dir, name, "raw")
        self.id_prefix, self.id_len, self.includes = id_prefix, id_len, includes

        self.patient_files = {}
        
        for filename in os.listdir(self.raw_data_dir):

            file_type = None
            for suffix in self.output_file_suffixes:
                if filename.endswith(suffix):
                    file_type = suffix.split(".")[0]
            if file_type is None or (self.includes is not None and self.includes not in filename):
                continue

            patient_id = filename
            for s in [self.id_prefix] + self.output_file_suffixes:
                patient_id = patient_id.replace(s, "")
            patient_id = patient_id.strip("_")
            if self.id_len is not None:
                patient_id = patient_id[:self.id_len]

            if patient_id not in self.patient_files:
                self.patient_files[patient_id] = {x.split(".")[0]: None for x in self.output_file_suffixes}
            self.patient_files[patient_id][file_type] = filename
        
        n_files = max([len(x) for x in self.patient_files.values()])
        for key, value in self.patient_files.items():
            if len(value) != n_files:
                raise ValueError(f"Missing files for subject with ID {key}, expected {n_files} got {len(value)}")
    
    def get_qc_data(self):
        self.qc_data = pd.read_csv(self.qc_data_path, dtype=str)
        self.qc_data["QC"] = self.qc_data["QC"].apply(pd.to_numeric, errors='coerce')
        self.qc_data = self.qc_data.dropna()
        if not all(self.qc_data["QC"] >= 0):
            raise ValueError(f"All QC ratings in file {self.qc_data_path} must greater than zero")
        self.qc_data["QC"] = np.where(self.qc_data["QC"] < 1, 1, 0)
        self.qc_data = self.qc_data[["ID", "QC"]]

    def get_tabular_data(self):
        self.tabular_data = {"ID": []}
        for patient_id in self.patient_files:
            self.tabular_data["ID"].append(patient_id)
            civet_qc = self.patient_files[patient_id]["civet_qc"]
            with open(os.path.join(self.raw_data_dir, civet_qc), 'r') as f:
                contents = f.read().strip().split("\n")
                for line in contents:
                    name, value = [x.strip() for x in line.split("=")]
                    try:
                        self.tabular_data[name].append(value)
                    except KeyError:
                        self.tabular_data[name] = [value]
        self.tabular_data = pd.DataFrame(self.tabular_data, dtype=str)
        self.tabular_data[self.tabular_features] = self.tabular_data[self.tabular_features].apply(pd.to_numeric, errors='coerce')
        self.tabular_data = self.tabular_data[["ID"] + self.tabular_features]
    
    def load_data(self):
        self.get_qc_data()
        self.get_tabular_data()
        self.df = pd.merge(self.qc_data, self.tabular_data, on="ID").dropna(axis=1, how="all")
        self.df["StudyName"] = self.name
    
    def train_test_split(self):
        features = self.df[self.tabular_features].to_numpy()
        target = self.df["QC"].to_numpy()
        return train_test_split(features, target, test_size=.3, random_state=0)

class Dataset:

    def __init__(self) -> None:
        
        studies = [
            Study(name="FEP", id_prefix="FEP"),
            Study(name="INSIGHT", id_prefix="Insight"),
            Study(name="LAM", id_prefix="LAM"),
            Study(name="NUSDAST", id_prefix=""),
            Study(name="TOPSY", id_prefix="", id_len=3, includes="V1_gradient_n4_anlm0.5r")
        ]

        self.train = {"Features": [], "Target": []}
        self.test = {"Features": [], "Target": []}  

        for study in studies:
            study.load_data()
            try:
                self.df = pd.concat([self.df, study.df], ignore_index=True)
            except AttributeError:
                self.df = study.df
            X_train, X_test, y_train, y_test = study.train_test_split()

            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.transform(X_test)

            self.train["Features"].append(X_train)
            self.train["Target"].append(y_train)
            self.test["Features"].append(X_test)
            self.test["Target"].append(y_test)

        self.train["Features"] = np.vstack(self.train["Features"])
        self.train["Target"] = np.hstack(self.train["Target"])
        self.test["Features"] = np.vstack(self.test["Features"])
        self.test["Target"] = np.hstack(self.test["Target"])

