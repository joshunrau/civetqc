import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import time

from . import RAW_DATA_DIRECTORY, ALL_FILE_SUFFIXES
from .images import AnglesImage, AtlasImage

class Study:
    
    idvar = "ID"
    qcvar = "QC"
    
    def __init__(self, name, id_prefix=None, id_len=None, includes=None) -> None:
        
        self.name = name
        self.id_prefix = id_prefix
        self.id_len = id_len
        self.includes = includes
        
        self.raw_data_dir = os.path.join(RAW_DATA_DIRECTORY, name, "raw")
        self.path_qc_ratings = os.path.join(RAW_DATA_DIRECTORY, name, f"{self.name}_QC.csv")
        
        # Get dictionary of all files associated with study
        self.files = {}
        for filename in os.listdir(self.raw_data_dir):
            for suffix in ALL_FILE_SUFFIXES:
                if filename.endswith(suffix):
                    file_type = suffix.split(".")[0]
                    subject_id = filename.strip(suffix)
            
            if self.includes is not None and self.includes not in filename:
                continue
                
            if self.id_prefix is not None:
                subject_id = subject_id.strip(self.id_prefix)
            
            subject_id = subject_id.strip("_")
            
            if self.id_len is not None:
                subject_id = subject_id[:self.id_len]
            
            filepath = os.path.join(self.raw_data_dir, filename)
            assert os.path.exists(filepath)
            
            try:
                self.files[subject_id][file_type] = filepath
            except KeyError:
                self.files[subject_id] = {}
                self.files[subject_id][file_type] = filepath

        complete_cases = [len(x) for x in self.files.values()].count(len(ALL_FILE_SUFFIXES))
        assert len(self.files) == complete_cases
    
    def get_qc_ratings(self):
        df = pd.read_csv(self.path_qc_ratings, dtype=str)
        df[self.qcvar] = df[self.qcvar].apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)
        if not all(df[self.qcvar] >= 0):
            raise ValueError(f"All QC ratings in file {self.path_qc_ratings} must greater than zero")
        df[self.qcvar] = np.where(df[self.qcvar] < 1, 1, 0)
        return df[[self.idvar, self.qcvar]]
        
    def subject_method(func):
        """ decorator for methods that retrieve information from files for each subject """
        def get_data(self):
            data = {self.idvar: []}
            for subject_id, subject_files in self.files.items():
                data[self.idvar].append(subject_id)
                result = func(self, subject_id, subject_files)
                for key, value in result.items():
                    try:
                        data[key].append(value)
                    except KeyError:
                        data[key] = [value]
            if all([len(v) == len(data[self.idvar]) for v in data.values()]):
                return pd.DataFrame(data, dtype=str)
            raise ValueError 
        return get_data
    
    @subject_method
    def get_text_features(self, subject_id, subject_files):
        features = {}
        with open(subject_files["civet_qc"], "r") as file:
            contents = file.read().strip().split("\n")
            for line in contents:
                name, value = [x.strip() for x in line.split("=")]
                features[name] = value
        return features
    
    @subject_method
    def get_image_features(self, subject_id, subject_files):
        
        images = {
            "angles": AnglesImage(subject_files["angles"]),
            "atlas": AtlasImage(subject_files["atlas"])
        }
        
        features = {}
        for label, img in images.items():
            blue, green, red = img.color_means
            features[f"{label}_blue_mean"] = blue
            features[f"{label}_green_mean"] = green
            features[f"{label}_red_mean"] = red
            features[f"{label}_laplacian_blur"] = img.blur

        return features
    
    def get_data(self, queue):
        df = self.get_qc_ratings()
        df = df.merge(self.get_text_features(), on="ID")
        df = df.merge(self.get_image_features(), on="ID")
        df = df.dropna(axis=1, how="all")
        queue.put(df)


def create_dataset():
    
    start = time.perf_counter()

    studies = [
        Study(name="FEP", id_prefix="FEP"),
        Study(name="INSIGHT", id_prefix="Insight"),
        Study(name="LAM", id_prefix="LAM"),
        Study(name="NUSDAST"),
        Study(name="TOPSY", id_len=3, includes="V1_gradient_n4_anlm0.5r")
    ]

    queue = mp.Queue()

    processes = []
    results = []

    for study in studies:
        process = mp.Process(target=study.get_data, args=(queue,))
        processes.append(process)
        process.start()
    
    for process in processes:
        df = queue.get() # will block
        results.append(df)

    for process in processes:
        process.join()
    
    for df in results:
        try:
            dataset = pd.concat([dataset, df], ignore_index=True)
        except NameError:
            dataset = df
    
    stop = time.perf_counter()
    print(f"Created dataset in {stop - start} seconds")
    
    return dataset.dropna(axis=1, how="any")