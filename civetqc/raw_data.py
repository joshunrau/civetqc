import os
import pandas as pd

class RawStudyData:
    """ class to prepare raw CIVET data into CSV file in case CSV is NA """

    civet_dict = {"ID": []}
    
    def __init__(self, dir_name) -> None:
        self.dir_name = dir_name
        self.get_patient_civet_qc()
        self.read_civet_qc()
    
    def get_patient_civet_qc(self) -> None:
        patients = {}
        for filename in os.listdir(self.dir_name):
            if "civet_qc" in filename:
                patients[filename.split("_")[0]] = filename
        self.patient_files = patients
    
    def read_civet_qc(self):
        for patient_id in self.patient_files:
            self.civet_dict["ID"].append(patient_id)
            with open(os.path.join(self.dir_name, self.patient_files[patient_id]), 'r') as f:
                for line in f:
                    var, value = line.split("=")
                    try:
                        self.civet_dict[var].append(value.strip("\n"))
                    except KeyError:
                        self.civet_dict[var] = [value.strip("\n")]
        
        req_len = len(self.civet_dict["ID"])
        for key in self.civet_dict:
            if len(self.civet_dict[key]) != req_len:
                raise ValueError("all values in dictionary are not equal!")
    
    def to_csv(self, path_out):
        pd.DataFrame(self.civet_dict).to_csv(path_out)


if __name__ == "__main__":
    rd = RawStudyData("/Users/joshua/Downloads")
    rd.to_csv("/Users/joshua/Developer/civetqc/data/studies/TOPSY/civet_TOPSY.csv")
