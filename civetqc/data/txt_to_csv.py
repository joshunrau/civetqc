import os
from typing import Union

import pandas as pd


def dict_values_equal(d: dict) -> bool:
    """ returns whether all values in dict are of equal length """
    req_len = len(list(d.values())[0])
    for key in d:
        if len(d[key]) != req_len:
            return False
    return True


def txt_to_csv(dir_name: str, output_dir: Union[None, str] = None) -> None:
    """ given a directory containing civet txt outputs, creates a single csv file """

    outfile = os.path.join(output_dir, "civetqc_txt2csv.csv")

    # Get list of patient files in directory
    patient_files = {}
    for filename in os.listdir(dir_name):
        if "civet_qc" in filename:
            patient_files[filename.split("_")[0]] = filename

    # Open each file and append data to dictionary
    civet_dict = {"ID": []}
    for patient_id in patient_files:
        civet_dict["ID"].append(patient_id)
        with open(os.path.join(dir_name, patient_files[patient_id]), 'r') as f:
            for line in f:
                var, value = line.split("=")
                try:
                    civet_dict[var].append(value.strip("\n"))
                except KeyError:
                    civet_dict[var] = [value.strip("\n")]

    # Verify length of all values are equal and write to csv
    if not dict_values_equal(civet_dict):
        raise ValueError("all values in dictionary are not equal!")
    df = pd.DataFrame(civet_dict).sort_values(by="ID")
    df.to_csv(outfile, index=False)
