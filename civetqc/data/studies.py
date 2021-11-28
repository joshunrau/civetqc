import os

class Studies:
    names = ["FEP", "LAM", "INSIGHT", "TOPSY"]
    dir_path = "/Users/joshua/Developer/civetqc/data"
    filepaths = {n: (os.path.join(dir_path, n, f"{n}_civet_data.csv"), os.path.join(dir_path, n, f"{n}_QC.csv")) for n in names}