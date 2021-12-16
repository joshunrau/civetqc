import os
from civetqc.utils.txt_to_csv import txt_to_csv

raw_data_dir = "/Users/joshua/Developer/civetqc/data/NUSDAST/raw"

def rm_study_name():
    for filename in os.listdir(raw_data_dir):
        os.rename(os.path.join(raw_data_dir, filename), os.path.join(raw_data_dir, filename.strip("NUSDAST_")))

if __name__ == "__main__":
    rm_study_name()
    txt_to_csv("/Users/joshua/Developer/civetqc/data/NUSDAST/raw", "/Users/joshua/Developer/civetqc/data/NUSDAST/NUSDAST_civet_data.csv", 2)