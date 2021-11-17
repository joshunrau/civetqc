import os

import civetqc as qc
from civetqc.data.create_dataset import MergedDataset
from civetqc.data.make_simulated import SimulatedCIVETData, SimulatedDataset, SimulatedQCData


SIMULATED_DIR = "/Users/joshua/Developer/civetqc/src/civetqc/data/simulated_data"


class StudyPaths:
    ROOT = "/Users/joshua/Developer/civetqc/data"
    FEP = os.path.join(ROOT, "FEP", "FEP_civet_data.csv"), os.path.join(ROOT, "FEP", "FEP_QC.csv"),
    LAM = os.path.join(ROOT, "LAM", "LAM_civet_data.csv"), os.path.join(ROOT, "LAM", "LAM_QC.csv"),
    INSIGHT = os.path.join(ROOT, "INSIGHT", "INSIGHT_civet_data.csv"), os.path.join(ROOT, "INSIGHT", "INSIGHT_QC.csv"),
    TOPSY = os.path.join(ROOT, "TOPSY", "TOPSY_civet_data.csv"), os.path.join(ROOT, "TOPSY", "TOPSY_QC.csv")
    ALL = [FEP, LAM, INSIGHT, TOPSY]


def main():

    # Merge data from all studies
    merged_data = MergedDataset(study_paths=StudyPaths.ALL, balanced=True)

    # Create simulated datasets for testing
    SimulatedDataset(study_paths=StudyPaths.ALL, balanced=True, simulated_dir=SIMULATED_DIR).to_csv()
    SimulatedCIVETData(study_paths=StudyPaths.ALL, balanced=True, simulated_dir=SIMULATED_DIR).to_csv()
    SimulatedQCData(study_paths=StudyPaths.ALL, balanced=True, simulated_dir=SIMULATED_DIR).to_csv()
    

if __name__ == "__main__":
    main()

