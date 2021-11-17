import os

import civetqc as qc
from civetqc import data
from civetqc.data.create_dataset import MergedDataset
from civetqc.data.make_simulated import SimulatedCIVETData, SimulatedDataset, SimulatedQCData
from civetqc.model.train import TrainModel


SIMULATED_DIR = "/Users/joshua/Developer/civetqc/src/civetqc/data/simulated_data"
MODEL_OUTPUT = "/Users/joshua/Developer/civetqc/src/civetqc/model/model.pkl"


def make_simulated() -> None:
    """ create simulated datasets for testing """
    SimulatedDataset(study_paths=StudyPaths.ALL, balanced=True, simulated_dir=SIMULATED_DIR).to_csv()
    SimulatedCIVETData(study_paths=StudyPaths.ALL, balanced=True, simulated_dir=SIMULATED_DIR).to_csv()
    SimulatedQCData(study_paths=StudyPaths.ALL, balanced=True, simulated_dir=SIMULATED_DIR).to_csv()


class StudyPaths:
    """ paths to studies on my computer """
    ROOT = "/Users/joshua/Developer/civetqc/data"
    FEP = os.path.join(ROOT, "FEP", "FEP_civet_data.csv"), os.path.join(ROOT, "FEP", "FEP_QC.csv"),
    LAM = os.path.join(ROOT, "LAM", "LAM_civet_data.csv"), os.path.join(ROOT, "LAM", "LAM_QC.csv"),
    INSIGHT = os.path.join(ROOT, "INSIGHT", "INSIGHT_civet_data.csv"), os.path.join(ROOT, "INSIGHT", "INSIGHT_QC.csv"),
    TOPSY = os.path.join(ROOT, "TOPSY", "TOPSY_civet_data.csv"), os.path.join(ROOT, "TOPSY", "TOPSY_QC.csv")
    ALL = [FEP, LAM, INSIGHT, TOPSY]


def main():
    merged_data = MergedDataset(study_paths=StudyPaths.ALL, balanced=True)
    model = TrainModel(data = merged_data)
    model.save(MODEL_OUTPUT)


if __name__ == "__main__":
    main()

