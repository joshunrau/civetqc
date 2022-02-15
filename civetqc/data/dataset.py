import datetime
import os
import pandas as pd

try:
    from study import StudyData
except ImportError:
    from .study import StudyData

class Dataset:

    processed_data_dir = os.path.join(StudyData.data_dir, "processed")

    def __init__(self) -> None:

        studies = [
            StudyData(study_name="FEP", id_prefix="FEP"),
            StudyData(study_name="INSIGHT", id_prefix="Insight"),
            StudyData(study_name="LAM", id_prefix="LAM"),
            StudyData(study_name="NUSDAST"),
            StudyData(study_name="TOPSY", id_len=3, includes="V1_gradient_n4_anlm0.5r")
        ]

        for study in studies:
            try:
                self.df = pd.concat([self.df, study.df], ignore_index=True)
            except AttributeError:
                self.df = study.df
    
    def make(self):
        filename = f"dataset_{datetime.date.today().isoformat()}.csv"
        filepath = os.path.join(self.processed_data_dir, filename)
        self.df.to_csv(filepath, index=False)
    
    @staticmethod
    def get_path():
        most_recent_dataset = None
        for filename in os.listdir(Dataset.processed_data_dir):
            if filename.startswith("dataset"):
                file_date = datetime.date.fromisoformat(filename.split("_")[1].split(".")[0])
                if most_recent_dataset is None or file_date > most_recent_dataset:
                    most_recent_dataset = file_date
        return os.path.join(Dataset.processed_data_dir, f"dataset_{most_recent_dataset}.csv")


        
        
    