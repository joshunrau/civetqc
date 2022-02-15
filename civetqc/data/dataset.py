import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import QCRatingsData, TabularCivetData, StudyData

class Dataset:
    
    feature_names = TabularCivetData.feature_names
    target_names = QCRatingsData.target_names
    
    def __init__(self) -> None:
        
        studies = [
            StudyData(study_name="FEP", id_prefix="FEP"),
            StudyData(study_name="INSIGHT", id_prefix="Insight"),
            StudyData(study_name="LAM", id_prefix="LAM"),
            StudyData(study_name="NUSDAST"),
            StudyData(study_name="TOPSY", id_len=3, includes="V1_gradient_n4_anlm0.5r")
        ] 
        
        self.df = studies[0].df
        for study in studies[1:]:
            self.df = pd.concat([self.df, study.df], ignore_index=True)
        
        self.features = self.df[self.feature_names].to_numpy()
        self.target = self.df[QCRatingsData.qcvar].to_numpy()
        
        self.train = {}
        self.test = {}
        self.train["Features"], self.test["Features"], self.train["Target"], self.test["Target"] = train_test_split(self.features, self.target, random_state=1)
        
        self.scaler = StandardScaler()
        self.train["Features"] = self.scaler.fit_transform(self.train["Features"])
        self.test["Features"] = self.scaler.transform(self.test["Features"])

