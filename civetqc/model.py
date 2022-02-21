from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data.dataset import Dataset

class Model:
    
    f2_scorer = make_scorer(fbeta_score, beta=2)
    path_saved_data = "/Users/joshua/Developer/civetqc/data/processed/civetqc_data_Feb20.csv"
    
    def __init__(self, clf, **kwargs):
        
        self.data = Dataset.from_file(self.path_saved_data)
        
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        
        self.grid = GridSearchCV(self.pipe, param_grid=kwargs, n_jobs=-1, scoring=self.f2_scorer)
        self.grid.fit(self.data.train["Features"], self.data.train["Target"])
        self.predicted = self.grid.predict(self.data.test["Features"])
        
    def __str__(self):
        return classification_report(self.data.test['Target'], self.predicted)
    
    @property
    def probabilities(self):
        try:
            return self.grid.predict_proba(self.data.test["Features"])
        except AttributeError:
            return None
    
    def decision_function(self):
        try:
            return self.grid.decision_function(self.data.test["Features"])
        except AttributeError:
            return None