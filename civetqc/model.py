from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Model:
    
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    def __init__(self, clf, df, **kwargs):
        
        self.df = df
        
        self.features = df.drop(["ID", "QC"], axis=1).to_numpy()
        self.target = df["QC"].to_numpy()
        
        x_train, x_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.33, random_state=0)
        self.train, self.test = {}, {}
        self.train["Features"], self.test["Features"] = x_train, x_test
        self.train["Target"], self.test["Target"] = y_train, y_test
        
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        
        self.grid = GridSearchCV(self.pipe, param_grid=kwargs, n_jobs=-1, scoring=self.f2_scorer)
        self.grid.fit(self.train["Features"], self.train["Target"])
        
    def __str__(self):
        return classification_report(self.test['Target'], self.predicted)
    
    @property
    def predicted(self):
        return self.grid.predict(self.test["Features"])

