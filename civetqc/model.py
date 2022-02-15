from abc import ABC, abstractmethod
from .data.dataset import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

class BaseModel(ABC):
    
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    def __init__(self) -> None:
        self.data = Dataset()
        gs = GridSearchCV(
            self.pipeline,
            param_grid=self.params,
            scoring=self.f2_scorer,
            cv=5)
        gs.fit(self.data.train['Features'], self.data.train['Target'])
        self.estimator = gs.best_estimator_
        self.predicted = self.estimator.predict(self.data.test["Features"])
    
    def __str__(self):
        return classification_report(self.data.test['Target'], self.predicted)
    
    @abstractmethod
    def pipeline(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    def score(self):
        return self.estimator.score(self.data.test['Features'], self.data.test['Target'])
    

class KNN(BaseModel):
    
    def __init__(self):
        super().__init__()
    
    @property
    def pipeline(self):
        return Pipeline([
            ('nca', NeighborhoodComponentsAnalysis()),
            ('knc', KNeighborsClassifier()),
        ])

    @property
    def params(self):
        return [{
            'knc__n_neighbors': list(range(1, 11)),
            'knc__leaf_size': [10, 30, 50]
        }]


class SVM(BaseModel):
    
    def __init__(self):
        super().__init__()
        
    @property
    def pipeline(self):
        return Pipeline([
            ('nca', NeighborhoodComponentsAnalysis()),
            ('svc', SVC(class_weight="balanced")),
        ])
    
    @property
    def params(self):
        return [{
            'svc__C': [0.1, 1, 10, 100, 1000],
            'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'svc__kernel': ['rbf', 'poly', 'sigmoid']
        }]

class RandomForest(BaseModel):
    
    def __init__(self):
        super().__init__()
    
    @property
    def pipeline(self):
        return Pipeline([
            ('rf', RandomForestClassifier()),
        ])
    
    @property
    def params(self):
        return [{
            'rf__max_depth': [2, 4, 6, 8],
            'rf__class_weight': [{0: 1, 1: 1}, {0: 2, 1: 1}, {0: 3, 1: 1}, {0: 10, 1: 1}, {0: 1, 1: 2}]
        }]

class NeuralNetwork(BaseModel):
    
    def __init__(self):
        super().__init__()
    
    @property
    def pipeline(self):
        return Pipeline([
            ('mlp', MLPClassifier(max_iter=1000)),
        ])
    
    @property
    def params(self):
        return [{
            'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'mlp__activation': ['tanh', 'relu'],
            'mlp__solver': ['sgd', 'adam'],
            'mlp__alpha': [0.0001, 0.05],
            'mlp__learning_rate': ['constant','adaptive'],
        }]