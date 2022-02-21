import multiprocessing as mp
import pandas as pd
import time

from sklearn.model_selection import train_test_split

from .study import Study


class Dataset:
    
    def __init__(self, df):
        
        self.df = df
        self.features = df.drop(["ID", "QC"], axis=1).to_numpy()
        self.target = df["QC"].to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.33, random_state=0)
        self.train, self.test = {}, {}
        self.train["Features"], self.test["Features"] = X_train, X_test
        self.train["Target"], self.test["Target"] = y_train, y_test
    
    @classmethod
    def make(cls):
        
        start = time.perf_counter()
        
        studies = [
            Study(name="FEP", id_prefix="FEP"),
            Study(name="INSIGHT", id_prefix="Insight"),
            Study(name="LAM", id_prefix="LAM"),
            Study(name="NUSDAST"),
            Study(name="TOPSY", id_len=3, includes="V1_gradient_n4_anlm0.5r")
        ]
        
        queue = mp.Queue()

        processes = []
        results = []

        for study in studies:
            process = mp.Process(target=study.get_data, args=(queue,))
            processes.append(process)
            process.start()

        for process in processes:
            df = queue.get() # will block
            results.append(df)

        for process in processes:
            process.join()
        
        df = results[0]
        print(df.shape)
        for study_df in results[1:]:
            df = pd.concat([df, study_df], ignore_index=True)
            print(df.shape)
        
        stop = time.perf_counter()
        print(f"Created dataset in {stop - start} seconds")
    
        return cls(df)
    
    @classmethod
    def from_file(cls, filepath):
        return cls(pd.read_csv(filepath))