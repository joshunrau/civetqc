import civetqc.dataset as dataset
import copy
import os
import pandas as pd
import random
import unittest


class SimulatedStudyData(dataset.StudyData):
    
    simulated_data_dir = "/Users/joshua/Developer/civetqc/data/simulated"

    def __init__(self, id_range: tuple, qc_range: tuple, seed: int):
        random.seed(seed)
        self.dict_data = {}
        self.n = len(range(id_range[0], id_range[1]))
        self.dict_data[self.idvar] = [x for x in range(id_range[0], id_range[1])]
        self.dict_data[self.qcvar] = [random.randint(qc_range[0], qc_range[1]) for _ in range(self.n)]
        self.df = pd.DataFrame(self.dict_data)
        for var in self.feature_names:
            self.dict_data[var] = [round(random.uniform(1, 10), 3) for _ in range(self.n)]
        self.df = pd.DataFrame(self.dict_data)
        self.min_id, self.max_id = self.df['ID'].min(), self.df['ID'].max()
        self.civet_path = os.path.join(self.simulated_data_dir, f"civet_{self.min_id}_{self.max_id}.csv")
        self.qc_path = os.path.join(self.simulated_data_dir, f"qc_{self.min_id}_{self.max_id}.csv")

    def duplicate_id(self):
        self.df.loc[0, "ID"] = self.df.loc[1, "ID"]

    def remove_qc_var(self):
        self.df = self.df.drop(['QC'], axis=1)

    def negative_qc_rating(self):
        self.df.loc[0, "QC"] = -1

    def write_data(self):
        for filepath, list_vars in zip(
            [self.civet_path, self.qc_path], [self.required_civet_vars, copy.copy(self.required_user_vars)]):
            if not os.path.isfile(filepath):
                try:
                    # noinspection PyTypeChecker
                    self.df.to_csv(path_or_buf=filepath, columns=list_vars, index=False)
                except KeyError:
                    list_vars.remove("QC")
                    # noinspection PyTypeChecker
                    self.df.to_csv(path_or_buf=filepath, columns=list_vars, index=False)


class TestStudyData(unittest.TestCase, dataset.Dataset):

    filepaths = {}
    error_filepaths = {}

    @classmethod
    def setUpClass(cls) -> None:

        for min_id in range(1, 101, 20):
            dat = SimulatedStudyData(id_range=(min_id, min_id + 20), qc_range=(0, 3), seed=min_id)
            dat.write_data()
            cls.filepaths[min_id] = (dat.civet_path, dat.qc_path)

        for min_id in range(101, 161, 20):
            dat = SimulatedStudyData(id_range=(min_id, min_id + 20), qc_range=(0, 3), seed=min_id)
            if min_id == 101:
                dat.duplicate_id()
            elif min_id == 121:
                dat.remove_qc_var()
            elif min_id == 141:
                dat.negative_qc_rating()
            dat.write_data()
            cls.error_filepaths[min_id] = (dat.civet_path, dat.qc_path)

    @classmethod
    def tearDownClass(cls) -> None:
        for d in cls.filepaths, cls.error_filepaths:
            for key in d:
                for i in range(len(d[key])):
                    os.remove(d[key][i])

    def setUp(self) -> None:
        self.datasets = {min_id: dataset.StudyData(self.filepaths[min_id][0], self.filepaths[min_id][1]) for min_id in self.filepaths}
        assert self.filepaths.keys() == self.datasets.keys()

    def test_data_shape(self):
        self.assertTrue(all([self.datasets[key].df.shape == (20, 31) for key in self.datasets]))

    def test_data_errors(self):
        self.assertRaises(dataset.DuplicateIdentifierError, dataset.StudyData, self.error_filepaths[101][0], self.error_filepaths[101][1])
        self.assertRaises(dataset.VariableNotFoundError, dataset.StudyData, self.error_filepaths[121][0], self.error_filepaths[121][1])
        self.assertRaises(dataset.NegativeQCRatingError, dataset.StudyData, self.error_filepaths[141][0], self.error_filepaths[141][1])
