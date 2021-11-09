import civetqc.dataset as dataset
import copy
import os
import pandas as pd
import random
import unittest


class SimulatedDataset(dataset.Dataset):
    
    simulated_data_dir = "/Users/joshua/Developer/civetqc/data/simulated"

    def __init__(self, id_range: tuple, qc_range: tuple, seed: int):
        random.seed(seed)
        self.dict_data = {}
        self.n = len(range(id_range[0], id_range[1]))
        self.dict_data[self.idvar] = [x for x in range(id_range[0], id_range[1])]
        self.dict_data[self.qcvar] = [random.randint(qc_range[0], qc_range[1]) for _ in range(self.n)]
        self.df = pd.DataFrame(self.dict_data)
        for var in self.civet_feature_names:
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
        for filepath, list_vars in zip([self.civet_path, self.qc_path],
                                       [self.required_civet_vars, copy.copy(self.required_user_vars)]):
            if not os.path.isfile(filepath):
                try:
                    # noinspection PyTypeChecker
                    self.df.to_csv(path_or_buf=filepath, columns=list_vars, index=False)
                except KeyError:
                    list_vars.remove("QC")
                    # noinspection PyTypeChecker
                    self.df.to_csv(path_or_buf=filepath, columns=list_vars, index=False)


class TestDataset(unittest.TestCase):

    filepaths = {}
    error_filepaths = {}

    @classmethod
    def setUpClass(cls) -> None:

        for min_id in range(1, 101, 20):
            dat = SimulatedDataset(id_range=(min_id, min_id + 20), qc_range=(0, 3), seed=min_id)
            dat.write_data()
            cls.filepaths[min_id] = {"CIVET": dat.civet_path, "QC": dat.qc_path}

        for min_id in range(101, 161, 20):
            dat = SimulatedDataset(id_range=(min_id, min_id + 20), qc_range=(0, 3), seed=min_id)
            if min_id == 101:
                dat.duplicate_id()
            elif min_id == 121:
                dat.remove_qc_var()
            elif min_id == 141:
                dat.negative_qc_rating()
            dat.write_data()
            cls.error_filepaths[min_id] = {"CIVET": dat.civet_path, "QC": dat.qc_path}

        assert len([f for f in os.listdir(SimulatedDataset.simulated_data_dir) if f.endswith(".csv")]) == 16

    @classmethod
    def tearDownClass(cls) -> None:
        for d in cls.filepaths, cls.error_filepaths:
            for key in d:
                for filepath in d[key]:
                    os.remove(d[key][filepath])

    def setUp(self) -> None:
        self.datasets = {min_id: dataset.Dataset(self.filepaths[min_id]["CIVET"], self.filepaths[min_id]["QC"], 1) for
                         min_id in self.filepaths}
        assert self.filepaths.keys() == self.datasets.keys()

    def test_data_shape(self):
        self.assertTrue(all([self.datasets[key].df.shape == (20, 31) for key in self.datasets]))

    def test_data_errors(self):
        self.assertRaises(dataset.DuplicateIdentifierError, dataset.Dataset, self.error_filepaths[101]["CIVET"],
                          self.error_filepaths[101]["QC"])
        self.assertRaises(dataset.VariableNotFoundError, dataset.Dataset, self.error_filepaths[121]["CIVET"],
                          self.error_filepaths[121]["QC"])
        self.assertRaises(dataset.NegativeQCRatingError, dataset.Dataset, self.error_filepaths[141]["CIVET"],
                          self.error_filepaths[141]["QC"])

    def test_equality(self):
        dataset_copy = copy.deepcopy(self.datasets[1])
        test_dataframe = pd.DataFrame({"x": [1, 2, 3]})
        self.assertFalse(id(self.datasets[1]) == id(dataset_copy))
        self.assertEqual(self.datasets[1], dataset_copy)
        self.assertNotEqual(self.datasets[1], test_dataframe)
        dataset_copy.idvar = "QC"
        self.assertNotEqual(self.datasets[1], dataset_copy)
        dataset_copy.idvar = "ID"
        self.assertEqual(self.datasets[1], dataset_copy)
        dataset_copy.qcvar = "ID"
        self.assertNotEqual(self.datasets[1], dataset_copy)
        dataset_copy.qcvar = "QC"
        self.assertEqual(self.datasets[1], dataset_copy)
        dataset_copy.df = test_dataframe
        self.assertNotEqual(self.datasets[1], dataset_copy)
        dataset_copy.df = self.datasets[1].df
        self.assertEqual(self.datasets[1], dataset_copy)
    
    def test_master_dataset(self):
        fpaths = {str(study): (self.filepaths[study]["CIVET"], self.filepaths[study]["QC"]) for study in self.filepaths}
        master_dataset = dataset.MasterDataset(fpaths)
        self.assertTrue(master_dataset.df.shape == (100, 31))
        self.assertTrue(master_dataset.df.drop_duplicates().shape == (100, 31))
        self.assertTrue(master_dataset.features.train.shape == (75, 29))
        self.assertTrue(master_dataset.features.test.shape == (25, 29))
        self.assertTrue(master_dataset.target.train.shape == (75,))
        self.assertTrue(master_dataset.target.test.shape == (25,))
