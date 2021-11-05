from civetqc.dataset import Dataset, VariableNotFoundError, DuplicateIdentifierError
from copy import deepcopy
import filepaths as paths
from numpy import NaN
import pandas as pd
import unittest


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = Dataset(paths.CIVET_OUTPUT, paths.USER_RATINGS, 1)

    def test_init(self):
        self.assertEqual(self.dataset.df.shape, (333, 42))
        self.assertRaises(FileNotFoundError, Dataset, paths.DOES_NOT_EXIST, paths.USER_RATINGS)
        self.assertRaises(DuplicateIdentifierError, Dataset, paths.DUPLICATED_ID, paths.USER_RATINGS)
        self.assertRaises(VariableNotFoundError, Dataset, paths.CIVET_OUTPUT, paths.INVALID_QCVAR_NAME)

    def test_equality_operator(self):
        test_df = pd.DataFrame({"x": [1, 2, 3]})
        test_dataset = deepcopy(self.dataset)
        self.assertNotEqual(test_dataset, test_df)
        self.assertEqual(test_dataset, self.dataset)
        test_dataset.idvar = "QC"
        self.assertNotEqual(test_dataset, self.dataset)
        test_dataset.idvar = "ID"
        self.assertEqual(test_dataset, self.dataset)
        test_dataset.qcvar = "ID"
        self.assertNotEqual(test_dataset, self.dataset)
        test_dataset.qcvar = "QC"
        self.assertEqual(test_dataset, self.dataset)
        test_dataset.df = test_df
        self.assertNotEqual(test_dataset, self.dataset)
        test_dataset.df = self.dataset.df
        self.assertEqual(test_dataset, self.dataset)
    
    def test_vars_in_cols(self):
        required_vars = [self.dataset.idvar, self.dataset.qcvar] + self.dataset.required_civet_vars
        self.assertIsNone(self.dataset.vars_in_cols(self.dataset.df, required_vars, 'file.csv'))
        required_vars.append("TEST")
        self.assertRaises(VariableNotFoundError, self.dataset.vars_in_cols, self.dataset.df, required_vars, 'file.csv')
