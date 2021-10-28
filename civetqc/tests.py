from copy import deepcopy
import os
from numpy import NaN
from pandas import Series
import unittest
from .core import Dataset

PATH_CIVET_OUTPUT = "/Users/joshua/Developer/civetqc/data/LAM/QC/civet_LAM_.csv"
PATH_USER_RATINGS = "/Users/joshua/Developer/civetqc/data/LAM/user_ratings.csv"
OUTPUT_DIR = "/Users/joshua/Developer/civetqc/data/tests"


class TestDataset(unittest.TestCase):

    dat = Dataset(PATH_CIVET_OUTPUT, PATH_USER_RATINGS)

    def test_init(self):
        self.assertRaises(FileNotFoundError, Dataset, "/Users/joshua/doesnotexist.csv", [])
        self.assertRaises(ValueError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/invalid.txt", [])
        self.assertEqual(self.dat.data.shape, (333, 44))
        self.assertRaises(RuntimeError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/invalid_var.csv", [])
        self.assertRaises(RuntimeError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/duplicated_id.csv", [])

    def test_vars_in_cols(self):
        vars = [self.dat.idvar, self.dat.qcvar] + self.dat.civet_vars
        self.assertTrue(self.dat.vars_in_cols(vars))
    
    def test_write_data(self):
        test_file = os.path.join(OUTPUT_DIR, "test.csv")
        self.dat.write_data(OUTPUT_DIR, "test.csv")
        self.assertTrue(os.path.isfile(test_file))
        os.remove(test_file)
    
    def test_col_to_numeric(self):
        dat = deepcopy(self.dat)
        self.assertTrue(all([isinstance(x, str) for x in dat.data["QC"]]))
        dat.col_to_numeric("QC")
        self.assertTrue(all([isinstance(x, (int, float, NaN)) for x in dat.data["QC"]]))