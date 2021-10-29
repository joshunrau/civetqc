from .core import Dataset
from copy import deepcopy
from numpy import NaN
import os
from .parse_args import parse_args
import pandas as pd
import unittest


PATH_CIVET_OUTPUT = "/Users/joshua/Developer/civetqc/data/LAM/QC/civet_LAM_.csv"
PATH_USER_RATINGS = "/Users/joshua/Developer/civetqc/data/LAM/user_ratings.csv"
OUTPUT_DIR = "/Users/joshua/Developer/civetqc/data/tests"


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dat = Dataset(PATH_CIVET_OUTPUT, PATH_USER_RATINGS)
    
    def test_parser(self):
        parser = parse_args([PATH_CIVET_OUTPUT, PATH_USER_RATINGS, OUTPUT_DIR])
        dat = Dataset(parser.civet_output, parser.user_ratings)
        self.assertEqual(dat, self.dat)

    def test_init(self):
        self.assertEqual(self.dat.data.shape, (333, 44))
        self.assertRaises(FileNotFoundError, Dataset, "/Users/joshua/doesnotexist.csv", [])
        self.assertRaises(ValueError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/invalid.txt", [])
        self.assertRaises(RuntimeError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/invalid_var.csv", [])
        self.assertRaises(RuntimeError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/duplicated_id.csv", [])
    
    def test__eq__(self):
        ex_df = pd.DataFrame({"x": [1, 2, 3]})
        dat1, dat2 = deepcopy(self.dat), deepcopy(self.dat)
        self.assertEqual(dat1, dat2)
        self.assertNotEqual(dat1, ex_df)
        dat1.idvar = ""
        self.assertNotEqual(dat1, dat2)
        dat1.idvar = "ID"
        self.assertEqual(dat1, dat2)
        dat1.qcvar = ""
        self.assertNotEqual(dat1, dat2)
        dat1.qcvar = "QC"
        self.assertEqual(dat1, dat2)
        dat1.data = ex_df
        self.assertNotEqual(dat1, dat2)
        dat1.data = dat2.data
        self.assertEqual(dat1, dat2)

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