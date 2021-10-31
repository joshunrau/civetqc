from .core import Dataset, Modeler
from copy import deepcopy
from numpy import NaN
import os
from .parse_args import parse_args
import pandas as pd
import unittest



PATH_CIVET_OUTPUT = "/Users/joshua/Developer/civetqc/data/LAM/civet_data.csv"
PATH_USER_RATINGS = "/Users/joshua/Developer/civetqc/data/LAM/user_ratings.csv"
OUTPUT_DIR = "/Users/joshua/Developer/civetqc/data/tests"



class TestDataset(unittest.TestCase):

    def setUp(self):
        self.civet = Dataset(PATH_CIVET_OUTPUT, [Dataset.idvar] + Dataset.civet_vars)
        self.user = Dataset(PATH_USER_RATINGS, [Dataset.idvar, Dataset.qcvar])

    def test_data_shape(self):
        self.assertEqual(self.civet.data.shape, (346, 40))
        self.assertEqual(self.user.data.shape, (410, 3))
    
    def test_error_handling(self):
        self.assertRaises(FileNotFoundError, Dataset, "/Users/joshua/doesnotexist.csv", [])
        self.assertRaises(ValueError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/invalid_format.txt", [])
        self.assertRaises(RuntimeError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/duplicated_id.csv", [])
        self.assertRaises(RuntimeError, Dataset, "/Users/joshua/Developer/civetqc/data/tests/invalid_var.csv", ["ID", "QC"])

    def test_equality_operator(self):
        example_df = pd.DataFrame({"x": [1, 2, 3]})
        civet_copy = deepcopy(self.civet)
        self.assertNotEqual(civet_copy, example_df)
        self.assertEqual(civet_copy, self.civet)
        civet_copy.idvar = "QC"
        self.assertNotEqual(civet_copy, self.civet)
        civet_copy.idvar = "ID"
        self.assertEqual(civet_copy, self.civet)
        civet_copy.qcvar = "ID"
        self.assertNotEqual(civet_copy, self.civet)
        civet_copy.qcvar = "QC"
        self.assertEqual(civet_copy, self.civet)
        civet_copy.data = example_df
        self.assertNotEqual(civet_copy, self.civet)
        civet_copy.data = self.civet.data
        self.assertEqual(civet_copy, self.civet)
    
    def test_vars_in_cols(self):
        usr_vars = [self.user.idvar, self.user.qcvar]
        self.assertTrue(self.user.vars_in_cols(usr_vars))
        self.assertFalse(self.civet.vars_in_cols(usr_vars))
    
    def test_write_data(self):
        test_file = os.path.join(OUTPUT_DIR, "test.csv")
        self.civet.write_data(OUTPUT_DIR, "test.csv")
        self.assertTrue(os.path.isfile(test_file))
        os.remove(test_file)
    
    def test_col_to_numeric(self):
        self.user.data["TEST"] = "-1"
        self.assertTrue(all([isinstance(x, str) for x in self.user.data["TEST"]]))
        self.user.col_to_numeric("TEST")
        self.assertTrue(all([isinstance(x, (int, float, NaN)) for x in self.user.data["TEST"]]))


class TestModeler(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Modeler(PATH_CIVET_OUTPUT, PATH_USER_RATINGS)

    def test_data(self):
        self.assertEqual(self.model.data.shape, (333, 42))
    

class TestParser(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Modeler(PATH_CIVET_OUTPUT, PATH_USER_RATINGS)
    
    def test_parser(self):
        parser = parse_args([PATH_CIVET_OUTPUT, PATH_USER_RATINGS, OUTPUT_DIR])
        model = Modeler(parser.civet_output, parser.user_ratings)
        self.assertEqual(model, self.model)