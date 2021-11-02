from .core import Modeler
from .dataset import Dataset, InvalidFileFormatError, VariableNotFoundError, DuplicateIdentifierError
from copy import deepcopy
from numpy import NaN
import os
from .parser import parse_args
import pandas as pd
import unittest
import subprocess


PATH_CIVET_OUTPUT = "/Users/joshua/Developer/civetqc/data/LAM/civet_data.csv"
PATH_USER_RATINGS = "/Users/joshua/Developer/civetqc/data/LAM/user_ratings.csv"
OUTPUT_DIR = "/Users/joshua/Developer/civetqc/data/tests"
NON_EXISTENT_FILE = "/Users/joshua/this_file_does_not_exist.csv"
INVALID_FILE_FORMAT = "/Users/joshua/Developer/civetqc/data/tests/invalid_format.txt"
DUPLICATED_ID = "/Users/joshua/Developer/civetqc/data/tests/duplicated_id.csv"
MISSING_QC_VAR = "/Users/joshua/Developer/civetqc/data/tests/qc_lower.csv"


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = Dataset(PATH_CIVET_OUTPUT, PATH_USER_RATINGS)

    def test_init(self):
        self.assertEqual(self.dataset.df.shape, (333, 42))
        self.assertRaises(FileNotFoundError, Dataset, NON_EXISTENT_FILE, PATH_USER_RATINGS)
        self.assertRaises(InvalidFileFormatError, Dataset, INVALID_FILE_FORMAT, PATH_USER_RATINGS)
        self.assertRaises(DuplicateIdentifierError, Dataset, DUPLICATED_ID, PATH_USER_RATINGS)
        self.assertRaises(VariableNotFoundError, Dataset, PATH_CIVET_OUTPUT, MISSING_QC_VAR)

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

    def test_write_data(self):
        test_file = os.path.join(OUTPUT_DIR, "test.csv")
        self.dataset.write_data(OUTPUT_DIR, "test.csv")
        self.assertTrue(os.path.isfile(test_file))
        os.remove(test_file)

    def test_col_to_numeric(self):
        self.dataset.df["TEST"] = "-1"
        self.assertTrue(all([isinstance(x, str) for x in self.dataset.df["TEST"]]))
        self.dataset.col_to_numeric("TEST")
        self.assertTrue(all([isinstance(x, (int, float, NaN)) for x in self.dataset.df["TEST"]]))
    
    def test_vars_in_cols(self):
        required_vars = [self.dataset.idvar, self.dataset.qcvar] + self.dataset.civet_vars
        self.assertTrue(self.dataset.vars_in_cols(self.dataset.df, required_vars))
        required_vars.append("TEST")
        self.assertFalse(self.dataset.vars_in_cols(self.dataset.df, required_vars))


class TestParser(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = Dataset(PATH_CIVET_OUTPUT, PATH_USER_RATINGS)

    def test_parser(self):
        parser = parse_args([PATH_CIVET_OUTPUT, PATH_USER_RATINGS])
        dataset = Dataset(parser.civet_output, parser.user_ratings)
        self.assertEqual(dataset, self.dataset)
    
    def test_shell(self):
        cmd = subprocess.run(["civetqc", PATH_CIVET_OUTPUT, PATH_USER_RATINGS])
        self.assertEqual(cmd.returncode, 0)
