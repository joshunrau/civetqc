from civetqc.dataset import Dataset, StudyData, VariableNotFoundError, DuplicateIdentifierError, CIVET_FILE_PATHS, QC_FILE_PATHS
from copy import deepcopy
from numpy import NaN
import pandas as pd
import unittest


TEST_DATA_DIR = "/Users/joshua/Developer/civetqc/data/tests"
DOES_NOT_EXIST = "/Users/joshua/this_file_does_not_exist.csv"
DUPLICATED_ID = "/Users/joshua/Developer/civetqc/data/tests/duplicated_id.csv"
INVALID_QCVAR_NAME = "/Users/joshua/Developer/civetqc/data/tests/qc_lower.csv"
INVALID_FILE_FORMAT = "/Users/joshua/Developer/civetqc/data/tests/invalid_format.txt"


class TestStudyData(unittest.TestCase):

    def setUp(self) -> None:
        self.sd = StudyData(CIVET_FILE_PATHS["LAM"], QC_FILE_PATHS["LAM"], 1)

    def test_init(self):
        self.assertEqual(self.sd.df.shape, (333, 31))
        self.assertRaises(FileNotFoundError, StudyData, DOES_NOT_EXIST, QC_FILE_PATHS["LAM"])
        self.assertRaises(DuplicateIdentifierError, StudyData, DUPLICATED_ID, QC_FILE_PATHS["LAM"])
        self.assertRaises(VariableNotFoundError, StudyData, CIVET_FILE_PATHS["LAM"], INVALID_QCVAR_NAME)

    def test_equality_operator(self):
        test_df = pd.DataFrame({"x": [1, 2, 3]})
        test_sd = deepcopy(self.sd)
        self.assertNotEqual(test_sd, test_df)
        self.assertEqual(test_sd, self.sd)
        test_sd.idvar = "QC"
        self.assertNotEqual(test_sd, self.sd)
        test_sd.idvar = "ID"
        self.assertEqual(test_sd, self.sd)
        test_sd.qcvar = "ID"
        self.assertNotEqual(test_sd, self.sd)
        test_sd.qcvar = "QC"
        self.assertEqual(test_sd, self.sd)
        test_sd.df = test_df
        self.assertNotEqual(test_sd, self.sd)
        test_sd.df = self.sd.df
        self.assertEqual(test_sd, self.sd)
    
    def test_vars_in_cols(self):
        required_vars = [self.sd.idvar, self.sd.qcvar] + self.sd.required_civet_vars
        self.assertIsNone(self.sd.vars_in_cols(self.sd.df, required_vars, 'file.csv'))
        required_vars.append("TEST")
        self.assertRaises(VariableNotFoundError, self.sd.vars_in_cols, self.sd.df, required_vars, 'file.csv')
