import unittest
from .core import Dataset


PATH_CIVET_OUTPUT = "/Users/joshua/Developer/civetqc/data/LAM/QC/civet_LAM_.csv"
PATH_USER_RATINGS = "/Users/joshua/Developer/civetqc/data/LAM/user_ratings.csv"
OUTPUT_DIR = "/Users/joshua/Developer/civetqc/data/Tests"


class TestDataset(unittest.TestCase):
    def test_core(self):
        dataset = Dataset(PATH_CIVET_OUTPUT, PATH_USER_RATINGS)
        self.assertEquals(dataset.data.shape, (333, 44))