import unittest
from civetqc.core import CivetOutput, UserRatings, CivetData


PATH_CIVET_OUTPUT = "/Users/joshua/Developer/civetqc/data/LAM/QC/civet_LAM_.csv"
PATH_USER_RATINGS = "/Users/joshua/Developer/civetqc/data/LAM/user_ratings.csv"
OUTPUT_DIR = "/Users/joshua/Developer/civetqc/data/Tests"


class TestCore(unittest.TestCase):
    def test_core(self):
        dataset = CivetData(CivetOutput(PATH_CIVET_OUTPUT), UserRatings(PATH_USER_RATINGS), drop_na=True)
        self.assertEquals(dataset.data.shape, (333, 44))