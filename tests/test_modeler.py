from civetqc.dataset import Dataset
from civetqc.modeler import Modeler
from tests.filepaths import CIVET_OUTPUT, USER_RATINGS
import unittest


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = Dataset(CIVET_OUTPUT, USER_RATINGS, 1)
        self.model = Modeler(CIVET_OUTPUT, USER_RATINGS, 1)

    def test(self):
        print(self.model)

    
