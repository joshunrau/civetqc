import unittest
import os
import shutil
from civetqc.core import main, UserData
from civetqc.config import SimulatedDataPaths


class TestPaths:
    root = os.path.abspath(os.path.dirname(__file__))
    data = os.path.join(root, "data")
    output = os.path.join(data, UserData.output_filename)

class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.isdir(TestPaths.data):
            os.mkdir(TestPaths.data)
    
    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(TestPaths.data):
            shutil.rmtree(TestPaths.data)
    
    def test_main(self):
        self.assertFalse(os.path.exists(TestPaths.output))
        main([SimulatedDataPaths.civet, TestPaths.data])
        self.assertTrue(os.path.exists(TestPaths.output))

