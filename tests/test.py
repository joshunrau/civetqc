import os
import unittest

from civetqc.main import main
from civetqc.resources import ResourceFilepaths

class Test(unittest.TestCase):

    def test_resource_filepaths_exist(self):
        for filepath in ResourceFilepaths.saved_models.values():
            self.assertTrue(os.path.isfile(filepath))
    
    def test_main_errors(self):
        self.assertRaises(FileNotFoundError, main, ["invalid/path/to/file.csv"])
        self.assertRaises(NotADirectoryError, main, [ResourceFilepaths.simulated_data['civet_data'], "-o", "invalid/path/to/directory"])