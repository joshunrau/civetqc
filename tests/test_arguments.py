import unittest

from civetqc.arguments import Arguments
from civetqc.dataset import Dataset
from filepaths import CIVET_OUTPUT, USER_RATINGS
import subprocess


class TestArguments(unittest.TestCase):
    
    def setUp(self) -> None:
        self.arguments = Arguments([CIVET_OUTPUT, USER_RATINGS])
    
    def test_init(self):
        d1 = Dataset(CIVET_OUTPUT, USER_RATINGS)
        d2 = Dataset(self.arguments.args.civet_output, self.arguments.args.user_ratings)
        self.assertEqual(d1, d2)
    
    def test_check_valid_files(self):
        pass

    def test_check_valid_cutoff(self):
        pass

    def test_check_output_dir(self):
        pass

    def test_shell(self):
        cmd = subprocess.run(["civetqc", CIVET_OUTPUT, USER_RATINGS])
        self.assertEqual(cmd.returncode, 0)
        
    
