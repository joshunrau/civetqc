import os
import shutil
import unittest

from pkg_resources import resource_filename
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from civetqc.app import App

class TestApp(unittest.TestCase):

  paths = SimpleNamespace(
    dummyData = resource_filename(__name__, 'resources/dummy_data.csv'),
    outputDir = os.path.abspath('tmp'),
    outputFilename = 'test.csv'
  )

  expected_results = {
    'PASS': 203,
    'FAIL': 197
  }

  @classmethod
  def setUpClass(cls) -> None:
    os.mkdir(cls.paths.outputDir)
  
  @classmethod
  def tearDownClass(cls) -> None:
    shutil.rmtree(cls.paths.outputDir)
  
  @patch('sys.argv', ['civetqc', 'this_file_does_not_exist.csv'])
  def test_input_path_does_not_exist(self):
    self.assertRaises(FileNotFoundError, App.main)
  
  @patch('sys.argv', ['civetqc', paths.dummyData, '--output_dir', 'probably_not_a_directory'])
  def test_output_dir_does_not_exist(self):
    self.assertRaises(NotADirectoryError, App.main)
  
  @patch('sys.argv', ['civetqc', paths.dummyData, '--output_dir', paths.outputDir, '--output_filename', paths.outputFilename])
  def test_valid_args(self):
    output_file = os.path.join(self.paths.outputDir, self.paths.outputFilename)
    output_file_exists = lambda : os.path.exists(output_file)
    self.assertFalse(output_file_exists())
    App.main()
    self.assertTrue(output_file_exists())
    df = pd.read_csv(output_file)
    self.assertEqual(sum(df['RESULT'] == 'PASS'), self.expected_results['PASS'])
    self.assertEqual(sum(df['RESULT'] == 'FAIL'), self.expected_results['FAIL'])