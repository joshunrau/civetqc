import os
import shutil
import unittest

from pkg_resources import resource_filename
from unittest.mock import patch

import numpy as np

from civetqc import utils

from civetqc.data import CivetData, QCRatingsData
from civetqc.main import main

DUMMY_DATA_PATHS = {
  'csv': resource_filename(__name__, 'dummy_data/dummy.csv'),
  'dir': resource_filename(__name__, 'dummy_data/verify')
}

EXPECTED_RESULTS = {
  'PASS': 203,
  'FAIL': 197
}

TEST_OUTPUT_DIR = os.path.abspath('tmp')

class TestData(unittest.TestCase):
  def test_data(self):
    data_from_csv = CivetData.from_csv(DUMMY_DATA_PATHS['csv'])
    data_from_output_files = CivetData.from_output_files(DUMMY_DATA_PATHS['dir'])
    self.assertTrue(np.array_equal(data_from_csv.subject_ids, data_from_output_files.subject_ids))
    self.assertTrue(np.array_equal(data_from_csv.features, data_from_output_files.features))


class TestMain(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    os.mkdir(TEST_OUTPUT_DIR)
  
  @classmethod
  def tearDownClass(cls) -> None:
    shutil.rmtree(TEST_OUTPUT_DIR)
  
  @patch('sys.argv', ['civetqc', 'this_file_does_not_exist.csv'])
  def test_input_path_does_not_exist(self):
    self.assertRaises(FileNotFoundError, main)
  
  @patch('sys.argv', ['civetqc', DUMMY_DATA_PATHS['csv'], '--output_dir', 'probably_not_a_directory'])
  def test_output_dir_does_not_exist(self):
    self.assertRaises(NotADirectoryError, main)
  
  @patch('sys.argv', ['civetqc', DUMMY_DATA_PATHS['csv'], '--output_dir', TEST_OUTPUT_DIR])
  def test_valid_args(self):
    output_file = os.path.join(TEST_OUTPUT_DIR, 'civetqc.csv')
    self.assertFalse(os.path.exists(output_file))
    main()
    self.assertTrue(os.path.exists(output_file))
    data = QCRatingsData.from_csv(output_file, qcvar='RATING', allow_non_numeric=True)
    self.assertEqual(sum(data.qc_ratings == 'PASS'), EXPECTED_RESULTS['PASS'])
    self.assertEqual(sum(data.qc_ratings == 'FAIL'), EXPECTED_RESULTS['FAIL'])

class TestUtils(unittest.TestCase):

  def test_get_non_unique(self):
    a1 = np.array([1, 2, 3, 4, 5])
    self.assertEqual(utils.get_non_unique(a1).size, 0)
    a2 = np.array([1, 1])
    self.assertEqual(utils.get_non_unique(a2).size, 1)
    a3 = np.array([1, 2, 3, 2, 3])
    self.assertEqual(utils.get_non_unique(a3).size, 2)
  
  def test_check_type(self):
    self.assertRaises(TypeError, utils.check_types, (5, int), ([1, 2, 3], np.ndarray))
    self.assertRaises(TypeError, utils.check_types, (np.ndarray([1, 2, 3]), list))
  
  def test_get_index(self):
    self.assertRaises(ValueError, utils.get_index, np.array([1, 2, 3]), 4)
    self.assertEqual(utils.get_index(np.array([1, 2, 3, 3]), 3), 2)

  def test_joint_sort(self):
    a1 = np.arange(10, 0, -1)
    a2 = np.arange(1, 31).reshape((10, 3))
    s1, s2 = utils.joint_sort(a1, a2, axis=0)
    self.assertTrue(np.array_equal(s1, np.arange(1, 11)))
    self.assertTrue(np.array_equal(s2, np.flip(a2, axis=0)))
    a3 = np.arange(1, 31).reshape((3, 10))
    _, s3 = utils.joint_sort(a1, a3, axis=1)
    self.assertTrue(np.array_equal(s3, np.flip(a3, axis=1)))