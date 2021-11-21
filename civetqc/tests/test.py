import os
import unittest

from ..resources import Filepaths


class MyTest(unittest.TestCase):

    def test(self):
        self.assertTrue(os.path.exists(Filepaths.saved_model))