import copy
import unittest

import numpy as np

from ..data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.dat = Dataset()

    def test_init(self):
        self.assertEqual(self.dat.features.shape, (1086, 33))
        self.assertEqual(self.dat.feature_names.shape, (33,))
        self.assertEqual(self.dat.target.shape, (1086,))
        self.assertEqual(self.dat.target_names.shape, (2,))
        self.assertEqual(self.dat.n_features, 33)
        self.assertEqual(self.dat.n_samples, 1086)

    def test_get_statistic_by_target(self):
        for method_name in [None, "apply_pca", "apply_isomap"]:
            dat_tmp = copy.deepcopy(self.dat)
            try:
                getattr(dat_tmp, method_name)()
            except TypeError:
                pass
            self.assertEqual(len(dat_tmp.get_statistic_by_target(np.mean)), dat_tmp.n_features)
            self.assertEqual(len(dat_tmp.get_statistic_by_target(np.std)), dat_tmp.n_features)

    def test_pca(self):
        self.dat.apply_pca()
        self.assertEqual(self.dat.features.shape, (1086, 2))
        self.assertEqual(self.dat.feature_names.shape, (2,))
        self.assertEqual(self.dat.n_features, 2)

    def test_isomap(self):
        self.dat.apply_isomap()
        self.assertEqual(self.dat.features.shape, (1086, 2))
        self.assertEqual(self.dat.feature_names.shape, (2,))
        self.assertEqual(self.dat.n_features, 2)

    def test_scatterplot(self):
        with self.assertRaises(ValueError):
            self.dat.scatterplot()
