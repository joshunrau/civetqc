import os

import pkg_resources as pkg


class ModulePaths:
    root = os.path.abspath(os.path.dirname(__file__))
    data = os.path.join(root, "data")
    features = os.path.join(root, "features")
    model = os.path.join(root, "model")


class SimulatedDataPaths:
    dataset = pkg.resource_filename(__name__, 'data/simulated_data/simulated_dataset.csv')
    civet = pkg.resource_filename(__name__, 'data/simulated_data/simulated_civet.csv')
    qc = pkg.resource_filename(__name__, 'data/simulated_data/simulated_qc.csv')
