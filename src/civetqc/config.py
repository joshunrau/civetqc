import os


class ModulePaths:
    root = os.path.abspath(os.path.dirname(__file__))
    data = os.path.join(root, "data")
    features = os.path.join(root, "features")
    model = os.path.join(root, "model")

