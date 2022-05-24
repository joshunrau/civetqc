import pickle

from .resources import ResourceFilepaths

def load_saved_model(model_name):
    with open(ResourceFilepaths.saved_models.get(model_name), 'rb') as f:
        return pickle.load(f)
