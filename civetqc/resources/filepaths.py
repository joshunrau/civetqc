import os

from pkg_resources import resource_filename, resource_listdir

class ResourceFilepaths:
    
    @classmethod
    @property
    def saved_models(cls):
        return cls.load("saved_models", ".pkl")
    
    @classmethod
    @property
    def simulated_data(cls):
        """ method from saved models """
        return cls.load("simulated_data", ".csv")
    
    @staticmethod
    def load(directory: str, file_extension: str):
        filepaths = {}
        for filename in resource_listdir(__name__, directory):
            if filename.endswith(file_extension):
                filepath = resource_filename(__name__, "/".join([directory, filename]))
                if not os.path.isfile(filepath):
                    raise FileNotFoundError(f"File not found: {filepath}")
                filepaths[filename[:-len(file_extension)]] = filepath
        if filepaths == {}:
            raise Exception("Could not find any files!")
        return filepaths