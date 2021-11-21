import os

from pkg_resources import resource_filename

from .. import __name__ as package_name


def get_resource_filepath(filepath: str):
    filepath = resource_filename(package_name, os.path.join('resources', filepath))
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Package resource in path '{filepath}' does not exist")
    return filepath


class Filepaths:

    simulated_datasets = {
        "DATASET": get_resource_filepath(os.path.join('datasets', 'simulated_dataset.csv')),
        "CIVET": get_resource_filepath(os.path.join('datasets', 'simulated_civet.csv')),
        "QC": get_resource_filepath(os.path.join('datasets', 'simulated_qc.csv'))
    }

    saved_model = get_resource_filepath(os.path.join('models', 'model.pkl'))
