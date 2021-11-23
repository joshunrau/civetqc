import sys

from .helpers import process_arguments, load_saved_model
from ..data.base import CIVETData


def main():
    path_csv, output_dir = process_arguments(sys.argv[1:])
    saved_model = load_saved_model()
    user_data = CIVETData(path_csv)
    user_data.predict_qc(saved_model)
    user_data.write_csv(output_dir)
