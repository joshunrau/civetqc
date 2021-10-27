import argparse

def parse_args(args) -> argparse.Namespace:
    """ parse arguments given at command line """
    parser = argparse.ArgumentParser(prog="civetqc")
    parser.add_argument("civet_output", type=str, nargs=1, help="Path to CSV file outputted by CIVET for QC")
    parser.add_argument("user_ratings", type=str, nargs=1, help="Path to CSV file containing user QC ratings")
    parser.add_argument("--output_dir", "-o", type=str, required=False, nargs=1, help="Path to directory where results will be outputted")
    return parser.parse_args(args)