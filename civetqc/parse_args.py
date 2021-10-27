import argparse

def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="civetqc",
        description='''
            This utility allows for semi-automated quality control of CIVET outputs. Based 
            on a set of known integer rankings, civetqc predicts the values for the remaining
            individuals. User-rated QC values must be contained in a CSV file (column “QC”), 
            with all individuals represented by a unique identifier (column “ID”). 
            ''')
    parser.add_argument("civet_output", type=str, nargs=1, help="Path to CSV file outputted by CIVET for QC")
    parser.add_argument("user_ratings", type=str, nargs=1, help="Path to CSV file containing user QC ratings")
    parser.add_argument("output_dir", type=str, nargs=1, help="Path to directory where results will be outputted")
    return parser.parse_args(args)