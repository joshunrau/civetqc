import argparse
from os import getcwd

def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="civetqc")
    parser.add_argument("civet_output", type=str, help="path to CSV file outputted by CIVET for QC")
    parser.add_argument("user_ratings", type=str, help="path to CSV file containing user QC ratings")
    parser.add_argument("--cutoff", metavar="<int>", type=int, default=False, help="cutoff value for acceptable QC")
    parser.add_argument("--output", metavar="<str>", type=str, default=getcwd(), help="path to output directory")
    return parser.parse_args(args)
