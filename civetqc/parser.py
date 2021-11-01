import argparse


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="civetqc")
    parser.add_argument("civet_output", type=str, help="Path to CSV file outputted by CIVET for QC")
    parser.add_argument("user_ratings", type=str, help="Path to CSV file containing user QC ratings")
    parser.add_argument("-b", metavar="int", type=int, help="Recode QC ratings to binary w/ acceptable cutoff int")
    parser.add_argument("-o", metavar="str", type=str, help="Path to output directory")
    return parser.parse_args(args)
