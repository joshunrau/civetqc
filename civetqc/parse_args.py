import argparse

def parse_args(args) -> argparse.Namespace:
    """ parse arguments given at command line """
    parser = argparse.ArgumentParser(prog="civetqc")
    parser.add_argument("civet_output", type=str, nargs=1, 
                        help="The path to the quality control csv file outputed by CIVET")
    parser.add_argument("user_ratings", type=str, nargs=1, 
                        help="The path to the csv file containing user quality control ratings")
    parser.add_argument("--output_dir", "-o", type=str, required=False, nargs=1, 
                        help="The path to directory where files should be outputted")
    return parser.parse_args(args)
    