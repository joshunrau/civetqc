import argparse
import os


class InvalidFileFormatError(Exception):
    """ raised when a user attempts to import a non-csv file """
    pass


class Arguments:

    def __init__(self, args) -> None:
        parser = argparse.ArgumentParser(prog="civetqc")
        parser.add_argument("civet_output", type=str, help="path to CSV file outputted by CIVET for QC")
        parser.add_argument("user_ratings", type=str, help="path to CSV file containing user QC ratings")
        parser.add_argument("--cutoff", metavar="<int>", type=int, default=1, help="cutoff value for acceptable QC")
        parser.add_argument("--output", metavar="<str>", type=str, default=os.getcwd(), help="path to output directory")
        parser.add_argument("--verbose", action="store_true", help="use verbose output")
        self.args = parser.parse_args(args)
        
        for filename in self.args.civet_output, self.args.user_ratings:
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist")
            if filename.split(".")[-1] != "csv":
                raise InvalidFileFormatError(f"File '{filename}' must be in csv format")

        if self.args.cutoff <= 0:
            raise ValueError("Cutoff value must be larger than zero")

        if not os.path.isdir(self.args.output):
            raise NotADirectoryError("Directory {self.args.output} does not exist")
