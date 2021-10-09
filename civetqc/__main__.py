from argparse import ArgumentParser
from civetqc import Subject

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prefix", type=str, nargs=1, required=True, help="file prefix")
    parser.add_argument("--id", type=str, nargs=1, required=True, help="subject id")
    args = parser.parse_args()
    print(args)