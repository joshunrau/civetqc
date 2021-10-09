from argparse import ArgumentParser
from civetqc import Subject

if __name__ == "__main__":

    parser = ArgumentParser(prog="civetqc")
    parser.add_argument("--csv_path", type=str, nargs=1, required=True, help="path to csv file")
    parser.add_argument("--prefix", type=str, nargs=1, required=True, help="file prefix")
    parser.add_argument("--id", type=str, nargs=1, required=True, help="subject id")
    args = parser.parse_args()
    
    test_subj = Subject(args.prefix[0], args.id[0])
    print(test_subj)
