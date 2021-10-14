from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser(prog="civetpy")
    parser.add_argument("--csv_path", type=str, nargs=1, required=True, help="path to csv file")
    parser.add_argument("--prefix", type=str, nargs=1, required=True, help="file prefix")
    parser.add_argument("--id", type=str, nargs=1, required=True, help="subject id")
    args = parser.parse_args()
