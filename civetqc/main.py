from __future__ import annotations

import argparse

from importlib.metadata import version
from pathlib import Path

from civetqc.data import CivetData, QCRatingsData
from civetqc.model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='civetqc')
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version('civetqc')}")
    parser.add_argument("input_path", help='path to file or directory with CIVET QC outputs', type=Path)
    parser.add_argument("--output_dir", default=Path.cwd(), help='default: %(default)s', type=Path, metavar='')
    parser.add_argument("--output_format", metavar='', default='csv', choices=['csv', 'json'],
                        type=str, help='options: csv (default), json')
    return parser.parse_args()


def verify_args(args: argparse.Namespace) -> None:
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")
    elif not args.output_dir.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {args.output_dir}")


def load_civet_data(input_path: Path) -> CivetData:
    if input_path.is_file():
        return CivetData.from_csv(input_path)
    return CivetData.from_output_files(input_path)


def main() -> None:
    args = parse_args()
    verify_args(args)

    civet_data = load_civet_data(args.input_path)
    model = Model.load()
    predicted_qc = model.predict(civet_data.features, labels={0: "PASS", 1: "FAIL"})
    qc_ratings = QCRatingsData(civet_data.subject_ids, predicted_qc)

    output_filepath = args.output_dir.joinpath(f"civetqc.{args.output_format}")
    if args.output_format == 'csv':
        qc_ratings.to_csv(output_filepath)
    elif args.output_format == 'json':
        qc_ratings.to_json(output_filepath)


if __name__ == '__main__':
    main()
