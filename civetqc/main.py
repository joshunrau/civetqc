from __future__ import annotations

import argparse

from importlib.metadata import version
from pathlib import Path

from civetqc.data import CivetData, QCRatingsData
from civetqc.model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="civetqc")
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {version('civetqc')}"
    )
    parser.add_argument(
        "input_path", type=Path, help="path to file or directory with CIVET QC outputs"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=Model.get_default_threshold(),
        metavar="",
        help="probability above which a failure will be predicted (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        metavar="",
        help="directory for results (default: %(default)s)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        metavar="",
        help="format for output file: csv, json (default: %(default)s)",
    )
    return parser.parse_args()


def verify_args(args: argparse.Namespace) -> None:
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")
    elif not args.output_dir.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {args.output_dir}")
    if not 1 > args.threshold > 0:
        raise ValueError(
            f"Threshold must be greater than zero and less than one, got {args.threshold}"
        )


def load_civet_data(input_path: Path) -> CivetData:
    if input_path.is_file():
        return CivetData.from_csv(input_path)
    return CivetData.from_output_files(input_path)


def main() -> None:

    args = parse_args()
    verify_args(args)

    civet_data = load_civet_data(args.input_path)
    model = Model.load()

    predicted_ratings = model.predict(civet_data.features, threshold=args.threshold)
    probabilities = model.predict_probabilities(civet_data.features)

    qc_data = QCRatingsData(civet_data.subject_ids, predicted_ratings, probabilities)

    output_filepath = args.output_dir.joinpath(f"civetqc.{args.output_format}")
    if args.output_format == "csv":
        qc_data.to_csv(output_filepath)

    elif args.output_format == "json":
        qc_data.to_json(output_filepath)


if __name__ == "__main__":
    main()
