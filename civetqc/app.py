from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .data import CIVETData
from .model import Model

class App:

  name = 'civetqc'
  version = version(name)

  default_output_filename = 'civetqc.csv'

  @classmethod
  def main(cls) -> None:

    parser = ArgumentParser(prog=cls.name, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {cls.version}")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output_dir", default=Path.cwd(), help="directory where results should be outputted", type=Path, metavar='')
    parser.add_argument("--output_filename", default=cls.default_output_filename, help="filename for results", type=Path, metavar='')
    args = parser.parse_args()
    
    if not args.input_path.exists():
      raise FileNotFoundError(f"Input path does not exist: {args.input_path}")
    if not args.output_dir.is_dir():
      raise NotADirectoryError(f"Output directory does not exist: {args.output_dir}")
    
    if args.input_path.is_dir():
      civet_data = CIVETData.from_output_files(args.input_path)
    elif args.input_path.is_file():
      civet_data = CIVETData.from_csv(args.input_path)
    else:
      raise AssertionError("Specified input path exists, but is neither a file or directory")
    
    model = Model.load()

    df = pd.DataFrame({
      "ID":  civet_data.df[civet_data.id_var],
      "CIVETQC_RESULT": model.predict(civet_data.features, labels={0: "PASS", 1: "FAIL"})
    })
    
    df.to_csv(args.output_dir.joinpath(args.output_filename), index=False)