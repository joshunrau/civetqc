import argparse
from pathlib import Path
from .resources import config

class App:

    name = config['app']['name']
    description = config['app']['description']
    version = config['app']['version']

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            prog = self.name,
            description = self.description,
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
        )
        self.parser.add_argument(
            "-v", "--version", 
            action="version",
            version=f"%(prog)s {self.version}"
        )
        self.parser.add_argument(
            "filepath", 
            help="path to csv file outputted by CIVET",
            type=Path
        )
        self.parser.add_argument(
            "-o", "--output_dir",
            default=Path.cwd(),
            help="directory where results should be outputted",
            type=Path,
            metavar=''
        )
        self._args = None
    
    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value: argparse.Namespace):
        if not value.filepath.is_file():
            raise FileNotFoundError(f"File not found: {value.filepath}")
        elif not value.output_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {value.output_dir}")
    
    def parse_args(self, argv: list):
        self.args = self.parser.parse_args(argv)