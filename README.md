# CivetQC

This repository is currently a work in progress. One major goal is to provide a command line tool to easily estimate quality control ratings for a large number of CIVET outputs by generalizing from a much smaller number of known examples. 

## Usage

    usage: civetqc [-h] [--output_dir OUTPUT_DIR] civet_output user_ratings

    positional arguments:
    civet_output          Path to CSV file outputted by CIVET for QC
    user_ratings          Path to CSV file containing user QC ratings

    optional arguments:
    -h, --help            show this help message and exit
    --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                            Path to directory where results will be outputted