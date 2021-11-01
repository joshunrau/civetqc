# CivetQC

This utility allows for semi-automated quality control of CIVET outputs. Based on a set of known integer rankings, civetqc predicts the values for the remaining individuals. User-rated QC values must be contained in a CSV file (column “QC”), with all individuals represented by a unique identifier (column “ID”).

## Usage

    usage: civetqc [-h] [-b <int>] [-o <str>] civet_output user_ratings

    positional arguments:
    civet_output  path to CSV file outputted by CIVET for QC
    user_ratings  path to CSV file containing user QC ratings

    optional arguments:
    -h, --help    show this help message and exit
    -b <int>      recode QC ratings with binary cutoff
    -o <str>      path to output directory