# CivetQC

This utility allows for semi-automated quality control of CIVET outputs. Based on a set of known integer rankings, civetqc predicts the values for the remaining individuals. User-rated QC values must be contained in a CSV file (column “QC”), with all individuals represented by a unique identifier (column “ID”).
## Usage

    usage: civetqc [-h] civet_output user_ratings output_dir

    positional arguments:
    civet_output  Path to CSV file outputted by CIVET for QC
    user_ratings  Path to CSV file containing user QC ratings
    output_dir    Path to directory where results will be outputted

    optional arguments:
    -h, --help    show this help message and exit
