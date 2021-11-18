# CivetQC

## About

Although the CIVET Cortical Surface Extraction Pipeline provides users with extensive data for quality control purposes, reviewing these data manually is time consuming and impractical when dealing with extremely large datasets. CivetQC is a fully automated quality control pipeline for CIVET outputs based on the random forest algorithm. Using data from our lab (N=1087), the algorithm was trained to classify CIVET outputs as either unacceptable (0) or acceptable (1). Although CivetQC is still in development, it can currently detect unacceptable outputs with an accuracy of approximately 85%.

## Installation

    pip install civetqc

## Usage

    usage: civetqc [-h] path_csv output_dir

    positional arguments:
    path_csv    path to csv file outputted by CIVET
    output_dir  path to directory where results should be outputted

    optional arguments:
    -h, --help  show this help message and exit


## Example

    civetqc /Users/joshua/Developer/civetqc/data/LAM/LAM_civet_data.csv /Users/joshua/Desktop
