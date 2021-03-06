# CIVETQC

## About

Although the CIVET Cortical Surface Extraction Pipeline provides users with extensive data for quality control purposes, reviewing these data manually is time consuming and impractical when dealing with extremely large datasets. CIVETQC is a fully automated quality control pipeline for CIVET outputs based on the scikit-learn. Using data from our lab (N=1215), the algorithm was trained to classify CIVET outputs as either acceptable (0) or unacceptable (1). CIVETQC is still in development, see below for a summary of model performance.

                    precision    recall  f1-score   support

               0        1.00      0.98      0.99       384
               1        0.70      0.94      0.80        17

        accuracy                            0.98       401
       macro avg        0.85      0.96      0.89       401
    weighted avg        0.98      0.98      0.98       401

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
