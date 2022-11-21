# CivetQC

![python-version](https://img.shields.io/pypi/pyversions/civetqc)
![pypi-version](https://img.shields.io/pypi/v/civetqc)
![license](https://img.shields.io/pypi/l/civetqc?color=green)
![tests](https://github.com/joshunrau/civetqc/actions/workflows/main.yml/badge.svg)

## About

CivetQC is a fully automated pipeline for quality control of [CIVET](https://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET) outputs. Using the random forest algorithm implemented in scikit-learn, CivetQC categorises outputs as acceptable or unacceptable based on a variety of quality metrics, including number of surface-surface intersections, self-intersections, and brain mask error, among others. The model was developed using 1216 T1-weighted scans from several studies affiliated with McGill University in Montreal, Canada. A separate sample of 120 individuals from the UK Biobank served to evaluate final model performance. 

We rated the output quality for each subject on a scale from 0 to 2 (0 = fail, 1 = questionable, 2 = pass) based on visual inspection of CIVET outputs. Ratings of 1 or 2 were considered acceptable, whereas ratings of less than one were considered unacceptable. The model was trained using stratified fivefold cross validation, and the optimal set of parameters were chosen based on the mean ROC-AUC score from fifty iterations of a randomised search of the hyperparameter space. Finally, we selected the optimum discrimination threshold in terms of maximising the F2 score.

The training data contained 1163 acceptable scans (95.6%) and 53 unacceptable scans (4.4%). During cross-validation, the mean AUC score for the best model was 0.91. As depicted below, the optimal F2 score was achieved with a discrimination threshold of 0.2, which yielded 1.00 precision and 0.98 recall for acceptable scans, and 0.71 precision and 0.98 recall for unacceptable scans. 

![Discrimination Thresholds](https://github.com/joshunrau/civetqc/blob/main/figures/thresholds_rfc.jpeg)

The testing data, on the other hand, included 98 scans of acceptable quality and 22 scans of unacceptable quality. The mean AUC score was 0.98, with 1.0 precision and 0.97 recall for acceptable scans, and 0.88 precision and 1.0 recall for unacceptable scans.

## Install

CivetQC is available via the Python Package Index (PyPI):

    pip install civetqc

## Usage

In most cases, the preferred method of using CivetQC is through the command line interface. Users must provide an input path, which may be either a file or a directory. If available, it is recommend to provide the file outputted by CIVET with aggregated tabular QC metrics. However, if this file is not available, users may instead provide a path to a directory containing files of the format prefix_id_civet_qc.txt, in which case CivetQC will attempt to extract the relevant metrics for each subject. 

    positional arguments:
    input_path        path to file or directory with CIVET QC outputs

    optional arguments:
    -h, --help        show this help message and exit
    -v, --version     show program's version number and exit
    --threshold       probability above which a failure will be predicted (default: 0.2)
    --output_dir      directory for results (default: $PWD)
    --output_format   format for output file: csv, json (default: csv)
