# CivetQC

This utility will allow for semi-automated quality control of CIVET outputs. It is a work in progress.

## Install

    $ pip install .

## Usage

    >>> import civetqc.dataset as qc
    >>> sample_study1 = qc.StudyData("path/to/civet_output1.csv", "path/to/qc_ratings1.csv")
    >>> sample_study2 = qc.StudyData("path/to/civet_output2.csv", "path/to/qc_ratings2.csv")
    >>> sample_dataset = qc.Dataset(sample_study1, sample_study2)
    >>> help(sample_dataset)
    
