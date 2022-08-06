# CivetQC

## About

### Background

The [CIVET Cortical Surface Extraction Pipeline](https://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET) provides users with extensive data for quality control (QC) purposes. However, manually reviewing these outputs can be time consuming and impractical when working with extremely large datasets. Previous research with other processing pipelines has demonstrated that supervised learning algorithms offer a feasible means for automated QC [(Klapwijk et al., 2019)](https://doi.org/10.1016/j.neuroimage.2019.01.014). Here, we present CivetQC, a fully automated QC pipeline for CIVET outputs based on scikit-learn.

### Methods
Data from five of our previous studies involving patients with psychotic disorders and healthy controls (N=1216) were processed using the CIVET pipeline. We rated the output quality for each subject on a scale from 0 to 2 (0 = fail, 1 = questionable, 2 = pass) based on visual inspection of CIVET outputs. Ratings of 1 or 2 were considered acceptable (n = 1163, 95.6%), whereas ratings of less than one were considered unacceptable (n = 53, 4.4%). We used the random forest algorithm to classify outputs as acceptable or unacceptable based on various quality metrics, including number of surface-surface intersections, self-intersections, and brain mask error, among others. Model training was performed using stratified fivefold cross validation. The optimal set of hyperparameters was selected based on the mean F2 score from among ten iterations of a randomized search of the hyperparameter space. Subsequently, the model was refit with the optimal set of hyperparameters on the entire training dataset. Finally, we determined the optimal discrimination threshold based on the F2 score. After finalizing the model, we evaluated its performance on a set of previously unseen data from the UK Biobank (N=120). 

### Results
The testing data included 98 scans of acceptable quality and 22 scans of unacceptable quality.
Overall model accuracy was 97%, with 0.99 precision and 0.98 recall for acceptable scans, and 0.91 precision and 0.95 recall for unacceptable scans. 

### Conclusion
These results demonstrate that comparably high recall and precision can be achieved for automated QC of CIVET outputs as has been demonstrated with FreeSurfer.

## Install

CivetQC is available via the Python Package Index (PyPI):

    pip install civetqc

## Usage

For most use cases, the preferred method of using CivetQC is through the command line interface. Users must provide an input path, which may be either a file or a directory.

If available, it is recommend to provide the file outputted by CIVET with aggregated tabular QC metrics (civet_{prefix}_.csv). However, if this file is not available, users may instead provide a path to a directory containing files of the format prefix_id_civet_qc.txt, in which case CivetQC will attempt to extract the relevant metrics for each subject. 

    usage: civetqc [-h] [-v] [--output_dir] [--output_filename] input_path

    positional arguments:
    input_path

    optional arguments:
    -h, --help          show this help message and exit
    -v, --version       show program's version number and exit
    --output_dir        directory where results should be outputted
    --output_filename   filename for results

