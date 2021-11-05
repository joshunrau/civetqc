#!/bin/bash
#
# Note that for the TOPSY data the file format, following 'April2020', refers to
# SubjectID/Timepoint/CIVETversion/civetprocessingparameter/filename
#
# Nov 5, Katie will look for NUSDAST
cic_server -d /data/lepage/LAM/data/processed/civet/2.1.1/QC/civet_LAM_.csv
cic_server -d /data/lepage/FEP/data/processed/civet/2.1.1/QC/civet_FEP_.csv
cic_server -d /data/lepage/FEP/info/FinalFEP_QC_2019.xlsx # <-- will send newer version
cic_server -d /data/lepage/INSIGHT/data/processed/civet/2.1.1/QC/civet_Insight_.csv
cic_server -d /data/lepage/INSIGHT/info/Final_Insight_QC_2019.xlsx # <-- will check
cic_server -d "/data/lepage/TOPSY/data/processed/civet/April2020/*/*/CIVET-2.1.0/*/verify/*.txt"
