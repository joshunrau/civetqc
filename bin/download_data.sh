#!/bin/bash

cic_server -d /data/lepage/LAM/data/processed/civet/2.1.1/QC/civet_LAM_.csv
cic_server -d /data/lepage/FEP/data/processed/civet/2.1.1/QC/civet_FEP_.csv
cic_server -d /data/lepage/FEP/info/FinalFEP_QC_2019.xlsx # <-- will send newer version
cic_server -d /data/lepage/INSIGHT/data/processed/civet/2.1.1/QC/civet_Insight_.csv
cic_server -d /data/lepage/INSIGHT/info/Final_Insight_QC_2019.xlsx # <-- will check
# 001/V1/CIVET-2.1.0/001_V1_gradient_n4_anlm0.5r/verify/001_V1_gradient_n4_anlm0.5r_surface_qc.txt 
# SubjectID/Timepoint/CIVETversion/civetprocessingparameter/filename
cic_server -d "/data/lepage/TOPSY/data/processed/civet/April2020/*/*/CIVET-2.1.0/*/verify/*.txt"
# Katie will look for NUSDAST

cic_server -d /data/lepage/TOPSY/data/processed/civet/April2020/*.txt

find /data/lepage/TOPSY/data/processed/civet/April2020/ -name *.txt