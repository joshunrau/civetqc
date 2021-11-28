#!/bin/bash
# 
# CIVET DATA
# TOPSY: CIVET output lost, must extract and merge txt files from CIC server, see utils.txt2csv
#   /data/lepage/TOPSY/data/processed/civet/April2020/*/*/CIVET-2.1.0/*/verify/*.txt
#
# QC DATA
# LAM: See Google Sheets
# FEP, INSIGHT, TOPSY: See emails from Nov 8

local_data_dir="/Users/joshua/Developer/civetqc/data"
declare -a study_names=("LAM" "FEP" "INSIGHT")

for study in "${study_names[@]}"; do

   local_study_dir="${local_data_dir}/${study}"
   if [[ -d $local_study_dir ]]; then
      continue
   else
      mkdir $local_study_dir
   fi

   cic_qc_dir="/data/lepage/${study}/data/processed/civet/2.1.1/QC/"

   autossh cic -d $cic_qc_dir $local_study_dir

done

