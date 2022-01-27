#!/bin/bash
#
# RECODE NUSDAST FILES THAT DO NOT CONTAIN "M" PREFIX

nusdat_dir="/Users/joshua/Developer/civetqc/data/NUSDAST/raw"
for filepath in ${nusdat_dir}/*; do
    filename=$(basename -- "$filepath")
    if [ ${filename:0:1} == "M" ]; then
        new_filepath="${nusdat_dir}/N${filename}"
        mv $filepath $new_filepath
    fi
done
