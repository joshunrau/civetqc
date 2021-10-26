#!/bin/bash

source ../venv/bin/activate
echo $(python --version) && echo $(which python)

civet_output="/Users/joshua/Developer/civetqc/data/LAM/QC/civet_LAM_.csv"
user_ratings="/Users/joshua/Developer/civetqc/data/LAM/user_ratings.csv"
output_dir="/Users/joshua/Developer/civetqc/data/Tests"
civetqc "$civet_output" "$user_ratings" -o "$output_dir"

