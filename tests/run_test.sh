#!/bin/bash

source ../venv/bin/activate
echo $(python --version) && echo $(which python)

civet_output="/Users/joshua/Developer/civetpy/data/LAM/QC/civet_LAM_.csv"
user_ratings="/Users/joshua/Developer/civetpy/data/LAM/user_ratings.csv"

civetqc "$civet_output" "$user_ratings"

