#!/bin/bash
#
# This script loads python and the scipy-stack module as well as
# loading a virtual environnment. Then, it runs a python file
# within this environment, e.g., ./run_py.sh main.py
#
# New packages can be installed with pip as normal after activating
# the virtual environment.
#
# source ENV/bin/activate
# pip install --no-index --upgrade pip
# pip install mypackage


module load python/3.9
module load scipy-stack
source ENV/bin/activate
python $1