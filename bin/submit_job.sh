#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-mlepage
#SBATCH --mem-per-cpu=16G
source ../venv/bin/activate
python select_data.py