#!/bin/bash
# 
# QC DATA
# LAM: See Google Sheets
# FEP, INSIGHT, TOPSY, NUSDAST: See emails from Nov 8

# CIVET DATA
autossh cic -d "/data/lepage/LAM/data/processed/civet/2.1.1/*/verify/*" "/Users/joshua/Developer/civetqc/data/LAM/raw"
autossh cic -d "/data/lepage/FEP/data/processed/civet/2.1.1/*/verify/*" "/Users/joshua/Developer/civetqc/data/FEP/raw"
autossh cic -d "/data/lepage/INSIGHT/data/processed/civet/2.1.1/*/verify/*" "/Users/joshua/Developer/civetqc/data/INSIGHT/raw"
autossh cic -d "/data/lepage/TOPSY/data/processed/civet/April2020/*/*/CIVET-2.1.0/*/verify/*" "/Users/joshua/Developer/civetqc/data/TOPSY/raw"
autossh cic -d "/data/lepage/NUSDAST/data/processed/civet2.1.0_bpipe_Niagara/civet/output/*/verify/*" "/Users/joshua/Developer/civetqc/data/NUSDAST/raw"