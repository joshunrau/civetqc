#!/usr/bin/env python
#
# RECODE TOPSY QC FILE TO INCLUDE LEADING ZEROS
# REMOVE UNUSED FILES

import os

qc_file = "/Users/joshua/Developer/civetqc/data/TOPSY/TOPSY_QC.csv"
data_dir = "/Users/joshua/Developer/civetqc/data/raw/TOPSY/raw"

with open(qc_file, "r") as f:
    contents = f.read().split("\n")

new_contents = [contents[0]]
for line in contents[1:]:
    line = line.split(",")
    while len(line[0]) != 3:
        line[0] = "0" + line[0]
    new_contents.append(",".join(line))
new_contents = "\n".join(new_contents)

with open(qc_file, "w") as f:
    f.write(new_contents)

for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    if "mp2rage" in filename or "t1" in filename:
        os.remove(filepath)
    


