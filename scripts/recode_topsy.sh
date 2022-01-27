#!/usr/bin/env python
#
# RECODE TOPSY QC FILE TO INCLUDE LEADING ZEROS

qc_file = "/Users/joshua/Developer/civetqc/data/TOPSY/TOPSY_QC.csv"

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


