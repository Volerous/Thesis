#!/usr/bin/env python3
import subprocess
import os
# os.chdir(f"/homes/pmcd/Peter_Patrick3/run-data{i+1}")
for i in range(3):
    os.chdir(f"/homes/pmcd/Peter_Patrick3/run-data{i+1}")
    subprocess.call("mp train --overwrite", shell=True)
    subprocess.call("mp train_fusion --overwrite", shell=True)
    subprocess.call("mp predict --overwrite", shell=True)