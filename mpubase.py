#!/usr/bin/env python3
import subprocess
import os
for i in range(3):
    os.chdir(f"/homes/pmcd/Peter_Patrick3/run{i+1}")
    subprocess.call("mp train --overwrite", shell=True)
    subprocess.call("mp train_fusion --overwrite", shell=True)
    subprocess.call("mp predict --overwrite", shell=True)
for i in range(3):
    os.chdir(f"/homes/pmcd/Peter_Patrick3/noaug-run{i+1}")
    subprocess.call("mp train --overwrite", shell=True)
    subprocess.call("mp train_fusion --overwrite", shell=True)
    subprocess.call("mp predict --overwrite", shell=True)