import numpy as np

import pandas as pd
from os import listdir
from os.path import isfile, isdir, join

import sys
mypath = "./raw_data/Questionnaire"
sleep_path = "./raw_data/Questionnaire/SLQ_I.XPT"

# find all XPT in Questionnaire
files = listdir(mypath)
files_xpt = []
for f in files:
    if f[-3:]=="XPT":
        full_path = join(mypath+"/"+f)
        files_xpt.append(full_path)

# read label
sleep_data = pd.read_sas(sleep_path)

# merge all data
print("Start Combining data ... ...\n")
for f in files_xpt:
    # ignore RXQ_RX_I.XPT : weird
    if f == mypath+'/RXQ_RX_I.XPT' or f == sleep_path:
        continue

    data2 = pd.read_sas(f)
    print(f,data2.shape)

    # merge by "SEQN" of sleep data
    sleep_data = pd.merge(sleep_data,data2,how='left',on='SEQN')

print(sleep_data.head())
print("data.shape",sleep_data.shape)
sleep_data.to_csv(mypath+str(".csv"))
