import sqlite3
import diffprivlib as dpl

import torch
import ctgan
import pandas as pd
import re
from pandas.api.types import is_numeric_dtype
import importlib

import importlib

from enum import Enum

conn = sqlite3.connect("database")

EPSILON = 0.01


class Mode(Enum):
    UNKNOWN = 0
    AVG = 'AVG'
    SUM = 'SUM'


class Dataset(Enum):
    UNKNOWN = 0
    DIAB = 'diabetes'
    KIDNEY = 'kidney'
    BREAST = 'breast'

class ResultSet():
    trueAnswer = False
    sampleAnswer = False
    unfilteredAnswer = False
    sampleUnfilteredAnswer = False

    def __init__(self):
        return

    def FillNext(self, value):
        if(not self.trueAnswer):
            self.trueAnswer = value
            return
        if(not self.sampleAnswer):
            self.sampleAnswer = value
            return
        if(not self.unfilteredAnswer):
            self.unfilteredAnswer = value
            return
        if(not self.sampleUnfilteredAnswer):
            self.sampleUnfilteredAnswer = value
            return
        print('All values have been submitted')
        return



query = input("Query:\n")

queryAnalysis = query.split(' ')

mode = Mode.UNKNOWN
dataset = Dataset.UNKNOWN


for allowedDataset in Dataset:
    if(allowedDataset.value in queryAnalysis):
        dataset = allowedDataset
        break
if(dataset == Dataset.UNKNOWN):
    raise Exception('Dataset unknown')

model = importlib.import_module(f"models.{dataset.value}.run")

sample = model.GetSyntheticData(100000)
realData = pd.read_sql(f"SELECT * FROM {dataset.value}", conn)

regAvg = re.compile('AVG(.*)', re.IGNORECASE)
regSum = re.compile('SUM(.*)', re.IGNORECASE)

matches = []

for allowedFunction in Mode:
    regex = re.compile(f"{allowedFunction.value}(.*)", re.IGNORECASE)
    match = list(filter(regex.match, queryAnalysis))
    if(len(match) == 1):
        mode = allowedFunction
    matches = matches + match

where = re.search('WHERE(.*)', query, re.IGNORECASE)
sampleFilter = sample.copy()
realDataFilter = realData.copy()
try:
    whereVals = re.findall(
        ' (.*?)(?: )*=(?: )?(?:\'|")(.+?)(?:\'|")(?: |$)', where.group(1), re.IGNORECASE)
    for((column, value)) in whereVals:
        sampleFilter = sampleFilter[sampleFilter[column] == value]
        realDataFilter = realDataFilter[realDataFilter[column] == value]
except:
    print('No where clause found')

if(len(matches) == 1):
    column = re.search('(\(.*\))', matches[0], re.IGNORECASE).group(1)[1:-1]
    sampleColumnDataFilter = pd.to_numeric(
        sampleFilter[column], errors='coerce')
    realColumnDataFilter = pd.to_numeric(
        realDataFilter[column], errors='coerce')
    sampleColumnData = pd.to_numeric(sample[column], errors='coerce')
    realColumnData = pd.to_numeric(realData[column], errors='coerce')
    allData = [realColumnDataFilter, sampleColumnDataFilter, realColumnData, sampleColumnData]
    results = ResultSet()
    if(len(listAvg) == 1):
        mode = Mode.AVG
        list(map(lambda d: results.FillNext(d.mean()), allData))
    elif(len(listSum) == 1):
        mode = Mode.SUM
        list(map(lambda d: results.FillNext(d.sum()), allData))
    else:
        raise Exception('Unsupported query')
else:
    raise Exception('Unsupported query')

conn.row_factory = lambda cursor, row: row[0]
cur = conn.cursor()

maxSample = sampleColumnData.max()
maxData = realColumnData.max()
if(mode == Mode.AVG):
    maxSample = maxSample/(len(sampleColumnData) - 1)
    maxData = maxData/(len(realColumnData) - 1)
maxValues = [maxSample, maxData]
print(f"Data max change: {maxData}")
print(f"Sample max change: {maxSample}")
print(f"Max change: {max(maxValues)}")
#data = cur.execute(query).fetchone()
# print(data)

print(f"Sample result with filters: {results.sampleAnswer}")
print(f"True result with filters: {results.trueAnswer}")
print(f"Sample result without filters: {results.sampleUnfilteredAnswer}")
print(f"True result without filters: {results.unfilteredAnswer}")

utility = [(str(maxSample), str(maxData), max(maxSample-maxData, 0.001))]
exp = dpl.mechanisms.Exponential()
exp.set_utility(utility)
exp.set_epsilon(EPSILON)

sensitivity = float(exp.randomise(str(max(maxValues))))
print(f"Sensitivity: {sensitivity}")
dp = dpl.mechanisms.Laplace()
dp.set_epsilon(EPSILON)
dp.set_sensitivity(sensitivity)

result = dp.randomise(results.trueAnswer)

print(f'Private result: {result}')
print(f'Private result error: {abs(result - results.trueAnswer)}')
