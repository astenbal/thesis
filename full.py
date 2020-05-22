import sqlite3
import diffprivlib as dpl

import torch
import ctgan
import pandas as pd
import re
from pandas.api.types import is_numeric_dtype

import importlib

from enum import Enum

conn = sqlite3.connect("database")


class Mode(Enum):
    UNKNOWN = 0
    AVG = 1
    SUM = 2

class Dataset(Enum):
    UNKNOWN = 0
    DIAB = 'diabetes'
    KIDNEY = 'kidney'
    BREAST = 'breast'

query = input("Query:\n")

queryAnalysis = query.split(' ')

mode = Mode.UNKNOWN
dataset = Dataset.UNKNOWN


if('diabetes' in queryAnalysis):
    import models.diabetes.run as model
    dataset = Dataset.DIAB
elif('kidney' in queryAnalysis):
    import models.kidney.run as model
    dataset = Dataset.KIDNEY
elif('breast' in queryAnalysis):
    import models.breast.run as model
    dataset = Dataset.BREAST
else:
    raise Exception('Dataset unknown')

sample = model.GetSyntheticData(100000)
realData = pd.read_sql(f"SELECT * FROM {dataset.value}", conn)

regAvg = re.compile('AVG(.*)', re.IGNORECASE)
regSum = re.compile('SUM(.*)', re.IGNORECASE)

listAvg = list(filter(regAvg.match, queryAnalysis))
listSum = list(filter(regSum.match, queryAnalysis))
matches = listAvg + listSum

where = re.search('WHERE(.*)', query, re.IGNORECASE)
whereVals = re.findall(' (.*?)(?: )*=(?: )?(?:\'|")(.+?)(?:\'|")(?: |$)', where.group(1), re.IGNORECASE)
sampleFilter = sample.copy()
realDataFilter = realData.copy()
for((column, value)) in whereVals:
    sampleFilter = sampleFilter[sampleFilter[column] == value]
    realDataFilter = realDataFilter[realDataFilter[column] == value]

if(len(matches) == 1):
    column = re.search('(\(.*\))', matches[0], re.IGNORECASE).group(1)[1:-1]
    sampleColumnDataFilter = pd.to_numeric(sampleFilter[column], errors = 'coerce')
    realColumnDataFilter = pd.to_numeric(realDataFilter[column], errors = 'coerce')
    sampleColumnData = pd.to_numeric(sample[column], errors = 'coerce')
    realColumnData = pd.to_numeric(realData[column], errors = 'coerce')
    if(len(listAvg) == 1):
        mode = Mode.AVG
        resultSampleDataFilter = sampleColumnDataFilter.mean()
        resultRealDataFilter = realColumnDataFilter.mean()
        resultSampleData = sampleColumnData.mean()
        resultRealData = realColumnData.mean()
    elif(len(listSum) == 1):
        mode = Mode.SUM
        resultSampleDataFilter = sampleColumnDataFilter.sum()
        resultRealDataFilter = realColumnDataFilter.sum()
        resultSampleData = sampleColumnData.sum()
        resultRealData = realColumnData.sum()
    else:
        raise Exception('Unsupported query')
else:
    raise Exception('Unsupported query')

conn.row_factory = lambda cursor, row: row[0]
cur = conn.cursor()

maxSample = sampleColumnDataFilter.max()
maxData = realColumnDataFilter.max()
print(f"Data max value: {maxData}")
print(f"Sample max value: {maxSample}")
#data = cur.execute(query).fetchone()
#print(data)

print(f"Sample result with filters: {resultSampleDataFilter}")
print(f"True result with filters: {resultRealDataFilter}")
print(f"Sample result without filters: {resultSampleData}")
print(f"True result without filters: {resultRealData}")