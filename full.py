import sqlite3
import diffprivlib as dpl

import torch
import ctgan
import pandas as pd
import re
from pandas.api.types import is_numeric_dtype
import importlib

from enum import Enum

# Connection to main database
conn = sqlite3.connect("database")

# Epsilon value for differential privacy
EPSILON = 0.01

# Enum to select the type of query being done
class Mode(Enum):
    UNKNOWN = 0
    AVG = 'AVG'
    SUM = 'SUM'

# Enum with all datasets and their names, used to select proper ctgan model and sql table
class Dataset(Enum):
    UNKNOWN = '?'
    DIAB = 'diabetes'
    KIDNEY = 'kidney'
    BREAST = 'breast'

# Class that contains all result values from different datasets
class ResultSet():
    trueAnswer = False
    sampleAnswer = False
    unfilteredAnswer = False
    sampleUnfilteredAnswer = False

    # Fill member variables in order
    def FillNext(self, value):
        if(not self.trueAnswer):
            self.trueAnswer = value
        elif(not self.sampleAnswer):
            self.sampleAnswer = value
        elif(not self.unfilteredAnswer):
            self.unfilteredAnswer = value
        elif(not self.sampleUnfilteredAnswer):
            self.sampleUnfilteredAnswer = value
        else:
            print('All values have been submitted')

# Ask the user to submit a query
query = input("Query:\n")

# Initialise mode and data variables
mode = Mode.UNKNOWN
dataset = Dataset.UNKNOWN


# Loop over known datasets and check which one is selected
for allowedDataset in Dataset:
    if(allowedDataset.value in query):
        dataset = allowedDataset
        break

# If no known dataset selected, raise an exception
if(dataset == Dataset.UNKNOWN):
    raise Exception('Dataset unknown')

# Import the model for the relevant dataset and create synthetic data
model = importlib.import_module(f"models.{dataset.value}.run")
sample = model.GetSyntheticData(100000)

# Read the real data from the table into a dataframe
realData = pd.read_sql(f"SELECT * FROM {dataset.value}", conn)

# Find the function(s) used in the query by looping over all allowed functions
matches = []
for allowedFunction in Mode:
    regex = re.compile(f"{allowedFunction.value}\((.*)\)", re.IGNORECASE)
    match = regex.match(query)
    if(match is not None):
        match = [match.group(1)]
        mode = allowedFunction
        matches = matches + match

# Find the where condition
where = re.search('WHERE(.*)', query, re.IGNORECASE)
sampleFilter = sample.copy()
realDataFilter = realData.copy()
# Try/catch as there might be no where condition
try:
    # Find all key = value combinations in the where (e.g. age = 60)
    whereVals = re.findall(
        ' (.*?)(?: )*=(?: )?(?:\'|")(.+?)(?:\'|")(?: |$)', where.group(1), re.IGNORECASE)
    # Filter the datasets using the column and value found
    for((column, value)) in whereVals:
        sampleFilter = sampleFilter[sampleFilter[column] == value]
        realDataFilter = realDataFilter[realDataFilter[column] == value]
except:
    print('No where clause found')

# Currently we only support queries with one function, here we check if there is only one
if(len(matches) == 1):
    # Set the column name we want to execute the query on
    column = matches[0]
    # Compile all different datasources in one list
    allData = [realDataFilter, sampleFilter, realData, sample]
    # Take the relevant column from all datasources and make it numeric, with non numeric values
    # changing to NaN
    allData = list(map(lambda d: pd.to_numeric(d[column], errors='coerce'), allData))
    # Create a new ResultSet to store all results in
    results = ResultSet()
    # Depending on the type of query we want to execute a different function on each data source
    if(mode == Mode.AVG):
        def func(d): return d.mean()
    elif(mode == Mode.SUM):
        def func(d): return d.sum()
    else:
        raise Exception('Unsupported query')
else:
    raise Exception('Unsupported query')
# Execute the correct function on each data source and put in results
list(map(lambda d: results.FillNext(func(d)), allData))

# Find the maximum values in the unfiltered sample and real data (this is the sensitivity of a sum query)
maxSample = allData[3].max()
maxData = allData[2].max()
# If the mode is average we divide these values by the sum of all columns - 1 to get the sensitivity
# both get divided by the size of real data, as this is the relevant sensitivity
if(mode == Mode.AVG):
    maxSample = maxSample/(len(allData[2]) - 1)
    maxData = maxData/(len(allData[2]) - 1)
# Compile into list for easy comparing
maxValues = [maxSample, maxData]
print(f"Data max change: {maxData}")
print(f"Sample max change: {maxSample}")
print(f"Max change: {max(maxValues)}")


print(f"Sample result with filters: {results.sampleAnswer}")
print(f"True result with filters: {results.trueAnswer}")
print(f"Sample result without filters: {results.sampleUnfilteredAnswer}")
print(f"True result without filters: {results.unfilteredAnswer}")

# TODO: Exponential mechanism utility (this probably isn't correct)
# Exponential mechanism to select a sensitivity in a differentially private way
utility = [(str(maxSample), str(maxData), max(maxSample-maxData, 0.001))]
exp = dpl.mechanisms.Exponential()
exp.set_utility(utility)
exp.set_epsilon(EPSILON)
sensitivity = float(exp.randomise(str(max(maxValues))))
print(f"Sensitivity: {sensitivity}")

# Execute differential privacy on the actual answer to get a differentially private answer
dp = dpl.mechanisms.Laplace()
dp.set_epsilon(EPSILON)
dp.set_sensitivity(sensitivity)

result = dp.randomise(results.trueAnswer)

print(f'Private result: {result}')
print(f'Private result error: {abs(result - results.trueAnswer)}')
