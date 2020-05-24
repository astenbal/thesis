import sqlite3
import diffprivlib as dpl

import torch
import ctgan
import pandas as pd
import re
import importlib

import helpers

# Connection to main database
conn = sqlite3.connect("database")

# Epsilon value for differential privacy
EPSILON = 0.01

# Ask the user to submit a query
query = input("Query:\n")

# Initialise mode and data variables
mode = helpers.Mode.UNKNOWN
dataset = helpers.Dataset.UNKNOWN


# Loop over known datasets and check which one is selected
for allowedDataset in helpers.Dataset:
    if(allowedDataset.value in query):
        dataset = allowedDataset
        break

# If no known dataset selected, raise an exception
if(dataset == helpers.Dataset.UNKNOWN):
    raise Exception('Dataset unknown')

# Import the model for the relevant dataset and create synthetic data
model = importlib.import_module(f"models.{dataset.value}.run")
sample = model.GetSyntheticData(100000)

# Read the real data from the table into a dataframe
realData = pd.read_sql(f"SELECT * FROM {dataset.value}", conn)

# Find the function(s) used in the query by looping over all allowed functions
matches = []
for allowedFunction in helpers.Mode:
    regex = re.compile(f"{allowedFunction.value['name']}\((.*)\)", re.IGNORECASE)
    match = regex.match(query)
    if(match is not None):
        match = [match.group(1)]
        mode = allowedFunction
        matches = matches + match

# Currently we only support queries with one function, here we check if there is only one
if(len(matches) != 1):
    raise Exception('Unsupported query')

# Find the where condition
where = re.search('WHERE(.*)', query, re.IGNORECASE)

# Create variables for filtering
sampleFilter = sample
realDataFilter = realData
# Check if there is a where condition
if(where is not None):
    functionNames = ''
    for allowedFunction in helpers.ComparisonType:
        if(functionNames != ''):
            functionNames += '|'
        functionNames += allowedFunction.value['regex']
    # Find all key = value combinations in the where (e.g. age = 60)
    whereVals = re.findall(
        f" (\w*?) *({functionNames}) *(?:'|\")?(\w+?)(?:'|\")?(?: |$)", where.group(1), re.IGNORECASE)
    # Filter the datasets using the column and value found
    for((column, funcType, value)) in whereVals:
        func = helpers.ComparisonType.UNKNOWN
        for allowedFunction in helpers.ComparisonType:
            if(allowedFunction.value['name'] == funcType):
                func = allowedFunction
                break
        if(func == helpers.ComparisonType.UNKNOWN):
            print(f"Unsupported comparison {funcType} on {column}")
        sampleFilter = sampleFilter[func.value['func'](sampleFilter[column], value)]
        realDataFilter = realDataFilter[func.value['func'](realDataFilter[column], value)]
else:
    print('No where clause found')

# Set the column name we want to execute the query on
column = matches[0]
# Compile all different datasources in one list
allData = [realDataFilter, sampleFilter, realData, sample]
# Take the relevant column from all datasources and make it numeric, with non numeric values
# changing to NaN
allData = list(map(lambda d: pd.to_numeric(d[column], errors='coerce'), allData))
# Create a new ResultSet to store all results in
results = helpers.ResultSet()
# Execute the correct function on each data source and put in results
list(map(lambda d: results.FillNext(mode.value['func'](d)), allData))

# Find the maximum values in the unfiltered sample and real data (this is the sensitivity of a sum query)
maxSample = allData[3].max()
maxData = allData[2].max()
# If the mode is average we divide these values by the sum of all columns - 1 to get the sensitivity
# both get divided by the size of real data, as this is the relevant sensitivity
if(mode == helpers.Mode.AVG):
    maxSample = maxSample/(len(allData[2]) - 1)
    maxData = maxData/(len(allData[2]) - 1)
# Compile into list for easy comparing
maxValues = [maxSample, maxData]
print(f"Query: {mode.value['name']}({column})")
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
