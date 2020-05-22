import torch
import ctgan
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pathlib

model = False

def SetModel(name: str = 'ctgan-diabetes-full-0'):
    global model
    model = torch.load(str(pathlib.Path(__file__).parent.absolute()) + '\\' + name)

def GetSyntheticData(amount: int):
    global model
    if(not model):
        modelName = input('Model is not currently set. To use default model, press y, to use a different model type the name.\n')
        if(modelName == 'y'):
            SetModel()
        else:
            SetModel(modelName)
        #raise Exception('Model not set. Use SetModel() to set a model')
    sample = model.sample(amount)
    for column in sample:
        sample[column] = pd.to_numeric(sample[column], errors = 'ignore')
        if(is_numeric_dtype(sample[column])):
            sample[column] = sample[column].abs()
    return sample
