import torch
import ctgan
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pathlib

model = torch.load(str(pathlib.Path(__file__).parent.absolute()) + '\ctgan-breast')

def GetSyntheticData(amount: int):
    sample = model.sample(amount)
    for column in sample:
        sample[column] = pd.to_numeric(sample[column], errors = 'ignore')
        if(is_numeric_dtype(sample[column])):
            sample[column] = sample[column].abs()
    return sample
