import torch
import ctgan
import pandas as pd
import pathlib

model = torch.load(str(pathlib.Path(__file__).parent.absolute()) + '\ctgan-kidney')

def GetSyntheticData(amount: int):
    sample = model.sample(amount)
    return sample
