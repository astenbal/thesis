from ctgan import CTGANSynthesizer
import pandas as pd

import pandas_profiling

import torch

data_kidney = pd.read_csv("../../data/kidney/raw.csv")

good_data_kidney = data_kidney.dropna()
#good_data_kidney.profile_report().to_file("data_kidney.html")

#print(data_kidney)

#print(data_kidney.columns.tolist())

ctgan = CTGANSynthesizer()
ctgan.fit(good_data_kidney, data_kidney.columns.tolist())
torch.save(ctgan, '../../models/kidney/ctgan-kidney')
samples = ctgan.sample(300)

#samples.profile_report().to_file("sample.html")

print(samples)