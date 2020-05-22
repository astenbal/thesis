from ctgan import CTGANSynthesizer
import pandas as pd

import pandas_profiling

import torch

data_cancer = pd.read_csv("../../data/cancer/raw.csv")

good_data_cancer = data_cancer.dropna()
#good_data_cancer.profile_report().to_file("data_breast.html")

#print(data_cancer)

#print(data_cancer.columns.tolist())

ctgan = CTGANSynthesizer()
ctgan.fit(good_data_cancer, ['id', 'diagnosis'])

#ctgan.fit(good_data_cancer, data_cancer.columns.tolist())
torch.save(ctgan, '../../models/breast/ctgan-breast')

samples = ctgan.sample(300)

print(samples)

print(samples.describe())

#samples.profile_report().to_file("sample_cancer.html")

#print(samples)