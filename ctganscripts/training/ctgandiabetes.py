from ctgan import CTGANSynthesizer
import pandas as pd

import pandas_profiling

import torch


#print(data_cancer)

#print(data_cancer.columns.tolist())
data_cancer = pd.read_csv("../../data/diabetes/raw.csv")


data_diab = data_cancer[50000:100000]
good_data_cancer = data_cancer.dropna()

good_data_cancer.profile_report().to_file("data_diab.html")
exit()
ctgan = CTGANSynthesizer()
ctgan.fit(good_data_cancer, ['race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted'])

torch.save(ctgan, '../../models/diabetes/ctgan-diabetes-50k-1')

samples = ctgan.sample(300)

print(samples)

print(samples.describe())

#samples.profile_report().to_file("sample_cancer.html")

#print(samples)