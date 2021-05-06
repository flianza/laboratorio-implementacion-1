import os
import csv
import pandas as pd
import datatable as dt
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def pad_dict_list(dict_list):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if ll < lmax:
            dict_list[lname] += [None] * (lmax - ll)
    return dict_list


weights = defaultdict(list)
gains = defaultdict(list)
for experimento in os.listdir(f'../experimentos/'):
    for file in os.listdir(f'../experimentos/{experimento}/'):
        if file.endswith('importance.csv'):
            with open(f'../experimentos/{experimento}/{file}') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    weights[row['variable']].append(float(row['weight']))
                    gains[row['variable']].append(float(row['gain']))

weights = pad_dict_list(weights)
gains = pad_dict_list(gains)

df_weights = dt.Frame(weights).mean().to_pandas().T
df_gains = dt.Frame(gains).mean().to_pandas().T
df_count = dt.Frame(weights)[:, dt.count(dt.f[:])].to_pandas().T

df = pd.DataFrame(columns=['variable', 'weight', 'gain'])
df['variable'] = df_weights.index.values
df['gain'] = df_gains[0].values
df['count'] = df_count[0].values
df['weight'] = df_weights[0].values

df = df.sort_values(by=['gain', 'count', 'weight'], ascending=False)

df.to_csv('../analisis_feature_importance.csv', index=False)





