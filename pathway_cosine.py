import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

# read data
datapath = '/account/tli/ratBodyMap/data'
train = pd.read_csv(datapath + '/rna_bodymap_train_log2.csv')
test = pd.read_csv(datapath + '/rna_bodymap_test_log2.csv')
# combine train and test
org = pd.concat([train, test])
org = org.iloc[:, 1:]
org = org.sort_values('id', ascending=True)
org = org.reset_index(drop = True)
org.columns= org.columns.str.lower()
orgFeatures = org.columns[3:]

pathwaySymbol = pd.read_csv('/account/tli/ratBodyMap/data/symbol_pathway_87.csv')
pathwaySymbol = pathwaySymbol.iloc[1:, 1:]
pathwaySymbol = pathwaySymbol.apply(lambda x: x.astype(str).str.lower())

pathways = np.array(pathwaySymbol.columns)

### test single file calculation
filepath = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/test/decode/decoded_prediction.csv'
targetPath = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/cosine'

predicted = pd.read_csv(filepath)
predicted = predicted.reset_index(drop=True)
predicted.columns= predicted.columns.str.lower()
filename = targetPath + '/pathway_cosine.csv' 

result = predicted.iloc[:, 2:9]

def pathway_cosine(feature, predicted=predicted, org=org):
    values = []
    for i in range(len(predicted)):       
        # true values
        targetSample = predicted.loc[i, 'targetlabel']
        true_values = org[org['sample'] == targetSample].loc[:, feature].values[0]
        # predicted values
        predicted_value = predicted.loc[i, feature].values
        cos=cosine_similarity(true_values.reshape(1,-1), predicted_value.reshape(1,-1))[0][0]
        values.append(cos)
    return values


for pathway in pathways:
    print(pathway)
    feature = pathwaySymbol[pathway].unique()
    print(len(feature))
    feature = [item for item in feature if item in orgFeatures]
    print('common length of feature', len(feature))
    valueCosine = pathway_cosine(feature) 
    result[pathway] = valueCosine
result.to_csv(filename)