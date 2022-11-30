import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

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

### test single file calculation
filepath = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/test/decode/decoded_prediction.csv'
targetPath = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/cosine'

predicted = pd.read_csv(filepath)
predicted = predicted.reset_index(drop=True)
predicted.columns= predicted.columns.str.lower()


### read the circadian gene list
sex36 = pd.read_csv('/account/tli/ratBodyMap/data/specific/sex_specific_36.csv')
a = sex36.genes.str.lower().unique()
b = list(set(a).intersection(orgFeatures))
print(len(b))



def generated_cosine(feature, predicted=predicted, org=org):
    values = []
    for i in range(len(predicted)):       
        # true values
        sourceSample = predicted.loc[i, 'targetlabel']
        source_values = org[org['sample'] == sourceSample].loc[:, feature].values[0]
        # predicted values
        predicted_value = predicted.loc[i, feature].values
        cos=cosine_similarity(source_values.reshape(1,-1), predicted_value.reshape(1,-1))[0][0]
        values.append(cos)
    return values

### for generated profiles
filename = targetPath + '/generated_sex36_cosine.csv' 
result = predicted.iloc[:, 1:8]
valueCosine = generated_cosine(b) 
result['cosine'] = valueCosine
result.to_csv(filename)

def control_cosine(feature, predicted=predicted, org=org):
    values = []
    for i in range(len(predicted)): 
        # source values
        sourceSample = predicted.loc[i, 'sample']
        source_values = org[org['sample'] == sourceSample].loc[:, feature].values[0]
        # target values
        targetSample = predicted.loc[i, 'targetlabel']
        target_values = org[org['sample'] == targetSample].loc[:, feature].values[0]
        # cosine similarity
        cos=cosine_similarity(source_values.reshape(1,-1), target_values.reshape(1,-1))[0][0]
        values.append(cos)
    return values

### for real profiles
filename = targetPath + '/control_sex36_cosine.csv' 
result = predicted.iloc[:, 1:8]
valueCosine = control_cosine(b) 
result['cosine'] = valueCosine
result.to_csv(filename)