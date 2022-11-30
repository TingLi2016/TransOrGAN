import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import math
import seaborn as sns
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import join, isfile

# read data
datapath = '/account/tli/ratBodyMap/data'
train = pd.read_csv(datapath + '/rna_bodymap_train_log2.csv')
test = pd.read_csv(datapath + '/rna_bodymap_test_log2.csv')
# combine train and test
org = pd.concat([train, test])
org = org.iloc[:, 1:]
org = org.sort_values('id', ascending=True)
org = org.reset_index(drop = True)
cols = org.columns[-40064:]

# calculate rmse value
def generated_rmse(org, predicted, features, filepath):
    result = predicted.iloc[:, :8]
    rmsevalues = []
    for i in range(len(result)):
        # true values
        targetSample = result.loc[i, 'targetLabel']
        true_value = org[org['sample'] == targetSample].loc[:, features].values[0]
        # predicted values
        predicted_value = predicted.loc[i, features].values
        mse = mean_squared_error(true_value, predicted_value)
        rmse = math.sqrt(mse)
        rmsevalues.append(rmse)
    result['rmse'] = rmsevalues
    result.to_csv(filepath)

# ### for the generated test samples
generatedTest = pd.read_csv('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/test/decode/decoded_prediction.csv')
generated_rmse(org, generatedTest, cols, '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/rmse/rmse_generated_test.csv')


# ### for the generated training samples
files = listdir('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/train/decode')
for file in files:
    predicted = pd.read_csv(join('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/train/decode', file), low_memory=False)
    # calculate and save the rmse
    filename = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/rmse/rmse_' + file
    generated_rmse(org, predicted, cols, filename)    
    
# True value rmse calculation
def control_rmse(predicted, features, filepath):
    # calculate pearson value
    df = pd.DataFrame(columns = ['sample1', 'sample2', 'rmse'])
    for i in range(len(predicted)-1):
        sample1 = predicted.loc[i,'sample']
        true_values1 = predicted.loc[i, features].values
        for j in range(i+1, len(predicted)):
            sample2 = predicted.loc[j, 'sample']
            true_values2 = predicted.loc[j, features].values
            mse = mean_squared_error(true_values1, true_values2)
            rmse = math.sqrt(mse)
            df.loc[len(df)] = [sample1, sample2, rmse]
    df.to_csv(filepath)
    
control_rmse(test, cols, '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/rmse/rmse_real_test.csv')
control_rmse(train, cols, '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/rmse/rmse_real_train.csv')
