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
features = org.columns[3:]
print(len(features))

# True value cosine calculation
def control_cosine(predicted, filename, features=features):
    # calculate pearson value
    df = pd.DataFrame(columns = ['sample1', 'sample2', 'cosine_similarity'])
    for i in range(len(predicted)-1):
        sample1 = predicted.loc[i,'sample']
        true_values1 = predicted.loc[i, features].values
        for j in range(i+1, len(predicted)):
            sample2 = predicted.loc[j, 'sample']
            true_values2 = predicted.loc[j, features].values
            cos=cosine_similarity(true_values1.reshape(1,-1), true_values2.reshape(1,-1))[0][0]
            df.loc[len(df)] = [sample1, sample2, cos]
    df.to_csv(filename)


# real data set for control consine with full feature set
resultPath = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/cosine/control'
control_cosine(train, resultPath + '/cosine_control_train.csv')
control_cosine(test, resultPath + '/cosine_control_test.csv')



