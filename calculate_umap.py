import pandas as pd
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import os
from os import listdir
from os.path import join, isfile

def read_data(path):
    data = pd.read_csv(path)
    data = data.iloc[:, 1:]
    categories = data['sample'].str.split('_')
    data.insert(3, 'sex', categories.str[1])
    data.insert(4, 'stage', categories.str[2])
    data.insert(5, 'bioCopy', categories.str[3])
    return data

def plot_helper(dataset, n):
    scaler = StandardScaler()
    scaler.fit(dataset.iloc[:, n:])
    scaled_data = scaler.transform(dataset.iloc[:, n:])

    reducer = umap.UMAP(random_state=11)
    reducer.fit(scaled_data)
    embedding = reducer.transform(scaled_data)

    
    plot_data = dataset.iloc[:, :n]
    plot_data['vec1'] = embedding[:, 0]
    plot_data['vec2'] = embedding[:, 1]
    return scaler, reducer, plot_data  


def data_umap(data, col, filename):
    dataUmap = data.iloc[:, :-len(col)]
    # scale and umap embeding the generated genes
    X_scaler = scaler.transform(data[cols])
    X_embedding = reducer.transform(X_scaler)
    # construct the dataframe for the predicted umap vectors
    dataUmap['vec1'] = X_embedding[:, 0]
    dataUmap['vec2'] = X_embedding[:, 1]
    # save the dataUmap
    dataUmap.to_csv('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/umap' + filename)
 

# read train and test set
trainSet = read_data('/account/tli/ratBodyMap/data/rna_bodymap_train_log2.csv')

# get the feature column start number
n = 6
# get the gene feature columns
cols = trainSet.columns[n:]
print(len(cols))
# UMAP helper
scaler, reducer, plot_data = plot_helper(trainSet, n)
plot_data.to_csv('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/umap/umap_train.csv')


### for the generated training samples
files = listdir('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/train/decode')
for file in files:
    tmp = pd.read_csv(join('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/train/decode', file), low_memory=False)
    data_umap(tmp, cols, '/umap_' + file)

### for the test samples
testSet = read_data('/account/tli/ratBodyMap/data/rna_bodymap_test_log2.csv')
data_umap(testSet, cols, '/umap_test.csv')

### for the generated test samples
generatedTest = pd.read_csv('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/test/decode/decoded_prediction.csv')
data_umap(generatedTest, cols, '/umap_generated_test.csv')


# ### for the generated training samples
# files = listdir('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/train/decode')
# for file in files:
#     tmp = pd.read_csv(join('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/train/decode', file), low_memory=False)
#     tmpUmap = tmp.iloc[:, 1:8]
#     # scale and umap embeding the generated genes
#     X_scaler = scaler.transform(tmp[cols])
#     X_embedding = reducer.transform(X_scaler)
#     # construct the dataframe for the predicted umap vectors
#     tmpUmap['vec1'] = X_embedding[:, 0]
#     tmpUmap['vec2'] = X_embedding[:, 1]
#     # save the tmpUmap
#     filename = '/umap_' + file
#     tmpUmap.to_csv('/account/tli/ratBodyMap/result/cyclegan/manuscript2022/umap' + filename)