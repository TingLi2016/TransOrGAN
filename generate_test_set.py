import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import GaussianNoise
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
#from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import LabelBinarizer
from random import randint
from sklearn.metrics.pairwise import cosine_similarity

def read_data(path):
    data = pd.read_csv(path)
    data = data.iloc[:, 1:]
    categories = data['sample'].str.split('_')
    data.insert(3, 'sex', categories.str[1])
    data.insert(4, 'stage', categories.str[2])
    data.insert(5, 'bioCopy', categories.str[3])
    return data

def binarizer(data):
    organBinarizer = LabelBinarizer().fit(data["organ"])
    sexBinarizer = LabelBinarizer().fit(data["sex"])
    stageBinarizer = LabelBinarizer().fit(data["stage"])
    bioCopyBinarizer = LabelBinarizer().fit(data["bioCopy"])
    return organBinarizer, sexBinarizer, stageBinarizer, bioCopyBinarizer

# read train and test set for generator
trainSet = read_data('/account/tli/ratBodyMap/data/rna_bodymap_train_log2_encoded.csv')
testSet = read_data('/account/tli/ratBodyMap/data/rna_bodymap_test_log2_encoded.csv')
# read data set
dataset = pd.concat([trainSet, testSet], axis = 0)
dataset = dataset.sort_values('id', ascending = True)

# initialize binarizer to transfer category data (organ, sex, stage, biological copy) to binary data
organBinarizer, sexBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)
# get the feature column start number
n = 6
# get the gene feature columns
cols = dataset.columns[n:]
print(len(cols))


# select a batch of random samples, returns images and target
def generate_test_real_samples(dataset, cols=cols, organBinarizer=organBinarizer, sexBinarizer=sexBinarizer, stageBinarizer=stageBinarizer, bioCopyBinarizer=bioCopyBinarizer):
    # duplicate all instances with the number of dataset
    dataset = dataset.reset_index(drop = True)
    samples = pd.concat([dataset]*len(dataset), ignore_index=True)
    # target samples label
    targetLabel = sorted(sorted(dataset['sample']) * len(dataset))
    # add target label to samples
    samples.insert(0, 'targetLabel', targetLabel)
    # reform source label
    sourceOrgan = organBinarizer.transform(samples['organ'])
    sex = sexBinarizer.transform(samples['sex'])
    stage = stageBinarizer.transform(samples['stage'])
    bioCopy = bioCopyBinarizer.transform(samples['bioCopy'])     
    # reform target label
    categories = samples['targetLabel'].str.split('_')
    targetOrgan = organBinarizer.transform(categories.str[0])
    targetSex = sexBinarizer.transform(categories.str[1])
    targetStage = stageBinarizer.transform(categories.str[2])
    targetBioCopy = bioCopyBinarizer.transform(categories.str[3])                 
    # combine the binarized representation to X
    X = np.hstack([sourceOrgan, sex, stage, bioCopy, targetOrgan, targetSex, targetStage, targetBioCopy, np.array(samples[cols])])
    return X, samples.drop([*cols], axis=1)


# read data for cosine calculation
train = pd.read_csv('/account/tli/ratBodyMap/data/rna_bodymap_train_log2.csv')
test = pd.read_csv('/account/tli/ratBodyMap/data/rna_bodymap_test_log2.csv')
# combine train and test
org = pd.concat([train, test])
org = org.iloc[:, 1:]
org = org.sort_values('id', ascending=True)
org = org.reset_index(drop = True)


def cal_cosine(values, true_values1, true_values2, pecentages=[0, 5, 25, 50, 75, 95]):
    for pecentage in pecentages:
        mask = np.where(true_values1 >= np.percentile(true_values1, pecentage), True, False)
        mask_true_values1 = true_values1[mask]
        mask_true_values2 = true_values2[mask]
        # cosine value
        cos=cosine_similarity(mask_true_values1.reshape(1,-1), mask_true_values2.reshape(1,-1))[0][0]
        values.append(cos)
    return values
   
def generated_cosine(predicted, filename, features, org=org, pecentages=[0, 5, 25, 50, 75, 95]):
    result = predicted.iloc[:, :-len(features)]
    # calculate pearson value
    df = pd.DataFrame(columns = ['c00', 'c05', 'c25', 'c50', 'c75', 'c95'])
    for i in range(len(result)):
        values = []
        # true values
        targetSample = result.loc[i, 'targetLabel']
        true_values = org[org['sample'] == targetSample].loc[:, features].values[0]
        # predicted values
        predicted_value = predicted.loc[i, features].values
        values1 = cal_cosine([], true_values, predicted_value)
        df.loc[len(df)] = values1
    df = pd.concat([result, df], axis = 1)
    df.to_csv(filename) 

def summarize_performance(g_model, X_in, orgDf1, decoder, train=train):
    ### make prediction
    X_out = g_model.predict(X_in)
    # save prediction
    features = []
    for i in range(len(X_out[0])):
        features.append('f'+str(i))
    X_out_df = pd.DataFrame(data=X_out, columns=features)
    X_out_df = pd.concat([orgDf1, X_out_df], axis = 1)
    filename1 = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/test/encode/encoded_prediction.csv'
    X_out_df.to_csv(filename1) 


    ### decode prediction
    decode_X = decoder.predict(X_out)
    
    ### scale back the decorded result
    #train = pd.read_csv('/account/tli/ratBodyMap/data/rna_bodymap_train_log2.csv')
    X_train = train.iloc[:, 4:]
    geneCols = train.columns[4:]
    print(X_train.shape)
    # scale data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    unscaled_decoderX = scaler.inverse_transform(decode_X)
    unscaled_decoderX_df = pd.DataFrame(data=unscaled_decoderX, columns=geneCols)
    unscaled_decoderX_df = pd.concat([orgDf1, unscaled_decoderX_df], axis = 1)
    filename2 = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/predictions/test/decode/decoded_prediction.csv'
    unscaled_decoderX_df.to_csv(filename2) 
    
    ### calculate cosine similarity
    filename3 = '/account/tli/ratBodyMap/result/cyclegan/manuscript2022/cosine/test/test_cosine_decoded_prediction.csv'
    generated_cosine(unscaled_decoderX_df, filename3, geneCols)
     
    

# prepare the test data set
X_in, orgDf= generate_test_real_samples(testSet)
# load the model
g_model = keras.models.load_model('/account/tli/ratBodyMap/script/cyclegan/manuscript05232022/github/g_model1_2161152.h5')
# load the decoder
decoder = tf.keras.models.load_model('/account/tli/ratBodyMap/script/cyclegan/manuscript05232022/github/decoder.h5')



mu, sigma = 0, 0.01 
# creating a noise with the same dimension as the dataset (2,2) 
noise = np.random.normal(mu, sigma, [len(X_in),len(X_in[0])]) 
X_in_noise = X_in + noise
summarize_performance(g_model, X_in_noise, orgDf, decoder)