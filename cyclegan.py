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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import LabelBinarizer
from random import randint

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

# read train and test set
trainSet = read_data('/account/tli/ratBodyMap/data/rna_bodymap_train_log2_encoded.csv')
testSet = read_data('/account/tli/ratBodyMap/data/rna_bodymap_test_log2_encoded.csv')
# read data set
dataset = pd.concat([trainSet, testSet], axis = 0)
dataset = dataset.sort_values('id', ascending = True)

# initialize binarizer to transfer category data (organ, sex, stage, biological copy) to binary data
organBinarizer, sexBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)
# get the feature column start number
n = 6
# UMAP helper
scaler, reducer, plot_data = plot_helper(trainSet, n)
# get the gene feature columns
cols = dataset.columns[n:]
print(len(cols))


# define the discriminator model
def define_discriminator(input_dim):
    # weight initialization
    init = RandomNormal(stddev=0.02)       

    # model structure
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define the standalone generator model1
def define_generator1(input_shape, output_shape):
    # gene input
    input_gene = Input(shape=input_shape)

    # weight initialization
    init = RandomNormal(stddev=0.01)
    # add GaussianNoise to input
    input_gene_withnoise = GaussianNoise(0.01)(input_gene)

    # l1
    g = Dense(256, activation="relu", kernel_initializer=init)(input_gene_withnoise)
    g = Activation('relu')(g)

    # l2
    g = Dense(256, activation="relu", kernel_initializer=init)(g)
    g = Activation('relu')(g)

    # l3
    g = Dense(128, activation="relu", kernel_initializer=init)(g)
    g = Activation('relu')(g)

    # output
    g = Dense(output_shape, activation='linear', kernel_initializer=init)(g)

    # define model
    model = Model(input_gene, g)
    return model

# define the standalone generator model2
def define_generator2(input_shape, label_shape, output_shape):
    # gene input
    input_gene = Input(shape=input_shape)
    input_label = Input(shape=label_shape)

    # weight initialization
    init = RandomNormal(stddev=0.01)
    
    # l1
    g = Dense(256, activation="relu", kernel_initializer=init)(Concatenate(axis=1)([input_label, input_gene]))
    g = Activation('relu')(g)

    # l2
    g = Dense(256, activation="relu", kernel_initializer=init)(g)
    g = Activation('relu')(g)
    
    # l3
    g = Dense(128, activation="relu", kernel_initializer=init)(g)
    g = Activation('relu')(g)

    # output
    g = Dense(output_shape, activation='linear', kernel_initializer=init)(g)

    # define model
    model = Model([input_label, input_gene], g)
    return model

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model1(g_model_1, d_model, g_model_2, input_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gene = Input(shape=input_shape)
    gen1_out = g_model_1(input_gene)
    output_d = d_model(gen1_out)
    # forward cycle
    input_target = Input(shape=(18, ))
    output_f = g_model_2([input_target, gen1_out])
    # define model graph
    model = Model([input_gene, input_target], [gen1_out, output_d, output_f])
    # define optimization algorithm configuration
    #opt = Adam(lr=0.001, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mae', 'binary_crossentropy', 'mae'], loss_weights=[10, 2, 5], optimizer='adam')
    return model

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model2(g_model1, g_model2, d_model, input_shape):
    # ensure the model we're updating is trainable
    g_model1.trainable = False
    # mark other generator model as not trainable
    g_model2.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # discriminator element
    input_gene = Input(shape=input_shape)
    gen1_out = g_model1(input_gene)
    # forward cycle
    input_target = Input(shape=(18, ))
    output_f = g_model2([input_target, gen1_out])
    output_d = d_model(output_f)
    # define model graph
    model = Model([input_gene, input_target], [gen1_out, output_f, output_d])
    # define optimization algorithm configuration
    #opt = Adam(lr=0.001, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mae', 'mae', 'binary_crossentropy'], loss_weights=[10, 5, 2], optimizer='adam')
    return model


# select a batch of random samples, returns genes and target
def generate_real_samples(dataset, n_samples, cols=cols, organBinarizer=organBinarizer, sexBinarizer=sexBinarizer, stageBinarizer=stageBinarizer, bioCopyBinarizer=bioCopyBinarizer):

    # choose random instances
    ix = [randint(0, dataset.shape[0] - 1) for i in range(n_samples)]
    # retrieve selected images
    samples = dataset.iloc[ix, :]

    # generate target sample
    j = randint(0, dataset.shape[0] - 1)
    # reset dataset index
    dataset = dataset.reset_index(drop = True)
    targetSample = dataset.loc[j, 'sample']

    # reform sample label in source
    sourceOrgan = organBinarizer.transform(samples['organ'])
    sex = sexBinarizer.transform(samples['sex'])
    stage = stageBinarizer.transform(samples['stage'])
    bioCopy = bioCopyBinarizer.transform(samples['bioCopy'])

    # reform sample label in target
    tmp = pd.DataFrame([targetSample] * n_samples, columns = ['label'])
    categories = tmp.label.str.split('_')  
    targetOrgan = organBinarizer.transform(categories.str[0])
    targetSex = sexBinarizer.transform(categories.str[1])
    targetStage = stageBinarizer.transform(categories.str[2])
    targetBioCopy = bioCopyBinarizer.transform(categories.str[3])      
    
    # combine the binarized representation to X
    X = np.hstack([sourceOrgan, sex, stage, bioCopy, targetOrgan, targetSex, targetStage, targetBioCopy, np.array(samples[cols])])
    # generate 'real' class labels (1)
    y = np.ones(len(X))
    return X, y, np.array(samples[cols]), np.hstack([sourceOrgan, sex, stage, bioCopy]), targetSample


# select a batch of random samples, returns genes and target
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

# select a batch of random samples, returns genes and target
def generate_target_samples(trainDataset, dataset, targetSample, cols=cols):
    tmp = pd.DataFrame()
    targetSamples = dataset[dataset['sample'] == targetSample]
    tmp = tmp.append([targetSamples] * len(trainDataset))
    X = np.array(tmp[cols])
    y = np.ones(len(X))
    return X, y

# generate a batch of images, returns genes and targets
def generate_fake_samples1(g_model, dataset):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = np.zeros(len(X))
    return X, y

# generate a batch of images, returns genes and targets
def generate_fake_samples2(g_model1, g_model2, dataset, orgOrgan):
    # generate fake instance from model 1
    gen1_out = g_model1.predict(dataset)
    # generate fake instance from model 2
    X = g_model2.predict([orgOrgan, gen1_out])
    # create 'fake' class labels (0)
    y = np.zeros(len(X))
    return X, y

# save the generator models to file
def save_models(step, g_model1, g_model2):
    # save the first generator model
    filename1 = '/account/tli/ratBodyMap/result/cyclegan/models/g_model1_%06d.h5' % (step+1)
    g_model1.save(filename1)
    print('>Saved: %s' %filename1)

    # save the second generator model
    #filename2 = '/account/tli/ratBodyMap/result/cyclegan/models/g_model2_%06d.h5' % (step+1)
    #g_model2.save(filename2)
    #print('>Saved: %s and %s' % (filename1, filename2))

# generate samples and save as a plot and save the model
def sub_plot(plot_data, ax, test_plot_data, targetOrgan, k):
    ax = sns.scatterplot(plot_data.loc[:, 'vec1'], plot_data.loc[:, 'vec2'], hue=plot_data.loc[:,'organ'])
    organ_specific_data = test_plot_data[test_plot_data['targetOrgan'] == targetOrgan]
    ax = sns.scatterplot(organ_specific_data.loc[:, 'vec1'], organ_specific_data.loc[:, 'vec2'], color='k', marker = 'o')
    ax.title.set_text('Target Organ is %s' %targetOrgan)
    if k < 6:
        ax.get_xaxis().set_visible(False)

def summarize_performance(step, g_model, X_in, orgDf, name, scaler=scaler, reducer=reducer, plot_data=plot_data):
    # make prediction
    X_out = g_model.predict(X_in)
    
    # save prediction
    features = []
    for i in range(len(X_out[0])):
        features.append('f'+str(i))
    X_out_df = pd.DataFrame(data=X_out, columns=features)
    X_out_df = pd.concat([orgDf, X_out_df], axis = 1)
    filename1 = '/account/tli/ratBodyMap/result/cyclegan/predictions_encoded/generator1_encoded_prediction_%06d.csv' %(step+1)
    X_out_df.to_csv(filename1)   
    
    # for visulization
    # scale and umap embeding the generated genes
    X_out_scaler = scaler.transform(X_out)
    X_out_embedding = reducer.transform(X_out_scaler)
    # construct the dataframe for the predicted umap vectors
    orgDf['vec1'] = X_out_embedding[:, 0]
    orgDf['vec2'] = X_out_embedding[:, 1]
    # compute subplot numbers   
    orgDf['targetOrgan'] = orgDf['targetLabel'].str.split('_').str[0]
    total = len(orgDf.targetOrgan.unique())
    ncolumns = 3
    # compute Rows required
    nrows = total // ncolumns
    nrows += total % ncolumns
    # create a position index
    position = range(1,total + 1)
    # plot
    fig = plt.figure(figsize = (15,12), dpi=300)
    for k, targetOrgan in enumerate(orgDf['targetOrgan'].unique()):
        # add every single subplot to the figure with a for loop
        ax = fig.add_subplot(nrows, ncolumns, position[k])
        sub_plot(plot_data, ax, orgDf, targetOrgan, k)
    # save plot to file
    plt.suptitle('Generated from %06d' %(step+1))
    filename2 = '/account/tli/ratBodyMap/result/cyclegan/picture/%s_generated_plot_%06d.png' % (name, (step+1))
    plt.savefig(filename2)
    plt.close()


# train cyclegan models
def train(d_model1, d_model2, g_model1, g_model2, composite_model1, composite_model2, trainSet, testSet):
    # define properties of the training run
    n_epochs, n_batch, = 10000, 1
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainSet) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # add the target label in the testSet for prediction
    X_in, orgDf= generate_test_real_samples(testSet)
    # manually enumerate epochs
    loss = []
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, y_realA, no_labelA, sourceSample, targetSample = generate_real_samples(trainSet, n_batch)
        X_realB, y_realB = generate_target_samples(X_realA, trainSet, targetSample)
        # generate a batch of fake samples
        X_fakeB, y_fakeB = generate_fake_samples1(g_model1, X_realA)
        X_fakeA, y_fakeA = generate_fake_samples2(g_model1, g_model2, X_realA, sourceSample)
        # update generator1 via adversarial and cycle loss
        g_loss1, _, _, _ = composite_model1.train_on_batch([X_realA, sourceSample], [X_realB, y_realB, no_labelA])
        # update discriminator for A -> [real/fake]
        dB_loss1 = d_model1.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model1.train_on_batch(X_fakeB, y_fakeB)
        # update generator1 via adversarial and cycle loss
        _, g_loss2, _, _ = composite_model2.train_on_batch([X_realA, sourceSample], [X_realB, no_labelA, y_realA])
        # update discriminator for B -> [real/fake]
        dA_loss1 = d_model2.train_on_batch(no_labelA, y_realA)
        dA_loss2 = d_model2.train_on_batch(X_fakeA, y_fakeA)
        loss.append([i, targetSample, g_loss1])
        # summarize performance
        print('>%d, %s, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, targetSample, dB_loss1[0], dB_loss2[0], dA_loss1[0], dA_loss2[0], g_loss1, g_loss2))
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            summarize_performance(i, g_model1, X_in, orgDf, 'Generator 1')
            # save the models
            save_models(i, g_model1, g_model2)
        if (i+1) % (bat_per_epo * 50) == 0: 
            loss_filename = '/account/tli/ratBodyMap/result/cyclegan/loss/loss_%06d.csv' % (i+1)
            pd.DataFrame(loss).to_csv(loss_filename)     
    pd.DataFrame(loss).to_csv('/account/tli/ratBodyMap/result/cyclegan/loss/loss.csv')        

# define input shape with source information and target label
input_shape1 = (36 + len(cols), )
# define input shape of target label 
target_shape = (18, )
# define shape without any label information
input_shape2 = (len(cols), )
# generator 1
g_model1 = define_generator1(input_shape1, input_shape2[0])
print(g_model1.summary())
# generator 2
g_model2 = define_generator2(input_shape2, target_shape, input_shape2[0])
print(g_model2.summary())
# discriminator 1
d_model1 = define_discriminator(input_shape2[0])
print(d_model1.summary())
# discriminator 2
d_model2 = define_discriminator(input_shape2[0])
# composite 1
composite_model1 = define_composite_model1(g_model1, d_model1, g_model2, input_shape1)
# composite 2
composite_model2 = define_composite_model2(g_model1, g_model2, d_model2, input_shape1)
# train models
train(d_model1, d_model2, g_model1, g_model2, composite_model1, composite_model2, trainSet, testSet)