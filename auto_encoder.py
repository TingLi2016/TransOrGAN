# train autoencoder 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, ActivityRegularization
from tensorflow.keras import activations
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
from keras import regularizers

# read data
datapath = '/account/tli/ratBodyMap/data'
train = pd.read_csv(datapath + '/rna_bodymap_train_log2.csv')
test = pd.read_csv(datapath + '/rna_bodymap_test_log2.csv')
features = train.columns[4:]
print(train.columns[:5])
# split into train test sets
X_train = train[features]
X_test = test[features]
# scale data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(max(X_train.max(0)))
print(max(X_test.max(0)))

print(min(X_train.min(0)))
print(min(X_test.min(0)))

#unscaled_train = scaler.inverse_transform(X_train)
#unscaled_test = scaler.inverse_transform(X_test)
#pd.DataFrame(unscaled_train).to_csv(datapath + '/unscaled_train.csv')
#pd.DataFrame(unscaled_test).to_csv(datapath + '/unscaled_test.csv')



# number of input columns
n_inputs = 2048

# define encoder
visible = Input(shape=(X_train.shape[1],))
# encoder level 1
e = Dense(n_inputs)(visible)

# bottleneck
n_bottleneck = 128
bottleneck = Dense(n_bottleneck)(e)

# decoder level 1
d = Dense(n_inputs)(bottleneck)

# output layer
output = Dense(X_train.shape[1], activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# print model summary
model.summary()
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=150, batch_size=16, verbose=2, validation_data=(X_test,X_test))
# save model
model.save('/account/tli/ratBodyMap/result/auto_encoder/autoencoder.h5')
# plot the autoencoder
plot_model(model, '/account/tli/ratBodyMap/result/auto_encoder/autoencoder_compress.png', show_shapes=True)

# save loss
filename1 = '/account/tli/ratBodyMap/result/auto_encoder/encoder_loss.csv'
lossDf = pd.DataFrame()
lossDf['loss'] = history.history['loss']
lossDf['val_loss'] = history.history['val_loss']
lossDf.to_csv(filename1)
# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
filename1 = '/account/tli/ratBodyMap/result/auto_encoder/encoder_loss_no_normalization.jpg'
pyplot.savefig(filename1)



# define an encoder model (without the decoder)
encoder = Model(inputs=model.input, outputs=model.layers[-3].output)
print('encoder')
encoder.summary()
plot_model(encoder, '/account/tli/ratBodyMap/result/auto_encoder/encoder_compress.png', show_shapes=True)
# save the encoder to file
encoder.save('/account/tli/ratBodyMap/result/auto_encoder/encoder.h5')



# get the representation of X_train and X_test
# encode the train data
X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)


features = []
for i in range(1, n_bottleneck+1):
    features.append('f'+str(i))

def encoded_representation(representation, org, features, filename):
    representation_df = pd.DataFrame(representation, columns = features)
    encoded_df = pd.concat([org.iloc[:, 1:4], representation_df], axis = 1)
    encoded_df.to_csv(filename)
    
# save the representation of encoded features
encoded_representation(X_train_encode, train, features, '/account/tli/ratBodyMap/data/rna_bodymap_train_log2_encoded.csv')
encoded_representation(X_test_encode, test, features, '/account/tli/ratBodyMap/data/rna_bodymap_test_log2_encoded.csv')


# define an decoder model
decoder_input = Input(shape=(n_bottleneck,))
decoder_layer1 = model.layers[-2]
decoder_layer2 = model.layers[-1]
decoder = Model(inputs=decoder_input, outputs=decoder_layer2(decoder_layer1(decoder_input)))
decoder.save('/account/tli/ratBodyMap/result/auto_encoder/decoder.h5')


print('decoder')
decoder.summary()

# dencoder the train data
X_train_decoder = decoder.predict(X_train_encode)
# encode the test data
X_test_decoder = decoder.predict(X_test_encode)

pd.DataFrame(X_train_decoder).to_csv('/account/tli/ratBodyMap/data/rna_bodymap_train_log2_decorded.csv')
pd.DataFrame(X_test_decoder).to_csv('/account/tli/ratBodyMap/data/rna_bodymap_test_log2_decorded.csv')


unscaled_train_decoder = scaler.inverse_transform(X_train_decoder)
unscaled_test_decoder = scaler.inverse_transform(X_test_decoder)
pd.DataFrame(unscaled_train_decoder).to_csv(datapath + '/unscaled_train_decoder.csv')
pd.DataFrame(unscaled_test_decoder).to_csv(datapath + '/unscaled_test_decoder.csv')
