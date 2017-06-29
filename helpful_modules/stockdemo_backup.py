%matplotlib notebook
%matplotlib inline

import os
import time
import datetime as dt
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import pandas
import pandas_datareader
import stock_data_preprocessing

import pandas as pd
data = pd.read_csv('SPY.csv')
# here we rearrange the columns and get rid of the non_adjusted close
data = data[data.columns[[0,1,2,3,6,5]]]

sequence_length = 50
result = []
#here we create a list of dataframes. that increment the day by one and and have sequence length sequence length
# this is important because it then allows us to apply a function to each sequence to normalize it into return space
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])

# This is an example of the top part of every dataframe
result[0].head()

# To Do, Turn this into a function and normalize all the sequences
def normalize_window(window_data):
    normalized_data =[]
    for window in window_data:
        window = window.drop(window.columns[[0]],axis=1)
        normalized_data.append(window.values/window.iloc[0].values-1)
    return normalized_data

result = np.array(normalize_window(result))
result[0]
print("The shape of the whole dataset is: {}".format(result.shape))

# set how you want to split train/validation/test set
train_split = 0.8
val_test_split = 0.5

# get the row that splits test data from the rest
train_row = int(round(train_split*result.shape[0]))

# get the training set (note that this is not split X_train or Y_train yet because we only
# care about the auto encoding and wont use this LSTM for prediction)
train = result[:train_row,:]

# Get the validation and testing portion of the data
val_test = result[train_row:,:]

#get the row we split test vs validation
test_row =int(round(val_test_split*val_test.shape[0]))

#get the validation set
validation =val_test[:test_row,:]

#get the test set
test = val_test[test_row:,:]

#now we print out the shape

print("Training set shape: {}".format(train.shape))
print("Validation set shape: {}".format(validation.shape))
print("Test set shape: {}".format(test.shape))


from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model


# This change of variables names is just to help me with understanding the network
# NOTE: LATENT_DIM IS A HYPERPARAMETER I PICKED 5 BECAUSE OF THE PAPER
timesteps = sequence_length
input_dim = train.shape[2]
latent_dim = 5

# My inputs
inputs = Input(shape=(timesteps, input_dim))
# They get encoded into a certain latent dimension
encoded = LSTM(latent_dim)(inputs)

#we then take the encoded vector and run it through another lstm
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

#we put it together in a single model
sequence_autoencoder = Model(inputs, decoded)

#create an incoder
encoder = Model(inputs, encoded)

# I TRIED TO CREATE A DECODER BUT I FAILED. TO DO create a decoder
# create a placeholder for an encoded (5-dimensional) input
#encoded_input = Input(shape=(latent_dim,))
# retrieve the last layer of the autoencoder model
#decoder_layer = sequence_autoencoder.layers[-1]
# create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))

from keras.callbacks import TensorBoard
n_epochs = 100
batch_size=32

sequence_autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')

sequence_autoencoder.fit(train, train,
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(validation, validation),
                verbose=1,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder',histogram_freq=5,
                                      write_images=True,
                                      embeddings_freq=5)])

