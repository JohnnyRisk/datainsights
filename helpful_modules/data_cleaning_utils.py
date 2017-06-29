# A function that normalizes all the sequences
import pandas as pd
import numpy as np
def normalize_window(window_data):
    normalized_data =[]
    for window in window_data:
        window = window.drop(window.columns[[0]],axis=1)
        normalized_data.append(window.values/window.iloc[0].values-1)
    return normalized_data

def normalize_pandas_window(window_data):
    normalized_data =[]
    for window in window_data:
        window = window.drop(window.columns[[0]],axis=1)
        normalized_data.append(np.array(window.pct_change().fillna(0)))
    return normalized_data

def load_stock_data(ticker, sequence_length=20, train_split =0.8, val_test_split=0.5, peak=0):
    data = pd.read_csv('./project_data/'+ticker + '.csv')
    result = []
    # here we rearrange the columns and get rid of the non_adjusted close
    data = data[data.columns[[0,1,2,3,6,5]]]
    
    #here we create a list of dataframes. that increment the day by one and and have sequence length sequence length
    # this is important because it then allows us to apply a function to each sequence to normalize it into return space
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    #normalize the data and get the shape
    result = np.array(normalize_pandas_window(result))
    print("The shape of the whole dataset is: {}".format(result.shape))

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

    print("Training Set shape: {}".format(train.shape))
    print("Validation set shape: {}".format(validation.shape))
    print("Test set shape: {}".format(test.shape))
    if peak ==1:
        print(data.head(3))
    return train, validation, test