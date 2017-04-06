"""
Author: Normando Ali Zubia Hernández

This file is created to explain the use of normalization
with different tools in sklearn library.

Every function contained in this file belongs to a different tool.
"""

from sklearn import preprocessing

import pandas as pd
import numpy

def z_score_normalization(data):
    # import data
    X = data[:,0:-2]
    Y = numpy.asarray(data[:,-1], dtype="int16")

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Data standarization
    standardized_data = preprocessing.scale(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])

def min_max_scaler(data):
    # import data
    X = data[:,0:-2]
    Y = numpy.asarray(data[:,-1], dtype="int16")

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(X)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        dict = numpy.unique(numpy_data[:,i])
        # print(dict)
        for j in range(len(dict)):
            # print(numpy.where(numpy_data[:,i] == dict[j]))
            temp[numpy.where(numpy_data[:,i] == dict[j])] = j

        numpy_data[:,i] = temp

    return numpy_data

if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    data = convert_data_to_numeric(data)
    z_score_normalization(data)
    min_max_scaler(data)
