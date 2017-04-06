"""
Author: Normando Ali Zubia Hernández
This file is created to explain the use of dimensionality reduction
with different tools in sklearn library.
Every function contained in this file belongs to a different tool.
"""
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import numpy

def get_feacture_subset(data, *args):
    featureDic = []
    for arg in args:
        featureDic.append(arg)

    subset = data[featureDic]
    return subset

def attribute_subset_selection_with_trees(data):
    # import data
    X = data[:,0:-1]
    Y = numpy.asarray(data[:,-1], dtype="int16")

    # First 10 rows
    print('Training Data:\n\n' + str(X[:20]))
    print('\n')
    print('Targets:\n\n' + str(Y[:20]))

    # Model declaration
    extra_tree = ExtraTreesClassifier()

    # Model training
    extra_tree.fit(X, Y)

    # Model information:
    print('\nModel information:\n')

    # display the relative importance of each attribute
    print('Importance of every feature: ' + str(extra_tree.feature_importances_))

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit = True)

    # Model transformation
    new_feature_vector = model.transform(X)

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
    # principal_components_analysis(2)
    # principal_components_analysis(.90)
    # recursive_feature_elimination(2)

    # select_k_best_features(2)

    data = pd.read_json('train.json')
    data = convert_data_to_numeric(data)
    attribute_subset_selection_with_trees(data)
    #print(data)

    #print(data.describe)
