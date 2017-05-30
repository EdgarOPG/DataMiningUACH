from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from utils import utils

import pandas as pd
import numpy

def attribute_subset_selection_with_trees(feature_vector, targets):
    # import data
    X = feature_vector
    Y = targets

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
    print('Importance of every feature:\n' + str(extra_tree.feature_importances_))

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit = True)

    # Model transformation
    new_feature_vector = model.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    return new_feature_vector

def principal_components_analysis(feature_vector, targets, n_components):
    # import data
    X = feature_vector
    Y = targets

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(X)

    # Model transformation
    new_feature_vector = pca.transform(X)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected:\n' + str(pca.n_components))
    print('New feature dimension:\n' + str(pca.n_components_))
    print('Variance of every feature:\n' + str(pca.explained_variance_ratio_))

    return new_feature_vector
    # First 10 rows of new feature vector
    #print('\nNew feature vector:\n')
    #print(new_feature_vector[:10])

    #print(pd.DataFrame(pca.components_,columns=columns[1:-1]))

    # Print complete dictionary
    # print(pca.__dict__)

def z_score_normalization(data):
    print('----- z_score_normalization -------\n')
    # import data
    X = data[:,0:-1]
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

    return standardized_data

def min_max_scaler(data):
    print('----- min_max_scaler -------\n')
    # import data
    X = data[:,0:-1]
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
    print('Data min: \n' + str(min_max_scaler.data_min_))
    print('Data max: \n' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    return new_feature_vector

def fill_missing_values_with_constant(data, column, constant):
    temp = data[column].fillna(constant)
    data[column] = temp
    return data

def fill_missing_values_with_mean(data, column):
    temp = data[column].fillna(data[column].mean())
    data[column] = temp
    return data

def fill_missing_values_with_mode(data, column):
    temp = data[column].fillna(data[column].mode()[0])
    data[column] = temp
    return data

def checkout_year(data, column):
    year = data[column]
    for x in data.index:
        if year[x] > 2017:
            data.set_value(x, column, data[column].mode()[0])
    return data

def convert_data_to_numeric(data):
    numpy_data = data.values
    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        if(type(temp[0]).__name__  == 'str'):
            dict = numpy.unique(numpy_data[:,i])
            # print(dict)
            for j in range(len(dict)):
                # print(numpy.where(numpy_data[:,i] == dict[j]))
                temp[numpy.where(numpy_data[:,i] == dict[j])] = j
            numpy_data[:,i] = temp
    return numpy_data

def save(data, name):
    data.to_csv('../data/' + name, index = False)

if __name__ == '__main__':

    print("DATA LOADING...")
    train_data = utils.load_data("../data/train.csv")
    test_data = utils.load_data("../data/test.csv")

    #Outliers
    print("FINDING OUTLIERS...")
    train_data['state'] = train_data['state'].replace(33, train_data['state'].mode()[0])
    train_data = checkout_year(train_data, 'build_year')

    #Cleaning
    print("DATA CLEANING...")
    train_data['product_type'] = train_data['product_type'].fillna('NA')
    train_data = train_data.fillna(-1)

    test_data['product_type'] = test_data['product_type'].fillna('NA')
    test_data = test_data.fillna(-1)

    save(train_data, 'train_clean.csv')
    save(test_data, 'test_clean.csv')
