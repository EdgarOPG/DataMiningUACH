"""
This file show the use of Neural Nettwork function of sklearn library
for more info: http://scikit-learn.org/stable/modules/neural_networks_supervised.html

Author: Normando Zubia
Universidad Autonoma de Chihuahua
"""

import numpy

from utils import utils
from data_preprocessing import normalization

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn import tree
from sklearn import model_selection



import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def mlp_classifier(data):
    """
    This function show the use of a neural network with iris dataset
    """

    data_features = data[:,0:-1]
    data_targets = numpy.asarray(data[:,-1], dtype="int16")

    # Data normalization
    data_features_normalized = normalization.z_score_normalization(data_features)

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = utils.data_splitting(
        data_features_normalized,
        data_targets,
        0.25)

    # Model declaration
    adaboost = AdaBoostRegressor(
        base_estimator=tree.DecisionTreeRegressor(
            criterion='mse',
            max_depth=10)
    )
    adaboost.fit(data_features_train, data_targets_train)
    # Model evaluation
    test_data_predicted = adaboost.predict(data_features_test)
    score = model_selection.cross_val_score(adaboost, data_features_normalized, data_targets, cv=10)

    logger.debug("Model Score: %s", score)

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

if __name__ == '__main__':

    print("DATA LOADING...")
    train_data = utils.load_data("../data/train_clean.csv")

    print("DATA CONVERTING...")
    train_data = convert_data_to_numeric(train_data)

    logger.info("###################---MLP Classifier---###################")
    # Classification example
    mlp_classifier(train_data)
