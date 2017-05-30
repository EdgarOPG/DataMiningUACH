"""
This file show the use of several algorithms to solve
Sberbank Russian Housing Market competition

Author: Normando Zubia
Universidad Autonoma de Chihuahua
"""

from utils import utils
from data_preprocessing import normalization
import numpy
import pandas
import matplotlib.pyplot as plt
import csv

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn import metrics
from sklearn import model_selection

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
    # Open data
    print("DATA LOADING...")
    train_data = utils.load_data("../data/train_clean.csv")
    test_data = utils.load_data("../data/test_clean.csv")

    # Outlier Example
    #plt.boxplot(train_data['full_sq'])
    #plt.show()

    # Get current column ordering to order concat dataset
    right_order = []
    for column in train_data.columns:
        right_order.append(column)

    print("DATA CLEANING...")
    #Cleaning


    test_data['product_type'] = test_data['product_type'].fillna('NA')
    test_data = test_data.fillna(-1)

    # Merge datasets
    frames = [train_data, test_data]

    complete_dataset = pandas.concat(frames)
    complete_dataset = complete_dataset[right_order]

    print("CONVERTING DATA...")
    # Convert data
    numpy_data = convert_data_to_numeric(complete_dataset)

    # Split features and target
    train_dataset = numpy_data[0:30470]
    test_dataset = numpy_data[30471: len(numpy_data)]

    feature_vector = train_dataset[:, 1:-1]
    targets = train_dataset[:, -1]

    test_temp = test_dataset[:, 1:-1]

    # Data normalization
    data_features_normalized = normalization.z_score_normalization(feature_vector)

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = utils.data_splitting(
        data_features_normalized,
        targets,
        0.25)

    # Algorithms declaration
    names = [
        "Random_Forest_Regressor",
        "Decision_Tree_Regressor",
    ]

    models = [
        RandomForestRegressor(
            criterion='mse',
            max_depth=10
        ),
        tree.DecisionTreeRegressor(
            criterion='mse'
        )    ]

    # Algorithm implementation
    for name, em_clf in zip(names, models):
        print("###################---" + name + "---###################")

        em_clf.fit(data_features_train, data_targets_train)

        # Model evaluation
        test_data_predicted = em_clf.predict(data_features_test)

        # Cross validation
        scores = model_selection.cross_val_score(em_clf, data_features_normalized, targets, cv=10)
        mean_error = scores.mean()

        print('Cross validation result: %s', mean_error)

        # Get predictions to Kaggle
        kaggle_predictions = em_clf.predict(test_dataset[:, 1:-1])

        # Generate CSV for Kaggle with csv package:
        path = "../data/predicted_kaggle_" + str(name) +".csv"
        # with open(path, "w") as csv_file:
        #     writer = csv.writer(csv_file, delimiter=',')
        #     writer.writerow(["id", "price_doc"])
        #
        #     for i in range(len(kaggle_predictions)):
        #         writer.writerow([test_dataset[i][0], kaggle_predictions[i]])

        # Generate CSV for Kaggle with pandas (easiest way)
        df_predicted = pandas.DataFrame({'id': test_dataset[:,0], 'price_doc': kaggle_predictions})

        df_predicted.to_csv(path, index=False)

        error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

        print('Total Error: %s', error)
