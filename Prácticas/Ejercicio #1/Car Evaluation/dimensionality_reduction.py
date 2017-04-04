import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier

dict = {'unacc' : 1,
        'acc' : 2,
        'good': 3,
        'vgood': 4}

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def principal_components_analysis(n_components):
    # import data
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

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
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    # Print complete dictionary
    # print(pca.__dict__)


def attribute_subset_selection_with_trees(data):
    # import data
    X = data[list(range(0,5))]
    Y = data[[6]]

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Model declaration
    extra_tree = DecisionTreeClassifier()

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


def recursive_feature_elimination(n_atributes):
    # import data
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Create a base classifier used to evaluate a subset of attributes
    # model_eval = ExtraTreesClassifier()

    # Note: Feature selection change with different models
    model_eval = LogisticRegression()

    # Create the RFE model and select 3 attributes
    rfe = RFE(model_eval, n_atributes)
    rfe = rfe.fit(X, Y)

    # Summarize the selection of the attributes
    # Model information:
    print('\nModel information:\n')
    print('New feature dimension: ' + str(rfe.n_features_))
    print('Feature Ranking: ' + str(rfe.ranking_))
    print('Selected features: ' + str(rfe.support_))

    # Model transformation
    new_feature_vector = rfe.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])


def select_k_best_features(n_atributes):
    # import data
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Model declaration
    kbest = SelectKBest(score_func = chi2, k = n_atributes)

    # Model training
    kbest.fit(X, Y)

    # Model transformation
    new_feature_vector = kbest.transform(X)

    # Summarize the selection of the attributes
    # Model information:
    print('\nModel information:\n')
    print('Feature Scores: ' + str(kbest.scores_))

    # Model transformation
    new_feature_vector = kbest.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])


if __name__ == '__main__':
    # principal_components_analysis(2)
    # principal_components_analysis(.93)
    data = open_file('train.csv')
    data['class'] = data['class'].replace(dict)
    #print(data[list(range(0,5))])
    attribute_subset_selection_with_trees(data)

    # recursive_feature_elimination(2)

    #select_k_best_features(2)
