import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def normalize(matrix):
    return (matrix - matrix.min(0)) / matrix.ptp(0)


def get_data(cols=None):
    """
    Gets data from the database file.
    :param cols: a list of desired column names
    :return: a numpy array of inputs and targets as well as a list of column names
    """
    with open('./MLDatabase.csv', 'r') as data_file:
        df = pd.read_csv(data_file)
    inputs = df.iloc[:, 4:-2]  # 4 because don't want some beg stuff or trgts
    targets = df.iloc[:, -2]  # -2 is nretention, -1 is pretention

    if cols:
        inputs = inputs[cols]
        col_names = cols

    inputs = np.array(inputs)
    targets = np.array(targets)

    if not cols:
        # Handle unknowns by ignoring that whole column
        col_names = df.columns.values[4:-2]
        _, num_ftrs = inputs.shape
        to_keep = []
        for i in range(num_ftrs):
            col = inputs[:, i]
            if not np.isnan(col).any():  # if there are NOT any nan vals
                to_keep.append(i)

        inputs = inputs[:, to_keep]
        col_names = col_names[to_keep]

    return normalize(inputs), targets, col_names


def get_folds(k, data=None):
    rand = np.random.permutation(49) * 3  # there are 49 different sites, each with three samplings
    num_in_fold = 49 // k
    num_extra = 49 % k

    # split permutations into sections
    kfold_inds = [[] for _ in range(k)]
    for i in range(k):
        inds = rand[i * num_in_fold:i * num_in_fold + num_in_fold]  # get num_in_fold rand indices
        kfold_inds[i] = list(inds)

    # add extras equally at the end
    extra_inds = rand[-num_extra:]
    for i in range(num_extra):
        insert_ind = i % k
        kfold_inds[insert_ind].append(extra_inds[-num_extra + i])

    if not data:
        return kfold_inds
    else:
        return kfold_inds_to_rows(data, kfold_inds)


def kfold_inds_to_rows(data, kfold_inds):
    k = len(kfold_inds)
    kfolds = [[] for _ in range(k)]

    data = np.hstack(data)  # put inputs and targets together
    _, num_cols = data.shape

    for i, inds in enumerate(kfold_inds):
        fold = np.zeros((len(inds) * 3, num_cols))
        fold[0 * len(inds):1 * len(inds)] = data[inds]
        inds = [x + 1 for x in inds]
        fold[1 * len(inds):2 * len(inds)] = data[inds]
        inds = [x + 1 for x in inds]
        fold[2 * len(inds):3 * len(inds)] = data[inds]
        kfolds[i] = fold

    kfolds = [np.array(x) for x in kfolds]  # now a list of 2d numpy arrays
    return kfolds


def get_split(kfolds, i, num_ftrs):
    X_test, y_test = kfolds[i][:, :num_ftrs], kfolds[i][:, num_ftrs:]  # split inputs and outputs
    training_data = kfolds[:i] + kfolds[i + 1:]
    training_data = np.vstack(training_data)
    X_train, y_train = training_data[:, :num_ftrs], training_data[:, num_ftrs:]  # remainder for training
    y_test = y_test.ravel()
    y_train = y_train.ravel()

    return X_train, y_train, X_test, y_test


def svm_r(X_train, y_train, X_test, y_test):
    classifier = svm.SVR()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    L1 = np.average(np.abs(np.subtract(y_pred, y_test)))
    L2 = np.average(np.square(np.subtract(y_pred, y_test)))
    return L1, L2, y_pred


def rf(X_train, y_train, X_test, y_test):
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    L1 = np.average(np.abs(np.subtract(y_pred, y_test)))
    L2 = np.average(np.square(np.subtract(y_pred, y_test)))
    return L1, L2, y_pred


def gbr(X_train, y_train, X_test, y_test):
    regressor = GradientBoostingRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    L1 = np.average(np.abs(np.subtract(y_pred, y_test)))
    L2 = np.average(np.square(np.subtract(y_pred, y_test)))
    return L1, L2, y_pred


def rr(X_train, y_train, X_test, y_test):
    classifier = linear_model.Ridge(alpha=.5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    L1 = np.average(np.abs(np.subtract(y_pred, y_test)))
    L2 = np.average(np.square(np.subtract(y_pred, y_test)))
    return L1, L2, y_pred
