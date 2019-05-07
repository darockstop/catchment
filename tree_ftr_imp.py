import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from collections import Counter
from mltools import get_data, get_folds


def rf(X_train, y_train, X_test, y_test):
    forest = RandomForestRegressor(n_estimators=10)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    L1 = np.average(np.abs(np.subtract(y_pred, y_test)))
    L2 = np.average(np.square(np.subtract(y_pred, y_test)))
    imp = forest.feature_importances_
    return L1, L2, y_pred, imp


def gbr(X_train, y_train, X_test, y_test):
    regressor = GradientBoostingRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    L1 = np.average(np.abs(np.subtract(y_pred, y_test)))
    L2 = np.average(np.square(np.subtract(y_pred, y_test)))
    imp = regressor.feature_importances_
    return L1, L2, y_pred, imp


def run(func):
    imps = []
    X, y, col_names = get_data()
    _, num_ftrs = X.shape
    for _ in range(333):
        kfolds = get_folds(X, y.reshape(len(X), 1), 3)
        for i in range(3):
            fold = kfolds[i]
            X_test, y_test = fold[:, :num_ftrs], fold[:, num_ftrs:]  # split inputs and outputs
            training_data = kfolds[:i] + kfolds[i + 1:]
            training_data = np.vstack(training_data)
            X_train, y_train = training_data[:, :num_ftrs], training_data[:, num_ftrs:]  # remainder for training
            y_test = y_test.ravel()
            y_train = y_train.ravel()

            L1, L2, pred, imp = func(X_train, y_train, X_test, y_test)
            ind = np.argpartition(imp, -5)[-5:]
            imps.extend(col_names[ind])
    count = Counter(imps)

    with open('imp_rf.csv', 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['attribute', 'freq'])
        for ftr_name, freq in count.items():
            w.writerow([ftr_name, freq])


run(rf)
