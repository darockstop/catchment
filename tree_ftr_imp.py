import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from collections import Counter
from mltools import get_data, get_folds
import matplotlib.pyplot as plt


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


def plot_imp(imp, x_names, ax):
    imp = 100 * (imp / imp.max())
    sorted_ind = np.argsort(imp)
    sorted_ind = sorted_ind[-10:]  # keep top 10 features only
    ax.barh(x_names[sorted_ind], imp[sorted_ind], color='#635a4d')


def run():
    imps = []
    X, ts, x_names, t_names = get_data()
    num_samples, num_ftrs = X.shape

    f, axarr = plt.subplots(len(t_names), 2, sharex='col')
    f.set_size_inches(18, 8)
    axarr[0, 0].set_title('Random Forest')
    axarr[0, 1].set_title('Gradient Boosting')


    for j in range(len(t_names)):
        axarr[j, 0].annotate(t_names[j],xy=(0, 0.5), xytext=(-axarr[j, 0].yaxis.labelpad-5,0), 
            xycoords=axarr[j, 0].yaxis.label, textcoords='offset points', 
            size='large', ha='right', va='center')

        t = ts[:, j]

        rf_imps = np.zeros(len(x_names))
        gb_imps = np.zeros(len(x_names))
        for _ in range(333):
            kfolds = get_folds(3, (X, t.reshape(num_samples, 1)))
            for i in range(3):
                fold = kfolds[i]
                X_test, y_test = fold[:, :num_ftrs], fold[:, num_ftrs:]  # split inputs and outputs
                training_data = kfolds[:i] + kfolds[i + 1:]
                training_data = np.vstack(training_data)
                X_train, y_train = training_data[:, :num_ftrs], training_data[:, num_ftrs:]  # remainder for training
                y_test = y_test.ravel()
                y_train = y_train.ravel()

                _, _, _, rf_imp = rf(X_train, y_train, X_test, y_test)
                _, _, _, gb_imp = gbr(X_train, y_train, X_test, y_test)

                rf_imps += rf_imp
                gb_imps += gb_imp

        plot_imp(rf_imps, np.array(x_names), axarr[j, 0])
        plot_imp(gb_imps, np.array(x_names), axarr[j, 1])
    f.savefig('./output/graphs/ftr_imp.png')


run()
