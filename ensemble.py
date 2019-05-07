import numpy as np
import csv
from mltools import get_data, gbr, rf, svm_r, rr, get_folds


def k_fold_cross(X, y, k, ensemble, filename):
    """
    Perform k fold cross validation on a training function. If no function, simply returns a list of
    2d numpy matrices (of k length) that you can then manually use. If a function is provided then
    returns the average accuracy score.
    :X: a numpy 2d array
    :y: a numpy 2d array
    :k: int, number of splits
    :train_func: the function that will do the training
    :args: the arguments for the function (assumes that it takes train and test as first two arguments)
    """
    kfolds = get_folds(k, (X, y.reshape(len(X), 1)))
    _, num_ftrs = X.shape

    for i, fold in enumerate(kfolds):  # for each fold
        print('Running fold {}'.format(i + 1))
        X_test, y_test = fold[:, :num_ftrs], fold[:, num_ftrs:]  # split inputs and outputs
        training_data = kfolds[:i] + kfolds[i + 1:]
        training_data = np.vstack(training_data)
        X_train, y_train = training_data[:, :num_ftrs], training_data[:, num_ftrs:]  # remainder for training
        y_test = y_test.ravel()
        y_train = y_train.ravel()

        all_preds = np.zeros((len(ensemble), len(y_test)))
        wts = np.zeros((len(ensemble), 1))
        L1s = []
        L2s = []
        for j, func in enumerate(ensemble):
            L1, L2, y_pred = func(X_train, y_train, X_test, y_test)  # get predictions from training
            weight = 1 / L2  # want to penalize error
            all_preds[j, :] = y_pred
            wts[j] = weight  # add to col
            L1s.append(L1)
            L2s.append(L2)
            print('{}:\tL1: {} L2: {}'.format(str(func).split()[1], round(L1, 3), round(L2, 3)))

        wts = wts / np.sum(wts)  # make wts sum to 1
        final_preds = np.sum(wts * all_preds, axis=0)  # sum columns
        final_L1 = np.average(np.abs(np.subtract(final_preds, y_test)))
        final_L2 = np.average(np.square(np.subtract(final_preds, y_test)))

        print('wts: {}'.format(wts.ravel()))
        print('Final L1 err: {}'.format(np.round(final_L1, 3)))

        with open(filename, 'a') as csvfile:
            writer = csv.writer(csvfile)
            L1s = np.round(L1s, 4)
            L2s = np.round(L2s, 4)
            final_L1 = round(final_L1, 4)
            final_L2 = round(final_L2, 4)
            wts = np.round(wts, 3)
            writer.writerow(['L1', L1s[0], L1s[1], L1s[2], L1s[3], final_L1])
            writer.writerow(['L2', L2s[0], L2s[1], L2s[2], L2s[3], final_L2])
            writer.writerow(['wts', wts[0, 0], wts[1, 0], wts[2, 0], wts[3, 0]])


def run_ensemble():
    filename = './output/results.csv'
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['', 'gb', 'rf', 'svm', 'r', 'ens'])
    # Qmm, SurplusP, Q?, Q?, RiverD, PMica, PGranite, Q?, Slope, d15-NO3
    # cols = [20, 16, 23, 22, 17, 7, 6, 21, 14, 32]
    # cols = ['Qmm', 'RiverD', 'Artif', 'PMother', 'd15N-NO3']
    cols = ['Qmm', 'SurplusP']
    X, y, col_names = get_data(cols)
    print('Ensemble with attributes: {}'.format(col_names))
    ensemble = [gbr, rf, svm_r, rr]
    k_fold_cross(X, y, 3, ensemble, filename)


run_ensemble()
