import numpy as np
import csv
from mltools import get_data, gbr, rf, svm_r, rr, get_folds, get_split, kfold_inds_to_rows
import itertools


def k_fold_cross(X, y, kfold_inds, ensemble, result_file):
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

    _, num_ftrs = X.shape
    kfolds = kfold_inds_to_rows((X, y), kfold_inds)

    final_L1s = []
    final_L2s = []
    for i in range(len(kfolds)):  # for each fold
        X_train, y_train, X_test, y_test = get_split(kfolds, i, num_ftrs)

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
            # print('{}:\tL1: {} L2: {}'.format(str(func).split()[1], round(L1, 3), round(L2, 3)))

        wts = wts / np.sum(wts)  # make wts sum to 1
        final_preds = np.sum(wts * all_preds, axis=0)  # sum columns
        final_L1 = np.average(np.abs(np.subtract(final_preds, y_test)))
        final_L2 = np.average(np.square(np.subtract(final_preds, y_test)))
        final_L1s.append(final_L1)
        final_L2s.append(final_L2)

        # print('wts: {}'.format(wts.ravel()))
        # print('Final L1 err: {}'.format(np.round(final_L1, 3)))

        with open(result_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            L1s = np.round(L1s, 4)
            L2s = np.round(L2s, 4)
            final_L1 = round(final_L1, 4)
            final_L2 = round(final_L2, 4)
            wts = np.round(wts, 3)
            writer.writerow(['L1', L1s[0], L1s[1], L1s[2], L1s[3], final_L1])
            writer.writerow(['L2', L2s[0], L2s[1], L2s[2], L2s[3], final_L2])
            writer.writerow(['wts', wts[0, 0], wts[1, 0], wts[2, 0], wts[3, 0]])
            writer.writerow([])
    return round(np.average(final_L1s), 4), round(np.average(final_L2s), 4)


def run_ensemble():
    res_filename = 'results.csv'
    # cols = ['Qmm', 'RiverD', 'Artif', 'Slope', 'perWet', 'SurplusP', 'B e (µg/L)']
    cols = ['Qmm','SurplusP', 'RiverD', 'perWet', 'Artif', 'Agri', 'Slope', 'Relief', 'PMother', 'PMmica', 'PMgranite', 'PMschiste', 'B e (µg/L)']

    ftr_combs = []
    for l in range(1, len(cols) + 1):
        ftr_combs.extend(list(itertools.combinations(cols, l)))
    ftr_combs = [list(x) for x in ftr_combs]

    kfold_inds = get_folds(3)

    for ftrs in ftr_combs:
        print('Ftrs: {}'.format(ftrs))
        if 'Qmm' not in ftrs:
            continue
        X, y, col_names = get_data(ftrs)
        with open(res_filename, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(ftrs)
            writer.writerow(['', 'gb', 'rf', 'svm', 'r', 'ens'])

        ensemble = [gbr, rf, svm_r, rr]
        L1, L2 = k_fold_cross(X, y.reshape(len(X), 1), kfold_inds, ensemble, res_filename)

        with open('summary.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ftrs, L1, L2])


run_ensemble()
