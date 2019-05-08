import numpy as np
import csv
from mltools import get_data, gbr, rf, svm, get_folds



def run(models):
    cols = ['Qmm']
    xs, ts, x_names, t_names = get_data(cols)

    for t_ind in range(len(t_names)):
        t = ts[:, t_ind]
        print('Predicting {}'.format(t_names[t_ind]))
        
        tr_L1s = [[] for _ in range(len(models))]
        ts_L1s = [[] for _ in range(len(models))]
        tr_L2s = [[] for _ in range(len(models))]
        ts_L2s = [[] for _ in range(len(models))]
        num_samples, num_ftrs = xs.shape
        kfolds = get_folds(3, (xs, t.reshape(num_samples, 1)))
        for i, fold in enumerate(kfolds):  # for each fold
            print('fold {}'.format(i+1))
            X_test, y_test = fold[:, :num_ftrs], fold[:, num_ftrs:]  # split inputs and outputs
            training_data = kfolds[:i] + kfolds[i + 1:]
            training_data = np.vstack(training_data)
            X_train, y_train = training_data[:, :num_ftrs], training_data[:, num_ftrs:]  # remainder for training
            y_test = y_test.ravel()
            y_train = y_train.ravel()

            
            for m_ind, model in enumerate(models):
                L1_tr, L2_tr, L1_ts, L2_ts, _ = model(X_train, y_train, X_test, y_test)
                tr_L1s[m_ind].append(L1_tr)
                ts_L1s[m_ind].append(L1_ts)
                tr_L2s[m_ind].append(L2_tr)
                ts_L2s[m_ind].append(L2_ts)
                print('tr {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_tr, 3), round(L2_tr, 3)))
                print('ts {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_ts, 3), round(L2_ts, 3)))
        
        with open('./output/res_{}.csv'.format(t_names[t_ind]), 'w') as csvfile:
            w = csv.writer(csvfile)
            for m_ind, model in enumerate(models):
                w.writerow([str(model).split()[1]])
                w.writerow(['avg train L1', sum(tr_L1s[m_ind])/len(tr_L1s[m_ind]), 'avg test L1', sum(ts_L1s[m_ind])/len(ts_L1s[m_ind])])
                w.writerow(['avg train L2', sum(tr_L2s[m_ind])/len(tr_L2s[m_ind]), 'avg test L2', sum(ts_L2s[m_ind])/len(ts_L2s[m_ind])])


if __name__ == '__main__':
    run([rf, gbr])
