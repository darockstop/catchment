import csv
from mltools import gbr, rf, svm
from data import *



def run(models):
    df = get_data(for_training=True)

    for t_ind in range(len(targets)):
        print('Predicting {}'.format(targets[t_ind]))
        
        tr_L1s = [[] for _ in range(len(models))]
        ts_L1s = [[] for _ in range(len(models))]
        tr_L2s = [[] for _ in range(len(models))]
        ts_L2s = [[] for _ in range(len(models))]
        kfolds = get_folds(df, 3)
        for i, fold in enumerate(kfolds):  # for each fold
            print('fold {}'.format(i+1))

            training, testing = preprocess(pd.concat(kfolds[:i] + kfolds[i + 1:]), fold)            
            X_train, y_train = training.drop(targets, axis=1), training[targets]
            X_test, y_test = testing.drop(targets, axis=1), testing[targets]
            X_train, X_test = X_train.drop('site', axis=1), X_test.drop('site', axis=1)

            for m_ind, model in enumerate(models):
                L1_tr, L2_tr, L1_ts, L2_ts, _ = model(X_train, y_train.iloc[:, t_ind], X_test, y_test.iloc[:, t_ind])
                tr_L1s[m_ind].append(L1_tr)
                ts_L1s[m_ind].append(L1_ts)
                tr_L2s[m_ind].append(L2_tr)
                ts_L2s[m_ind].append(L2_ts)
                print('tr {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_tr, 3), round(L2_tr, 3)))
                print('ts {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_ts, 3), round(L2_ts, 3)))
        
        with open('./output/r_imp_{}.csv'.format(targets[t_ind]), 'w') as csvfile:
            w = csv.writer(csvfile)
            for m_ind, model in enumerate(models):
                w.writerow([str(model).split()[1]])
                w.writerow(['avg train L1', sum(tr_L1s[m_ind])/len(tr_L1s[m_ind]), 
                            'avg test L1', sum(ts_L1s[m_ind])/len(ts_L1s[m_ind])])
                w.writerow(['avg train L2', sum(tr_L2s[m_ind])/len(tr_L2s[m_ind]), 
                            'avg test L2', sum(ts_L2s[m_ind])/len(ts_L2s[m_ind])])


if __name__ == '__main__':
    run([gbr, rf, svm])
