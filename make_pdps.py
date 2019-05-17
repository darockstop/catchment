from matplotlib import pyplot as plt
from pdpbox import pdp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from data import *


def run(model, target, logfile):
    print('Using model: {}'.format(model), file=logfile)
    df = get_data(for_training=True)

    kfolds = get_folds(df, 3)
    for f_ind, fold in enumerate(kfolds):  # for each fold

        training, testing = preprocess(pd.concat(kfolds[:f_ind] + kfolds[f_ind + 1:]), fold)            
        X_train, y_train = training.drop(target, axis=1), training[target]
        X_test, y_test = testing.drop(target, axis=1), testing[target]
        X_train, X_test = X_train.drop('site', axis=1), X_test.drop('site', axis=1)

        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        L1_tr = np.average(np.abs(np.subtract(pred_train, y_train.values.ravel())))
        L1_ts = np.average(np.abs(np.subtract(pred_test, y_test.values.ravel())))
        print('fold {}: trL1: {} tsL1: {}'.format(f_ind+1, round(L1_tr, 3), round(L1_ts, 3)), file=logfile)

        imp = model.feature_importances_
        inds = np.argsort(imp)[::-1]  

        for ind in inds[:10]:
            pdp_goals = pdp.pdp_isolate(model=model, dataset=X_test, 
                model_features=X_train.columns.values, feature=X_train.columns.values[ind])

            f, ax = pdp.pdp_plot(pdp_goals, X_train.columns.values[ind])
            plt.savefig('./output/pdp/{}_{}_{}.png'.format(target, X_train.columns.values[ind], str(f_ind+1)))


if __name__ == '__main__':
    model = GradientBoostingRegressor(n_estimators=20)
    target = 'retentionP'
    logfilename = './output/pdp/{}_zlog.txt'.format(target)
    with open(logfilename, 'w') as logfile:
        run(model, target, logfile)
