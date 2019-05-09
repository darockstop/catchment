from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mltools import *


def run(datafile, model, target, logfile):
    print('Using model: {}'.format(model), file=logfile)
    with open(datafile, 'r') as data_file:
        df = pd.read_csv(data_file)
        print('Starting with {} samples and {} features'.format(df.shape[0], df.shape[1]))
        df = df.drop(['QMar2016', 'QNov2015', 'QJune2018'], axis=1)
        df = df.dropna(subset=['retentionN'])
        df = df[df['retentionN'] != 0]

    xs = df.iloc[:, 4:-2]  # 4 because don't want some beg stuff or trgts
    xs = xs.dropna(axis=1)
    x_names = xs.columns.values

    t = df.iloc[:, -2]  # -2 is nretention, -1 is pretention

    xs = np.array(xs)
    t = np.array(t)

    num_samples, num_ftrs = xs.shape
    print('using {} samples and {} features'.format(num_samples, num_ftrs))
    kfolds = get_folds(3, (xs, t.reshape(num_samples, 1)))
    for f_ind, fold in enumerate(kfolds):  # for each fold
        xs_test, y_test = fold[:, :num_ftrs], fold[:, num_ftrs:]  # split inputs and outputs
        training_data = kfolds[:f_ind] + kfolds[f_ind + 1:]
        training_data = np.vstack(training_data)
        xs_train, y_train = training_data[:, :num_ftrs], training_data[:, num_ftrs:]  # remainder for 
        y_test = y_test.ravel()
        y_train = y_train.ravel()

        xs_train = pd.DataFrame(xs_train, columns=x_names)
        xs_test = pd.DataFrame(xs_test, columns=x_names)
        y_train = pd.DataFrame(y_train, columns=[target])
        y_test = pd.DataFrame(y_test, columns=[target])

        # import pdb; pdb.set_trace()

        model.fit(xs_train, y_train)
        pred_train = model.predict(xs_train)
        pred_test = model.predict(xs_test)
        L1_tr = np.average(np.abs(np.subtract(pred_train, y_train.values.ravel())))
        L1_ts = np.average(np.abs(np.subtract(pred_test, y_test.values.ravel())))
        print('fold {}: trL1: {} tsL1: {}'.format(f_ind+1, round(L1_tr, 3), round(L1_ts, 3)), file=logfile)

        imp = model.feature_importances_
        inds = np.argsort(imp)[::-1]  

        for ind in inds[:5]:
            pdp_goals = pdp.pdp_isolate(model=model, dataset=xs_test, model_features=x_names, feature=x_names[ind])

            f, ax = pdp.pdp_plot(pdp_goals, x_names[ind])
            plt.savefig('./output/pdp/{}_{}_{}.png'.format(target, ind, str(f_ind+1)))


if __name__ == '__main__':
    import sys
    datafile = sys.argv[1]
    model = GradientBoostingRegressor(n_estimators=50)
    target = 'retentionN'
    logfilename = './output/pdp/{}_log.txt'.format(target)
    with open(logfilename, 'w') as logfile:
        run(datafile, model, target, logfile)
