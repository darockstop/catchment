import numpy as np
from mltools import gbr, rf, svm
import matplotlib.pyplot as plt
from data import *



def plot_imp(imp, x_names, ax):
    imp = 100 * (imp / np.sum(imp))
    sorted_ind = np.argsort(imp)
    sorted_ind = sorted_ind[-10:]  # keep top 10 features only
    ax.barh(x_names[sorted_ind], imp[sorted_ind], color='#635a4d')


def run():
    imps = []
    df = get_data(for_training=True)

    f, axarr = plt.subplots(len(targets), 2, sharex='col')
    f.set_size_inches(18, 8)
    axarr[0, 0].set_title('Random Forest')
    axarr[0, 1].set_title('Gradient Boosting')


    for t_ind in range(len(targets)):

        axarr[t_ind, 0].annotate(targets[t_ind],xy=(0, 0.5), xytext=(-axarr[t_ind, 0].yaxis.labelpad-5,0), 
            xycoords=axarr[t_ind, 0].yaxis.label, textcoords='offset points', 
            size='large', ha='right', va='center')

        rf_imps = np.zeros(len(df.columns)-3)  # too big? small?
        gb_imps = np.zeros(len(df.columns)-3)

        for _ in range(33):
            kfolds = get_folds(df, 3)
            for i, fold in enumerate(kfolds):  # for each fold

                training, testing = preprocess(pd.concat(kfolds[:i] + kfolds[i + 1:]), fold)            
                X_train, y_train = training.drop(targets, axis=1), training[targets]
                X_test, y_test = testing.drop(targets, axis=1), testing[targets]
                X_train, X_test = X_train.drop('site', axis=1), X_test.drop('site', axis=1)

                _, _, _, _, rf_imp = rf(X_train, y_train.iloc[:, t_ind], X_test, y_test.iloc[:, t_ind], get_imp=True)
                _, _, _, _, gb_imp = gbr(X_train, y_train.iloc[:, t_ind], X_test, y_test.iloc[:, t_ind], get_imp=True)

                rf_imps += rf_imp
                gb_imps += gb_imp

        plot_imp(rf_imps, np.array(X_train.columns.values), axarr[t_ind, 0])
        plot_imp(gb_imps, np.array(X_train.columns.values), axarr[t_ind, 1])
    f.savefig('./output/graphs/ftr_imp_imputed.png')


run()
