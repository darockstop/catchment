import numpy as np
from sklearn import svm as svm_r
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import csv



def train(clf, X_train, y_train, X_test, y_test, get_imp=False):
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    L1_tr = np.average(np.abs(np.subtract(pred_train, y_train)))
    L2_tr = np.average(np.square(np.subtract(pred_train, y_train)))
    
    pred_test = clf.predict(X_test)
    L1_ts = np.average(np.abs(np.subtract(pred_test, y_test)))
    L2_ts = np.average(np.square(np.subtract(pred_test, y_test)))
    
    with open('./output/pred.csv', 'a') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(pred_test)

    imp = clf.feature_importances_ if get_imp else None
    
    return L1_tr, L2_tr, L1_ts, L2_ts, imp


def rf(X_train, y_train, X_test, y_test, get_imp=False):
    forest = RandomForestRegressor(n_estimators=10)
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(forest, X_train, y_train, X_test, y_test, get_imp)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def gbr(X_train, y_train, X_test, y_test, get_imp=False):
    regressor = GradientBoostingRegressor(n_estimators=20)
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(regressor, X_train, y_train, X_test, y_test, get_imp)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def svm(X_train, y_train, X_test, y_test, get_imp=False):
    classifier = svm_r.SVR(gamma='scale')
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(classifier, X_train, y_train, X_test, y_test, False)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
