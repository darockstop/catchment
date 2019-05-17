import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer



data_path = './data'
targets = ['retentionN', 'retentionP']


def get_data(for_training=False):
    with open(os.path.join(data_path, 'MLDatabase2.csv'), 'r') as data_file:
        df = pd.read_csv(data_file)

    if for_training:
        df = df.dropna(subset=targets)  # if no target, drop
        df = df.drop(['sort', 'catchment', 'sampling', 'date', 'time', 'code', 'cation'], axis=1)
        df = df.drop(['QJune2018', 'QMar2016', 'QNov2015', 'no3rank'], axis=1)
        # df = df.dropna()
        return df

    else:
        return df


def preprocess(train, valid):

    train, obj_train = train.select_dtypes(exclude=['object']), train.select_dtypes(['object'])
    valid, obj_valid = valid.select_dtypes(exclude=['object']), valid.select_dtypes(['object'])

    # impute with median
    imputer = SimpleImputer(strategy='median')
    imp_train = pd.DataFrame(imputer.fit_transform(train))
    imp_train.columns = train.columns
    imp_train.index = train.index
    imp_valid = pd.DataFrame(imputer.transform(valid))
    imp_valid.columns = valid.columns
    imp_valid.index = valid.index

    imp_train_norm = ((imp_train - imp_train.min()) / (imp_train.max() - imp_train.min()))  # norm train 0-1
    imp_valid_norm = ((imp_valid - imp_train.min()) / (imp_train.max() - imp_train.min()))  # norm valid according to train

    imp_train_norm = pd.concat([imp_train_norm, obj_train], axis=1)
    imp_valid_norm = pd.concat([imp_valid_norm, obj_valid], axis=1)

    return imp_train_norm, imp_valid_norm


def get_folds(data, k):
    unique_sites = set(data['site'])  # get unique sites
    kfolds_sites = np.array_split(list(unique_sites), k)  # split sites into k
    kfolds = [data.loc[data['site'].isin(vals)] for vals in kfolds_sites]  # get k separate dataframes
    return kfolds

