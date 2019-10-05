import time
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from contextlib import contextmanager
import multiprocessing as mp
from joblib import Parallel, delayed
from multiprocessing import cpu_count, Pool

import warnings
warnings.filterwarnings('ignore')

def parallelize(data, func, nparallel = 5):
    nparallel = nparallel
    data_split = np.array_split(data, nparallel)
    pool = Pool(nparallel)
    result = pool.map(func, data_split)
    pool.close()
    pool.join()
    ans = pd.concat(result, axis = 0)
    del result; gc.collect();
    return ans


def count_and_merge(df, func, ids_list):
    t1 = time.time()
    res = Parallel(n_jobs=-1, require='sharedmem', verbose=0) \
            (delayed(func)(df, ids) for ids in ids_list)
    ans = pd.concat(res, axis = 1)
    t2 = time.time()
    print('time = ', t2 - t1)
    del res
    gc.collect()
    return ans

def cross_count_helper(df, i):
    name = "count_" + "_".join(i)
    # df[name] = df.groupby(i)[i[0]].transform('count')
    df[name] = df.groupby(i)[i[0]].transform('count')/df.shape[0]
    return df[[name]]

def linear_split(X, feature):
    X['req_time'] = pd.to_datetime(X['req_time'])

    X_train = X.loc[X['req_time'] < '2018-11-23', feature].reset_index(drop = True)
    X_valid = X.loc[(X['req_time'] > '2018-11-23') &
                    (X['req_time'] < '2018-12-01'), feature].reset_index(drop = True)

    y_train = X.loc[X['req_time'] < '2018-11-23', 'click_mode'].reset_index(drop = True)
    y_valid = X.loc[(X['req_time'] > '2018-11-23') &
                    (X['req_time'] < '2018-12-01'), 'click_mode'].reset_index(drop = True)
    return X_train, X_valid, y_train, y_valid


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def correlation_reduce(df, threshold = 0.8):
    threshold = threshold
    print('Original Training shape: ', df.shape)
    # Absolute value correlation matrix
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) if column not in ['SK_ID_CURR']]
    print('There are %d columns to remove.' % (len(to_drop)))
    df.drop(to_drop,axis = 1, inplace = True)
    print('Training shape: ', df.shape)
    return df

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def get_datetime(train, col, f = "%Y-%m-%d %H:%M:%S"):
    train_fe = pd.DataFrame([])
    s = pd.to_datetime(train[col])
    train_fe[col + '_dayofweek'] = s.dt.dayofweek
    train_fe[col + '_weekend'] = train_fe[col + '_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    train_fe[col + '_hour'] = s.dt.hour
    train_fe[col + '_elapsed_time'] = s.dt.hour * 60 + s.dt.minute
    if col == 'req_time':
        train_fe[col + '_minute'] = s.dt.minute
        train_fe[col + '_weekofyear'] = s.dt.weekofyear
    train_fe[col + '_hour_4'] = s.dt.hour // 4
    train_fe[col + '_hour_05_10'] = train_fe[col + '_hour'].apply(lambda x: 1 if x >= 5 and x < 10 else 0)
    train_fe[col + '_hour_10_16'] = train_fe[col + '_hour'].apply(lambda x: 1 if x >= 10 and x < 16 else 0)
    train_fe[col + '_hour_16_20'] = train_fe[col + '_hour'].apply(lambda x: 1 if x >= 16 and x < 20 else 0)
    train_fe[col + '_hour_20_05'] = train_fe[col + '_hour'].apply(lambda x: 1 if x >= 20 or x < 5 else 0)
    return train_fe
