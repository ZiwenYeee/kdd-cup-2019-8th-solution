import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import gc
import geohash2 as geohash
from multiprocessing import cpu_count, Pool

from Basic_function import timer
from Basic_function import get_datetime
def get_fe_lag_feature(data):
    lag_col = [
        ['req_time','pid'],# 1.时间。 2.类别维度，连续两次行为时间差
        ]
    g = lag_feature(data, lag_col)
    return g

def concat(L):
    """
    tools for concat new dataframe
    """
    result = None
    for l in L:
        if l is None:
            continue
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except Exception as err:
                print(err)
                print(l.head())
    return result

def universe_mp_generator(gen_func, feat_list):
    """
    tools for multy thread generator
    """
    pool = Pool(7)
    result = [pool.apply_async(gen_func, feats) for feats in feat_list]
    pool.close()
    pool.join()
    return [aresult.get() for aresult in result]
    
def lag_id_helper(df, time_col, col):
    df['ke_cnt_' + col] = df.groupby(col)[time_col].rank(ascending=False,method = 'first')
    df2 = df[[col, 'ke_cnt_' + col, time_col]].copy()
    df2['ke_cnt_' + col] = df2['ke_cnt_' + col] - 1
    df3 = pd.merge(df, df2, on=[col, 'ke_cnt_' + col], how='left')
    df['LAG_{}_{}'.format(col, time_col)] = (df3[time_col +'_x'] - df3[time_col + '_y'])
    df['LAG_{}_{}'.format(col, time_col)]  = df['LAG_{}_{}'.format(col, time_col)] .values.astype(np.int64) // 10 ** 9
    del df2,df3
    gc.collect()
    return df[['LAG_{}_{}'.format(col, time_col)]]


def lag_feature(df, col):
    num_df0 = df.shape[1]
    count_columns = []
    for i in col:
        count_columns.append([df[i], i[0], i[1]])
    count_result = universe_mp_generator(lag_id_helper, count_columns)
    del count_columns
    gc.collect()
    count_result.append(df)
    df = concat(count_result)
    num_df1 = df.shape[1]
    print("number of LAG feature is {}".format(num_df1 - num_df0))
    return df
