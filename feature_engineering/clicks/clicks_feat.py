
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import time
from Basic_function import timer
from sklearn.model_selection import KFold, StratifiedKFold
from Basic_function import get_datetime
from multiprocessing import cpu_count, Pool

def clicks_feat(train, test, clicks, train_queries, test_queries, train_plans, test_plans):
    train_queries = get_queries_basic(train_queries)
    test_queries = get_queries_basic(test_queries)
    
    data = pd.merge(train_queries, clicks, on = ['sid'], how = 'left')
    
    data.click_mode.fillna(0, inplace = True)
    data.pid.fillna(-1, inplace = True)
    data['values'] = 1
    with timer("time ratio encoding:"):
        key = ['o_geohash', 'd_geohash']
        df = oof_fe_by(data, key, 'cnt')
        tr = pd.merge(train_queries[['sid'] + key], df, on = key, how = 'left')
        ts = pd.merge(test_queries[['sid'] + key], df, on = key, how = 'left')
        train = pd.merge(train, tr, on = ['sid'], how = 'left')
        test = pd.merge(test, ts, on = ['sid'], how = 'left')
    return train, test

def oof_fe_by(data, key, op = 'cnt'):
    tr = pd.DataFrame()
    folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)
    for n_fold, (tr_idx, ts_idx) in enumerate(folds.split(data, data['click_mode'])):
        sys.stdout.write('{},'.format(n_fold))
        X_train, X_test = data.iloc[tr_idx], data.iloc[ts_idx]
        encode = pd.pivot_table(X_train, index= key,values=['values'],columns=['click_mode'],aggfunc='sum')
        encode.columns = ["_".join(key) + "_" + str(col[1]) for col in encode.columns.ravel()]
        tr = tr.append(encode)
    tr = tr.groupby(key).mean()
    if op == 'ratio':
        ori_col = tr.columns
        for col in ori_col:
            tr[col] = tr[col]/tr[ori_col].sum(axis = 1)
    tr.reset_index(inplace = True)
    return tr

def get_queries_basic(queries):
    df_all = parallelize(queries, queries_extend)
    return df_all

def parallelize(data, func):
    cores = cpu_count() #Number of CPU cores on your system
    partitions = cores #Define as many partitions as you want
    data_split = np.array_split(data, partitions - 1)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def queries_extend(queries):
    def sid_location(x):
        if x < 499999:
            return 0
        elif x < 1000000:
            return 1
        elif x < 1500000:
            return 3
        elif x < 2000000:
            return 2
        elif x < 2094358:
            return 0
        elif x < 2180312:
            return 1
        elif x < 2304916:
            return 3
        else:
            return 2
    # def city_location(location):
    #     x = float(location.split(",")[0])
    #     y = float(location.split(",")[1])
    #     a = []
    #     a.append((x - 116.41) ** 2 + (y - 39.91) ** 2) #北京
    #     a.append((x - 121.43) ** 2 + (y - 31.20) ** 2) #上海
    #     a.append((x - 114.059560) ** 2 + (y - 22.542860) ** 2) #深圳
    #     a.append((x - 113.264360) ** 2 + (y - 23.129080) ** 2) #广州
    #     return a.index(min(a))
    
    if 'o' in queries:
        queries['o_x'] = queries['o'].apply(lambda x: float(x.split(',')[0]))
        queries['o_y'] = queries['o'].apply(lambda x: float(x.split(',')[1]))
        queries['d_x'] = queries['d'].apply(lambda x: float(x.split(',')[0]))
        queries['d_y'] = queries['d'].apply(lambda x: float(x.split(',')[1]))
    queries['o_count_totle'] = queries.groupby(['o'])['o'].transform('count')
    queries['d_count_totle'] = queries.groupby(['d'])['d'].transform('count')
    queries['city'] = queries['sid'].apply(lambda x: sid_location(x) ) 
    # queries['city'] = queries['o'].apply(lambda x: city_location(x) ) 
    queries = geohash_encode(queries, 'o_x', 'o_y', 'o', 5)
    queries = geohash_encode(queries, 'd_x', 'd_y', 'd', 5)
    return queries
