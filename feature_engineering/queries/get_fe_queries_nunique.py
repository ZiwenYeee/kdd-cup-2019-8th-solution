import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import gc
from Basic_function import timer
import multiprocessing as mp
from joblib import Parallel, delayed

def get_fe_queries_nunique(queries):
    unique_encoding = [
    (['pid'],['o', 'd']),
    (['pid', 'o'], ['d']),
    (['pid', 'd'], ['o']),
    (['o'], ['d','pid']),
    (['o', 'd'], ['pid']),
    (['d'], ['o', 'pid']),
    (['pid'], ['haversine','manhattan','bearing_array']),
    # (['o', 'req_time_hour'], ['d','haversine','manhattan','bearing_array']),
    # (['o','req_time_weekofyear'], ['d','haversine','manhattan','bearing_array']),
    ]
    
    t1 = time.time()
    res = Parallel(n_jobs=-1, require='sharedmem', verbose=0) \
            (delayed(get_fe_position_count)(queries, ids[0], ids[1]) for ids in unique_encoding)
    ans = pd.concat(res, axis = 1)
    t2 = time.time()
    print('time = ', t2 - t1)
    del res
    gc.collect()
    return ans

def get_fe_position_count(queries, key, merge_list):
    data = queries[['sid'] + key]
    for col in merge_list:
        temp = queries.groupby(key)[[col]].nunique()
        temp.columns = ["_".join(key) + "_" + col + '_nunique_count']
        temp["_".join(key) + "_" + col + '_nunique_count'] = \
        temp["_".join(key) + "_" + col + '_nunique_count']/temp.shape[0]
        temp.reset_index(inplace = True)
        data = data.merge(temp, on = key, how = 'left')
    data.drop(key, axis = 1, inplace = True)
    data.fillna(0, inplace = True)
    return data
