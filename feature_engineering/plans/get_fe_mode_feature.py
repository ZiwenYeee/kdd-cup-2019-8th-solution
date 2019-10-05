import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from Basic_function import timer
import gc
def get_fe_mode_feature(plans):
    data = pd.DataFrame([])
    plans['mode'] = plans['plans'].apply(lambda x: x[0]['transport_mode'])
    df = parallel_expand(plans)
    df.replace("", 0, inplace = True)
    stat1 = mode_feature(plans, df, ['mode'])
    data[stat1.columns] = stat1
    return data
    

def mode_feature(main, df, key):
    data = main[['sid'] + key]
    stat1 = df.groupby(key).agg({k:['mean', 'min', 'max', 'std', 'sum', 'skew'] for k in ['price', 'distance', 'eta']})
    stat1.columns = ["_".join(col) for col in stat1.columns.ravel()]
    stat1.reset_index(inplace = True)
    data = pd.merge(data, stat1, on = key, how = 'left')
    data.drop(key, axis = 1, inplace = True)
    return data


def expand_func(df):
    tdf = df.set_index(['sid'])\
        .plans.apply(pd.Series).stack()\
        .reset_index(level=-1, drop=True)\
        .reset_index()\
        .rename(columns={0:'plan'})
    
    def list2dic(row):
        if isinstance(row.plan, dict):
            return row['plan']['distance'], row['plan']['price'], \
                    row['plan']['eta'], row['plan']['transport_mode']
        else:
            return np.nan, np.nan, np.nan, np.nan
    
    tdf = tdf.join(
            tdf.apply(lambda x: list2dic(x), axis = 1).apply(pd.Series)\
            .rename(columns = {0: 'distance', 1: 'price', 2: 'eta', 3: 'mode'})
    )

    return tdf


from multiprocessing import Pool
def parallel_expand(df, nparallel = 12):
    t1 = time.time()
    df_list = [df.loc[idx] for idx in np.array_split(df.index, nparallel)]
    
    pool = Pool(nparallel)
    result = pool.map(expand_func, df_list)
    pool.close()
    pool.join()
    
    t2 = time.time()
    print('paralle_expand time = ', t2 - t1)
    
    ans = pd.concat(result, axis = 0)
    ans.reset_index(drop = True, inplace = True)
    del result
    
    gc.collect()
    return ans

