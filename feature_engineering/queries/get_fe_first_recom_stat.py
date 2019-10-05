import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import count_and_merge
from sklearn.decomposition import TruncatedSVD

def get_fe_first_recom_stat(data):
    # data1 = data[['sid']]
    col_list = [['pid'],
            
            # ['recom_mode_0', 'city']
            # ['city','pid']
             ]
    g = count_and_merge(data, get_fe_id_stat, col_list)
    # data1[g.columns] = g
    # data1.drop(['sid'], axis = 1, inplace = True)
    return g

def get_fe_id_stat(data, key):
    temp = data.groupby(key).agg({k:['mean', 'min', 'max', 'sum', 'std']
                                      for k in ['distance','eta','price',
                                      # 'distance_eta','price_eta','distance_price',
                                        'manhattan'
                                               
                                               ]})
    temp.columns = ["_".join(key) + '_level_' + "_".join(j) for j in temp.columns.ravel()]
    temp.reset_index(inplace = True)
    # temp.fillna(0, inplace = True)
    # pca_cols = [col for col in temp.columns if col not in key]
    # n = 5
    # svd = TruncatedSVD(n_components = n, n_iter = 50, random_state = 777)
    # df_svd = pd.DataFrame(svd.fit_transform(temp[pca_cols].values))
    # df_svd.columns = ['svd_{}_{}'.format('transport_' + "_".join(key), i) for i in range(n)]
    # df_svd[key] = temp[key]
    # df = pd.merge(data[['sid'] + key], df_svd, on = key, how = 'left')
    
    df = pd.merge(data[['sid'] + key], temp, on = key, how = 'left')
    df.drop(key, axis = 1, inplace = True)
    df.drop(['sid'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df
