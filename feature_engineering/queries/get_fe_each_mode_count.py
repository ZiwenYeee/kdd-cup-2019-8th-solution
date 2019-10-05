import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import count_and_merge
from Basic_function import timer
from sklearn.decomposition import TruncatedSVD

def get_fe_each_mode_count(data):
    data1 = data[['sid']]
    col_list = [
                # ['o', 'd'],
                ['o'] , ['d'],
                ['pid']
                # ['o', 'd'], ['o', 'pid'], ['d', 'pid'],
                # ['o', 'd', 'pid'],
                # ['pid', 'haversine'],['pid', 'manhattan'],['pid','bearing_array'],
                # ['haversine'],['manhattan'],['bearing_array'],
                
                # ['req_time_dayofweek', 'req_time_hour']

                ]
    g = count_and_merge(data, get_fe_different_mode_count, col_list)
    return g
    # data1[g.columns] = g
    # g = count_and_merge(data, get_fe_first_recom_count, [['o', 'd'],['o', 'd', 'pid']])
    # data1[g.columns] = g
    # data1.fillna(0, inplace = True)
    # data1.drop(['sid'], axis = 1, inplace = True)
    # return data1

def get_fe_different_mode_count(data, key):
    g = data.groupby(key).agg({
    k:['sum', 'mean'] for k in ["mode_flag_" + str(i) for i in range(1,12)] })
    g.columns = ["count_" + "_".join(key) + "_" + "_".join(col) for col in g.columns.ravel()]
    g.reset_index(inplace = True)
    g.fillna(0, inplace = True)
    # pca_cols = [col for col in g.columns if col not in key]
    # n = 3
    # svd = TruncatedSVD(n_components = n, n_iter = 100, random_state = 777)
    # df_svd = pd.DataFrame(svd.fit_transform(g[pca_cols].values))
    # df_svd.columns = ['svd_{}_{}'.format('counting_' + "_".join(key), i) for i in range(n)]
    # df_svd[key] = g[key]
    df = pd.merge(data[['sid'] + key], g, on = key, how = 'left')
    df.drop(key, axis = 1, inplace = True)
    df.drop(['sid'], axis = 1, inplace = True)
    return df

def get_fe_first_recom_count(data, key):
    data.pid.fillna(-1, inplace = True)
    df = data[['sid'] + key]
    g = data.groupby(key)[['recom_distance_0']].nunique()
    g.columns = ["_".join(key) + "_recom_distance_0" + '_nunique_count']
    # g = g/g.shape[0]
    g.reset_index(inplace = True)
    df = pd.merge(df, g, on = key, how = 'left')
    g = data.groupby(key)[['recom_eta_0']].nunique()
    g.columns = ["_".join(key) + "_recom_eta_0" + '_nunique_count']
    # g = g/g.shape[0]
    g.reset_index(inplace = True)
    df = pd.merge(df, g, on = key, how = 'left')
    g = data.groupby(key)[['recom_mode_0']].nunique()
    g.columns = ["_".join(key) + "_recom_mode_0" + '_nunique_count']
    # g = g/g.shape[0]
    g.reset_index(inplace = True)
    df = pd.merge(df, g, on = key, how = 'left')
    g = data.groupby(key)[['recom_price_0']].nunique()
    g.columns = ["_".join(key) + "_recom_price_0" + '_nunique_count']
    # g = g/g.shape[0]
    g.reset_index(inplace = True)
    df = pd.merge(df, g, on = key, how = 'left')

    df["_".join(key) + 'eta_dis_diff'] = df["_".join(key) + "_recom_eta_0" + '_nunique_count'] - \
                                            df["_".join(key) + "_recom_distance_0" + '_nunique_count']
    df["_".join(key) + 'eta_price_diff'] = df["_".join(key) + "_recom_eta_0" + '_nunique_count'] - \
                                            df["_".join(key) + "_recom_price_0" + '_nunique_count']
    df["_".join(key) + 'dis_price_diff'] = df["_".join(key) + "_recom_distance_0" + '_nunique_count'] - \
                                            df["_".join(key) + "_recom_price_0" + '_nunique_count']
    df.drop(key, axis = 1, inplace = True)
    df.drop(['sid'], axis = 1, inplace = True)
    return df


# group = get_fe_different_eta_count(data, col)
# df = pd.merge(df, group, on = ['sid'], how = 'left')

def get_fe_different_eta_count(data, key):
    g = data.groupby(key).agg({
    k:['mean', 'sum','min', 'max', 'std'] for k in ["eta_flag_" + str(i) for i in range(1,12)] })
    g.columns = ["eta_" + "_".join(key) + "_" + "_".join(col) for col in g.columns.ravel()]
    g.reset_index(inplace = True)
    g.fillna(0, inplace = True)
    df = pd.merge(data[['sid'] + key], g, on = key, how = 'left')
    df.drop([['sid'] + key], axis = 1, inplace = True)
    return df
