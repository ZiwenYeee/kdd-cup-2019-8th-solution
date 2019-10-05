import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import count_and_merge
from sklearn.decomposition import TruncatedSVD

#speical
#haversine
def get_fe_loc_mode_count(data):
    data['values'] = 1.0
    col_list = [
        ['o'] , ['d'], ['pid'],
                # ['o', 'd'], 
        # ['o', 'pid'], ['d', 'pid'],
                # ['o', 'd', 'pid'],
                # ['city', 'pid'],
                # ['haversine'],['manhattan'],['bearing_array'],

                ]
    g = count_and_merge(data, get_fe_transport_count, col_list)
    return g

def get_fe_transport_count(data, key):
    od_mode_feat = pd.pivot_table(data,index= key,values=['values'],columns=['recom_mode_0'],aggfunc='sum')
    od_mode_feat.columns = ["_".join(key) + '_transport_mode_' + str(x[1]) for x in od_mode_feat.columns]
    # od_mode_feat = data.groupby(key + ['recom_mode_0']).agg({"values":['mean', 'sum']})
    od_mode_feat.fillna(0, inplace = True)
    # col_ = od_mode_feat.columns
    # sum_ = od_mode_feat.sum(axis = 1)
    # for i in col_:
    #     od_mode_feat[i + "_mean"] = od_mode_feat[i]/sum_
    od_mode_feat.reset_index(inplace = True)
    # pca_cols = [col for col in od_mode_feat.columns if col not in key]
    # n = 3
    # svd = TruncatedSVD(n_components = n, n_iter = 50, random_state = 777)
    # df_svd = pd.DataFrame(svd.fit_transform(od_mode_feat[pca_cols].values))
    # df_svd.columns = ['svd_{}_{}'.format('transport_' + "_".join(key), i) for i in range(n)]
    # df_svd[key] = od_mode_feat[key]
    # df = pd.merge(data[['sid'] + key], df_svd, on = key, how = 'left')
    df = pd.merge(data[['sid'] + key], od_mode_feat, on = key, how = 'left')
    df.drop(key, axis = 1, inplace = True)
    df.drop(['sid'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df
