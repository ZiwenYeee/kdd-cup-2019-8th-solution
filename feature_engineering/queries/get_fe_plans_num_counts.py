import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Basic_function import count_and_merge


#don't use pid.
def get_fe_plans_num_counts(data):
    # data = pd.merge(queries, plans, on = ['sid'], how = 'left')
    data['values'] = 1.0
    col_list = [['o'], ['d'], ['o', 'd'], ['pid'], 
            ['o', 'pid'], ['d', 'pid'], ['o', 'd', 'pid'],
            # ['haversine'],['manhattan'],['bearing_array'],

            ]
    # data1 = data[['sid']]
    # data1 = pd.DataFrame([])
    g = count_and_merge(data, get_fe_plans_num, col_list)
    return g
    # data1[g.columns] = g
    # g = count_and_merge(data, get_fe_plans_num_count, col_list)
    # data1[g.columns] = g
    # g = count_and_merge(data, get_fe_plans_num_nunique, col_list)
    # data1[g.columns] = g
    # data1.drop(['sid'], axis = 1, inplace = True)
    # return data1

def get_fe_plans_num_nunique(data, key):
    od_mode_feat = data.groupby(key)[['recom_num']].nunique()
    od_mode_feat.columns = ["_".join(key) + '_plans_num_nunique_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat.reset_index(inplace = True)
    df = pd.merge(data[['sid'] + key], od_mode_feat, on = key, how = 'left')
    df.drop(key, axis = 1, inplace = True)
    df.drop(['sid'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df

def get_fe_plans_num(data, key):
    od_mode_feat = pd.pivot_table(data,index = key,values=['values'],columns=['recom_num'],aggfunc='sum')
    od_mode_feat.columns = ["_".join(key) + '_plans_num_' + str(x[1]) for x in od_mode_feat.columns]
    df = pd.merge(data[['sid'] + key], od_mode_feat, on = key, how = 'left')
    df.drop(key, axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df

def get_fe_plans_num_count(data, key):
    od_mode_feat = pd.pivot_table(data,index = key,values=['recom_num'],columns=['values'],aggfunc='sum')
    od_mode_feat.columns = ["_".join(key) + '_plans_num_count_' + str(x[1]) for x in od_mode_feat.columns]
    df = pd.merge(data[['sid'] + key], od_mode_feat, on = key, how = 'left')
    df.drop(key, axis = 1, inplace = True)
    df.drop(['sid'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df
