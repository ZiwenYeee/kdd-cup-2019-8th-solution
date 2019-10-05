
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from Basic_function import timer


def data_cleaning(tr_q, tr_p, tr_c):
    data = tr_p.copy()
    with timer("basic:"):   
        tr_q = get_queries_basic(tr_q)

    for i in range(1):
        data['recom_mode_' + str(i)] = data['plans'].apply(lambda x: x[i]['transport_mode'] if len(x) > i else 0)
        data['recom_price_' + str(i)] = data['plans'].apply(lambda x: x[i]['price'] if len(x) > i else 0)
        data['recom_eta_' + str(i)] = data['plans'].apply(lambda x: x[i]['eta']//60 if len(x) > i else 0)
        data['recom_distance_' + str(i)] = data['plans'].apply(lambda x: x[i]['distance']//1000 if len(x) > i else 0)
        data['recom_price_' + str(i)].replace("", 0, inplace = True)
        data['recom_price_' + str(i)] = \
        np.where(data['recom_mode_' + str(i)] == 3, -2, data['recom_price_' + str(i)])
        data['recom_price_' + str(i)] = \
        np.where(data['recom_mode_' + str(i)] == 5, -1, data['recom_price_' + str(i)])

    data = data.merge(tr_q, on = ['sid'], how = 'left')
    data = data.merge(tr_c, on = ['sid'], how = 'left')
    data.click_mode.fillna(0, inplace = True)
    id_list = count_list(data)
    tr_q = tr_q.merge(id_list, on = ['sid'], how = 'left')
    tr_q.flag.fillna(0, inplace = True)
    tr_q = tr_q.loc[tr_q.flag == 0].reset_index(drop = True)
    tr_q.drop(['flag'], axis = 1, inplace = True)

    tr_p = tr_p.merge(id_list, on = ['sid'], how = 'left')
    tr_p.flag.fillna(0, inplace = True)
    tr_p = tr_p.loc[tr_p.flag == 0].reset_index(drop = True)
    tr_p.drop(['flag'], axis = 1, inplace = True)

    tr_c = tr_c.merge(id_list, on = ['sid'], how = 'left')
    tr_c.flag.fillna(0, inplace = True)
    tr_c = tr_c.loc[tr_c.flag == 0].reset_index(drop = True)
    tr_c.drop(['flag'], axis = 1, inplace = True)
    return tr_q, tr_p, tr_c


def count_list(data):
    var_list = [
        ['recom_price_0'],
        ['recom_eta_0'],
        ['recom_distance_0'],
        ['recom_price_0','recom_mode_0'],
        ['recom_eta_0','recom_mode_0'],
        ['recom_distance_0','recom_mode_0'],
        ['recom_price_0','click_mode'],
        ['recom_eta_0','click_mode'],
        ['recom_distance_0','click_mode']
    ]
    t = pd.DataFrame([])
    for col in var_list:
        g = count_cleaning(data, col)
        t = t.append(g)
    t = t.drop_duplicates(['sid'])
    t['flag'] = 1
    return t

def count_cleaning(data, key, scale = 1):
    g = data.groupby(key)[[key[0]]].transform('count')
    g['sid'] = data['sid']
    delete_idx = g.loc[g[key[0]] <= scale, ['sid']].reset_index(drop = True)
    return delete_idx


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
    def city_location(location):
        x = float(location.split(",")[0])
        y = float(location.split(",")[1])
        a = []
        a.append((x - 116.41) ** 2 + (y - 39.91) ** 2) #北京
        a.append((x - 121.43) ** 2 + (y - 31.20) ** 2) #上海
        a.append((x - 114.059560) ** 2 + (y - 22.542860) ** 2) #深圳
        a.append((x - 113.264360) ** 2 + (y - 23.129080) ** 2) #广州
        return a.index(min(a))
    
    if 'o' in queries:
        queries['o_x'] = queries['o'].apply(lambda x: float(x.split(',')[0]))
        queries['o_y'] = queries['o'].apply(lambda x: float(x.split(',')[1]))
        queries['d_x'] = queries['d'].apply(lambda x: float(x.split(',')[0]))
        queries['d_y'] = queries['d'].apply(lambda x: float(x.split(',')[1]))
    queries['o_count_totle'] = queries.groupby(['o'])['o'].transform('count')
    queries['d_count_totle'] = queries.groupby(['d'])['d'].transform('count')
    queries['city'] = queries['o'].apply(lambda x: city_location(x) ) 
    # queries = geohash_encode(queries, 'o_x', 'o_y', 'o')
    # queries = geohash_encode(queries, 'd_x', 'd_y', 'd')
    return queries