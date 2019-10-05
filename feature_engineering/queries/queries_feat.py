import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import gc
import geohash2 as geohash
from multiprocessing import cpu_count, Pool

from Basic_function import timer
from Basic_function import get_datetime
from Basic_function import parallelize
from queries.profiles_feat import get_fe_profiles
from queries.time_diff_feat import time_diff_feat
from queries.get_fe_first_recom_stat import get_fe_first_recom_stat
from queries.get_fe_different_dist import get_fe_different_dist
from queries.get_fe_queries_cross_count import get_fe_queries_cross_count
from queries.get_fe_queries_nunique import get_fe_queries_nunique
from queries.get_fe_loc_mode_count import get_fe_loc_mode_count
from queries.get_fe_plans_num_counts import get_fe_plans_num_counts
from queries.get_fe_each_mode_count import get_fe_each_mode_count
from queries.get_fe_encode_geo_time import get_fe_encode_geo_time


def queries_feat(train, test, train_plans, train_queries, test_plans, test_queries, profiles):
    with timer("basic:"):
        train_num = train.shape[0]
        df = pd.concat([train[['sid']], test[['sid']]]).reset_index(drop = True)
        plans = pd.concat([train_plans, test_plans]).reset_index(drop = True)
        plans = get_plans_basic(plans)
        queries = pd.concat([train_queries, test_queries]).reset_index(drop = True)    
        queries = get_queries_basic(queries)
    with timer("get request time feature"):
        time_feature = get_datetime(queries, 'req_time')
        df[time_feature.columns] = time_feature
        queries[time_feature.columns] = time_feature
        queries.pid.fillna(-1, inplace = True)
        del time_feature
        gc.collect()
    with timer("profiles feature:"):
        pro_feat = get_fe_profiles(profiles)
        pro_feat = pd.merge(queries[['sid', 'pid']], pro_feat, on = ['pid'], how = 'left')
        queries = pd.merge(queries, pro_feat, on = ['sid','pid'], how = 'left')
        df = pd.merge(df, pro_feat, on = ['sid'], how = 'left')
        del pro_feat
        gc.collect()
    with timer("time diff feature:"):
        time_feat = time_diff_feat(queries, plans)
        queries[time_feat.columns] = time_feat
        df[time_feat.columns] = time_feat
        del time_feat
        gc.collect()
    with timer("distance feature:"):
        distance_feat = get_fe_different_dist(queries)
        queries[distance_feat.columns] = distance_feat
        df[distance_feat.columns] = distance_feat
        del distance_feat
        gc.collect()
    
    # queries_new = pd.get_dummies(queries['city'], columns= ['city'], dummy_na= False)
    # queries[queries_new.columns] = queries_new
    # df[queries_new.columns] = queries_new
    data = pd.merge(queries, plans, on = ['sid'], how = 'left')
    del plans;gc.collect()
    queries = queries[['sid', 'o_x', 'o_y', 'd_x', 'd_y','o_count_totle','d_count_totle','city']]
    df['recom_distance_haversine'] = data['recom_distance_0']/(data['haversine'] + 0.01)
    df['recom_distance_manhanttan'] = data['recom_distance_0']/(data['manhattan'] + 0.01)
    df['recom_distance_bearing_array'] = data['recom_distance_0']/(data['bearing_array'] + 0.01)

    data['recom_distance_haversine'] = data['recom_distance_0']/(data['haversine'] + 0.01)
    data['recom_distance_manhanttan'] = data['recom_distance_0']/(data['manhattan'] + 0.01)
    data['recom_distance_bearing_array'] = data['recom_distance_0']/(data['bearing_array'] + 0.01)

    with timer("get_fe_first_recom_stat feature:"):
        pid_stat = get_fe_first_recom_stat(data)
        df[pid_stat.columns] = pid_stat
    with timer("get_fe_queries_cross_count count:"):
        cross_count = get_fe_queries_cross_count(data)
        df[cross_count.columns] = cross_count
    with timer("get_fe_queries_nunique nunique:"):
        nunique = get_fe_queries_nunique(data)
        df[nunique.columns] = nunique
    with timer("get_fe_location_mode_count mode count:"):
        mode_loc = get_fe_loc_mode_count(data)
        df[mode_loc.columns] = mode_loc
    # with timer("get_fe_each_mode_count:"):
    #     mode_count = get_fe_each_mode_count(data)
    #     df[mode_count.columns] = mode_count
    # with timer("geo time encoding feature:"):
    #     geo_feat = get_fe_encode_geo_time(queries)
    #     df[geo_feat.columns] = geo_feat
    #     del geo_feat
    #     gc.collect()
    # with timer("get_fe_plans_num_counts:"):
    #     num = get_fe_plans_num_counts(data)
    #     df[num.columns] = num    
    del data;gc.collect()
    
    basic_feat = ['sid', 'o_x', 'o_y', 'd_x', 'd_y','o_count_totle','d_count_totle','city']
    df = pd.merge(df, queries[basic_feat], on = ['sid'], how = 'left')
    df.drop(['sid'], axis = 1, inplace = True)
    train[df.columns] = df[:train_num].reset_index(drop = True)
    test[df.columns] = df[train_num:].reset_index(drop = True)
    return train, test

def get_plans_basic(plans):
    df_all = parallelize(plans, plans_extend)
    return df_all

def get_queries_basic(queries):
    df_all = parallelize(queries, queries_extend)
    return df_all

def queries_extend(queries):
    # def sid_location(x):
    #     if x < 499999:
    #         return 0
    #     elif x < 1000000:
    #         return 1
    #     elif x < 1500000:
    #         return 3
    #     elif x < 2000000:
    #         return 2
    #     elif x < 2094358:
    #         return 0
    #     elif x < 2180312:
    #         return 1
    #     elif x < 2249266:
    #         return 3
    #     else:
    #         return 2
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
    # queries['city'] = queries['sid'].apply(lambda x: sid_location(x) ) 
    queries['city'] = queries['o'].apply(lambda x: city_location(x) ) 
    # queries = geohash_encode(queries, 'o_x', 'o_y', 'o', 5)
    # queries = geohash_encode(queries, 'd_x', 'd_y', 'd', 5)
    return queries

def plans_extend(plans):
    def basic_func(x):
        a = np.zeros((1,12))
        for j in x:
            a[0,j['transport_mode']] += 1
        return a.tolist()[0]
    def eta_func(x):
        a = np.zeros((1,12))
        b = np.zeros((1,12))
        for j in x:
            a[0,j['transport_mode']] += 1
            b[0,j['transport_mode']] += j['eta']/60
            # b[0,j['transport_mode']] += j['eta']/( (a[0,j['transport_mode']] + 1) * 60)
        return b.tolist()[0]
    plans['flag_list'] = plans['plans'].apply(lambda x: basic_func(x))
    for i in range(5):
        plans['recom_mode_' + str(i)] = plans['plans'].apply(lambda x: x[i]['transport_mode'] if len(x) > i else 0)
        plans['recom_price_' + str(i)] = plans['plans'].apply(lambda x: x[i]['price'] if len(x) > i else 0)
        plans['recom_eta_' + str(i)] = plans['plans'].apply(lambda x: x[i]['eta']/60 if len(x) > i else 0)
        plans['recom_distance_' + str(i)] = plans['plans'].apply(lambda x: x[i]['distance']/1000 if len(x) > i else 0)
        plans['recom_price_' + str(i)].replace("", 0, inplace = True)
        plans['recom_price_' + str(i)] = \
        np.where(plans['recom_mode_' + str(i)] == 3, -2, plans['recom_price_' + str(i)])
        plans['recom_price_' + str(i)] = \
        np.where(plans['recom_mode_' + str(i)] == 5, -1, plans['recom_price_' + str(i)])  
    plans['recom_num'] = plans['plans'].apply(lambda x: len(x))
    for i in range(1,12):
        plans['mode_flag_' + str(i)] = plans.flag_list.apply(lambda x: 1 if x[i] > 0 else 0)
    plans['price'] = plans['plans'].apply(lambda x: x[0]['price'] if x[0]['price'] != "" else 0)
    plans['eta'] = plans['plans'].apply(lambda x: x[0]['eta'])
    plans['distance'] = plans['plans'].apply(lambda x: x[0]['distance'])
    plans['plans_num'] = plans['plans'].apply(lambda x: len(x))
       
    plans['distance_eta'] = plans['distance'] / (plans['eta'] + 0.01)
    plans['price_eta'] = plans['price'] / (plans['eta'] + 0.01)
    plans['distance_price'] = plans['distance'] / (plans['price'] + 0.01)
    return plans

def geohash_encode(df, longitude, latitude, prefix="", precision=9):
    df_copy = df.copy()
    if prefix:
        prefix = prefix + "_"
    df_copy[prefix + 'geohash'] = df_copy[[longitude, latitude]].apply(lambda x: geo(x, precision), axis=1)
    return df_copy


def geo(lat_lon, precision):
    longitude = lat_lon[0]
    latitude = lat_lon[1]
    return geohash.encode(latitude, longitude, precision)
