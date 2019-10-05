import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from Basic_function import cross_count_helper
from Basic_function import count_and_merge
from sklearn.decomposition import TruncatedSVD

# def cross_count_helper(df, i):
#     # name = "count_" + "_".join(i)
#     # g = pd.DataFrame([])
#     temp = df.groupby(i)[i[0]].count().unstack()
#     temp.fillna(0, inplace = True)
#     temp.columns = ["_".join(i) +"_count_" + str(city) for city in range(4)]
#     i.remove('city')
#     svd = TruncatedSVD(n_components = 1, n_iter = 50, random_state = 777)
#     df_svd = pd.DataFrame(svd.fit_transform(temp.values))
#     df_svd.columns = ['svd_{}_{}'.format('_'.join(i), 1)]
#     temp.reset_index(inplace = True)
#     df_svd[i] = temp[i]
#     g = pd.merge(df[i], df_svd, on = i, how = 'left')
#     g.drop(i, axis = 1, inplace = True)
#     # g[name] = df.groupby(i)[i[0]].transform('count')
#     # g[name] = df.groupby(i)[i[0]].transform('count')/df.shape[0]
#     return g
                             
def cross_count_helper(df, i):
    name = "count_" + "_".join(i)
    g = pd.DataFrame([])
    # g[name] = df.groupby(i)[i[0]].transform('count')
    g[name] = df.groupby(i)[i[0]].transform('count')/df.shape[0]
    return g

def get_fe_queries_cross_count(data):
    cross_count = [
    ['o'],
    ['d'],
    ['o','d'],
    ['o','pid'],
    ['d','pid'],
    ['o','d','pid'],
    ['req_time_hour', 'o', 'd'],
    ['req_time_hour', 'o'],
    ['req_time_hour', 'd'],
    ['req_time_hour', 'pid'],
    ['o_x'],
    ['o_y'],
    ['d_x'],
    ['d_y'],
    ['haversine', 'o', 'd'],
    ['haversine', 'o'],
    ['haversine', 'd'],
    ['haversine', 'pid'],
    ['haversine'],
    ['manhattan'],
    ['bearing_array'],

    ['o', 'recom_mode_0'],
    ['d', 'recom_mode_0'],
    ['o','d', 'recom_mode_0'],
    ['o','pid', 'recom_mode_0'],
    ['d','pid', 'recom_mode_0'],
    ['o','d','pid', 'recom_mode_0'],
        
    # ['req_time_dayofweek','req_time_hour'],
    # ['req_time_weekend', 'req_time_hour'],
        
    ['haversine', 'recom_mode_0'],
    ['manhattan', 'recom_mode_0'],
    ['bearing_array', 'recom_mode_0'],
    ['req_time_hour','haversine'],
    ['req_time_hour','manhattan'],
    ['req_time_hour','bearing_array'],

    ['req_time_dayofweek','req_time_hour', 'haversine', 'recom_mode_0'],
    ['req_time_dayofweek','req_time_hour', 'manhattan', 'recom_mode_0'],
    ['req_time_dayofweek','req_time_hour', 'bearing_array', 'recom_mode_0'],

#     ['req_time_hour', 'haversine', 'recom_mode_0'],
#     ['req_time_hour', 'manhattan', 'recom_mode_0'],
#     ['req_time_hour', 'bearing_array', 'recom_mode_0'],

#     ['req_time_dayofweek', 'haversine', 'recom_mode_0'],
#     ['req_time_dayofweek', 'manhattan', 'recom_mode_0'],
#     ['req_time_dayofweek', 'bearing_array', 'recom_mode_0'],

    # ['req_time_dayofweek','req_time_hour','recom_mode_0'],
    # ['req_time_weekend', 'req_time_hour','recom_mode_0'],                    
        
    ['pid', 'manhattan'],
    ['pid', 'bearing_array'],
    ['pid', 'recom_mode_0'],
    
        
        

    ['req_time_dayofweek', 'req_time_hour', 'o'],
    ['req_time_dayofweek', 'req_time_hour', 'd'],
    ['req_time_dayofweek', 'req_time_hour', 'pid'],

    ['req_time_weekend', 'req_time_hour', 'o'],
    ['req_time_weekend', 'req_time_hour', 'd'],
    ['req_time_weekend', 'req_time_hour', 'pid'],
               

    ['city', 'recom_mode_0'],
    ['city', 'haversine'],
    ['city', 'manhattan'],
    ['city', 'bearing_array'],
    ['city', 'pid'],
    
    ['o', 'req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['o', 'req_time_weekofyear', 'req_time_dayofweek'],
    
    ['d', 'req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['d', 'req_time_weekofyear', 'req_time_dayofweek'],

    ['o', 'pid', 'req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['o', 'pid', 'req_time_weekofyear', 'req_time_dayofweek'],
    
    ['d', 'pid','req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['d', 'pid','req_time_weekofyear', 'req_time_dayofweek'],

    ['o', 'req_time_minute', 'req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['o', 'req_time_minute','req_time_weekofyear', 'req_time_dayofweek'],
    
    ['d', 'req_time_minute','req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['d', 'req_time_minute','req_time_weekofyear', 'req_time_dayofweek'],

    ['o', 'd', 'req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['o', 'd','req_time_weekofyear', 'req_time_dayofweek'],

    ['pid', 'req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    ['pid', 'req_time_weekofyear', 'req_time_dayofweek'],

    # ['pid', 'req_time_minute','req_time_weekofyear', 'req_time_dayofweek', 'req_time_hour'],
    # ['pid', 'req_time_minute','req_time_weekofyear', 'req_time_dayofweek'],

    # ['req_time_dayofweek', 'city'],
    # ['req_time_hour', 'city'],

#     ['req_time_dayofweek', 'req_time_hour', 'haversine'],
#     ['req_time_dayofweek', 'req_time_hour', 'manhattan'],
#     ['req_time_dayofweek', 'req_time_hour', 'bearing_array'],

#     ['req_time_weekend', 'req_time_hour', 'haversine'],
#     ['req_time_weekend', 'req_time_hour', 'manhattan'],
#     ['req_time_weekend', 'req_time_hour', 'bearing_array'],
 
        
    # ['req_time_dayofweek', 'req_time_hour', 'haversine', 'city'],
    # ['req_time_dayofweek', 'req_time_hour', 'manhattan', 'city'],
    # ['req_time_dayofweek', 'req_time_hour', 'bearing_array', 'city'],

    # ['req_time_hour','haversine', 'city'],
    # ['req_time_hour','manhattan', 'city'],
    # ['req_time_hour','bearing_array', 'city'],
        
    # ['req_time_dayofweek', 'req_time_hour', 'req_time_minute', 'o'],
    # ['req_time_dayofweek', 'req_time_hour', 'req_time_minute', 'd'],

    # ['plans_num', 'haversine'],
    # ['plans_num', 'manhattan'],
    # ['plans_num', 'bearing_array'],
    ]
    # cross_count_new = [ col + ['city'] for col in cross_count if 'city' not in col]
    # g = count_and_merge(data, cross_count_helper, cross_count_new)
    g = count_and_merge(data, cross_count_helper, cross_count)
    g['cross_count_sum'] = g.sum(axis = 1)
    # g.fillna(0, inplace = True)
    # svd = TruncatedSVD(n_components = n, n_iter = 50, random_state = 777)
    # df_svd = pd.DataFrame(svd.fit_transform(g.values))
    # df_svd.columns = ['svd_{}_{}'.format('counting_pca_', i) for i in range(n)]
    return g
