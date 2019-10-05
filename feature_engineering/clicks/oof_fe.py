import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from Basic_function import timer
from sklearn.model_selection import KFold, StratifiedKFold




def oof_fe_by(clicks, train_queries, test_queries, train_plans, test_plans, key, op = 'cnt'):
    if key == 'get_idx_dist':
        train_idx = get_idx_dist(train_queries, train_plans)
        test_idx = get_idx_dist(test_queries, test_plans)
    if key == 'get_idx_dist_day_hour':
        train_dist = get_idx_dist(train_queries, train_plans)
        test_dist = get_idx_dist(test_queries, test_plans)
        train_time = get_idx_day_hour(train_plans)
        test_time = get_idx_day_hour(test_plans)
        train_idx = pd.merge(train_dist, train_time, on = ['sid'], how = 'left')
        test_idx = pd.merge(test_dist, test_time, on = ['sid'], how = 'left')
    if key == 'get_idx_dist_hour':
        train_dist = get_idx_dist(train_queries, train_plans)
        test_dist = get_idx_dist(test_queries, test_plans)
        train_time = get_idx_hour(train_plans)
        test_time = get_idx_hour(test_plans)
        train_idx = pd.merge(train_dist, train_time, on = ['sid'], how = 'left')
        test_idx = pd.merge(test_dist, test_time, on = ['sid'], how = 'left')
    if key == 'get_idx_day_hour':
        train_idx = get_idx_day_hour(train_plans)
        test_idx = get_idx_day_hour(test_plans)
    if key == 'get_idx_hour':
        train_idx = get_idx_hour(train_plans)
        test_idx = get_idx_hour(test_plans)
    if key == 'get_idx_day':
        train_idx = get_idx_day(train_plans)
        test_idx = get_idx_day(test_plans)
    if key == 'get_idx_grid_o':
        train_idx = get_idx_grid(train_queries, train_plans, 'o')
        test_idx = get_idx_grid(test_queries, test_plans, 'o')
    if key == 'get_idx_grid_d':
        train_idx = get_idx_grid(train_queries, train_plans, 'd')
        test_idx = get_idx_grid(test_queries, test_plans, 'd')
    if key == 'get_idx_pid':
        train_idx = get_idx_pid(train_plans, train_queries, profiles)
        test_idx = get_idx_pid(test_plans, test_queries, profiles)
    tr = pd.DataFrame()
    ts = pd.DataFrame()
    folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=0)
    clicks_index = pd.merge(train_idx[['sid']], clicks[['sid', 'click_mode']], on = ['sid'], how = 'left')
    clicks_index.fillna(0, inplace = True)
    for n_fold, (tr_idx, ts_idx) in enumerate(folds.split(clicks_index, clicks_index['click_mode'])):
        sys.stdout.write('{},'.format(n_fold))
        X_train, X_test = clicks_index.iloc[tr_idx], clicks_index.iloc[ts_idx]
        X_test = pd.merge(X_test[['sid']], train_idx, on = ['sid'], how = 'left')
        tr_index = X_test[['sid']]
        ts_index = test_idx[['sid']]
        train_fe, test_fe = get_fe_cnt_ratio_by(X_train, train_idx, test_idx, op)
        tr_index = pd.merge(tr_index, train_fe, on = ['sid'], how = 'left')
        tr = tr.append(tr_index)
        ts = ts.append(test_fe)
    print('')
    ts = ts.groupby(['sid']).mean()
    tr = pd.merge(train_idx[['sid']], tr, on = ['sid'], how = 'left')
    ts = pd.merge(test_idx[['sid']], ts, on = ['sid'], how = 'left')
    return tr,ts


def Descartes(data, col_list):
    q = pd.DataFrame(data[col_list[0]].unique(), columns = [col_list[0]])
    q['key'] = 0
    for col in [col for col in col_list if col not in col_list[0]]:
        q1 = pd.DataFrame(data[col].unique(), columns = [col])
        q1['key'] = 0
        q = pd.merge(q, q1, on = ['key'])
    q.drop(['key'], axis = 1, inplace = True)
    return q
def get_fe_cnt_ratio_by(click, train_idx, test_idx, op = 'cnt'):
    train_fe, test_fe = pd.DataFrame([]), pd.DataFrame([])
    col_idx = [col for col in train_idx if col != 'sid']
    c = pd.DataFrame(click['click_mode'].unique(), columns = ['click_mode'])
    d = Descartes(train_idx, col_idx)
    c['key'] = 0
    d['key'] = 0
    main = pd.merge(c, d, on =['key'])
    main.drop(['key'], axis = 1, inplace = True)
    group = pd.merge(click[['sid', 'click_mode']], train_idx, on = ['sid'], how = 'left')
    group = group.groupby(col_idx + ['click_mode'])[['sid']].count()
    group.columns = ['click_' + "_".join(col_idx) + "_" + op]
    group.reset_index(inplace = True)
    main = pd.merge(main, group, on = col_idx + ['click_mode'], how = 'left')
    main.set_index(col_idx + ['click_mode'], inplace = True)
    main = main.unstack()
    main.columns = [col[0] + "_" + str(int(col[1])) for col in main.columns]
    main.fillna(0, inplace = True)
    if op == 'ratio':
        ratio_cols = main.columns
        main_sum = main[ratio_cols].sum(axis = 1)
        for i in ratio_cols:
            main[i] = main[i]/main_sum
    main.reset_index(inplace = True)
    train_fe = pd.merge(train_idx, main, on = col_idx, how = 'left')
    test_fe = pd.merge(test_idx, main, on = col_idx, how = 'left')
    train_fe.drop(col_idx, axis = 1, inplace = True)
    test_fe.drop(col_idx, axis = 1, inplace = True)
    return train_fe, test_fe

def get_idx_day_hour(data):
    df_idx = pd.DataFrame([])
    s = pd.to_datetime(data['plan_time'])
    df_idx['sid'] = data['sid']
    df_idx['wday'] = s.dt.dayofweek
    df_idx['hour'] = s.dt.hour
    return df_idx

def get_idx_day(data):
    df_idx = pd.DataFrame([])
    s = pd.to_datetime(data['plan_time'])
    df_idx['sid'] = data['sid']
    df_idx['day'] = s.dt.dayofweek
    return df_idx

def get_idx_hour(data):
    df_idx = pd.DataFrame([])
    s = pd.to_datetime(data['plan_time'])
    df_idx['sid'] = data['sid']
    df_idx['hour'] = s.dt.hour
    return df_idx

def get_idx_pid(plans, queries, profiles):
    pro = profiles.copy()
    pro['p_sum'] = pro[[col for col in pro.columns if col not in ['pid']]].sum(axis = 1)
    main = pd.merge(plans[['sid']], queries[['sid', 'pid']], on = ['sid'], how = 'left')
    main = pd.merge(main, pro[['pid', 'p_sum']], on = ['pid'], how = 'left')
    main.p_sum.fillna(0, inplace = True)
    main = main[['sid', 'p_sum']]
    return main

def get_idx_dist(query, plans):
    queries = query.copy()
    queries['o_x'] = queries['o'].apply(lambda x: float(x.split(',')[0]))
    queries['o_y'] = queries['o'].apply(lambda x: float(x.split(',')[1]))
    queries['d_x'] = queries['d'].apply(lambda x: float(x.split(',')[0]))
    queries['d_y'] = queries['d'].apply(lambda x: float(x.split(',')[1]))
    df = pd.merge(plans[['sid']], queries, on = ['sid'], how = 'left')
    df['dist_grid'] = abs(df['o_x'] - df['d_x']) + abs(df['o_y'] - df['d_y'])
    df['dist_grid_idx'] = (df['dist_grid'] * 111111) // 300
    return df[['sid','dist_grid_idx']]

def get_idx_grid(query, plans, od):
    od_x_min = 115.4
    od_y_min = 39.46
    od_x_max = 117.48
    od_y_max = 40.97
    queries = query.copy()
    queries['o_x'] = queries['o'].apply(lambda x: float(x.split(',')[0]))
    queries['o_y'] = queries['o'].apply(lambda x: float(x.split(',')[1]))
    queries['d_x'] = queries['d'].apply(lambda x: float(x.split(',')[0]))
    queries['d_y'] = queries['d'].apply(lambda x: float(x.split(',')[1]))
    df = pd.merge(plans[['sid']], queries, on = ['sid'], how = 'left')
    if od == 'o':
        df['o_x_id'] = abs(df['o_x'] - od_x_min) * 111111 // 300
        df['o_y_id'] = abs(df['o_y'] - od_y_min) * 111111 // 300
        df['o_x_y_id'] = df['o_x_id'] * 100 + df['o_y_id']
        return df[['sid','o_x_y_id']]
    else:
        df['d_x_id'] = abs(df['d_x'] - od_x_min) * 111111 // 300
        df['d_y_id'] = abs(df['d_y'] - od_y_min) * 111111 // 300
        df['d_x_y_id'] = df['d_x_id'] * 100 + df['d_y_id']
        return df[['sid','d_x_y_id']]
