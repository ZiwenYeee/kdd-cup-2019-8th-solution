import numpy as np
import pandas as pd
import time
from Basic_function import timer

def get_fe_count_plan(df, train_plan, test_plan):
    group = get_key_hour_day(train_plan, test_plan)
    df = pd.merge(df, group, on = ['sid'], how = 'left')
    group = get_key_day(train_plan, test_plan)
    df = pd.merge(df, group, on = ['sid'], how = 'left')
    return df

def get_key_day(train_plan, test_plan, ratio = True):
    plans = pd.concat([train_plan, test_plan])
    data = plans[['sid']]
    data['recom_mode_0'] = plans['plans'].apply(lambda x: x[0]['transport_mode'])
    data['plans_time'] = pd.to_datetime(plans['plan_time'])
    data['day'] = data.plans_time.dt.dayofweek
    data['values'] = 1
    data = get_fe_plans_by(data, ['day'], ratio)
    data.drop(['plans_time', 'day', 'values'], axis = 1, inplace = True)
    return data


def get_key_hour_day(train_plan, test_plan, ratio = True):
    plans = pd.concat([train_plan, test_plan])
    data = plans[['sid']]
    data['recom_mode_0'] = plans['plans'].apply(lambda x: x[0]['transport_mode'])
    data['plans_time'] = pd.to_datetime(plans['plan_time'])
    data['day'] = data.plans_time.dt.dayofweek
    data['hour'] = data.plans_time.dt.hour
    data['values'] = 1
    data = get_fe_plans_by(data, ['day', 'hour'], ratio)
    data.drop(['plans_time', 'day', 'hour', 'values'], axis = 1, inplace = True)
    return data



def get_fe_plans_by(data, key, ratio = True):
    temp = pd.pivot_table(data, index = key, values = ['values'], columns = ['recom_mode_0'], aggfunc = 'sum')
    prefix = "_".join(key)
    temp.columns = [prefix + "_" + str(col[1]) for col in temp.columns.ravel()]
    temp.fillna(0, inplace = True)
    if ratio:
        ratio_col = temp.columns
        temp[prefix + "_" + 'sum'] = temp.sum(axis = 1)
        for i in ratio_col:
            temp[i + "_ratio"] = temp[i]/temp[prefix + "_" + 'sum']
    temp.reset_index(inplace = True)
    data = pd.merge(data, temp, on = key, how = 'left')
    return data
