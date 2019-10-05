import numpy as np
import pandas as pd

from Basic_function import parallelize

from multiprocessing import cpu_count, Pool

def plans_split(plans):
    df_all = parallelize(plans, work_split)

    return df_all


def work_split(plans):
    def basic_func(x):
        a = np.zeros((1,12))
        for j in x:
            a[0,j['transport_mode']] += 1
        return a.tolist()[0]

    def advanced_func(x):
        a = np.zeros((1,12))
        c = 1
        for j in x:
            a[0,j['transport_mode']] = c
            c += 1
        return a.tolist()[0]

    data = pd.DataFrame([])
    for i in range(5):
        data['recom_mode_' + str(i)] = plans['plans'].apply(lambda x: x[i]['transport_mode'] if len(x) > i else 0)
        data['recom_price_' + str(i)] = plans['plans'].apply(lambda x: x[i]['price'] if len(x) > i else 0)
        data['recom_eta_' + str(i)] = plans['plans'].apply(lambda x: x[i]['eta']/60 if len(x) > i else 0)
        data['recom_distance_' + str(i)] = plans['plans'].apply(lambda x: x[i]['distance']/1000 if len(x) > i else 0)
        data['recom_price_' + str(i)].replace("", 0, inplace = True)
        data['recom_price_' + str(i)] = \
        np.where(data['recom_mode_' + str(i)] == 3, -2, data['recom_price_' + str(i)])
        # data['recom_price_' + str(i)] = \
        # np.where(data['recom_mode_' + str(i)] == 6, -2, data['recom_price_' + str(i)])
        data['recom_price_' + str(i)] = \
        np.where(data['recom_mode_' + str(i)] == 5, -1, data['recom_price_' + str(i)])
        # if i >= 1:
        #     data['recom_mode_' + str(i)] = \
        #     np.where(data['recom_mode_' + str(i)] == -1, data['recom_mode_' + str(i - 1)], data['recom_mode_' + str(i)])
    data['plan_len'] = plans['plans'].apply(lambda x: len(x))
    data['flag_list'] = plans['plans'].apply(lambda x: basic_func(x))
    data['rank_list'] = plans['plans'].apply(lambda x: advanced_func(x))

    for i in range(1,12):
        # data['mode_flag_' + str(i)] = data.flag_list.apply(lambda x: 1 if x[i] > 0 else 0)
        # data['mode_flag_ratio_' + str(i)] = data['mode_flag_' + str(i)]/(0.1 + data['plan_len'])
        data['mode_cnt_' + str(i)] = data.flag_list.apply(lambda x: x[i])
        data['mode_cnt_ratio_' + str(i)] = data['mode_cnt_' + str(i)]/(0.1 + data['plan_len'])
        data['mode_rank_' + str(i)] = data.rank_list.apply(lambda x: x[i])
    data.drop(['flag_list', 'rank_list'], axis = 1, inplace = True)

    return data
