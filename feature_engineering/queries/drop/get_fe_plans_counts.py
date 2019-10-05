

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from Basic_function import cross_count_helper
from Basic_function import timer

def get_fe_plans_counts(plans_split, queries):
    split = plans_split.copy()
    data = split[['sid']]
    split = split.merge(queries, on = ['sid'], how = 'left')
    # with timer("price part counter:"):
    #     group = price_counter(split, False)
    #     data[group.columns] = group

    with timer("mode part counter:"):
        group = mode_counter(split)
        data[group.columns] = group
    return data




def mode_counter(split_main):
    data = split_main[['sid','recom_price_0', 'recom_eta_0', 'recom_distance_0']]
    data['recom_price_0'] = data['recom_price_0'].apply(lambda x: x//100 if x > 0 else x)
    data['recom_eta_0'] = data['recom_eta_0']  * 60//60
    data['recom_distance_0'] = data['recom_distance_0'] * 1000//1000
    cross_count = [['recom_mode_0'],
                   ['recom_price_0'],
                   ['recom_eta_0'],
                   ['recom_distance_0'],
                   ['recom_price_0','recom_mode_0'],
                   ['recom_eta_0', 'recom_mode_0'],
                   ['recom_distance_0', 'recom_mode_0'],
                #    ['o', 'recom_eta_0'],
                #    ['d', 'recom_eta_0'],
                #    ['o','recom_mode_0'],
                #    ['d', 'recom_mode_0'],
                #    ['o', 'd', 'recom_mode_0'],
                  ]
    weight_list = []
    for col in cross_count:
        group = cross_count_helper(split_main, col, False)
        data[group.columns] = group
        weight_list.append(group.columns[0])
    data.drop(['recom_price_0'], axis = 1, inplace = True)
    data.rename({"recom_eta_0":"recom_eta_0_group",
                "recom_distance_0":"recom_distance_0_group"}, axis = 1, inplace = True)
    data.fillna(0, inplace = True)
    return data

def price_counter(split_main, weight = False):
    data = split_main[['sid','recom_price_0']]
    data['recom_price_0'] = data['recom_price_0']//100
    cross_count = [['recom_price_0'],
                   ['recom_mode_0','recom_price_0'],
                   ['o','recom_price_0'],
                   ['d', 'recom_price_0'],
                   ['o', 'd', 'recom_price_0'],
                   ['pid', 'recom_price_0'],
                   ['o', 'recom_mode_0', 'recom_price_0'],
                   ['d', 'recom_mode_0', 'recom_price_0'],
                   ['o','d', 'recom_mode_0', 'recom_price_0']
                  ]
    weight_list = []
    for col in cross_count:
        group = cross_count_helper(split_main, col, False)
        data[group.columns] = group
        weight_list.append(group.columns[0])
    if weight:
        for i in weight_list:
            data[i + "_weight_price"] = data[i] * data['recom_price_0']
    data.drop(['recom_price_0'], axis = 1, inplace = True)
    data.fillna(0, inplace = True)
    return data
