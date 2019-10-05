
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import gc
from Basic_function import timer
from plans.plans_split import plans_split
from plans.get_fe_diff_div import get_fe_diff_div
from Basic_function import get_datetime
from plans.get_fe_different_recomend import get_fe_different_recomend
from plans.get_fe_mode_feature import get_fe_mode_feature
def plans_split_feat(train, test, plans, test_plans, queries, test_queries):
    with timer("plans split main table:"):
        split_feat = plans_split(plans)
        split_feat_test = plans_split(test_plans)
        split_feat['sid'] = plans['sid']
        split_feat_test['sid'] = test_plans['sid']
        split_main = pd.concat([split_feat, split_feat_test]).reset_index(drop = True)
        plans_main = pd.concat([plans, test_plans]).reset_index(drop = True)
    with timer("get diff related feature"):
        diff_main = get_fe_diff_div(split_main)
        split_main[diff_main.columns] = diff_main
    # with timer("get plans time feature"):
    #     time_main = get_datetime(plans_main, 'plan_time')
    #     split_main[time_main.columns] = time_main
    with timer("get plans differnt type recommend:"):
        diff_main = get_fe_different_recomend(plans_main)
        split_main[diff_main.columns] = diff_main
    # with timer("get_fe_mode_feature:"):
    #     mode_main = get_fe_mode_feature(plans_main)
    #     split_main[mode_main.columns] = mode_main
    del plans_main;gc.collect()
    train.set_index(['sid'], inplace = True)
    test.set_index(['sid'], inplace = True)
    split_main.set_index(['sid'], inplace = True)
    train = train.join(split_main, how = 'left')
    test = test.join(split_main, how = 'left')
    train.reset_index(inplace = True)
    test.reset_index(inplace = True)
                          
    # train = pd.merge(train, split_main, on = ['sid'], how = 'left')
    # test = pd.merge(test, split_main, on = ['sid'], how = 'left')

    return train, test
