import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool

from Basic_function import parallelize

def get_fe_different_recomend(plans):
    df_all = parallelize(plans, work_recomend)
    return df_all


def work_recomend(plans):
    def stat_func(x, var):
        a = []
        for j in x:
            if j[var] != "":
                a.append(j[var])
            else:
                a.append(0)
        return a
    data = pd.DataFrame([])
    for i in ['distance', 'eta', 'price']:
        data[i + "_list"] = plans['plans'].apply(lambda x: stat_func(x, i))
        data["max_" + i + "_stat"] = data[i + "_list"].apply(lambda x: max(x))
        data["min_" + i + "_stat"] = data[i + "_list"].apply(lambda x: min(x))
        data["std_" + i + "_stat"] = data[i + "_list"].apply(lambda x: np.std(x))
        data["mean_" + i + "_stat"] = data[i + "_list"].apply(lambda x: np.mean(x))
        # data["median_" + i + "_stat"] = data[i + "_list"].apply(lambda x: np.median(x))

        data[i + 'min_diff_max'] = data['max_' + i + '_stat'] - data['min_' + i + '_stat']
        data[i + 'min_div_max'] = data['max_' + i + '_stat'] /(0.01 + data['min_' + i + '_stat'])
        data.drop([i + "_list"], axis = 1, inplace = True)
        # for j in ['min', 'max']:
        #     data['mode_' + i + "_" + j] = plans['plans'].apply(lambda x: basic_func(x, i, j))
        #     data['mode_rank_' + i + "_" + j] = plans['plans'].apply(lambda x: basic_func_rank(x, i, j))
    return data
