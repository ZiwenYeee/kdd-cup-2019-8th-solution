import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def get_fe_cnt_rank_ratio(train_plans_split):
    df_fe = pd.DataFrame([])
    col_mode_cnt = ['mode_cnt_' + str(i) for i in range(1, 12)]
    for col in col_mode_cnt:
        df_fe[col + '_ratio'] = \
            train_plans_split[col] / (7 - train_plans_split['cnt_999999'] // 4)
        df_fe[col + '_ratio'] = \
            train_plans_split[col] / (7 - train_plans_split['cnt_999999'] // 4)

    col_mode_rank = ['mode_rank_' + str(i) for i in range(1, 12)]
    for col in col_mode_rank:
        df_fe[col + '_ratio'] = \
            train_plans_split[col] / (7 - train_plans_split['cnt_999999'] // 4)
        df_fe[col + '_ratio'] = \
            train_plans_split[col] / (7 - train_plans_split['cnt_999999'] // 4)
    return df_fe
