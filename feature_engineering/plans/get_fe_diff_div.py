import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def get_fe_diff_div(df):
    df_fe = pd.DataFrame([])

    top_m = 2
    for i in range(1, top_m):
        df_fe['diff_eta_{}_{}'.format(0, i)] = df['recom_eta_{}'.format(0)] - df['recom_eta_{}'.format(i)]
        df_fe['diff_distance_{}_{}'.format(0, i)] = df['recom_distance_{}'.format(0)] - df['recom_distance_{}'.format(i)]
        df_fe['diff_price_{}_{}'.format(0, i)] = df['recom_price_{}'.format(0)] - df['recom_price_{}'.format(i)]

        df_fe['div_eta_{}_{}'.format(0, i)] = \
        df['recom_eta_{}'.format(0)] / (df['recom_eta_{}'.format(i)] + 0.01)
        df_fe['div_distance_{}_{}'.format(0, i)] = \
        df['recom_distance_{}'.format(0)] / (df['recom_distance_{}'.format(i)] + 0.01)
        df_fe['div_price_{}_{}'.format(0, i )] = \
        df['recom_price_{}'.format(0)] / (df['recom_price_{}'.format(i)] + 0.01)

        df_fe['div_price_eta_{}_{}'.format(i, i)] = \
        df['recom_price_{}'.format(i)]/(df['recom_eta_{}'.format(i)] + 0.01)
        df_fe['diff_price_distance_{}_{}'.format(i, i)] = \
        df['recom_distance_{}'.format(i)]/(0.01 + df['recom_price_{}'.format(i)])
        df_fe['diff_distance_eta_{}_{}'.format(i, i)] = \
        df['recom_distance_{}'.format(i)]/(0.01 + df['recom_eta_{}'.format(i)])



    return df_fe
