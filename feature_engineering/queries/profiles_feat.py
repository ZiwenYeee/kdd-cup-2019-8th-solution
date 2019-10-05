from sklearn.decomposition import TruncatedSVD
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def df_to_svd(df, name, n = 10):
    svd = TruncatedSVD(n_components = n, n_iter = 50, random_state = 777)

    df_svd = pd.DataFrame(svd.fit_transform(df.values))
    df_svd.columns = ['svd_{}_{}'.format(name, i) for i in range(n)]
    return df_svd

def get_fe_profiles(df):
    df_fe = df[['pid']]
    col_p = [col for col in df if col != 'pid']
    df_fe['p_sum'] = df[col_p].sum(axis = 1)

    col_p = ['p' + str(i) for i in range(0, 10)]
    df_fe['p_sum_00_10'] = df[col_p].sum(axis = 1)

    col_p = ['p' + str(i) for i in range(10,26)]
    df_fe['p_sum_10_26'] = df[col_p].sum(axis = 1)

    col_p = ['p' + str(i) for i in range(26,38)]
    df_fe['p_sum_26_38'] = df[col_p].sum(axis = 1)

    col_p = ['p' + str(i) for i in range(38,66)]
    df_fe['p_sum_38_66'] = df[col_p].sum(axis = 1)


    df_svd = df_to_svd(df[['p' + str(i) for i in range(66)]], name = 'p', n = 10)
    df_fe[df_svd.columns] = df_svd
    # df_fe[col_p] = df[col_p]
    return df_fe
