import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def get_fe_o_d(queries):
    o_f = queries.groupby(['o'])['d'].nunique().reset_index()
    o_f.columns = ['o','d_nunique']
    d_f = queries.groupby(['d'])['o'].nunique().reset_index()
    d_f.columns = ['d','o_nunique']
    data = pd.merge(queries, o_f, on = 'o', how = 'left')
    data = pd.merge(data, d_f, on = ['d'], how = 'left')
    data = data[['sid','d_nunique','o_nunique']]
    return data
