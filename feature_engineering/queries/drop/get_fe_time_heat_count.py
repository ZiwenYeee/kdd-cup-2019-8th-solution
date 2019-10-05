
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


def get_fe_time_heat_count(queries):
    df = queries[['sid']]
    g = get_fe_time_10_count(queries)
    df = df.merge(g, on = ['sid'], how = 'left')
    # g = get_fe_time_o_nunique(queries)
    # df = df.merge(g, on = ['sid'], how = 'left')
    # g = get_fe_time_d_nunique(queries)
    # df = df.merge(g, on = ['sid'], how = 'left')
    return df



def get_fe_time_d_nunique(queries):
    tr_q = queries.copy()
    tr_q['minute_10'] = tr_q['req_time_minute']//20
    gr = tr_q.groupby(['req_time_dayofweek', 'req_time_hour','minute_10'])[['d']].nunique()
    gr.columns = ['d_unique']
    merge_col = []
    merge_col = merge_col + ['d_unique']
    gr.reset_index(inplace = True)
    tr_q = tr_q.merge(gr, on = ['req_time_dayofweek', 'req_time_hour','minute_10'], how = 'left')
    gr = gr.groupby(['req_time_dayofweek', 'req_time_hour']).agg(
        {"d_unique":['mean','sum','std','min','max']})
    gr.columns = ["_".join(col) for col in gr.columns.ravel()]
    merge_col = merge_col + list(gr.columns)
    gr.reset_index(inplace = True)
    tr_q = tr_q.merge(gr, on = ['req_time_dayofweek', 'req_time_hour'], how = 'left')
    data = tr_q[['sid'] + merge_col]
    data.fillna(0, inplace = True)
    return data


def get_fe_time_o_nunique(queries):
    tr_q = queries.copy()
    tr_q['minute_10'] = tr_q['req_time_minute']//20
    gr = tr_q.groupby(['req_time_dayofweek', 'req_time_hour','minute_10'])[['o']].nunique()
    gr.columns = ['o_unique']
    merge_col = []
    merge_col = merge_col + ['o_unique']
    gr.reset_index(inplace = True)
    tr_q = tr_q.merge(gr, on = ['req_time_dayofweek', 'req_time_hour','minute_10'], how = 'left')
    gr = gr.groupby(['req_time_dayofweek', 'req_time_hour']).agg(
        {"o_unique":['mean','sum','std','min','max']})
    gr.columns = ["_".join(col) for col in gr.columns.ravel()]
    merge_col = merge_col + list(gr.columns)
    gr.reset_index(inplace = True)
    tr_q = tr_q.merge(gr, on = ['req_time_dayofweek', 'req_time_hour'], how = 'left')
    data = tr_q[['sid'] + merge_col]
    data.fillna(0, inplace = True)
    return data


def get_fe_time_10_count(queries):
    tr_q = queries.copy()
    tr_q['minute_10'] = tr_q['req_time_minute']//20
    gr = tr_q.groupby(['req_time_dayofweek', 'req_time_hour','minute_10'])[['sid']].count()
    gr.columns = ['10_minute_count']
    merge_col = []
    merge_col = merge_col + ['10_minute_count']
    gr.reset_index(inplace = True)
    tr_q = tr_q.merge(gr, on = ['req_time_dayofweek', 'req_time_hour','minute_10'], how = 'left')
    gr = gr.groupby(['req_time_dayofweek', 'req_time_hour']).agg(
        {"10_minute_count":['mean','sum','std','min','max']})
    gr.columns = ["_".join(col) for col in gr.columns.ravel()]
    merge_col = merge_col + list(gr.columns)
    gr.reset_index(inplace = True)
    tr_q = tr_q.merge(gr, on = ['req_time_dayofweek', 'req_time_hour'], how = 'left')
    data = tr_q[['sid'] + merge_col]
    data.fillna(0, inplace = True)
    return data
