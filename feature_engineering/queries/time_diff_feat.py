
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time


def time_diff_feat(queries, plans):
    df = pd.merge(queries[['sid', 'req_time']], plans[['sid', 'plan_time']], on = ['sid'], how = 'left')
    df['req_time'] = pd.to_datetime(df['req_time'])
    df['plan_time'] = pd.to_datetime(df['plan_time'])
    df['diff_time'] = df['req_time'] - df['plan_time']
    df['diff_time'] = df['diff_time'].dt.total_seconds()
    return df[['diff_time']]
