import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def get_fe_od_count(queries, plan):
    df = queries[['sid']]
    plans = plan.copy()
    plans['recom_mode_0'] = plans['plans'].apply(lambda x: x[0]['transport_mode'])
    data = pd.merge(queries[['sid', 'o', 'd']], plans, on = ['sid'], how = 'left')
    data['od'] = data['o'] + '_' + data['d']
    data['values'] = 1.0
    od_mode_feat = pd.pivot_table(data,index=['od'],values=['values'],columns=['recom_mode_0'],aggfunc='sum')
    od_mode_feat.columns = ['od_transport_mode_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    df = pd.merge(data[['sid', 'od']], od_mode_feat, on = ['od'], how = 'left')
    df.drop(['od'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df
