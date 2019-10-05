import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def get_fe_pid_count(queries, plan):
    df = queries[['sid']]
    plans = plan.copy()
    plans['recom_mode_0'] = plans['plans'].apply(lambda x: x[0]['transport_mode'])
    data = pd.merge(queries[['sid', 'pid']], plans, on = ['sid'], how = 'left')
    data.pid.fillna(-1, inplace = True)
    data['values'] = 1.0
    od_mode_feat = pd.pivot_table(data,index=['pid'],values=['values'],columns=['recom_mode_0'],aggfunc='sum')
    od_mode_feat.columns = ['pid_transport_mode_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    df = pd.merge(data[['sid', 'pid']], od_mode_feat, on = ['pid'], how = 'left')
    df.drop(['pid'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df
